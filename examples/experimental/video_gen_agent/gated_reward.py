"""Phase 1's gated video reward: ``--custom-rm-path examples.experimental.video_gen_agent.gated_reward.reward_func``,
and for training ``--custom-reward-post-process-path examples.experimental.video_gen_agent.gated_reward.group_normalized_rewards``.

Each signal term is a reward pool's score of the submitted clip against the request (HPSv3 on frames, VQAScore on the
request, PickScore, ...), mapped to [0, 1] by a fixed affine map fit on the calibration clips. Gates act on every
term before anything else sees it:

  motion     every term 0 when the motion judge pool's score is below ``--video-reward-motion-threshold``
  ocr        every term times ``--video-reward-ocr-factor`` when the OCR pool counts text the request does not ask for
  adherence  off by default; every term 0 when the adherence judge pool's score is below its threshold

``reward_func`` gives each sample the mean of its gated terms, which eval reports. ``group_normalized_rewards``
trains on the sum of each term standardized over the episodes of a prompt's group, then hands that sum to Miles'
own GRPO normalization, which standardizes it again; with one term the two agree with plain GRPO on that term.

Each sample records every score, term and gate under ``metadata["video_reward"]``, and
``multiturn.episode_metrics`` logs how often each gate fires. An episode without a submitted clip scores 0 on every
term; any pool failing for a clip scores it None, so its group is dropped rather than trained on as a failure.
"""

import asyncio
import logging

import torch

from miles.ray.rollout.train_data_conversion import _normalize_rewards_by_rollout
from miles.rollout.reward_shaping import reward_group_segments, rollout_segments
from miles.utils.args.schema import A, Arg, BaseConfig
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

RECORD_KEY = "video_reward"
GATES = ("motion", "ocr", "adherence")


class GatedVideoRewardConfig(BaseConfig):
    video_reward_signal_pools: A[
        list[str], Arg(type_parser=str, nargs="+", help="The reward pools whose scores of the clip are the terms.")
    ]
    video_reward_signal_ranges: A[
        list[float],
        Arg(
            type_parser=float,
            nargs="+",
            help="Per signal pool, the scores that map to 0 and 1, fit on the calibration clips; outside is clipped.",
        ),
    ]
    video_reward_motion_pool: A[str, Arg(help="A judge pool weighting only the motion axis.")] = "motion"
    video_reward_motion_threshold: A[float, Arg(help="A motion score below this zeroes every term.")] = 0.4
    video_reward_ocr_pool: A[str, Arg(help="A pool counting text in the clip that the request does not ask for.")] = (
        "ocr"
    )
    video_reward_ocr_factor: A[float, Arg(help="Every term's factor when the OCR pool counts any such text.")] = 0.5
    video_reward_adherence_pool: A[
        str | None, Arg(help="A judge pool weighting only prompt adherence; unset leaves the gate off.")
    ] = None
    video_reward_adherence_threshold: A[float, Arg(help="An adherence score below this zeroes every term.")] = 0.0


async def reward_func(args, samples: list[Sample], **pools) -> list[float | None]:
    names = _pool_names(args)
    missing = sorted(name for name in names.values() if f"{name}_pool" not in pools)
    assert not missing, f"reward pools {missing} are not declared, got {sorted(pools)}"
    submitted = [(sample.metadata.get("video_gen") or {}).get("submitted") for sample in samples]
    requests = {
        s["path"]: sample.metadata["video_gen"]["request"] for s, sample in zip(submitted, samples, strict=True) if s
    }
    records = await _score(args, requests, {part: pools[f"{name}_pool"] for part, name in names.items()})
    unsubmitted = {"submitted": False, "terms": dict.fromkeys(args.video_reward_signal_pools, 0.0), "reward": 0.0}
    for sample, s in zip(samples, submitted, strict=True):
        sample.metadata[RECORD_KEY] = records[s["path"]] if s else unsubmitted
    return [sample.metadata[RECORD_KEY]["reward"] for sample in samples]


def group_normalized_rewards(args, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """The raw rewards, and per sample the sum of its episode's terms each standardized within its prompt's group,
    normalized again as Miles' GRPO normalizes any reward."""
    raw = [sample.get_reward_value(args) for sample in samples]
    summed = [0.0] * len(samples)
    for rows in reward_group_segments(args, samples, None):
        episodes = rollout_segments(samples, rows)
        for pool in args.video_reward_signal_pools:
            terms = torch.tensor([samples[siblings[0]].metadata[RECORD_KEY]["terms"][pool] for siblings in episodes])
            centered = terms - terms.mean()
            if len(terms) > 1 and terms.std() > 0:
                centered = centered / terms.std()
            for siblings, value in zip(episodes, centered.tolist(), strict=True):
                for row in siblings:
                    summed[row] += value
    if args.advantage_estimator in ("grpo", "gspo", "reinforce_plus_plus_baseline") and args.rewards_normalization:
        return raw, _normalize_rewards_by_rollout(args, samples, summed, None)
    return raw, summed


def _pool_names(args) -> dict[str, str]:
    pools, ranges = args.video_reward_signal_pools, args.video_reward_signal_ranges
    assert pools, "--video-reward-signal-pools names no pool"
    assert len(ranges) == 2 * len(pools), f"--video-reward-signal-ranges needs a low and a high per pool, got {ranges}"
    assert all(
        low < high for low, high in zip(ranges[::2], ranges[1::2], strict=True)
    ), f"a signal range is empty: {ranges}"
    names = {f"signal:{pool}": pool for pool in pools}
    names |= {"motion": args.video_reward_motion_pool, "ocr": args.video_reward_ocr_pool}
    if args.video_reward_adherence_pool is not None:
        names["adherence"] = args.video_reward_adherence_pool
    return names


async def _score(args, requests: dict[str, str], pools: dict) -> dict[str, dict]:
    """Each distinct clip's record: every pool's score, the gated terms, which gates fired, and the reward."""
    if not requests:
        return {}
    paths, texts = list(requests), list(requests.values())
    try:
        scores = await asyncio.gather(*(pool.score(paths, texts) for pool in pools.values()))
    except Exception:
        logger.exception("scoring %d submitted clips failed", len(paths))
        return {path: {"submitted": True, "reward": None} for path in paths}
    return {
        path: clip_record(args, dict(zip(pools, parts, strict=True)))
        for path, *parts in zip(paths, *scores, strict=True)
    }


def clip_record(args, scores: dict[str, float | None]) -> dict:
    record = {"submitted": True, "scores": scores}
    if any(score is None for score in scores.values()):
        return record | {"reward": None}
    fired = {
        "motion": scores["motion"] < args.video_reward_motion_threshold,
        "ocr": scores["ocr"] > 0,
        "adherence": "adherence" in scores and scores["adherence"] < args.video_reward_adherence_threshold,
    }
    factor = 0.0 if fired["motion"] or fired["adherence"] else 1.0
    if fired["ocr"]:
        factor *= args.video_reward_ocr_factor
    ranges = args.video_reward_signal_ranges
    terms = {
        pool: factor * min(max((scores[f"signal:{pool}"] - low) / (high - low), 0.0), 1.0)
        for pool, low, high in zip(args.video_reward_signal_pools, ranges[::2], ranges[1::2], strict=True)
    }
    return record | {"terms": terms, "fired": fired, "reward": sum(terms.values()) / len(terms)}


reward_func.config_class = GatedVideoRewardConfig
group_normalized_rewards.config_class = GatedVideoRewardConfig
