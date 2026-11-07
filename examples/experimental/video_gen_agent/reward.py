"""The video-generation env's reward step: ``--custom-rm-path examples.experimental.video_gen_agent.reward.reward_func``
with ``--video-reward-pool <name>``, a pool from ``--reward-pool-config``.

The env records the submitted clip under ``metadata["video_gen"]``; this scores it in the rollout process, where
the reward pool lives. Every kept sample of one episode carries the same clip, so each distinct clip is scored once.
"""

import hashlib
import logging
from pathlib import Path

from miles.rollout.rm_hub.video.reward import VideoRewardConfig
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class StubClipScorer:
    """A reward pool scorer for CPU smoke runs, with the pool's ``score_batch(outputs, prompts)`` shape.

    The score is a fixed function of the clip's bytes in [0, 1), so equal clips score equally and a test can
    predict it; it says nothing about quality.
    """

    def score_batch(self, outputs: list[str], prompts: list[str]) -> list[float]:
        return [stub_score(output) for output in outputs]


def stub_score(path: str) -> float:
    return int.from_bytes(hashlib.sha256(Path(path).read_bytes()).digest()[:8], "big") / 2**64


async def reward_func(args, samples: list[Sample], **pools) -> list[float | None]:
    """Score each sample's submitted clip against the user's request on ``--video-reward-pool``.

    A sample without a submitted clip scores 0: the policy ran out of turns or never submitted. A pool failure
    scores None, so the sample is dropped rather than trained on as a failure.
    """
    pool_kwarg = f"{args.video_reward_pool}_pool"
    assert (
        pool_kwarg in pools
    ), f"--video-reward-pool {args.video_reward_pool} is not a declared pool, got {sorted(pools)}"
    submitted = [(sample.metadata.get("video_gen") or {}).get("submitted") for sample in samples]
    requests = {
        s["path"]: sample.metadata["video_gen"]["request"] for s, sample in zip(submitted, samples, strict=True) if s
    }
    scores = {}
    if requests:
        try:
            scores = dict(
                zip(requests, await pools[pool_kwarg].score(list(requests), list(requests.values())), strict=True)
            )
        except Exception:
            logger.exception("scoring %d submitted clips failed", len(requests))
            scores = dict.fromkeys(requests)
    return [scores[s["path"]] if s else 0.0 for s in submitted]


reward_func.config_class = VideoRewardConfig
