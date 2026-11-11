"""Phase 1's gated reward: terms, gates, records, and the per-term group standardization before GRPO's."""

from types import SimpleNamespace

import pytest
import torch
from examples.experimental.video_gen_agent.gated_reward import RECORD_KEY, group_normalized_rewards, reward_func
from examples.experimental.video_gen_agent.multiturn.episode_metrics import episode_metrics

from miles.ray.rollout.train_data_conversion import _normalize_rewards_by_rollout
from miles.utils.types import Sample


class _Pool:
    """A reward pool scoring each clip from a table; it records every call."""

    def __init__(self, table: dict[str, float | None] | None = None, *, broken: bool = False):
        self.table, self.broken, self.calls = table or {}, broken, []

    async def score(self, paths, requests):
        self.calls.append((list(paths), list(requests)))
        if self.broken:
            raise RuntimeError("pool down")
        return [self.table.get(path) for path in paths]


def _args(**overrides):
    values = dict(
        video_reward_signal_pools=["hps", "ur"],
        video_reward_signal_ranges=[0.0, 10.0, 1.0, 5.0],
        video_reward_motion_pool="motion",
        video_reward_motion_threshold=0.4,
        video_reward_ocr_pool="ocr",
        video_reward_ocr_factor=0.5,
        video_reward_adherence_pool=None,
        video_reward_adherence_threshold=0.0,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        reward_key=None,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
    )
    return SimpleNamespace(**(values | overrides))


def _sample(path: str | None, request: str = "a fox", **fields) -> Sample:
    submitted = {"path": path} if path else None
    return Sample(metadata={"video_gen": {"request": request, "submitted": submitted}, "session_id": path}, **fields)


def _pools(**tables):
    defaults = {"hps": {}, "ur": {}, "motion": {}, "ocr": {}}
    return {f"{name}_pool": _Pool(table) for name, table in (defaults | tables).items()}


async def test_the_reward_is_the_mean_of_the_mapped_terms_and_is_recorded():
    pools = _pools(hps={"c": 7.5}, ur={"c": 6.0}, motion={"c": 0.9}, ocr={"c": 0.0})
    sample = _sample("c")

    assert await reward_func(_args(), [sample], **pools) == [pytest.approx((0.75 + 1.0) / 2)]
    record = sample.metadata[RECORD_KEY]
    assert record["terms"] == {"hps": 0.75, "ur": 1.0}
    assert record["scores"] == {"signal:hps": 7.5, "signal:ur": 6.0, "motion": 0.9, "ocr": 0.0}
    assert record["fired"] == {"motion": False, "ocr": False, "adherence": False}


@pytest.mark.parametrize(
    "motion, ocr, adherence, reward, fired",
    [
        (0.39, 0.0, None, 0.0, {"motion": True, "ocr": False, "adherence": False}),
        (0.4, 2.0, None, 0.25, {"motion": False, "ocr": True, "adherence": False}),
        (0.4, 0.0, 0.1, 0.0, {"motion": False, "ocr": False, "adherence": True}),
        (0.4, 0.0, 0.3, 0.5, {"motion": False, "ocr": False, "adherence": False}),
    ],
)
async def test_each_gate_acts_on_every_term(motion, ocr, adherence, reward, fired):
    args = _args(video_reward_adherence_pool="adh", video_reward_adherence_threshold=0.2)
    pools = _pools(hps={"c": 5.0}, ur={"c": 3.0}, motion={"c": motion}, ocr={"c": ocr}) | {
        "adh_pool": _Pool({"c": 1.0 if adherence is None else adherence})
    }
    sample = _sample("c")

    assert await reward_func(args, [sample], **pools) == [reward]
    assert sample.metadata[RECORD_KEY]["fired"] == fired
    assert sample.metadata[RECORD_KEY]["terms"] == {"hps": reward, "ur": reward}


async def test_the_adherence_gate_is_off_by_default_and_its_pool_is_not_called():
    pools = _pools(hps={"c": 5.0}, ur={"c": 3.0}, motion={"c": 1.0}, ocr={"c": 0.0}) | {"adh_pool": _Pool()}

    await reward_func(_args(), [_sample("c")], **pools)

    assert pools["adh_pool"].calls == []


async def test_scores_outside_the_range_are_clipped():
    pools = _pools(hps={"c": 12.0}, ur={"c": 0.0}, motion={"c": 1.0}, ocr={"c": 0.0})
    sample = _sample("c")

    await reward_func(_args(), [sample], **pools)

    assert sample.metadata[RECORD_KEY]["terms"] == {"hps": 1.0, "ur": 0.0}


async def test_each_distinct_clip_is_scored_once_against_its_request_and_no_submission_scores_0():
    pools = _pools(hps={"c": 5.0}, ur={"c": 3.0}, motion={"c": 1.0}, ocr={"c": 0.0})
    samples = [_sample("c", "a fox"), _sample("c", "a fox"), _sample(None)]

    rewards = await reward_func(_args(), samples, **pools)

    assert rewards == [0.5, 0.5, 0.0]
    for pool in pools.values():
        assert pool.calls == [(["c"], ["a fox"])]
    assert samples[2].metadata[RECORD_KEY] == {"submitted": False, "terms": {"hps": 0.0, "ur": 0.0}, "reward": 0.0}


@pytest.mark.parametrize("broken", ["hps_pool", "motion_pool"])
async def test_a_failing_or_missing_score_drops_the_sample(broken):
    pools = _pools(hps={"c": 5.0, "d": 5.0}, ur={"c": 3.0, "d": 3.0}, motion={"c": 1.0}, ocr={"c": 0.0, "d": 0.0})
    pools[broken].broken = True

    assert await reward_func(_args(), [_sample("c"), _sample(None)], **pools) == [None, 0.0]
    pools[broken].broken = False
    assert await reward_func(_args(), [_sample("d")], **pools) == [None]


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(video_reward_signal_ranges=[0.0, 1.0]), "a low and a high per pool"),
        (dict(video_reward_signal_ranges=[1.0, 1.0, 1.0, 5.0]), "a signal range is empty"),
        (dict(video_reward_motion_pool="judge"), "reward pools \\['judge'\\] are not declared"),
    ],
)
async def test_a_bad_config_is_refused(overrides, message):
    with pytest.raises(AssertionError, match=message):
        await reward_func(_args(**overrides), [_sample("c")], **_pools())


def _scored(terms: dict[str, float], group: int, episode: int, branches: int = 1) -> list[Sample]:
    record = {"submitted": True, "terms": terms, "reward": sum(terms.values()) / len(terms)}
    return [
        Sample(
            group_index=group,
            index=episode,
            rollout_id=episode,
            reward=record["reward"],
            metadata={RECORD_KEY: record},
        )
        for _ in range(branches)
    ]


def test_each_term_is_standardized_within_its_group_then_grpo_normalizes_the_sum():
    samples = [
        *_scored({"hps": 0.2, "ur": 1.0}, group=0, episode=0, branches=2),
        *_scored({"hps": 0.8, "ur": 0.0}, group=0, episode=1),
        *_scored({"hps": 0.6, "ur": 0.5}, group=0, episode=2),
        *_scored({"hps": 0.1, "ur": 0.3}, group=1, episode=3),
        *_scored({"hps": 0.1, "ur": 0.9}, group=1, episode=4),
    ]
    args = _args()

    raw, normalized = group_normalized_rewards(args, samples)

    assert raw == [s.reward for s in samples]

    def z(values):
        t = torch.tensor(values)
        return ((t - t.mean()) / t.std()).tolist() if t.std() > 0 else (t - t.mean()).tolist()

    group_0 = [a + b for a, b in zip(z([0.2, 0.8, 0.6]), z([1.0, 0.0, 0.5]), strict=True)]
    group_1 = [a + b for a, b in zip([0.0, 0.0], z([0.3, 0.9]), strict=True)]
    summed = [group_0[0], *group_0, *group_1]
    assert normalized == pytest.approx(_normalize_rewards_by_rollout(args, samples, summed, None))
    assert normalized[0] == normalized[1]
    assert normalized != pytest.approx(_normalize_rewards_by_rollout(args, samples, raw, None), abs=1e-3)


def test_with_one_term_it_is_plain_grpo_on_that_term():
    samples = [s for i, v in enumerate((0.1, 0.7, 0.4)) for s in _scored({"hps": v}, group=0, episode=i)]
    args = _args(video_reward_signal_pools=["hps"], video_reward_signal_ranges=[0.0, 1.0])

    _, normalized = group_normalized_rewards(args, samples)

    assert normalized == pytest.approx(_normalize_rewards_by_rollout(args, samples, [0.1, 0.7, 0.4], None), abs=1e-5)


def test_without_reward_normalization_the_summed_terms_are_the_training_rewards():
    samples = [*_scored({"hps": 0.0}, group=0, episode=0), *_scored({"hps": 1.0}, group=0, episode=1)]
    args = _args(video_reward_signal_pools=["hps"], video_reward_signal_ranges=[0.0, 1.0], rewards_normalization=False)

    assert group_normalized_rewards(args, samples)[1] == pytest.approx([-0.7071, 0.7071], abs=1e-4)


def test_episode_metrics_report_how_often_each_gate_fires():
    def sample(session: str, motion: bool, ocr: bool) -> Sample:
        record = {
            "submitted": True,
            "scores": {"signal:hps": 4.0, "motion": 0.1 if motion else 0.9, "ocr": float(ocr)},
            "terms": {"hps": 0.0 if motion else 0.4},
            "fired": {"motion": motion, "ocr": ocr, "adherence": False},
            "reward": 0.0 if motion else 0.4,
        }
        video_gen = {"turn_stats": [], "submitted": {"clip_id": "video_1", "prompt": "a fox runs"}}
        video_gen["images"] = [{"prompt": "a fox"}, {"prompt": "a red fox in snow"}]
        return Sample(metadata={"session_id": session, "video_gen": video_gen, RECORD_KEY: record})

    unsubmitted = Sample(metadata={"session_id": "d", "video_gen": {"turn_stats": [], "submitted": None}})
    unsubmitted.metadata[RECORD_KEY] = {"submitted": False, "terms": {"hps": 0.0}, "reward": 0.0}
    samples = [sample("a", True, False), sample("a", True, False), sample("b", False, True), sample("c", False, False)]

    metrics = episode_metrics([*samples, unsubmitted])

    assert metrics["reward/motion_fired"] == pytest.approx(1 / 3)
    assert metrics["reward/ocr_fired"] == pytest.approx(1 / 3) and metrics["reward/adherence_fired"] == 0
    assert metrics["reward/score_hps"] == 4.0 and metrics["reward/term_hps"] == pytest.approx(0.8 / 3)
    assert (metrics["episode/video_prompt_words"], metrics["episode/image_prompt_words"]) == (3.0, 3.5)


async def test_without_a_motion_pool_the_motion_gate_is_off_and_no_motion_pool_is_asked():
    pools = _pools(hps={"c": 5.0}, ur={"c": 3.0}, ocr={"c": 1.0})
    del pools["motion_pool"]
    sample = _sample("c")

    (reward,) = await reward_func(_args(video_reward_motion_pool=None), [sample], **pools)

    assert reward == pytest.approx(0.25)
    assert sample.metadata[RECORD_KEY]["fired"] == {"motion": False, "ocr": True, "adherence": False}
    assert "motion" not in sample.metadata[RECORD_KEY]["scores"]
