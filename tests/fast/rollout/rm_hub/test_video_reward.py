"""A text run's reward step scores each sample's generated clip on a reward pool, through `video_rm`."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tests.fast.fixtures.args_fixtures import parser_defaults
from tests.fast.rollout.inference_rollout.conftest import make_eval_args

import miles.rollout.inference_rollout.inference_rollout_common as inference_rollout_common
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput, RolloutFnConstructorInput
from miles.rollout.filter_hub.common_filters import apply_missing_reward_filter
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn, generate_and_rm_group
from miles.rollout.rm_hub.video.reward import CLIP_PATH_KEY, REQUEST_KEY
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.types import Sample


class FakePool:
    """Stands in for the executor's client of a reward pool: a clip named `broken` scores None."""

    def __init__(self):
        self.calls = []

    async def score(self, outputs, prompts):
        self.calls.append((list(outputs), list(prompts)))
        return [
            None if "broken" in path else float(len(request)) for path, request in zip(outputs, prompts, strict=True)
        ]


async def _generate_clip(input: GenerateFnInput) -> GenerateFnOutput:
    """Generates `clip-<index>.mp4` for each sample, and no clip for the one whose prompt is `fail`."""
    sample = input.sample
    sample.status = Sample.Status.COMPLETED
    sample.metadata[REQUEST_KEY] = f"request {sample.index}"
    if sample.prompt != "fail":
        sample.metadata[CLIP_PATH_KEY] = f"/clips/{sample.prompt}-{sample.index}.mp4"
    return GenerateFnOutput(samples=sample)


def _rollout_fn(monkeypatch, *, group_rm: bool, reward_pools) -> InferenceRolloutFn:
    monkeypatch.setattr(inference_rollout_common, "load_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference_rollout_common, "load_processor", lambda *args, **kwargs: None)
    args = ImmutableNamespace(
        **parser_defaults()
        | make_eval_args()
        | dict(
            group_rm=group_rm,
            custom_generate_function_path=f"{__name__}._generate_clip",
            custom_rm_path=CustomFunctionConfig(
                path="miles.rollout.rm_hub.video.reward.video_rm", config={"video_reward_pool": "judge"}
            ),
            inference_runtime_mut_state=SimpleNamespace(engine_count=1),
        )
    )
    return InferenceRolloutFn(RolloutFnConstructorInput(args=args, data_source=MagicMock(), reward_pools=reward_pools))


def _group(*prompts: str) -> list[Sample]:
    return [Sample(index=index, prompt=prompt, metadata={}) for index, prompt in enumerate(prompts)]


@pytest.mark.parametrize("group_rm", [False, True])
async def test_the_reward_step_scores_each_clip_against_its_request_on_the_named_pool(monkeypatch, group_rm):
    pool = FakePool()
    fn = _rollout_fn(monkeypatch, group_rm=group_rm, reward_pools={"judge": pool, "other": FakePool()})

    group = await generate_and_rm_group(fn.state, _group("cat", "dog"), sampling_params={})

    assert [sample.reward for sample in group] == [9.0, 9.0]
    clips = sorted(path for outputs, _ in pool.calls for path in outputs)
    assert clips == ["/clips/cat-0.mp4", "/clips/dog-1.mp4"]
    assert len(pool.calls) == (1 if group_rm else 2)


@pytest.mark.parametrize("group_rm", [False, True])
async def test_a_sample_without_a_clip_or_with_one_the_scorer_cannot_read_scores_none_and_drops_its_group(
    monkeypatch, group_rm
):
    pool = FakePool()
    fn = _rollout_fn(monkeypatch, group_rm=group_rm, reward_pools={"judge": pool})

    group = await generate_and_rm_group(fn.state, _group("cat", "fail", "broken"), sampling_params={})

    assert [sample.reward for sample in group] == [9.0, None, None]
    assert all("fail" not in path for outputs, _ in pool.calls for path in outputs)
    assert apply_missing_reward_filter(fn.state.args, group).reason == "group_has_missing_reward"


async def test_a_pool_name_the_run_did_not_declare_fails_the_reward_step(monkeypatch):
    fn = _rollout_fn(monkeypatch, group_rm=True, reward_pools={"other": FakePool()})

    with pytest.raises(AssertionError, match="judge"):
        await generate_and_rm_group(fn.state, _group("cat"), sampling_params={})
