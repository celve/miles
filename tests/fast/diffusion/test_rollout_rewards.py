import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from tests.fast.fixtures.args_fixtures import parser_defaults

from miles.rollout.base_types import RolloutFnConstructorInput
from miles.rollout.diffusion_rollout import rollout_fn
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.diffusion.types import DiffusionRollout
from miles.utils.types import Sample


def fixed_steps(args, sample, num_steps, seed):
    return [0, 1], None


class _FakeOcrPool:
    def __init__(self, num_workers):
        self.scored = []

    async def score(self, images, prompts):
        self.scored.append(images[0].shape)
        return [float(len(prompts[0]))]

    def shutdown(self):
        pass


def _rollout(monkeypatch, *, prompt_metadata=(), **overrides):
    args = Namespace(
        **parser_defaults()
        | dict(
            sglang_router_ip="127.0.0.1",
            sglang_router_port=1,
            diffusion_step_strategy_path=f"{__name__}.fixed_steps",
            diffusion_microgroup_size=2,
            n_samples_per_prompt=4,
            rm_type="ocr",
            # What startup validation makes of the parser's default loss type.
            loss_type="flow_grpo",
        )
        | overrides
    )
    monkeypatch.setattr(rollout_fn, "OcrRewardPool", _FakeOcrPool)
    monkeypatch.setattr(
        rollout_fn,
        "parse_rollout_response",
        lambda raw: [DiffusionRollout(generated_output=torch.zeros(3, 1, 4, 4)) for _ in range(raw)],
    )
    data_source = SimpleNamespace(dataset=[("p", metadata, None) for metadata in prompt_metadata])
    fn = rollout_fn.DiffusionRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))
    fn.semaphore = asyncio.Semaphore(1)

    async def post(payload):
        return payload["num_outputs_per_prompt"]

    fn._post = post
    reward_calls = []

    async def spy(args, samples, **kwargs):
        reward_calls.append(len(samples))
        return await real(args, samples, **kwargs)

    real = rollout_fn.batched_async_rm
    monkeypatch.setattr(rollout_fn, "batched_async_rm", spy)
    return fn, reward_calls


def _group():
    return [Sample(group_index=0, index=i, prompt='write "abc"') for i in range(4)]


@pytest.mark.parametrize("group_rm,calls", [(False, [2, 2]), (True, [4])])
async def test_rewards_come_from_rm_hub_per_microgroup_or_per_group(monkeypatch, group_rm, calls):
    fn, reward_calls = _rollout(monkeypatch, group_rm=group_rm)

    group = await fn._generate_group(_group())

    assert reward_calls == calls
    assert [sample.reward for sample in group] == [float(len('write "abc"'))] * 4
    assert fn.reward_kwargs["ocr_pool"].scored == [(4, 4, 3)] * 4


async def constant_reward(args, samples):
    return [0.5] * len(samples)


async def test_a_custom_reward_replaces_ocr_without_receiving_the_ocr_pool(monkeypatch):
    fn, _ = _rollout(monkeypatch, custom_rm_path=CustomFunctionConfig(path=f"{__name__}.constant_reward"))
    # rm_hub builds the hook's config from the run config, which iterates like the parsed one.
    fn.args = ImmutableNamespace(**vars(fn.args))

    group = await fn._generate_group(_group())

    assert fn.reward_kwargs == {}
    assert [sample.reward for sample in group] == [0.5] * 4


class _FakePickScorePool:
    def __init__(self, args):
        self.calls = []

    async def score(self, outputs, prompts):
        self.calls.append(len(outputs))
        return [0.25] * len(outputs)

    def shutdown(self):
        pass


async def test_pickscore_scores_each_microgroup_in_one_batched_call(monkeypatch):
    monkeypatch.setattr(rollout_fn, "PickScorePool", _FakePickScorePool)
    fn, reward_calls = _rollout(monkeypatch, rm_type="pickscore")

    group = await fn._generate_group(_group())

    assert reward_calls == [2, 2] and fn.reward_kwargs["pickscore_pool"].calls == [2, 2]
    assert [sample.reward for sample in group] == [0.25] * 4


async def test_a_sample_that_overrides_the_reward_leaves_the_pickscore_batch():
    from miles.rollout.rm_hub import batched_async_rm

    pool = _FakePickScorePool(None)
    args = Namespace(**parser_defaults() | dict(rm_type="pickscore", custom_rm_path=None))
    samples = [Sample(index=0, prompt="p"), Sample(index=1, prompt="p", metadata={"rm_type": "deterministic_random"})]
    for sample in samples:
        sample.diffusion = DiffusionRollout(generated_output=torch.zeros(3, 1, 4, 4))
        sample.tokens, sample.response = [], ""

    rewards = await batched_async_rm(args, samples, pickscore_pool=pool)

    assert pool.calls == [1] and rewards[0] == 0.25 and rewards[1] in (0, 1)


async def test_a_prompt_that_overrides_pickscore_with_ocr_is_scored_on_an_ocr_pool(monkeypatch):
    monkeypatch.setattr(rollout_fn, "PickScorePool", _FakePickScorePool)
    fn, _ = _rollout(monkeypatch, rm_type="pickscore", prompt_metadata=[{}, {"rm_type": "ocr"}])
    group = _group()
    for sample in group[2:]:
        sample.metadata = {"rm_type": "ocr"}

    group = await fn._generate_group(group)

    assert fn.reward_kwargs["pickscore_pool"].calls == [2]
    assert [sample.reward for sample in group] == [0.25, 0.25] + [float(len('write "abc"'))] * 2


def test_a_prompt_cannot_select_pickscore_under_another_launch_reward(monkeypatch):
    with pytest.raises(ValueError, match="launch with --rm-type pickscore"):
        _rollout(monkeypatch, rm_type="ocr", prompt_metadata=[{"rm_type": "pickscore"}])
