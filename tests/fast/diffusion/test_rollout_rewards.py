import asyncio
from argparse import Namespace

import pytest
import torch
from tests.fast.fixtures.args_fixtures import parser_defaults

from miles.diffusion.types import DiffusionRollout
from miles.rollout.base_types import RolloutFnConstructorInput
from miles.rollout.diffusion_rollout import rollout_fn
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import ImmutableNamespace
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


def _rollout(monkeypatch, **overrides):
    args = Namespace(
        **parser_defaults()
        | dict(
            sglang_router_ip="127.0.0.1",
            sglang_router_port=1,
            diffusion_step_strategy_path=f"{__name__}.fixed_steps",
            diffusion_microgroup_size=2,
            n_samples_per_prompt=4,
            rm_type="ocr",
        )
        | overrides
    )
    monkeypatch.setattr(rollout_fn, "OcrRewardPool", _FakeOcrPool)
    monkeypatch.setattr(
        rollout_fn,
        "parse_rollout_response",
        lambda raw: [DiffusionRollout(generated_output=torch.zeros(3, 1, 4, 4)) for _ in range(raw)],
    )
    fn = rollout_fn.DiffusionRolloutFn(RolloutFnConstructorInput(args=args, data_source=None))
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
    assert fn.reward_pool.scored == [(4, 4, 3)] * 4


async def constant_reward(args, samples):
    return [0.5] * len(samples)


async def test_a_custom_reward_replaces_ocr_without_receiving_the_ocr_pool(monkeypatch):
    fn, _ = _rollout(monkeypatch, custom_rm_path=CustomFunctionConfig(path=f"{__name__}.constant_reward"))
    # rm_hub builds the hook's config from the run config, which iterates like the parsed one.
    fn.args = ImmutableNamespace(**vars(fn.args))

    group = await fn._generate_group(_group())

    assert fn.reward_pool is None
    assert [sample.reward for sample in group] == [0.5] * 4
