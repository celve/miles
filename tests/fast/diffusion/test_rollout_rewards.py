import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from tests.fast.fixtures.args_fixtures import parser_defaults

from miles.diffusion.rollout import rollout_fn
from miles.diffusion.types import DiffusionRollout
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.utils.types import Sample


def fixed_steps(args, sample, num_steps, seed):
    return [0, 1], None


class _FakeOcrPool:
    """Stands in for the executor's client of the `ocr` reward pool."""

    def __init__(self):
        self.scored = []

    async def score(self, images, prompts):
        self.scored.append(images[0].shape)
        return [float(len(prompts[0]))]

    async def ensure_healthy(self):
        pass


def _rollout(monkeypatch, *, prompt_metadata=(), reward_pools=None, **overrides):
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
    monkeypatch.setattr(
        rollout_fn,
        "parse_rollout_response",
        lambda raw: [DiffusionRollout(generated_output=torch.zeros(3, 1, 4, 4)) for _ in range(raw)],
    )
    data_source = SimpleNamespace(dataset=[("p", metadata, None) for metadata in prompt_metadata])
    fn = rollout_fn.DiffusionRolloutFn(
        RolloutFnConstructorInput(
            args=args,
            data_source=data_source,
            reward_pools={"ocr": _FakeOcrPool()} if reward_pools is None else reward_pools,
        )
    )
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


class _FakePickScorePool:
    """Stands in for the executor's client of the `pickscore` reward pool."""

    def __init__(self):
        self.calls = []
        self.health_checks = 0

    async def score(self, outputs, prompts):
        self.calls.append(len(outputs))
        return [0.25] * len(outputs)

    async def ensure_healthy(self):
        self.health_checks += 1


def _pickscore_pools(*, with_ocr=False):
    return {"pickscore": _FakePickScorePool()} | ({"ocr": _FakeOcrPool()} if with_ocr else {})


async def test_pickscore_scores_each_microgroup_in_one_batched_call(monkeypatch):
    fn, reward_calls = _rollout(monkeypatch, rm_type="pickscore", reward_pools=_pickscore_pools())

    group = await fn._generate_group(_group())

    assert reward_calls == [2, 2] and fn.reward_kwargs["pickscore_pool"].calls == [2, 2]
    assert [sample.reward for sample in group] == [0.25] * 4


async def test_a_sample_that_overrides_the_reward_leaves_the_pickscore_batch():
    from miles.rollout.rm_hub import batched_async_rm

    pool = _FakePickScorePool()
    args = Namespace(**parser_defaults() | dict(rm_type="pickscore", custom_rm_path=None))
    samples = [Sample(index=0, prompt="p"), Sample(index=1, prompt="p", metadata={"rm_type": "deterministic_random"})]
    for sample in samples:
        sample.diffusion = DiffusionRollout(generated_output=torch.zeros(3, 1, 4, 4))
        sample.tokens, sample.response = [], ""

    rewards = await batched_async_rm(args, samples, pickscore_pool=pool)

    assert pool.calls == [1] and rewards[0] == 0.25 and rewards[1] in (0, 1)


async def test_a_prompt_that_overrides_pickscore_with_ocr_is_scored_on_an_ocr_pool(monkeypatch):
    fn, _ = _rollout(
        monkeypatch,
        rm_type="pickscore",
        prompt_metadata=[{}, {"rm_type": "ocr"}],
        reward_pools=_pickscore_pools(with_ocr=True),
    )
    group = _group()
    for sample in group[2:]:
        sample.metadata = {"rm_type": "ocr"}

    group = await fn._generate_group(group)

    assert fn.reward_kwargs["pickscore_pool"].calls == [2]
    assert [sample.reward for sample in group] == [0.25, 0.25] + [float(len('write "abc"'))] * 2


@pytest.mark.parametrize(
    "rm_type,override,pools",
    [("ocr", "pickscore", {"ocr"}), ("pickscore", "ocr", {"pickscore"})],
)
def test_a_prompt_that_selects_a_reward_whose_pool_was_not_started_is_rejected_at_startup(
    monkeypatch, rm_type, override, pools
):
    """Pools start from the launch flags, before the prompts are read, so the error names the flag that starts one."""
    with pytest.raises(ValueError, match=f"--diffusion-reward-pools {override}"):
        _rollout(
            monkeypatch,
            rm_type=rm_type,
            prompt_metadata=[{"rm_type": override}],
            reward_pools={name: _FakeOcrPool() for name in pools},
        )


def test_a_prompt_may_select_pickscore_under_an_ocr_launch_once_its_pool_is_declared(monkeypatch):
    fn, _ = _rollout(
        monkeypatch,
        rm_type="ocr",
        prompt_metadata=[{"rm_type": "pickscore"}],
        reward_pools=_pickscore_pools(with_ocr=True),
    )

    assert sorted(fn.reward_kwargs) == ["ocr_pool", "pickscore_pool"]


def test_a_global_custom_reward_takes_every_sample_so_prompt_overrides_need_no_pool(monkeypatch):
    fn, _ = _rollout(
        monkeypatch,
        rm_type="pickscore",
        custom_rm_path=f"{__name__}.fixed_steps",
        diffusion_output_num_frames=5,
        prompt_metadata=[{"rm_type": "ocr"}],
        reward_pools=_pickscore_pools(),
    )

    assert list(fn.reward_kwargs) == ["pickscore_pool"]


async def test_every_rollout_first_checks_that_each_reward_pool_is_whole_and_dispose_leaves_the_pools_alone(
    monkeypatch,
):
    """The executor owns the pools' workers; the rollout only restarts what died, never shuts them down."""
    pools = _pickscore_pools()
    fn, _ = _rollout(monkeypatch, rm_type="pickscore", reward_pools=pools)
    fn.data_source.get_samples = lambda num_groups: [_group()]

    await fn(RolloutFnTrainInput(rollout_id=0))
    await fn(RolloutFnTrainInput(rollout_id=1))
    await fn.dispose()

    assert pools["pickscore"].health_checks == 2
    assert fn.reward_kwargs["pickscore_pool"] is pools["pickscore"]
