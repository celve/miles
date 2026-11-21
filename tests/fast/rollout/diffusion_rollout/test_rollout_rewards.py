import asyncio
import json
from argparse import Namespace
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import msgpack
import pytest
import torch
from safetensors.torch import save
from tests.fast.fixtures.args_fixtures import parser_defaults

from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.diffusion_rollout import rollout_fn
from miles.rollout.diffusion_rollout.data_source import DiffusionPromptDataSource
from miles.rollout.submission_scheduler import SampleBackfillSubmission
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.diffusion.types import DiffusionRollout
from miles.utils.eval_config import EvalDatasetConfig
from miles.utils.function_registry import function_registry
from miles.utils.types import Sample


def fixed_steps(args, sample, num_steps, seed):
    return [0, 1], None


class _FakeOcrPool:
    """Stands in for the executor's client of the `ocr` reward pool."""

    def __init__(self):
        self.scored = []
        self.health_checks = 0

    async def score(self, images, prompts):
        self.scored.append(images[0].shape)
        return [float(len(prompts[0]))]

    async def ensure_healthy(self):
        self.health_checks += 1


def _rollout(*, prompt_metadata=(), reward_pools=None, **overrides):
    args = ImmutableNamespace(
        **parser_defaults()
        | dict(
            sglang_router_ip="127.0.0.1",
            sglang_router_port=1,
            sglang=SimpleNamespace(models=[], common_value=lambda name: False),
            inference_runtime_mut_state=SimpleNamespace(engine_count=1),
            custom_generate_function_path="miles.rollout.diffusion_rollout.generate.generate",
            diffusion_step_strategy_path=f"{__name__}.fixed_steps",
            diffusion_microgroup_size=2,
            n_samples_per_prompt=4,
            rollout_batch_size=1,
            over_sampling_batch_size=1,
            rm_type="ocr",
            loss_type="flow_grpo",
        )
        | overrides
    )
    data_source = SimpleNamespace(
        dataset=[("p", metadata, None) for metadata in prompt_metadata],
        get_samples=lambda num_groups: [_group()],
        sample_index_offset=0,
    )
    return rollout_fn.DiffusionRolloutFn(
        RolloutFnConstructorInput(
            args=args,
            data_source=data_source,
            reward_pools=(
                ({} if args.custom_rm_path is not None else {"ocr": _FakeOcrPool()})
                if reward_pools is None
                else reward_pools
            ),
        )
    )


@asynccontextmanager
async def _fake_engine(fn):
    calls = []
    tensor = save({"t": torch.zeros(3, 1, 5, 6)})
    pool = next(iter(fn.reward_pools.values()), None)

    async def respond(request):
        if pool:
            assert pool.health_checks > 0
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(
            200,
            content=msgpack.packb(
                [{"generated_output": {"__tensor__": True, "data": tensor}}] * body["num_outputs_per_prompt"]
            ),
        )

    await fn.client._http.aclose()
    fn.client._http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        yield calls
    finally:
        await fn.dispose()


def _group():
    return [Sample(group_index=0, index=i, prompt='write "abc"') for i in range(4)]


@pytest.mark.parametrize("group_rm", [False, True])
async def test_ocr_runs_through_rm_hub_with_a_pool_in_both_reward_modes(group_rm):
    fn = _rollout(group_rm=group_rm)
    async with _fake_engine(fn) as requests:
        output = await fn(RolloutFnTrainInput(rollout_id=8))
    (group,) = output.samples

    assert [request["num_outputs_per_prompt"] for request in requests] == [2, 2]
    assert [sample.reward for sample in group] == [float(len('write "abc"'))] * 4
    assert [sample.diffusion.rollout_id for sample in group] == [8] * 4
    assert [sample.lineage.source_sample_index for sample in group] == list(range(4))
    assert fn.reward_pools["ocr"].scored == [(5, 6, 3)] * 4


async def constant_reward(args, samples):
    return [0.5] * len(samples)


@pytest.mark.parametrize("group_rm", [False, True])
async def test_a_custom_reward_replaces_ocr_without_receiving_the_ocr_pool(group_rm):
    fn = _rollout(custom_rm_path=CustomFunctionConfig(path=f"{__name__}.constant_reward"), group_rm=group_rm)

    async with _fake_engine(fn):
        (group,) = (await fn(RolloutFnTrainInput(rollout_id=0))).samples

    assert fn.reward_pools == {}
    assert [sample.reward for sample in group] == [0.5] * 4


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


@pytest.mark.parametrize("group_rm,batch_sizes", [(False, [2, 2]), (True, [4])])
async def test_pickscore_scores_in_position_order_at_the_chosen_granularity(group_rm, batch_sizes):
    pools = _pickscore_pools()
    fn = _rollout(rm_type="pickscore", reward_pools=pools, group_rm=group_rm)

    async with asyncio.timeout(5), _fake_engine(fn):
        (group,) = (await fn(RolloutFnTrainInput(rollout_id=0))).samples

    assert pools["pickscore"].calls == batch_sizes
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


async def test_a_prompt_overriding_pickscore_with_ocr_uses_its_declared_pool():
    pools = _pickscore_pools(with_ocr=True)
    fn = _rollout(rm_type="pickscore", prompt_metadata=[{}, {"rm_type": "ocr"}], reward_pools=pools)
    group = _group()
    for sample in group:
        sample.metadata = {"rm_type": "ocr"}
    fn.data_source.get_samples = lambda num_groups: [group]

    async with _fake_engine(fn):
        (group,) = (await fn(RolloutFnTrainInput(rollout_id=0))).samples

    assert pools["pickscore"].calls == []
    assert [sample.reward for sample in group] == [float(len('write "abc"'))] * 4


@pytest.mark.parametrize(
    "rm_type,override,pools",
    [("ocr", "pickscore", {"ocr"}), ("pickscore", "ocr", {"pickscore"})],
)
def test_a_prompt_that_selects_a_reward_whose_pool_was_not_started_is_rejected_at_startup(rm_type, override, pools):
    """Pools start from the launch flags, before the prompts are read, so the error names the flag that starts one."""
    with pytest.raises(ValueError, match=f"--diffusion-reward-pools {override}"):
        _rollout(
            rm_type=rm_type,
            prompt_metadata=[{"rm_type": override}],
            reward_pools={name: _FakeOcrPool() for name in pools},
        )


async def test_a_prompt_may_select_pickscore_under_an_ocr_launch_once_its_pool_is_declared():
    fn = _rollout(
        rm_type="ocr",
        prompt_metadata=[{"rm_type": "pickscore"}],
        reward_pools=_pickscore_pools(with_ocr=True),
    )

    await fn.dispose()
    assert sorted(fn.reward_pools) == ["ocr", "pickscore"]


async def test_a_global_custom_reward_builds_no_pool_for_the_overrides_it_replaces():
    fn = _rollout(
        rm_type="pickscore",
        custom_rm_path=f"{__name__}.fixed_steps",
        diffusion_output_num_frames=5,
        prompt_metadata=[{"rm_type": "ocr"}],
    )

    await fn.dispose()
    assert fn.reward_pools == {}


async def test_every_rollout_checks_pool_health_and_keeps_the_executor_owned_pools():
    pools = _pickscore_pools()
    fn = _rollout(rm_type="pickscore", reward_pools=pools)
    async with _fake_engine(fn) as requests:
        await fn(RolloutFnTrainInput(rollout_id=0))
        await fn(RolloutFnTrainInput(rollout_id=1))

    assert len(requests) == 4
    assert pools["pickscore"].health_checks == 2
    assert fn.reward_pools["pickscore"] is pools["pickscore"]


async def test_sample_submission_and_all_samples_hook_use_the_shared_loop(monkeypatch, tmp_path):
    fn = _rollout(
        rollout_submission_granularity="sample",
        rollout_all_samples_process_path="test_process",
    )
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"input": "first"}\n{"input": "second"}\n')
    fn.data_source = DiffusionPromptDataSource(
        Namespace(**(dict(fn.state.args) | {"prompt_data": str(prompts), "rollout_shuffle": False}))
    )
    completions = []
    calls = []
    sample_done = SampleBackfillSubmission.sample_done_callback

    def on_done(scheduler):
        sample_done(scheduler)
        completions.append(scheduler.samples_in_flight)

    def process_all(args, groups, data_source):
        assert data_source == fn.data_source.get_samples
        calls.append(("all", [sample.index for sample in groups[0]]))

    monkeypatch.setattr(SampleBackfillSubmission, "sample_done_callback", on_done)
    async with _fake_engine(fn):
        with function_registry.temporary("test_process", process_all):
            (group,) = (await fn(RolloutFnTrainInput(rollout_id=0))).samples

    assert completions[:4] == [3, 2, 1, 0]
    assert calls == [("all", [0, 1, 2, 3])]
    assert [sample.index for sample in group] == [0, 1, 2, 3]


async def test_resuming_with_a_new_group_size_keeps_seeds_and_positions(tmp_path):
    pool = _FakeOcrPool()
    fn = _rollout(reward_pools={"ocr": pool})
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"input": "p"}\n')
    first = DiffusionPromptDataSource(
        Namespace(**(dict(fn.state.args) | {"prompt_data": str(prompts), "n_samples_per_prompt": 3}))
    )
    first.get_samples(1)
    first.save(tmp_path / "checkpoint")
    resumed = DiffusionPromptDataSource(Namespace(**(dict(fn.state.args) | {"prompt_data": str(prompts)})))
    resumed.load(tmp_path / "checkpoint")
    fn.data_source = resumed

    async with _fake_engine(fn) as requests:
        (group,) = (await fn(RolloutFnTrainInput(rollout_id=1))).samples

    assert [sample.index for sample in group] == [3, 4, 5, 6]
    assert [sample.diffusion.seed for sample in group] == list(
        range(fn.state.args.rollout_seed + 4, fn.state.args.rollout_seed + 8)
    )
    assert [request["num_outputs_per_prompt"] for request in requests] == [2, 2]


@pytest.mark.parametrize("eval_steps", [None, 3])
async def test_executor_scores_image_only_eval_with_a_seed_per_repeat(tmp_path, monkeypatch, eval_steps):
    prompts = tmp_path / "eval.jsonl"
    prompts.write_text('{"question":"first","attrs":{"tag":1},"target":"a"}\n{"question":"second","target":"b"}\n')
    dataset = EvalDatasetConfig(
        name="ocr",
        path=str(prompts),
        input_key="question",
        metadata_key="attrs",
        label_key="target",
        n_samples_per_eval_prompt=3,
    )
    fn = _rollout(eval_datasets=[dataset], diffusion_eval_num_steps=eval_steps, eval_uses_snapshots=False)
    executor = object.__new__(RolloutExecutor)
    executor.args = fn.state.args
    executor.eval_generate_rollout = fn
    executor.use_legacy_rollout_v1 = False
    executor._refresh_inference_runtime_mut_state = AsyncMock()
    executor._metric_checker = None
    executor.custom_convert_samples_to_train_data_func = Mock(side_effect=AssertionError("converted eval images"))
    results = {}
    monkeypatch.setattr(
        "miles.ray.rollout.rollout_executor.log_eval_rollout_data",
        lambda rollout_id, args, data, metrics: results.update(data),
    )

    async with _fake_engine(fn) as requests:
        await executor.eval(0)

    samples = results["ocr"]["samples"]
    seeds = [fn.state.args.rollout_seed + j for j in range(3)]
    assert sorted((r["prompt"], r["seed"]) for r in requests) == [(p, s) for p in ("first", "second") for s in seeds]
    assert all(r["rollout"] is False and r["num_outputs_per_prompt"] == 1 for r in requests)
    assert all(r["num_inference_steps"] == (eval_steps or fn.state.args.diffusion_num_steps) for r in requests)
    assert all(not r["rollout_return_dit_trajectory"] and not r["rollout_return_denoising_env"] for r in requests)
    assert [(s.index, s.diffusion.seed) for s in samples] == list(enumerate(seeds * 2))
    assert all(s.diffusion.dit_trajectory is None for s in samples)
    assert [(s.label, s.metadata) for s in samples] == [("a", {"tag": 1})] * 3 + [("b", {})] * 3
    assert results["ocr"]["rewards"] == [5.0] * 3 + [6.0] * 3
    executor.custom_convert_samples_to_train_data_func.assert_not_called()


async def filter_test_reward(args, samples):
    return [float(sample.index % 2) if sample.prompt == "keep" else 0.0 for sample in samples]


async def test_reward_std_filter_refills_a_dropped_diffusion_group(tmp_path):
    from tests.fast.utils.diffusion.test_validation import _parse_diffusion_run

    filter_path = "miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter"
    config = _parse_diffusion_run("--global-batch-size", "64", "--dynamic-sampling-filter-path", filter_path)
    fn = _rollout(
        dynamic_sampling_filter_path=config.dynamic_sampling_filter_path,
        custom_rm_path=CustomFunctionConfig(path=f"{__name__}.filter_test_reward"),
    )
    prompts = tmp_path / "train.jsonl"
    prompts.write_text('{"input":"drop"}\n{"input":"keep"}\n')
    fn.data_source = DiffusionPromptDataSource(
        Namespace(**(dict(fn.state.args) | {"prompt_data": str(prompts), "rollout_shuffle": False}))
    )
    async with asyncio.timeout(5), _fake_engine(fn) as requests:
        output = await fn(RolloutFnTrainInput(rollout_id=1))

    (group,) = output.samples
    assert output.metrics["rollout/dynamic_filter/drop_zero_std_0.0"] == 1
    assert [sample.index for sample in group] == [4, 5, 6, 7]
    assert [r["prompt"] for r in requests] == ["drop", "drop", "keep", "keep"]
