import asyncio
import json
from argparse import Namespace
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import msgpack
import pytest
import torch
from safetensors.torch import save
from tests.fast.fixtures.args_fixtures import parser_defaults

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.diffusion_rollout import rollout_fn
from miles.rollout.diffusion_rollout.data_source import DiffusionPromptDataSource
from miles.rollout.submission_scheduler import SampleBackfillSubmission
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.function_registry import function_registry
from miles.utils.retry_utils import NonRetryableError


def fixed_steps(args, sample, num_steps, seed):
    return [0, 1], None


class _FakeOcrPool:
    def __init__(self, num_workers):
        self.scored = []
        self.closed = False

    async def score(self, images, prompts):
        self.scored.append(images[0].shape)
        return [float(len(prompts[0]))]

    def shutdown(self):
        self.closed = True


def _rollout(monkeypatch, tmp_path, **overrides):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"input": 'write "abc"'}) + "\n")
    args = ImmutableNamespace(
        **parser_defaults()
        | dict(
            prompt_data=str(prompts),
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
        )
        | overrides
    )
    monkeypatch.setattr(rollout_fn, "OcrRewardPool", _FakeOcrPool)
    return rollout_fn.DiffusionRolloutFn(
        RolloutFnConstructorInput(args=args, data_source=DiffusionPromptDataSource(args))
    )


@asynccontextmanager
async def _fake_engine(fn):
    calls = []
    tensor = save({"t": torch.zeros(3, 1, 5, 6)})

    async def respond(request):
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
        async with asyncio.timeout(5):
            yield calls
    finally:
        await fn.dispose()


@pytest.mark.parametrize("group_rm", [False, True])
async def test_ocr_runs_through_rm_hub_with_the_rollouts_pool(monkeypatch, tmp_path, group_rm):
    fn = _rollout(monkeypatch, tmp_path, group_rm=group_rm)
    async with _fake_engine(fn) as requests:
        (group,) = (await fn(RolloutFnTrainInput(rollout_id=8))).samples

    assert [request["num_outputs_per_prompt"] for request in requests] == [2, 2]
    assert [sample.reward for sample in group] == [float(len('write "abc"'))] * 4
    assert [sample.diffusion.rollout_id for sample in group] == [8] * 4
    assert [sample.lineage.source_sample_index for sample in group] == list(range(4))
    assert fn.reward_pool.scored == [(5, 6, 3)] * 4
    assert fn.reward_pool.closed


async def reward_by_batch_size(args, samples):
    return [len(samples) / 10] * len(samples)


@pytest.mark.parametrize("group_rm,reward", [(False, 0.2), (True, 0.4)])
async def test_a_custom_reward_preserves_the_scoring_granularity(monkeypatch, tmp_path, group_rm, reward):
    fn = _rollout(
        monkeypatch,
        tmp_path,
        custom_rm_path=CustomFunctionConfig(path=f"{__name__}.reward_by_batch_size"),
        group_rm=group_rm,
    )
    async with _fake_engine(fn):
        (group,) = (await fn(RolloutFnTrainInput(rollout_id=0))).samples

    assert fn.reward_pool is None
    assert [sample.reward for sample in group] == [reward] * 4


async def failing_reward(args, samples):
    raise RuntimeError("group scorer unavailable")


async def test_group_reward_failure_stops_the_diffusion_rollout(monkeypatch, tmp_path):
    fn = _rollout(
        monkeypatch,
        tmp_path,
        group_rm=True,
        custom_rm_path=CustomFunctionConfig(path=f"{__name__}.failing_reward"),
    )
    get_samples = fn.data_source.get_samples

    def get_once(count):
        assert fn.data_source.sample_group_index == 0, "unexpected refill after group reward failure"
        return get_samples(count)

    monkeypatch.setattr(fn.data_source, "get_samples", get_once)
    async with _fake_engine(fn):
        with pytest.raises(NonRetryableError, match="group scorer unavailable") as failure:
            await fn(RolloutFnTrainInput(rollout_id=0))

    assert isinstance(failure.value.__cause__, RuntimeError)


async def test_sample_submission_and_all_samples_hook_use_the_shared_loop(monkeypatch, tmp_path):
    fn = _rollout(
        monkeypatch, tmp_path, rollout_submission_granularity="sample", rollout_all_samples_process_path="test_process"
    )
    completions = []
    processed = []
    sample_done = SampleBackfillSubmission.sample_done_callback

    def on_done(scheduler):
        sample_done(scheduler)
        completions.append(scheduler.samples_in_flight)

    def process_all(args, groups, data_source):
        assert data_source == fn.data_source.get_samples
        processed.extend((sample.index, sample.reward) for group in groups for sample in group)

    monkeypatch.setattr(SampleBackfillSubmission, "sample_done_callback", on_done)
    async with _fake_engine(fn):
        with function_registry.temporary("test_process", process_all):
            (group,) = (await fn(RolloutFnTrainInput(rollout_id=0))).samples

    assert completions == [3, 2, 1, 0]
    assert processed == [(sample.index, sample.reward) for sample in group]
    assert [sample.index for sample in group] == [0, 1, 2, 3]


async def test_resuming_with_a_new_group_size_keeps_seeds_and_positions(monkeypatch, tmp_path):
    fn = _rollout(monkeypatch, tmp_path)
    previous_args = Namespace(**(dict(fn.state.args) | {"n_samples_per_prompt": 3}))
    previous_source = DiffusionPromptDataSource(previous_args)
    previous_source.get_samples(1)
    previous_source.save(tmp_path / "checkpoint")
    fn.data_source.load(tmp_path / "checkpoint")

    async with _fake_engine(fn) as requests:
        (group,) = (await fn(RolloutFnTrainInput(rollout_id=1))).samples

    assert [sample.index for sample in group] == [3, 4, 5, 6]
    assert [sample.diffusion.seed for sample in group] == list(
        range(fn.state.args.rollout_seed + 4, fn.state.args.rollout_seed + 8)
    )
    assert [request["num_outputs_per_prompt"] for request in requests] == [2, 2]
