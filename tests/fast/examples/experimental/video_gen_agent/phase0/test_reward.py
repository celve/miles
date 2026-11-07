"""Phase 0's reward step, driven through the text rollout's `generate_and_rm_group`, with the real
`DiffusionApiClient` behind the real Miles router and a fake multimodal_gen server, and a fake reward pool."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgpack
import pytest
import torch
from examples.experimental.video_gen_agent.phase0 import reward
from examples.experimental.video_gen_agent.phase0.reward import (
    GENERATOR_PROMPT_KEY,
    SEND_REQUEST_KEY,
    PlannerRewardConfig,
    log_rollout_data,
    request_seed,
)
from fastapi import FastAPI, Request, Response
from safetensors.torch import save
from tests.fast.fixtures.args_fixtures import parser_defaults
from tests.fast.rollout.inference_rollout.conftest import make_eval_args

import miles.rollout.inference_rollout.inference_rollout_common as inference_rollout_common
import miles.rollout.inference_rollout.inference_rollout_train as inference_rollout_train
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput, RolloutFnConstructorInput
from miles.rollout.filter_hub.common_filters import apply_missing_reward_filter
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.rollout.rm_hub.video.reward import CLIP_PATH_KEY, REQUEST_KEY
from miles.router.config import MilesRouterConfig
from miles.router.router import MilesRouter
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.eval_config import EvalDatasetConfig
from miles.utils.http_utils import find_available_port
from miles.utils.retry_utils import NonRetryableError
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

_FRAMES, _HEIGHT, _WIDTH = 9, 32, 48


def _fake_multimodal_gen(seen: list) -> FastAPI:
    """Answers `/rollout/generate` as sglang.multimodal_gen does, and fails any prompt that mentions an outage."""
    app = FastAPI()

    @app.post("/rollout/generate")
    async def generate(request: Request):
        body = await request.json()
        seen.append(body)
        if "outage" in body["prompt"]:
            return Response(status_code=503)
        generator = torch.Generator().manual_seed(body["seed"] + len(body["prompt"]))
        video = torch.randint(0, 256, (3, _FRAMES, _HEIGHT, _WIDTH), dtype=torch.uint8, generator=generator)
        output = {"__tensor__": True, "data": save({"t": video})}
        return Response(content=msgpack.packb([{"seed": body["seed"], "generated_output": output}]))

    return app


@pytest.fixture
def generator_router():
    seen = []
    workers = [
        UvicornThreadServer(_fake_multimodal_gen(seen), host="127.0.0.1", port=find_available_port(33000 + 100 * i))
        for i in range(2)
    ]
    port = find_available_port(23000)
    config = MilesRouterConfig(
        host="127.0.0.1",
        port=port,
        health_check_interval=60.0,
        health_check_failure_threshold=3,
        max_connections=100,
        timeout=None,
    )
    router = MilesRouter(config, verbose=False)
    for worker in workers:
        worker.start()
        router._add_worker_url(worker.url)
    router_server = UvicornThreadServer(router.app, host="127.0.0.1", port=port)
    router_server.start()
    yield ("127.0.0.1", port), seen
    router_server.stop()
    for worker in workers:
        worker.stop()


class FakePool:
    """Stands in for the judge pool's client: scores the request's length, and None for a clip named in `unreadable`."""

    def __init__(self, unreadable=()):
        self.calls = []
        self.unreadable = set(unreadable)

    async def score(self, outputs, prompts):
        self.calls.append((list(outputs), list(prompts)))
        return [
            None if path in self.unreadable else float(len(request))
            for path, request in zip(outputs, prompts, strict=True)
        ]


async def _plan(input: GenerateFnInput) -> GenerateFnOutput:
    """A planner whose answer is its prompt, after an empty reasoning block."""
    sample = input.sample
    sample.status = Sample.Status.COMPLETED
    sample.response = f"<think>\n\n</think>\n\n{sample.prompt}"
    return GenerateFnOutput(samples=sample)


_EVAL_DATASETS = [
    EvalDatasetConfig(name="heldout", path="heldout.jsonl", n_samples_per_eval_prompt=1),
    EvalDatasetConfig(name="heldout_raw", path="heldout_raw.jsonl", n_samples_per_eval_prompt=1),
]


def _rollout_fn(monkeypatch, tmp_path, router, *, group_rm, pool, num_frames=_FRAMES) -> InferenceRolloutFn:
    monkeypatch.setattr(inference_rollout_common, "load_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference_rollout_common, "load_processor", lambda *args, **kwargs: None)
    rm_config = PlannerRewardConfig(
        video_reward_pool="judge",
        video_clip_dir=str(tmp_path),
        video_num_frames=num_frames,
        video_fps=8,
        video_width=_WIDTH,
        video_height=_HEIGHT,
    )
    args = ImmutableNamespace(
        **parser_defaults()
        | make_eval_args()
        | dict(
            group_rm=group_rm,
            custom_generate_function_path=f"{__name__}._plan",
            custom_rm_path=CustomFunctionConfig(
                path="examples.experimental.video_gen_agent.phase0.reward.planner_rm", config=rm_config
            ),
            sglang_model_routers={"policy": ("127.0.0.1", 1), "generator": router},
            inference_runtime_mut_state=SimpleNamespace(engine_count=1),
            eval_datasets=_EVAL_DATASETS,
            rollout_batch_size=1,
            over_sampling_batch_size=1,
            n_samples_per_prompt=2,
        )
    )
    return InferenceRolloutFn(
        RolloutFnConstructorInput(args=args, data_source=MagicMock(), reward_pools={"judge": pool})
    )


def _group(request: str, *answers: str, send_request: bool = False) -> list[Sample]:
    metadata = {REQUEST_KEY: request} | ({SEND_REQUEST_KEY: True} if send_request else {})
    return [Sample(index=i, prompt=answer, metadata=dict(metadata)) for i, answer in enumerate(answers)]


@pytest.mark.parametrize("group_rm", [False, True])
async def test_each_answer_is_one_generator_call_whose_clip_is_scored_against_the_request(
    monkeypatch, tmp_path, generator_router, group_rm
):
    router, seen = generator_router
    pool = FakePool()
    fn = _rollout_fn(monkeypatch, tmp_path, router, group_rm=group_rm, pool=pool)

    group = await generate_and_rm_group(fn.state, _group("a cat", "a tabby cat", "a cat at dusk"), sampling_params={})

    assert sorted(body["prompt"] for body in seen) == ["a cat at dusk", "a tabby cat"]
    assert {body["seed"] for body in seen} == {request_seed("a cat")}
    assert [sample.reward for sample in group] == [5.0, 5.0]
    assert sorted(path for outputs, _ in pool.calls for path in outputs) == sorted(
        sample.metadata[CLIP_PATH_KEY] for sample in group
    )
    assert {request for _, requests in pool.calls for request in requests} == {"a cat"}


async def test_a_raw_row_sends_the_request_itself_with_the_same_seed(monkeypatch, tmp_path, generator_router):
    router, seen = generator_router
    fn = _rollout_fn(monkeypatch, tmp_path, router, group_rm=True, pool=FakePool())

    group = await generate_and_rm_group(
        fn.state, _group("a cat", "a tabby cat", send_request=True), sampling_params={}
    )

    ((body,),) = [seen]
    assert body["prompt"] == "a cat" and body["seed"] == request_seed("a cat")
    assert group[0].metadata[GENERATOR_PROMPT_KEY] == "a cat" and group[0].reward == 5.0


@pytest.mark.parametrize("group_rm", [False, True])
async def test_a_generator_outage_or_an_unreadable_clip_scores_none_and_drops_the_group(
    monkeypatch, tmp_path, generator_router, group_rm
):
    router, _ = generator_router
    pool = FakePool()
    fn = _rollout_fn(monkeypatch, tmp_path, router, group_rm=group_rm, pool=pool)

    group = await generate_and_rm_group(fn.state, _group("a cat", "a cat", "outage"), sampling_params={})
    pool.unreadable = {group[0].metadata[CLIP_PATH_KEY]}
    unreadable = await generate_and_rm_group(fn.state, _group("a cat", "a cat"), sampling_params={})

    assert [sample.reward for sample in group] == [5.0, None]
    assert CLIP_PATH_KEY not in group[1].metadata
    assert [sample.reward for sample in unreadable] == [None]
    assert apply_missing_reward_filter(fn.state.args, group).reason == "group_has_missing_reward"


async def test_the_held_out_eval_scores_the_planner_and_the_raw_request_baseline(
    monkeypatch, tmp_path, generator_router
):
    router, seen = generator_router
    fn = _rollout_fn(monkeypatch, tmp_path, router, group_rm=False, pool=FakePool())
    args = fn.state.args
    cache = {
        dataset.cache_key
        + (args.hf_checkpoint, args.apply_chat_template, args.chat_template_path): SimpleNamespace(
            samples=_group("a cat", "a tabby cat", send_request=dataset.name == "heldout_raw")
        )
        for dataset in _EVAL_DATASETS
    }

    results = await run_eval_datasets(fn.state, cache)

    assert {name: result["rewards"] for name, result in results.items()} == {"heldout": [5.0], "heldout_raw": [5.0]}
    assert sorted(body["prompt"] for body in seen) == ["a cat", "a tabby cat"]
    assert {body["seed"] for body in seen} == {request_seed("a cat")}


async def test_an_empty_answer_scores_zero_without_calling_the_generator(monkeypatch, tmp_path, generator_router):
    router, seen = generator_router
    fn = _rollout_fn(monkeypatch, tmp_path, router, group_rm=True, pool=FakePool())

    group = await generate_and_rm_group(fn.state, _group("a cat", "", "a cat"), sampling_params={})

    assert [sample.reward for sample in group] == [0.0, 5.0]
    assert len(seen) == 1
    assert apply_missing_reward_filter(fn.state.args, group).keep


class _BoundedScheduler:
    """Submits one group at a time, at most three, so a driver that swallows the error ends instead of spinning."""

    sample_done_callback = None

    def __init__(self, submitted):
        self.submitted = submitted

    def has_capacity(self, *, pending_groups: int, group_budget: int) -> bool:
        return pending_groups == 0 and len(self.submitted) < 3

    def on_submit(self, samples) -> None:
        pass

    async def wait_for_progress(self, pendings):
        assert pendings, f"the driver dropped {len(self.submitted)} failed groups and kept asking for more"
        return await asyncio.wait(pendings)


@pytest.mark.parametrize("group_rm", [False, True])
async def test_a_clip_the_generator_resized_stops_the_training_rollout(
    monkeypatch, tmp_path, generator_router, group_rm
):
    router, _ = generator_router
    fn = _rollout_fn(monkeypatch, tmp_path, router, group_rm=group_rm, pool=FakePool(), num_frames=_FRAMES + 4)
    submitted = []

    def submit(state, samples, callback):
        submitted.extend(tasks := real_submit(state, samples, callback))
        return tasks

    async def no_dumper(args) -> None:
        pass

    real_submit = inference_rollout_train.submit_generate_tasks
    monkeypatch.setattr(inference_rollout_train, "submit_generate_tasks", submit)
    monkeypatch.setattr(
        inference_rollout_train, "make_submission_scheduler", lambda *a, **k: _BoundedScheduler(submitted)
    )
    monkeypatch.setattr(inference_rollout_train.dumper_utils, "configure_sglang", no_dumper)

    with pytest.raises(NonRetryableError, match="asked for 13 frames"):
        await inference_rollout_train.generate_rollout_async(fn.state, 0, lambda n: [_group("a cat", "a cat")])
    assert len(submitted) == 1


def test_the_rollout_log_adds_phase0_timings_and_keeps_the_default_metrics(monkeypatch):
    logged = []
    monkeypatch.setattr(reward.tracking, "log", lambda args, metrics, step_key: logged.append((metrics, step_key)))
    samples = [
        Sample(metadata={GENERATOR_PROMPT_KEY: "a tabby cat", reward.TIMING_KEY: {"generate": 4.0, "score": 1.0}}),
        Sample(metadata={GENERATOR_PROMPT_KEY: "", reward.TIMING_KEY: {"generate": 2.0, "score": 3.0}}),
    ]
    args = SimpleNamespace(wandb_always_use_train_step=False)

    assert log_rollout_data(7, args, samples, {}, 10.0) is False

    ((metrics, step_key),) = logged
    assert step_key == "rollout/step" and metrics["rollout/step"] == 7
    assert metrics["rollout/phase0/generate_seconds"] == 3.0
    assert metrics["rollout/phase0/score_seconds"] == 2.0
    assert metrics["rollout/phase0/empty_answer_ratio"] == 0.5
    assert metrics["rollout/phase0/generator_prompt_words"] == 1.5
