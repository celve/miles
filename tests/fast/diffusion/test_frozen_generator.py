import asyncio
import hashlib
import json
import os
import shlex
import sys
from types import ModuleType, SimpleNamespace

import imageio.v3 as iio
import msgpack
import pytest
import torch
from fastapi import FastAPI, Request, Response
from safetensors.torch import save
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

from miles.diffusion import engine_main
from miles.diffusion.api_client import DiffusionApiClient
from miles.diffusion.engine import compute_diffusion_engine_env_vars, compute_diffusion_engine_launch_cmd
from miles.router.config import MilesRouterConfig
from miles.router.router import MilesRouter
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

_GENERATOR = {"worker_type": "regular", "num_gpus": 2, "num_gpus_per_engine": 2, "engine_kind": "diffusion"}


def _sglang_yaml(*, generator_first=False, update_weights=False, generator_groups=(_GENERATOR,)) -> str:
    policy = ["  - name: policy", "    server_groups:", "      - {worker_type: regular, num_gpus: 1}"]
    generator = [
        "  - name: generator",
        "    model_path: /models/Wan2.2-TI2V-5B-Diffusers",
        f"    update_weights: {str(update_weights).lower()}",
        "    server_groups:",
        *(f"      - {json.dumps(group)}" for group in generator_groups),
    ]
    models = generator + policy if generator_first else policy + generator
    return "\n".join(["sglang:", *models]) + "\n"


def _parse(tmp_path, *extra, **yaml_options):
    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(_sglang_yaml(**yaml_options))
    return parse_fsdp_test_config(
        *("--rollout-num-gpus", "3", "--num-gpus-per-node", "8", "--sglang-config", str(config_path)), *extra
    )


def test_a_text_run_serves_a_frozen_diffusion_model_beside_its_policy(tmp_path):
    config = _parse(tmp_path, "--use-miles-router")

    policy, generator = config.sglang.models
    assert policy.update_weights and not generator.update_weights
    assert [group.engine_kind for group in generator.server_groups] == ["diffusion"]


@pytest.mark.parametrize(
    "yaml_options, extra, message",
    [
        (dict(update_weights=True), ("--use-miles-router",), "update_weights: true"),
        (dict(generator_first=True), ("--use-miles-router",), "the first place"),
        (dict(), (), "--use-miles-router"),
        (
            dict(
                generator_groups=(
                    {**_GENERATOR, "num_gpus": 1, "num_gpus_per_engine": 1},
                    {**_GENERATOR, "num_gpus": 1, "num_gpus_per_engine": 1, "engine_kind": "text"},
                )
            ),
            ("--use-miles-router",),
            "engine_kind text groups",
        ),
        (dict(generator_groups=({**_GENERATOR, "worker_type": "prefill"},)), ("--use-miles-router",), "prefill"),
    ],
)
def test_a_frozen_diffusion_model_outside_the_supported_layout_is_rejected(tmp_path, yaml_options, extra, message):
    with pytest.raises(ValueError, match=message):
        _parse(tmp_path, *extra, **yaml_options)


def test_a_frozen_diffusion_model_on_gpus_shared_with_training_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="GPUs shared with training"):
        _parse(tmp_path, "--use-miles-router", "--colocate", "--actor-num-gpus-per-node", "3")


def _launched_server_args(*, update_weights, overrides=None, num_gpus_per_engine=4):
    args = SimpleNamespace(
        hf_checkpoint="/models/policy",
        diffusion_lora_target_modules=None if not update_weights else ["attn.to_q"],
        diffusion_dit_precision=None,
        diffusion_flow_shift=None,
        diffusion_forward_dtype="bf16",
    )
    argv = shlex.split(
        compute_diffusion_engine_launch_cmd(
            args,
            interpreter_prefix=["python"],
            base_gpu_id=0,
            num_gpus_per_engine=num_gpus_per_engine,
            sglang_overrides={"model_path": "/models/wan", **(overrides or {})},
            update_weights=update_weights,
            nccl_port=15001,
            host="127.0.0.1",
            port=15000,
        )
    )
    return json.loads(argv[argv.index("--server-args") + 1]), compute_diffusion_engine_env_vars(
        args, update_weights=update_weights
    )


def test_a_frozen_engine_gets_no_lora_merge_args_and_leaves_its_gpu_split_to_sglang():
    server_args, env_vars = _launched_server_args(update_weights=False)

    assert not {key for key in server_args if key.startswith("lora")}
    assert "tp_size" not in server_args and server_args["num_gpus"] == 4
    assert env_vars == {}


def test_a_trained_engine_keeps_its_lora_merge_args():
    server_args, env_vars = _launched_server_args(update_weights=True, num_gpus_per_engine=1)

    assert server_args["lora_merge_mode"] == "merge" and server_args["lora_target_modules"] == ["attn.to_q"]
    assert "SGLANG_DIFFUSION_LORA_MERGE_FP32" in env_vars


def test_the_groups_sequence_and_cfg_parallel_overrides_reach_the_engine():
    parallel = {"sp_degree": 2, "ulysses_degree": 2, "enable_cfg_parallel": True}
    server_args, _ = _launched_server_args(update_weights=False, overrides=parallel)

    assert parallel.items() <= server_args.items()


def test_an_engine_of_several_gpus_sees_all_of_them(monkeypatch):
    launched = []
    launch_module = ModuleType("sglang.multimodal_gen.runtime.launch_server")
    launch_module.launch_server = launched.append
    args_module = ModuleType("sglang.multimodal_gen.runtime.server_args")
    args_module.ServerArgs = SimpleNamespace(from_kwargs=lambda **kwargs: kwargs)
    monkeypatch.setattr(engine_main.dataclasses, "fields", lambda _: [SimpleNamespace(name="num_gpus")])
    monkeypatch.setitem(sys.modules, launch_module.__name__, launch_module)
    monkeypatch.setitem(sys.modules, args_module.__name__, args_module)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setattr(sys, "argv", ["engine_main", "--base-gpu-id", "1", "--server-args", '{"num_gpus": 2}'])

    engine_main.main()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5,6"
    assert launched == [{"num_gpus": 2}]


def _fake_multimodal_gen(*, frames: int, height: int, width: int, seen: list) -> FastAPI:
    """Serves only what sglang.multimodal_gen serves to Miles: no pause, continue or abort routes."""
    app = FastAPI()

    @app.post("/rollout/generate")
    async def generate(request: Request):
        body = await request.json()
        seen.append(body)
        generator = torch.Generator().manual_seed(body["seed"])
        video = torch.randint(0, 256, (3, frames, height, width), dtype=torch.uint8, generator=generator)
        output = {"__tensor__": True, "data": save({"t": video})}
        return Response(content=msgpack.packb([{"seed": body["seed"], "generated_output": output}]))

    return app


@pytest.fixture
def generator_behind_router():
    seen = []
    workers = [
        UvicornThreadServer(
            _fake_multimodal_gen(frames=9, height=32, width=48, seen=seen),
            host="127.0.0.1",
            port=find_available_port(31000 + 100 * index),
        )
        for index in range(2)
    ]
    port = find_available_port(21000)
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
    yield DiffusionApiClient(server_url=router_server.url), seen
    router_server.stop()
    for worker in workers:
        worker.stop()


def _generate(client, tmp_path, **overrides):
    request = dict(prompt="a red kite", seed=7, num_frames=9, fps=8, width=48, height=32, output_dir=str(tmp_path))
    return asyncio.run(client.generate_clip(**(request | overrides)))


def test_a_clip_is_generated_behind_the_router_and_stored_by_its_sha256(generator_behind_router, tmp_path):
    client, seen = generator_behind_router

    path = _generate(client, tmp_path, guidance_scale=4.0, extra_sampling_params={"guidance_scale_2": 3.0})

    with open(path, "rb") as f:
        assert os.path.basename(path) == f"{hashlib.sha256(f.read()).hexdigest()}.mp4"
    assert iio.imread(path, plugin="pyav").shape == (9, 32, 48, 3)
    assert os.listdir(tmp_path) == [os.path.basename(path)]
    (body,) = seen
    assert body["rollout"] is False and body["rollout_video_dtype"] == "uint8"
    assert body["guidance_scale"] == 4.0 and body["extra_sampling_params"] == {"guidance_scale_2": 3.0}


def test_different_clips_never_share_a_path(generator_behind_router, tmp_path):
    client, _ = generator_behind_router

    assert _generate(client, tmp_path, seed=1) != _generate(client, tmp_path, seed=2)


def test_a_clip_the_engine_resized_is_refused(generator_behind_router, tmp_path):
    client, _ = generator_behind_router

    with pytest.raises(ValueError, match="asked for 10 frames"):
        _generate(client, tmp_path, num_frames=10)


def test_pausing_or_aborting_a_multimodal_gen_engine_sends_it_nothing(generator_behind_router):
    client, _ = generator_behind_router

    async def lifecycle():
        await client.pause_generation(mode="retract")
        await client.continue_generation()
        await client.abort_all_requests()

    asyncio.run(lifecycle())
