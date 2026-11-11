"""Make the pairwise reward's anchors before a run: each request's one-shot clip (B0-ref in reference mode), made on
the run's own image and video models served on this machine's GPUs, and placed in the run's ``--anchor-dir``.

  python -m examples.experimental.video_gen_agent.keyframe.eval.anchors --gpus 0,1,2,3 --reward pairwise \\
      [the launcher's flags: --video-mode, --requests, --model-dir, --data-dir, --anchor-dir, ...]

Run the launcher's ``prepare`` first for the models and request files; the launcher's flags then pick the same
generators, requests, clip shape and directory its run reads. The first GPU serves the image model, and the others
video engines of the run's --video-gpus-per-engine each (two consecutive GPUs an engine in h3-ref mode). Requests that already have an anchor are skipped, the count and an estimate are printed before anything is
served, and it exits non-zero if any request is still without one.
"""

import asyncio
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import httpx
import typer
from examples.experimental.video_gen_agent.keyframe.eval import arms
from examples.experimental.video_gen_agent.keyframe.generator import RouterImageGenerator
from examples.experimental.video_gen_agent.keyframe.run_qwen35_9b_keyframe import (
    ScriptArgs,
    _agent_env,
    _anchor_request_files,
    _sglang_config,
)

from miles.backends.sglang_utils.sglang_config import DiffusionLora
from miles.diffusion.api_client import DiffusionApiClient
from miles.diffusion.engine import compute_diffusion_engine_launch_cmd
from miles.router.config import MilesRouterConfig
from miles.router.router import MilesRouter
from miles.utils.external_utils import command_utils
from miles.utils.function_registry import load_function
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

# GPU-seconds per anchor on H200: reference mode made 295 in about 55 min with Z-Image-Turbo and VACE-14B sharing one
# GPU (LIN-2858); h3-ref's is two GPUs for 8.1 s a 4 s clip plus the image (LIN-2860). No other mode is measured.
SECONDS_PER_ANCHOR = {"reference": 11.0, "h3-ref": 17.0}
# The arms' one-shot arm for each env mode, which h3-ref shares with reference mode.
_ARMS_MODE = {"preview": "t2v", "i2v": "i2v", "reference": "reference"}


@dataclass
class AnchorArgs(ScriptArgs):
    gpus: str = "0,1"
    port_base: int = 42000

    def __post_init__(self):
        super().__post_init__()
        assert self.anchor_dir is not None, "anchors are for --reward pairwise"
        assert len(self.gpu_ids) >= 2, "--gpus names the image model's GPU, then one or more video GPUs"
        assert (
            len(self.gpu_ids) - 1
        ) % self.video_gpus_per_engine == 0, (
            f"--gpus' video GPUs do not split into engines of {self.video_gpus_per_engine}"
        )

    @property
    def gpu_ids(self) -> list[int]:
        return [int(gpu) for gpu in self.gpus.split(",")]


def plan(args: AnchorArgs) -> tuple[list[dict], str]:
    """The one-shot rows of the requests without an anchor, and what making them costs."""
    requests = [request for path in _anchor_request_files(args) for request in arms.heldout_requests(Path(path))]
    missing = arms.missing_anchors(requests, Path(args.anchor_dir))
    rows = [arms.one_shot_row(request, _ARMS_MODE[args.env_mode]) for request in missing]
    video_gpus = len(args.gpu_ids) - 1
    if (rate := SECONDS_PER_ANCHOR.get(args.video_mode)) is None:
        cost = f"no measured rate for {args.video_mode} mode"
    else:
        hours = len(rows) * rate / 3600
        cost = f"about {hours:.1f} GPU-h, {hours / video_gpus:.1f} h on {video_gpus} video GPUs (H200)"
    return rows, f"{len(rows)} of {len(set(requests))} requests have no anchor in {args.anchor_dir}: {cost}"


@command_utils.dataclass_cli
def main(args: AnchorArgs):
    make_anchors(args)


def make_anchors(args: AnchorArgs) -> None:
    rows, summary = plan(args)
    print(summary, flush=True)
    if rows:
        asyncio.run(_make(args, rows))
    rows, summary = plan(args)
    if rows:
        sys.exit(summary)


async def _make(args: AnchorArgs, rows: list[dict]) -> None:
    ports = iter(range(args.port_base, args.port_base + 1000, 10))
    image_gpu, *video_gpus = args.gpu_ids
    # The run's own video group: the diffusers backend in reference mode, H3's Ulysses split, the LoRAs to set.
    (group,) = next(model for model in _sglang_config(args)["sglang"] if model["name"] == "video")["server_groups"]
    size = group["num_gpus_per_engine"]
    image = [_launch(args.local(args.image_model), [image_gpu], {}, ports)]
    videos = [
        _launch(args.video_path, video_gpus[i : i + size], group.get("overrides", {}), ports)
        for i in range(0, len(video_gpus), size)
    ]
    servers = []
    try:
        await _wait_healthy([url for _, url in image + videos], timeout_s=3600)
        if group.get("loras"):
            loras = [DiffusionLora(**lora) for lora in group["loras"]]
            await asyncio.gather(*(DiffusionApiClient(server_url=url).set_lora(loras) for _, url in videos))
        servers = [_router([url for _, url in group], next(ports)) for group in (image, videos)]
        os.environ.update(
            _agent_env(args)
            | {
                "AGENT_VIDEO_GEN_IMAGE_GENERATOR_URL": servers[0].url,
                "AGENT_VIDEO_GEN_GENERATOR_URL": servers[1].url,
                "AGENT_VIDEO_GEN_IMAGE_DIR": f"{args.anchor_dir}/.images",
                "AGENT_VIDEO_GEN_CLIP_DIR": f"{args.anchor_dir}/.clips",
            }
        )
        generator = load_function(os.environ["AGENT_VIDEO_GEN_GENERATOR"])()
        shape = arms.SHAPE | {"seconds": args.seconds, "fps": args.video.fps}
        made = await arms.generate(rows, generator, RouterImageGenerator(), concurrency=4 * len(videos), shape=shape)
        print(f"placed {arms.place_anchors(made, Path(args.anchor_dir))} anchors", flush=True)
    finally:
        for server in servers:
            server.stop()
        for process, _ in image + videos:
            process.terminate()
        for process, _ in image + videos:
            process.wait(timeout=60)


def _launch(model_path: str, gpus: list[int], overrides: dict, ports) -> tuple[subprocess.Popen, str]:
    assert gpus == list(range(gpus[0], gpus[0] + len(gpus))), f"an engine's GPUs are consecutive, not {gpus}"
    port = next(ports)
    cmd = compute_diffusion_engine_launch_cmd(
        SimpleNamespace(hf_checkpoint=model_path, diffusion_dit_precision=None, diffusion_flow_shift=None),
        interpreter_prefix=[sys.executable],
        base_gpu_id=gpus[0],
        num_gpus_per_engine=len(gpus),
        sglang_overrides={"model_path": model_path, **overrides},
        update_weights=False,
        nccl_port=next(ports),
        host="127.0.0.1",
        port=port,
    )
    return subprocess.Popen(shlex.split(cmd)), f"http://127.0.0.1:{port}"


async def _wait_healthy(urls: list[str], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=10) as client:
        for url in urls:
            while not await _healthy(client, url):
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{url} never became healthy")
                await asyncio.sleep(5)


async def _healthy(client: httpx.AsyncClient, url: str) -> bool:
    try:
        return (await client.get(f"{url}/health_generate")).status_code == 200
    except httpx.HTTPError:
        return False


def _router(urls: list[str], port: int) -> UvicornThreadServer:
    config = MilesRouterConfig(
        host="127.0.0.1",
        port=port,
        health_check_interval=60.0,
        health_check_failure_threshold=3,
        max_connections=None,
        timeout=None,
    )
    router = MilesRouter(config, verbose=False)
    for url in urls:
        router._add_worker_url(url)
    server = UvicornThreadServer(router.app, host="127.0.0.1", port=port)
    server.start()
    return server


if __name__ == "__main__":
    typer.run(main)
