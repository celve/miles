"""The H3 clip generator against fake H3 engines behind a real Miles router (LIN-2860)."""

import asyncio
import itertools
from pathlib import Path

import httpx
import pytest
from examples.experimental.video_gen_agent.generator import ClipRequest, GeneratorUnavailable
from examples.experimental.video_gen_agent.keyframe import h3_generator
from examples.experimental.video_gen_agent.keyframe.h3_generator import H3ClipGenerator
from fastapi import FastAPI, HTTPException

from miles.router.router import MilesRouter, MilesRouterConfig
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

_ids = itertools.count()


def _fake_h3_engine(name: str, seen: list, *, delay: float = 0.0, fail: bool = False, refuse: str = "") -> FastAPI:
    """sglang's async video API as H3 runs it: a job per POST, finished after ``delay``, written under output_path."""
    app = FastAPI()
    jobs: dict[str, dict] = {}

    @app.post("/v1/videos")
    async def create(body: dict):
        if refuse:
            raise HTTPException(400, refuse)
        # H3's own refusals (sglang f8d0201), of fields the planner controls.
        if body.get("negative_prompt"):
            raise HTTPException(400, "negative_prompt is not supported: MiniMax H3 serves only the CFG-distilled ...")
        if body.get("seed", 0) < 0:
            raise HTTPException(400, f"seed must be a non-negative int or list of ints, got {body['seed']}")
        job = f"{name}-{next(_ids)}"
        seen.append((name, body))
        jobs[job] = {"id": job, "status": "queued", "done_at": asyncio.get_running_loop().time() + delay, "body": body}
        return {"id": job, "status": "queued"}

    @app.get("/v1/videos")
    async def listed(limit: int = 20, after: str | None = None):
        ids = list(jobs)
        ids = ids[ids.index(after) + 1 :] if after else ids
        page = [await status(job) for job in ids[:limit]]
        return {"data": page, "last_id": page[-1]["id"] if page else None, "has_more": len(ids) > limit}

    @app.get("/v1/videos/{job}")
    async def status(job: str):
        record = jobs.get(job)
        if record is None:
            raise HTTPException(404)
        if asyncio.get_running_loop().time() < record["done_at"]:
            return {"id": job, "status": "in_progress"}
        if fail:
            return {"id": job, "status": "failed", "error": "reference-image preparation failed"}
        path = Path(record["body"]["output_path"]) / f"{job}.mp4"
        path.write_bytes(b"mp4")
        return {"id": job, "status": "completed", "file_path": str(path)}

    @app.delete("/v1/videos/{job}")
    async def delete(job: str):
        jobs.pop(job)
        seen.append((name, "deleted", job))
        return {"id": job, "status": "deleted"}

    @app.get("/health")
    async def health():
        return {}

    return app


@pytest.fixture
def h3(monkeypatch, tmp_path):
    """Starts fake engines and a router listing them; yields the generator class's env and each engine's log."""
    servers = []

    def start(*engines: dict) -> list:
        seen = []
        router = MilesRouter(
            MilesRouterConfig(
                host="127.0.0.1",
                port=find_available_port(37000),
                max_connections=None,
                timeout=None,
                health_check_interval=60.0,
                health_check_failure_threshold=3,
            ),
            verbose=False,
        )
        for index, options in enumerate(engines):
            server = UvicornThreadServer(
                _fake_h3_engine(f"e{index}", seen, **options),
                host="127.0.0.1",
                port=find_available_port(37100 + 50 * index),
            )
            server.start()
            servers.append(server)
            router._add_worker_url(server.url)
        server = UvicornThreadServer(router.app, host="127.0.0.1", port=router.config.port)
        server.start()
        servers.append(server)
        monkeypatch.setenv(h3_generator.URL_ENV, server.url)
        monkeypatch.setenv(h3_generator.CLIP_DIR_ENV, str(tmp_path))
        monkeypatch.setattr(h3_generator, "POLL_S", 0.01)
        return seen, [server.url for server in servers[-len(engines) - 1 : -1]]

    yield start
    for server in servers:
        server.stop()


def _request(**fields) -> ClipRequest:
    defaults = dict(
        prompt="a corgi chases a ball", negative_prompt="", seed=7, seconds=4.0, width=832, height=480, fps=24
    )
    return ClipRequest(**(defaults | {"reference_paths": ("/assets/corgi.png", "/assets/ball.png")} | fields))


async def test_a_clip_is_one_ref2va_job_with_the_assets_written_where_the_trainer_reads(h3, tmp_path):
    seen, _ = h3({})

    clip = await H3ClipGenerator().generate(_request())

    (_, body), (_, deleted, job) = seen
    assert (body["task"], body["seed"], body["seconds"], body["output_path"]) == ("ref2va", 7, 4, str(tmp_path))
    assert isinstance(body["seconds"], int)
    assert body["conditions"] == [
        {"type": "image", "role": "reference", "uri": "file:///assets/corgi.png"},
        {"type": "image", "role": "reference", "uri": "file:///assets/ball.png"},
    ]
    assert body["target"] == {"short_edge": 480, "aspect_ratio": "16:9", "duration_seconds": 4.0}
    assert body["reference_resize_mode"] == "match"
    assert body["prompt"] == (
        "<Subject 1> is the subject shown in <Picture 1>, without its plain background.\n"
        "<Subject 2> is the subject shown in <Picture 2>, without its plain background.\n"
        "[reference generation] a corgi chases a ball"
    )
    assert "num_inference_steps" not in body and "negative_prompt" not in body
    assert clip.path == str(tmp_path / f"{job}.mp4") and deleted == "deleted"


async def test_each_clip_goes_to_the_engine_whose_own_queue_is_shortest(h3):
    seen, (busy, _) = h3({"delay": 30.0}, {"delay": 0.05})
    # Jobs another episode's process left queued, which only the engine itself can count: 150 spans two list pages.
    async with httpx.AsyncClient() as client:
        for _ in range(150):
            (await client.post(f"{busy}/v1/videos", json={"output_path": "/unused"})).raise_for_status()

    for seed in range(3):
        await H3ClipGenerator().generate(_request(seed=seed))

    assert [entry[0] for entry in seen[150:] if isinstance(entry[1], dict)] == ["e1", "e1", "e1"]


async def test_engines_with_equal_queues_share_the_episodes_that_start_together(h3, monkeypatch):
    seen, _ = h3({"delay": 0.2}, {"delay": 0.2})
    monkeypatch.setattr(h3_generator.random, "choice", lambda engines: engines[-1])

    await H3ClipGenerator().generate(_request())

    (name, _), _ = seen
    assert name == "e1"


async def test_a_failed_job_or_a_dead_router_is_an_outage(h3, monkeypatch):
    seen, _ = h3({"fail": True})

    with pytest.raises(GeneratorUnavailable, match="reference-image preparation failed"):
        await H3ClipGenerator().generate(_request())
    assert seen[-1][1] == "deleted"

    monkeypatch.setenv(h3_generator.URL_ENV, f"http://127.0.0.1:{find_available_port(38000)}")
    with pytest.raises(GeneratorUnavailable, match="an H3 engine failed"):
        await H3ClipGenerator().generate(_request())


async def test_a_negative_prompt_or_seed_from_the_planner_is_no_outage(h3):
    seen, _ = h3({})

    await H3ClipGenerator().generate(_request(negative_prompt="blurry, watermark", seed=-1))

    (_, body), _ = seen
    assert "negative_prompt" not in body and body["seed"] == 2**32 - 1


async def test_a_refused_job_says_why(h3):
    h3({"refuse": "reference image is unreadable"})

    with pytest.raises(GeneratorUnavailable, match="refused an H3 job: 400 .*reference image is unreadable"):
        await H3ClipGenerator().generate(_request())


@pytest.mark.parametrize(
    "fields, message",
    [
        (dict(reference_paths=()), "needs at least one reference image"),
        (dict(seconds=2.0), "4 to 15 whole-second clips, not 2"),
        (dict(seconds=4.5), "4 to 15 whole-second clips, not 4.5"),
    ],
)
async def test_a_request_h3_cannot_make_is_refused_before_any_engine_sees_it(h3, fields, message):
    seen, _ = h3({})

    with pytest.raises(ValueError, match=message):
        await H3ClipGenerator().generate(_request(**fields))
    assert seen == []


def test_the_canvas_is_the_nearest_h3_aspect_ratio():
    assert [h3_generator.aspect_ratio(w, h) for w, h in ((832, 480), (480, 832), (512, 512), (640, 480))] == [
        "16:9",
        "9:16",
        "1:1",
        "4:3",
    ]
