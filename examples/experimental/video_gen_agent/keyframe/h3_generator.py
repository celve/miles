"""MiniMax-H3 reference-to-video (FastH3 Omni) behind the env's ``ClipGenerator`` interface:
``AGENT_VIDEO_GEN_GENERATOR=examples.experimental.video_gen_agent.keyframe.h3_generator.H3ClipGenerator``.

sglang serves H3's ref2va only through its asynchronous ``/v1/videos``: ``/rollout/generate`` skips H3's pre-queue
reference preparation, and H3 delivers an MP4 file rather than the frames that endpoint returns (LIN-2860). A job lives
on the engine that took it, so rather than going through the router, this asks the router for its engines once and
sends each clip to the engine with the fewest clips in flight from this process, then polls that engine until the clip
is written. The MP4 keeps H3's audio track; the reward reads only its video stream.

Options come from ``AGENT_VIDEO_GEN_*`` env vars, as the env's own do:

  AGENT_VIDEO_GEN_GENERATOR_URL        the video model's router, asked for its engines; ``generate.generate`` sets it
  AGENT_VIDEO_GEN_CLIP_DIR             where engines write clips, a path the engines and trainer can read (required)
  AGENT_VIDEO_GEN_NUM_INFERENCE_STEPS  denoising steps (default: the model's; FastH3 Omni takes only 8)
"""

import asyncio
import os

import httpx

from examples.experimental.video_gen_agent.generator import Clip, ClipRequest, GeneratorUnavailable

URL_ENV = "AGENT_VIDEO_GEN_GENERATOR_URL"
CLIP_DIR_ENV = "AGENT_VIDEO_GEN_CLIP_DIR"
STEPS_ENV = "AGENT_VIDEO_GEN_NUM_INFERENCE_STEPS"

# H3's canvases (sglang's MINIMAX_H3_FINITE_ASPECT_RATIOS); a request takes the one nearest its width over height.
ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
SECONDS_RANGE = (4.0, 15.0)
POLL_S = 0.5

# Clips in flight per engine, shared by every generator in this process, since the env makes one per episode.
_in_flight: dict[str, int] = {}
_engines: dict[str, list[str]] = {}
_engines_lock = asyncio.Lock()


def h3_prompt(prompt: str, num_references: int) -> str:
    """H3's reference prompt form (its VIDEO_PROMPT_WRITING_GUIDE_ref_en.md): one subject per <Picture k>, then the
    task type, then the planner's prompt."""
    subjects = "\n".join(
        f"<Subject {k}> is the subject shown in <Picture {k}>, without its plain background."
        for k in range(1, num_references + 1)
    )
    return f"{subjects}\n[reference generation] {prompt}"


def aspect_ratio(width: int, height: int) -> str:
    def ratio(name: str) -> float:
        w, h = name.split(":")
        return int(w) / int(h)

    return min(ASPECT_RATIOS, key=lambda name: abs(ratio(name) - width / height))


class H3ClipGenerator:
    def __init__(self):
        self.router_url = os.environ[URL_ENV]
        self.clip_dir = os.environ[CLIP_DIR_ENV]
        steps = os.environ.get(STEPS_ENV)
        self.num_inference_steps = int(steps) if steps else None

    async def generate(self, request: ClipRequest) -> Clip:
        if not request.reference_paths:
            raise ValueError("H3's ref2va needs at least one reference image")
        low, high = SECONDS_RANGE
        if not low <= request.seconds <= high:
            raise ValueError(f"H3 makes {low:g} to {high:g} second clips, not {request.seconds:g}")
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                engine = min(await self._engines(client), key=lambda url: _in_flight.get(url, 0))
                _in_flight[engine] = _in_flight.get(engine, 0) + 1
                try:
                    return await self._generate_on(client, engine, request)
                finally:
                    _in_flight[engine] -= 1
        except httpx.HTTPError as error:
            raise GeneratorUnavailable(f"an H3 engine failed: {error!r}") from error

    async def _engines(self, client: httpx.AsyncClient) -> list[str]:
        async with _engines_lock:
            if not _engines.get(self.router_url):
                response = await client.get(f"{self.router_url}/list_workers")
                response.raise_for_status()
                urls = response.json()["urls"]
                if not urls:
                    raise GeneratorUnavailable("the H3 router lists no engines")
                _engines[self.router_url] = urls
            return _engines[self.router_url]

    async def _generate_on(self, client: httpx.AsyncClient, engine: str, request: ClipRequest) -> Clip:
        body = {
            "prompt": h3_prompt(request.prompt, len(request.reference_paths)),
            "task": "ref2va",
            "conditions": [{"type": "image", "role": "reference", "uri": f"file://{path}"} for path in request.reference_paths],
            "target": {
                "short_edge": min(request.width, request.height),
                "aspect_ratio": aspect_ratio(request.width, request.height),
                "duration_seconds": float(request.seconds),
            },
            "seconds": request.seconds,
            "seed": request.seed,
            "num_outputs_per_prompt": 1,
            # FastH3 Omni's default upscales each reference to a 2048 short edge, about 10 s of 2xB300 per reference
            # with no gain in reward or subject fidelity (LIN-2860).
            "reference_resize_mode": "match",
            "output_path": self.clip_dir,
        }
        if request.negative_prompt:
            body["negative_prompt"] = request.negative_prompt
        if self.num_inference_steps is not None:
            body["num_inference_steps"] = self.num_inference_steps
        response = await client.post(f"{engine}/v1/videos", json=body)
        response.raise_for_status()
        job = response.json()["id"]
        try:
            while (status := await self._status(client, engine, job))["status"] not in ("completed", "failed"):
                await asyncio.sleep(POLL_S)
        finally:
            # The engine keeps every finished job in memory until it is deleted; the clip file stays.
            await client.delete(f"{engine}/v1/videos/{job}")
        if status["status"] == "failed" or not status.get("file_path"):
            raise GeneratorUnavailable(f"H3 job {job} on {engine} failed: {status.get('error')!r}")
        return Clip(path=status["file_path"])

    async def _status(self, client: httpx.AsyncClient, engine: str, job: str) -> dict:
        response = await client.get(f"{engine}/v1/videos/{job}")
        response.raise_for_status()
        return response.json()
