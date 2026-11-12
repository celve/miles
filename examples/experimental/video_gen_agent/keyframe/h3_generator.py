"""MiniMax-H3 reference-to-video (FastH3 Omni) behind the env's ``ClipGenerator`` interface:
``AGENT_VIDEO_GEN_GENERATOR=examples.experimental.video_gen_agent.keyframe.h3_generator.H3ClipGenerator``.

sglang serves H3's ref2va only through its asynchronous ``/v1/videos``: ``/rollout/generate`` skips H3's pre-queue
reference preparation, and H3 delivers an MP4 file rather than the frames that endpoint returns (LIN-2860). A job lives
on the engine that took it, so rather than going through the router, this asks the router for its engines and sends
each clip to the engine with the fewest unfinished jobs, then polls that engine until the clip is written. The engines
count their own jobs, since each episode runs in a process of its own. The MP4 keeps H3's audio track; the reward reads
only its video stream. H3 refuses a negative prompt (its checkpoint is CFG-distilled, so one would do nothing, as in the
other modes' CFG-1 generators) and a negative seed, both of which the planner may send, so neither reaches it as such.

The eval arms ask for more than the env does. A request's ``image_path`` and ``last_image_path`` go to H3 as keyframes at
its first and last frame, beside the references; with no reference the job is fl2va, and with neither t2va. FastH3 Omni
serves ref2va alone, so fl2va and t2va need the base model's FL2VA partition (``--model-variant fl2va``).

Options come from ``AGENT_VIDEO_GEN_*`` env vars, as the env's own do:

  AGENT_VIDEO_GEN_GENERATOR_URL        the video model's router, asked for its engines; ``generate.generate`` sets it
  AGENT_VIDEO_GEN_CLIP_DIR             where engines write clips, a path the engines and trainer can read (required)
  AGENT_VIDEO_GEN_NUM_INFERENCE_STEPS  denoising steps (default: the model's; FastH3 Omni takes only 8)
"""

import asyncio
import os
import random

import httpx

from examples.experimental.video_gen_agent.generator import Clip, ClipRequest, GeneratorUnavailable

URL_ENV = "AGENT_VIDEO_GEN_GENERATOR_URL"
CLIP_DIR_ENV = "AGENT_VIDEO_GEN_CLIP_DIR"
STEPS_ENV = "AGENT_VIDEO_GEN_NUM_INFERENCE_STEPS"

# H3's canvases (sglang's MINIMAX_H3_FINITE_ASPECT_RATIOS); a request takes the one nearest its width over height.
ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
# sglang's video request takes whole seconds.
SECONDS_RANGE = (4, 15)
# A queued clip waits for every clip ahead of it, so a slow poll costs an episode at most this much and no throughput.
POLL_S = 2.0


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


def keyframes(request: ClipRequest) -> list[dict]:
    """The request's first and last frames as H3 keyframe conditions, at frame 0 and -1, the two it takes."""
    frames = ((0, request.image_path), (-1, request.last_image_path))
    return [
        {"type": "image", "role": "keyframe", "uri": f"file://{path}", "frame_index": index}
        for index, path in frames
        if path is not None
    ]


class H3ClipGenerator:
    def __init__(self):
        self.router_url = os.environ[URL_ENV]
        self.clip_dir = os.environ[CLIP_DIR_ENV]
        steps = os.environ.get(STEPS_ENV)
        self.num_inference_steps = int(steps) if steps else None

    async def generate(self, request: ClipRequest) -> Clip:
        low, high = SECONDS_RANGE
        if request.seconds not in range(low, high + 1):
            raise ValueError(f"H3 makes {low} to {high} whole-second clips, not {request.seconds:g}")
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
                return await self._generate_on(client, await self._least_queued(client), request)
        except httpx.HTTPError as error:
            raise GeneratorUnavailable(f"an H3 engine failed: {error!r}") from error

    async def _least_queued(self, client: httpx.AsyncClient) -> str:
        response = await client.get(f"{self.router_url}/list_workers")
        response.raise_for_status()
        engines = response.json()["urls"]
        if not engines:
            raise GeneratorUnavailable("the H3 router lists no engines")
        queues = await asyncio.gather(*(self._unfinished(client, engine) for engine in engines))
        # Episodes that start together all see the same queues, so ties go to a random engine.
        shortest = min(queues)
        return random.choice([engine for engine, queue in zip(engines, queues, strict=True) if queue == shortest])

    async def _unfinished(self, client: httpx.AsyncClient, engine: str) -> int:
        """The engine's jobs not yet completed or failed; finished jobs are deleted, so few pages are live ones."""
        count, after = 0, None
        while True:
            response = await client.get(
                f"{engine}/v1/videos", params={"limit": 100} | ({"after": after} if after else {})
            )
            response.raise_for_status()
            page = response.json()
            count += sum(job["status"] not in ("completed", "failed") for job in page["data"])
            if not page.get("has_more"):
                return count
            after = page["last_id"]

    async def _generate_on(self, client: httpx.AsyncClient, engine: str, request: ClipRequest) -> Clip:
        references = request.reference_paths
        body = {
            "prompt": h3_prompt(request.prompt, len(references)) if references else request.prompt,
            "task": "ref2va" if references else "fl2va" if keyframes(request) else "t2va",
            # H3 numbers only the references as <Picture k>; keyframes go first, as its own clients send them.
            "conditions": keyframes(request)
            + [{"type": "image", "role": "reference", "uri": f"file://{path}"} for path in references],
            "target": {
                "short_edge": min(request.width, request.height),
                "aspect_ratio": aspect_ratio(request.width, request.height),
                "duration_seconds": float(request.seconds),
            },
            "seconds": int(request.seconds),
            "seed": request.seed % 2**32,
            "num_outputs_per_prompt": 1,
            "output_path": self.clip_dir,
        }
        if references:
            # FastH3 Omni's default upscales each reference to a 2048 short edge, about 10 s of 2xB300 per reference
            # with no gain in reward or subject fidelity (LIN-2860).
            body["reference_resize_mode"] = "match"
        if self.num_inference_steps is not None:
            body["num_inference_steps"] = self.num_inference_steps
        response = await client.post(f"{engine}/v1/videos", json=body)
        if response.is_client_error:
            raise GeneratorUnavailable(f"{engine} refused an H3 job: {response.status_code} {response.text}")
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
