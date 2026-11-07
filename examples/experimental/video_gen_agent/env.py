"""Video-generation env: an agent function for ``agentic_tool_call.generate`` under session v2.

The planner turns a user's video request into clips with a frozen generator, through two tools:
``generate(prompt, negative_prompt, seed, seconds)`` and ``submit(clip_id)``. The env fixes resolution and fps,
caps clip length, generations and turns, and states the remaining budget in every tool reply.

It does not score. It returns the submitted clip under ``metadata["video_gen"]``, which v2 copies into every
kept sample, and ``examples.experimental.video_gen_agent.reward.reward_func`` scores it in the rollout's reward step.

Run-wide options come from ``AGENT_VIDEO_GEN_*`` env vars, which worker and sandbox modes forward by default:

  AGENT_VIDEO_GEN_GENERATOR        import path of a zero-argument callable returning a ``ClipGenerator`` (required)
  AGENT_VIDEO_GEN_SUBMITTED_DIR    a durable dir submitted clips are copied to, out of the run media dir that
                                   is deleted at run end (required)
  AGENT_VIDEO_GEN_MAX_GENERATIONS  generations per episode, at most 3 (default 3)
  AGENT_VIDEO_GEN_MAX_TURNS        policy turns per episode (default 8)
  AGENT_VIDEO_GEN_MAX_SECONDS      longest clip in seconds; longer requests are cut to it (default 5)
  AGENT_VIDEO_GEN_WIDTH / _HEIGHT / _FPS  every clip's shape (default 832x480 at 16 fps)
  AGENT_VIDEO_GEN_TIMEOUT_S        a generation slower than this is an outage (default 600)
  AGENT_VIDEO_GEN_CLIP_OBSERVATIONS  1 to return each clip as a video_url part; needs video through TITO (default 0)
"""

import asyncio
import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from examples.experimental.video_gen_agent.generator import Clip, ClipGenerator, ClipRequest, GeneratorUnavailable
from miles.rollout.agentic.agent_function import InfraAbort
from miles.utils.function_registry import load_function
from miles.utils.http_utils import post

_ENV_PREFIX = "AGENT_VIDEO_GEN_"
_GENERATIONS_LIMIT = 3

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate",
            "description": "Generate a video clip from a text prompt. Returns the clip's id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "What the clip shows, in detail."},
                    "negative_prompt": {"type": "string", "description": "What the clip must not show."},
                    "seed": {"type": "integer", "description": "Random seed; the same arguments give the same clip."},
                    "seconds": {"type": "number", "description": "Clip length in seconds."},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit one generated clip as the answer. Ends the episode.",
            "parameters": {
                "type": "object",
                "properties": {"clip_id": {"type": "string", "description": "The id generate returned."}},
                "required": ["clip_id"],
            },
        },
    },
]


@dataclass(frozen=True)
class VideoGenEnvOptions:
    generator: str = ""
    max_generations: int = 3
    max_turns: int = 8
    max_seconds: float = 5.0
    width: int = 832
    height: int = 480
    fps: int = 16
    timeout_s: float = 600.0
    clip_observations: bool = False
    submitted_dir: str = ""

    def __post_init__(self) -> None:
        if not self.generator:
            raise ValueError(f"{_ENV_PREFIX}GENERATOR must name the clip generator's import path")
        if not self.submitted_dir:
            raise ValueError(f"{_ENV_PREFIX}SUBMITTED_DIR must name a durable dir for submitted clips")
        if self.max_generations > _GENERATIONS_LIMIT:
            raise ValueError(
                f"{_ENV_PREFIX}MAX_GENERATIONS is at most {_GENERATIONS_LIMIT}, got {self.max_generations}"
            )
        for name in ("max_generations", "max_turns", "max_seconds", "width", "height", "fps", "timeout_s"):
            if getattr(self, name) <= 0:  # config-access-exempt: iterates the options dataclass's own fields
                raise ValueError(
                    f"{_ENV_PREFIX}{name.upper()} must be positive, got {getattr(self, name)}"
                )  # config-access-exempt: iterates the options dataclass's own fields

    @classmethod
    def from_env(cls) -> "VideoGenEnvOptions":
        values = {}
        for f in fields(cls):
            raw = os.environ.get(_ENV_PREFIX + f.name.upper())
            if raw is not None:
                values[f.name] = raw == "1" if f.type is bool else type(f.default)(raw)
        return cls(**values)


@dataclass
class _GeneratedClip:
    clip_id: str
    request: ClipRequest
    path: str


@dataclass
class _Episode:
    options: VideoGenEnvOptions
    generator: ClipGenerator
    clips: dict[str, _GeneratedClip] = field(default_factory=dict)
    submitted: dict[str, Any] | None = None
    turns: int = 0
    generation_time: float = 0.0

    @property
    def generations_left(self) -> int:
        return self.options.max_generations - len(self.clips)

    def budget(self) -> str:
        return f"Generations left: {self.generations_left}. Turns left: {self.options.max_turns - self.turns}."


async def run_agent(base_url: str, prompt: list[dict], request_kwargs: dict, metadata: dict, **kwargs) -> dict:
    options = VideoGenEnvOptions.from_env()
    episode = _Episode(options=options, generator=load_function(options.generator)())
    messages = [{"role": "system", "content": _system_prompt(options)}, *prompt]
    while episode.submitted is None and episode.turns < options.max_turns:
        payload = {**request_kwargs, "model": "default", "messages": messages, "tools": TOOLS}
        reply = (await post(f"{base_url}/v1/chat/completions", payload))["choices"][0]["message"]
        episode.turns += 1
        messages.append(reply)
        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:
            messages.append({"role": "user", "content": f"Call generate or submit. {episode.budget()}"})
        for call in tool_calls:
            if episode.submitted is not None:
                break
            content = await _call_tool(episode, call["function"]["name"], call["function"].get("arguments"))
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": content})
    return {
        "video_gen": {
            "request": _user_request(prompt),
            "clips": [{"clip_id": c.clip_id, "path": c.path, **asdict(c.request)} for c in episode.clips.values()],
            "submitted": episode.submitted,
            "turns": episode.turns,
        },
        "agent_metrics": {"total_tool_time": episode.generation_time},
    }


def _system_prompt(options: VideoGenEnvOptions) -> str:
    return (
        "You turn a user's video request into a video clip, using a text-to-video generator through tools. "
        "Write a detailed generation prompt and call generate; when you have a clip that fulfils the request, "
        "call submit with its id. "
        f"Clips are {options.width}x{options.height} at {options.fps} fps and at most {options.max_seconds:g} "
        f"seconds long. You have {options.max_generations} generations and {options.max_turns} turns. "
        "An episode that ends without a submitted clip fails."
    )


def _user_request(prompt: list[dict]) -> str:
    return next(m["content"] for m in prompt if m["role"] == "user")


async def _call_tool(episode: _Episode, name: str, raw_arguments: str | dict | None) -> str | list[dict]:
    try:
        arguments = json.loads(raw_arguments or "{}") if isinstance(raw_arguments, str) else (raw_arguments or {})
    except json.JSONDecodeError:
        return f"Error: the arguments of {name} are not valid JSON. {episode.budget()}"
    if not isinstance(arguments, dict):
        return f"Error: the arguments of {name} must be a JSON object. {episode.budget()}"
    if name == "generate":
        return await _generate(episode, arguments)
    if name == "submit":
        return _submit(episode, arguments)
    return f"Error: unknown tool {name!r}; the tools are generate and submit. {episode.budget()}"


async def _generate(episode: _Episode, arguments: dict) -> str | list[dict]:
    options = episode.options
    if episode.generations_left <= 0:
        return f"Error: no generations left. Submit one of {', '.join(episode.clips)}. {episode.budget()}"
    request, error = _clip_request(options, arguments, default_seed=len(episode.clips))
    if error:
        return f"Error: {error}. {episode.budget()}"
    started = time.monotonic()
    try:
        clip: Clip = await asyncio.wait_for(episode.generator.generate(request), timeout=options.timeout_s)
    except (GeneratorUnavailable, TimeoutError) as e:
        raise InfraAbort("GeneratorUnavailable", f"generating a clip failed: {e!r}") from e
    finally:
        episode.generation_time += time.monotonic() - started
    clip_id = f"clip_{len(episode.clips) + 1}"
    episode.clips[clip_id] = _GeneratedClip(clip_id=clip_id, request=request, path=clip.path)
    cut = f" (cut to the {options.max_seconds:g} s limit)" if arguments.get("seconds", 0) > options.max_seconds else ""
    receipt = f"Generated {clip_id}: {request.seconds:g} s{cut}, seed {request.seed}. {episode.budget()}"
    if not options.clip_observations:
        return receipt
    return [{"type": "text", "text": receipt}, {"type": "video_url", "video_url": {"url": f"file://{clip.path}"}}]


def _clip_request(
    options: VideoGenEnvOptions, arguments: dict, *, default_seed: int
) -> tuple[ClipRequest | None, str | None]:
    prompt = arguments.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, "generate needs a non-empty prompt"
    negative_prompt = arguments.get("negative_prompt") or ""
    seed = arguments.get("seed", default_seed)
    seconds = arguments.get("seconds", options.max_seconds)
    if not isinstance(negative_prompt, str):
        return None, "negative_prompt must be a string"
    if not isinstance(seed, int) or isinstance(seed, bool):
        return None, "seed must be an integer"
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds <= 0:
        return None, "seconds must be a positive number"
    request = ClipRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        seconds=min(float(seconds), options.max_seconds),
        width=options.width,
        height=options.height,
        fps=options.fps,
    )
    return request, None


def _submit(episode: _Episode, arguments: dict) -> str:
    clip_id = arguments.get("clip_id")
    clip = episode.clips.get(clip_id) if isinstance(clip_id, str) else None
    if clip is None:
        known = ", ".join(episode.clips) or "none yet"
        return f"Error: unknown clip_id {clip_id!r}; generated clips: {known}. {episode.budget()}"
    path = _keep(clip.path, episode.options.submitted_dir)
    episode.submitted = {"clip_id": clip.clip_id, "path": path, "prompt": clip.request.prompt}
    return f"Submitted {clip.clip_id}."


def _keep(path: str, submitted_dir: str) -> str:
    """Copy a clip into ``submitted_dir`` under its content hash, which outlives the run media dir."""
    source = Path(path)
    destination = Path(submitted_dir) / f"{hashlib.sha256(source.read_bytes()).hexdigest()}{source.suffix}"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        # episodes of one group often submit the same clip at once; a reader must never see a half-copied file
        partial = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.part")
        shutil.copyfile(source, partial)
        os.replace(partial, destination)
    return str(destination)
