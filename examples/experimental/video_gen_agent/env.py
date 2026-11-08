"""Video-generation env: an agent function for ``agentic_tool_call.generate`` under session v2.

The planner turns a user's video request into clips with a frozen generator, through two tools:
``generate(prompt, negative_prompt, seed, seconds)`` and ``submit(clip_id)``. The env fixes resolution and fps,
caps clip length, generations and turns, and states the remaining budget in every tool reply.

Two more tools are opt-in. ``extend(clip_id, prompt, ...)`` continues a clip as a new shot, image-to-video from its
last frame, and submitting an extended clip submits its shots joined into one video. With clip observations on,
``AGENT_VIDEO_GEN_KEEP_CLIPS`` keeps only the last K clips in view: each older one is folded to a note or its last
frame in the request the env sends, and ``recall(clip_id)`` shows it again. Session v2 branches at every fold, and
each branch becomes a sample with the episode's reward.

It does not score. It returns the submitted clip under ``metadata["video_gen"]``, which v2 copies into every
kept sample, and ``examples.experimental.video_gen_agent.reward.reward_func`` scores it in the rollout's reward step.
A turn cut at the token limit ends the episode unsubmitted. Any other failure raises ``InfraAbort``, so the group is
dropped rather than trained at reward 0, and ``--max-consecutive-aborted-groups`` stops a run whose env keeps failing.

Run-wide options come from ``AGENT_VIDEO_GEN_*`` env vars, which worker and sandbox modes forward by default:

  AGENT_VIDEO_GEN_GENERATOR        import path of a zero-argument callable returning a ``ClipGenerator`` (required)
  AGENT_VIDEO_GEN_SUBMITTED_DIR    a durable dir submitted clips are copied to, out of the run media dir that
                                   is deleted at run end (required)
  AGENT_VIDEO_GEN_MAX_GENERATIONS  generations per episode, at most 3 (default 3)
  AGENT_VIDEO_GEN_MAX_EXTENDS      extends per episode, beside the generations; 0 offers no extend tool (default 0)
  AGENT_VIDEO_GEN_MIN_SHOTS        shots a submitted video needs; a shorter chain is refused (default 0: any)
  AGENT_VIDEO_GEN_MAX_TURNS        policy turns per episode (default 8)
  AGENT_VIDEO_GEN_MAX_SECONDS      longest clip in seconds; longer requests are cut to it (default 5)
  AGENT_VIDEO_GEN_WIDTH / _HEIGHT / _FPS  every clip's shape (default 832x480 at 16 fps)
  AGENT_VIDEO_GEN_TIMEOUT_S        a generation slower than this is an outage (default 600)
  AGENT_VIDEO_GEN_CLIP_OBSERVATIONS  1 to return each clip as a video_url part; needs video through TITO (default 0)
  AGENT_VIDEO_GEN_KEEP_CLIPS       clips kept in view; older ones fold, and a recall tool is offered (default 0: all)
  AGENT_VIDEO_GEN_FOLD             what a folded clip becomes: ``text`` (a note) or ``keyframe`` (default text)
"""

import asyncio
import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from examples.experimental.video_gen_agent.clip_media import join_shots, last_frame
from examples.experimental.video_gen_agent.generator import Clip, ClipGenerator, ClipRequest, GeneratorUnavailable
from miles.rollout.agentic.agent_function import InfraAbort
from miles.utils.function_registry import load_function
from miles.utils.http_utils import post

_ENV_PREFIX = "AGENT_VIDEO_GEN_"
_GENERATIONS_LIMIT = 3
_FOLDS = ("text", "keyframe")

_CLIP_PARAMETERS = {
    "prompt": {"type": "string", "description": "What the clip shows, in detail."},
    "negative_prompt": {"type": "string", "description": "What the clip must not show."},
    "seed": {"type": "integer", "description": "Random seed; the same arguments give the same clip."},
    "seconds": {"type": "number", "description": "Clip length in seconds."},
}
_CLIP_ID = {"clip_id": {"type": "string", "description": "The id generate returned."}}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate",
            "description": "Generate a video clip from a text prompt. Returns the clip's id.",
            "parameters": {"type": "object", "properties": _CLIP_PARAMETERS, "required": ["prompt"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit one generated clip as the answer. Ends the episode.",
            "parameters": {"type": "object", "properties": _CLIP_ID, "required": ["clip_id"]},
        },
    },
]

EXTEND_TOOL = {
    "type": "function",
    "function": {
        "name": "extend",
        "description": "Continue a clip as a new shot, starting from its last frame. Returns the new clip's id.",
        "parameters": {
            "type": "object",
            "properties": _CLIP_ID | _CLIP_PARAMETERS,
            "required": ["clip_id", "prompt"],
        },
    },
}

RECALL_TOOL = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": "Show a folded clip again.",
        "parameters": {"type": "object", "properties": _CLIP_ID, "required": ["clip_id"]},
    },
}


@dataclass(frozen=True)
class VideoGenEnvOptions:
    generator: str = ""
    max_generations: int = 3
    max_extends: int = 0
    min_shots: int = 0
    max_turns: int = 8
    max_seconds: float = 5.0
    width: int = 832
    height: int = 480
    fps: int = 16
    timeout_s: float = 600.0
    clip_observations: bool = False
    keep_clips: int = 0
    fold: str = "text"
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
        for name in ("max_extends", "min_shots", "keep_clips"):
            if getattr(self, name) < 0:  # config-access-exempt: iterates the options dataclass's own fields
                raise ValueError(
                    f"{_ENV_PREFIX}{name.upper()} must not be negative, got {getattr(self, name)}"
                )  # config-access-exempt: iterates the options dataclass's own fields
        if self.min_shots > 1 + self.max_extends:
            raise ValueError(
                f"{_ENV_PREFIX}MIN_SHOTS {self.min_shots} needs MAX_EXTENDS of at least {self.min_shots - 1}, "
                f"got {self.max_extends}"
            )
        if self.fold not in _FOLDS:
            raise ValueError(f"{_ENV_PREFIX}FOLD is one of {', '.join(_FOLDS)}, got {self.fold!r}")
        if self.keep_clips and not self.clip_observations:
            raise ValueError(f"{_ENV_PREFIX}KEEP_CLIPS folds clip observations, so it needs CLIP_OBSERVATIONS=1")

    @classmethod
    def from_env(cls) -> "VideoGenEnvOptions":
        values = {}
        for f in fields(cls):
            raw = os.environ.get(_ENV_PREFIX + f.name.upper())
            if raw is not None:
                values[f.name] = raw == "1" if f.type is bool else type(f.default)(raw)
        return cls(**values)

    @property
    def tools(self) -> list[dict]:
        return [
            *TOOLS,
            *([EXTEND_TOOL] if self.max_extends else []),
            *([RECALL_TOOL] if self.keep_clips else []),
        ]


@dataclass
class _GeneratedClip:
    clip_id: str
    request: ClipRequest
    path: str
    parent: str | None = None


@dataclass
class _Episode:
    options: VideoGenEnvOptions
    generator: ClipGenerator
    clips: dict[str, _GeneratedClip] = field(default_factory=dict)
    # (message index, clip id) of every tool reply that shows a clip, oldest first.
    observations: list[tuple[int, str]] = field(default_factory=list)
    submitted: dict[str, Any] | None = None
    turns: int = 0
    recalls: int = 0
    turn_stats: list[dict] = field(default_factory=list)
    generation_time: float = 0.0

    @property
    def generations_left(self) -> int:
        return self.options.max_generations - sum(clip.parent is None for clip in self.clips.values())

    @property
    def extends_left(self) -> int:
        return self.options.max_extends - sum(clip.parent is not None for clip in self.clips.values())

    @property
    def folded(self) -> list[tuple[int, str]]:
        keep = self.options.keep_clips
        return self.observations[:-keep] if keep else []

    def budget(self) -> str:
        extends = f" Extends left: {self.extends_left}." if self.options.max_extends else ""
        return (
            f"Generations left: {self.generations_left}.{extends} Turns left: {self.options.max_turns - self.turns}."
        )


async def run_agent(base_url: str, prompt: list[dict], request_kwargs: dict, metadata: dict, **kwargs) -> dict:
    options = VideoGenEnvOptions.from_env()
    episode = _Episode(options=options, generator=load_function(options.generator)())
    messages = [{"role": "system", "content": _system_prompt(options)}, *prompt]
    try:
        await _play(episode, messages, base_url, request_kwargs)
    except InfraAbort:
        raise
    except Exception as e:
        # agentic_tool_call trains a failed agent's turns at reward 0, but an env fault is not the policy's doing.
        raise InfraAbort("EnvError", f"the video-generation env failed: {e!r}") from e
    return {
        "video_gen": {
            "request": _user_request(prompt),
            "clips": [
                {"clip_id": c.clip_id, "path": c.path, "parent": c.parent, **asdict(c.request)}
                for c in episode.clips.values()
            ],
            "submitted": episode.submitted,
            "turns": episode.turns,
            "folded_clips": len(episode.folded),
            "recalls": episode.recalls,
            "turn_stats": episode.turn_stats,
        },
        "agent_metrics": {"total_tool_time": episode.generation_time},
    }


async def _play(episode: _Episode, messages: list[dict], base_url: str, request_kwargs: dict) -> None:
    while episode.submitted is None and episode.turns < episode.options.max_turns:
        payload = {**request_kwargs, "model": "default", "messages": await _view(episode, messages)}
        payload["tools"] = episode.options.tools
        started = time.monotonic()
        response = await post(f"{base_url}/v1/chat/completions", payload)
        episode.turn_stats.append(
            {
                "prompt_tokens": (response.get("usage") or {}).get("prompt_tokens"),
                "seconds": time.monotonic() - started,
            }
        )
        choice = response["choices"][0]
        reply = choice["message"]
        episode.turns += 1
        messages.append(reply)
        if choice.get("finish_reason") == "length":
            # Session v2 refuses to extend a length-truncated turn (409), so the episode ends here unsubmitted.
            return
        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:
            messages.append({"role": "user", "content": f"Call generate or submit. {episode.budget()}"})
        for call in tool_calls:
            if episode.submitted is not None:
                break
            content, shown = await _call_tool(episode, call["function"]["name"], call["function"].get("arguments"))
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": content})
            if shown is not None:
                episode.observations.append((len(messages) - 1, shown))


def _system_prompt(options: VideoGenEnvOptions) -> str:
    text = (
        "You turn a user's video request into a video clip, using a text-to-video generator through tools. "
        "Write a detailed generation prompt and call generate; when you have a clip that fulfils the request, "
        "call submit with its id. "
        f"Clips are {options.width}x{options.height} at {options.fps} fps and at most {options.max_seconds:g} "
        f"seconds long. You have {options.max_generations} generations and {options.max_turns} turns. "
        "An episode that ends without a submitted clip fails."
    )
    if options.max_extends:
        text += (
            " To make a longer video, call extend with a clip's id: it continues that clip as a new shot from its "
            "last frame, and submitting the last shot submits all its shots joined in order, as one video. "
            f"You have {options.max_extends} extends."
        )
    if options.min_shots > 1:
        text += f" The video must have at least {options.min_shots} shots: extend until it does, then submit its last shot."
    if options.keep_clips:
        folded_to = "a note" if options.fold == "text" else "their last frame"
        text += (
            f" Only your last {options.keep_clips} clips stay in view; older ones are folded to {folded_to}, "
            "and recall shows one again."
        )
    return text


def _user_request(prompt: list[dict]) -> str:
    return next(m["content"] for m in prompt if m["role"] == "user")


async def _view(episode: _Episode, messages: list[dict]) -> list[dict]:
    """The history as the policy sees it: every clip observation but the last K folded."""
    folded = {index: await _folded_content(episode, messages[index], clip_id) for index, clip_id in episode.folded}
    return [{**m, "content": folded[i]} if i in folded else m for i, m in enumerate(messages)]


async def _folded_content(episode: _Episode, message: dict, clip_id: str) -> str | list[dict]:
    receipt = " ".join(part["text"] for part in message["content"] if part["type"] == "text")
    clip = episode.clips[clip_id]
    if episode.options.fold == "text":
        return (
            f'{receipt} {clip_id} is folded out of view; it was generated from the prompt "{clip.request.prompt}". '
            "Call recall to watch it again."
        )
    frame = await asyncio.to_thread(last_frame, clip.path)
    return [
        {"type": "text", "text": f"{receipt} {clip_id} is folded to its last frame; call recall to watch it again."},
        _media_part("image_url", frame),
    ]


def _media_part(kind: str, path: str) -> dict:
    return {"type": kind, kind: {"url": f"file://{path}"}}


async def _call_tool(
    episode: _Episode, name: str, raw_arguments: str | dict | None
) -> tuple[str | list[dict], str | None]:
    """The tool's reply, and the id of the clip it shows, if any."""
    try:
        arguments = json.loads(raw_arguments or "{}") if isinstance(raw_arguments, str) else (raw_arguments or {})
    except json.JSONDecodeError:
        return f"Error: the arguments of {name} are not valid JSON. {episode.budget()}", None
    if not isinstance(arguments, dict):
        return f"Error: the arguments of {name} must be a JSON object. {episode.budget()}", None
    if name == "generate":
        return await _generate(episode, arguments)
    if name == "submit":
        return await _submit(episode, arguments), None
    if name == "extend" and episode.options.max_extends:
        return await _extend(episode, arguments)
    if name == "recall" and episode.options.keep_clips:
        return _recall(episode, arguments)
    names = [tool["function"]["name"] for tool in episode.options.tools]
    listed = f"{', '.join(names[:-1])} and {names[-1]}"
    return f"Error: unknown tool {name!r}; the tools are {listed}. {episode.budget()}", None


async def _generate(episode: _Episode, arguments: dict) -> tuple[str | list[dict], str | None]:
    if episode.generations_left <= 0:
        return f"Error: no generations left. Submit one of {', '.join(episode.clips)}. {episode.budget()}", None
    request, error = _clip_request(episode.options, arguments, default_seed=len(episode.clips))
    if error:
        return f"Error: {error}. {episode.budget()}", None
    return await _make_clip(episode, request, arguments, parent=None)


async def _extend(episode: _Episode, arguments: dict) -> tuple[str | list[dict], str | None]:
    if episode.extends_left <= 0:
        return f"Error: no extends left. {episode.budget()}", None
    parent, error = _known_clip(episode, arguments)
    if error:
        return error, None
    request, error = _clip_request(episode.options, arguments, default_seed=len(episode.clips))
    if error:
        return f"Error: {error}. {episode.budget()}", None
    frame = await asyncio.to_thread(last_frame, parent.path)
    return await _make_clip(episode, replace(request, image_path=frame), arguments, parent=parent.clip_id)


async def _make_clip(
    episode: _Episode, request: ClipRequest, arguments: dict, *, parent: str | None
) -> tuple[str | list[dict], str | None]:
    options = episode.options
    started = time.monotonic()
    try:
        clip: Clip = await asyncio.wait_for(episode.generator.generate(request), timeout=options.timeout_s)
    except (GeneratorUnavailable, TimeoutError) as e:
        raise InfraAbort("GeneratorUnavailable", f"generating a clip failed: {e!r}") from e
    finally:
        episode.generation_time += time.monotonic() - started
    clip_id = f"clip_{len(episode.clips) + 1}"
    episode.clips[clip_id] = _GeneratedClip(clip_id=clip_id, request=request, path=clip.path, parent=parent)
    cut = f" (cut to the {options.max_seconds:g} s limit)" if arguments.get("seconds", 0) > options.max_seconds else ""
    extending = f" extending {parent}" if parent else ""
    receipt = f"Generated {clip_id}{extending}: {request.seconds:g} s{cut}, seed {request.seed}. {episode.budget()}"
    if not options.clip_observations:
        return receipt, None
    return [{"type": "text", "text": receipt}, _media_part("video_url", clip.path)], clip_id


def _recall(episode: _Episode, arguments: dict) -> tuple[str | list[dict], str | None]:
    clip, error = _known_clip(episode, arguments)
    if error:
        return error, None
    if clip.clip_id in {clip_id for _, clip_id in episode.observations[-episode.options.keep_clips :]}:
        return f"Error: {clip.clip_id} is already in view. {episode.budget()}", None
    episode.recalls += 1
    # The clip's own path, so the engines' feature cache serves it again.
    receipt = f"Recalled {clip.clip_id}. {episode.budget()}"
    return [{"type": "text", "text": receipt}, _media_part("video_url", clip.path)], clip.clip_id


def _known_clip(episode: _Episode, arguments: dict) -> tuple[_GeneratedClip | None, str | None]:
    clip_id = arguments.get("clip_id")
    clip = episode.clips.get(clip_id) if isinstance(clip_id, str) else None
    if clip is None:
        known = ", ".join(episode.clips) or "none yet"
        return None, f"Error: unknown clip_id {clip_id!r}; generated clips: {known}. {episode.budget()}"
    return clip, None


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
    # `not >` also rejects NaN, which the qwen3_coder parser passes through as a JSON number.
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not seconds > 0:
        return None, "seconds must be a positive number"
    request = ClipRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        seconds=float(min(seconds, options.max_seconds)),
        width=options.width,
        height=options.height,
        fps=options.fps,
    )
    return request, None


async def _submit(episode: _Episode, arguments: dict) -> str:
    clip, error = _known_clip(episode, arguments)
    if error:
        return error
    shots = _shots(episode, clip)
    if len(shots) < episode.options.min_shots:
        return (
            f"Error: the video needs at least {episode.options.min_shots} shots, and {clip.clip_id} has "
            f"{len(shots)}; extend it first. {episode.budget()}"
        )
    submitted_dir = episode.options.submitted_dir
    if len(shots) == 1:
        path = _keep(clip.path, submitted_dir)
    else:
        paths = [shot.path for shot in shots]
        path = await asyncio.to_thread(join_shots, paths, fps=episode.options.fps, output_dir=submitted_dir)
    episode.submitted = {"clip_id": clip.clip_id, "path": path, "prompt": clip.request.prompt}
    if len(shots) > 1:
        episode.submitted["shots"] = [shot.clip_id for shot in shots]
    return f"Submitted {clip.clip_id}."


def _shots(episode: _Episode, clip: _GeneratedClip) -> list[_GeneratedClip]:
    """The clip and the shots it extends, the first shot first."""
    chain = [clip]
    while chain[-1].parent is not None:
        chain.append(episode.clips[chain[-1].parent])
    return chain[::-1]


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
