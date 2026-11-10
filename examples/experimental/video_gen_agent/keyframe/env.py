"""Keyframe env: the planner makes up to three keyframe images, sees each, then makes one video, which is its answer.

An agent function for ``agentic_tool_call.generate`` under session v2, with two tools:
``generate_image(prompt, negative_prompt, seed)`` returns the image as an ``image_url`` part beside a receipt, and
``generate_video(image_id, prompt, negative_prompt, seed)`` makes one clip and ends the episode with it as the
submission. In ``preview`` mode the clip is text-to-video from the chosen image's prompt followed by that prompt,
which says what moves and how the camera moves; in ``i2v`` mode it is image-to-video from the chosen image. In
``reference`` mode the images are assets, one per subject or object isolated on a plain background plus one for the
setting, and ``generate_video`` takes ``image_ids``, one or more watched assets: the clip is reference-to-video, showing
them without starting from any of them, from the planner's prompt alone, which composes them. The planner never
watches the clip.

Like ``examples.experimental.video_gen_agent.env``, it does not score: the clip is copied out by content hash and
recorded under ``metadata["video_gen"]["submitted"]``, which ``reward.reward_func`` scores. A turn cut at the token
limit ends the episode without a video, which scores 0. A generator outage or timeout raises
``InfraAbort("GeneratorUnavailable")`` and any other env fault ``InfraAbort("EnvError")``, so the group is dropped.

Run-wide options come from ``AGENT_VIDEO_GEN_*`` env vars, which worker and sandbox modes forward by default:

  AGENT_VIDEO_GEN_GENERATOR        import path of a zero-argument callable returning a ``ClipGenerator`` (required)
  AGENT_VIDEO_GEN_IMAGE_GENERATOR  import path of a zero-argument callable returning an ``ImageGenerator`` (required)
  AGENT_VIDEO_GEN_SUBMITTED_DIR    a durable dir submitted clips are copied to (required)
  AGENT_VIDEO_GEN_MAX_IMAGES       images per episode, at most 4 (default 3)
  AGENT_VIDEO_GEN_MAX_TURNS        policy turns per episode (default 6)
  AGENT_VIDEO_GEN_MAX_PROMPT_WORDS a longer generator prompt gets an error reply; 0, the default, sets no cap. In
                                   preview mode the video's prompt is the image's and the planner's together
  AGENT_VIDEO_GEN_SECONDS          the video's length (default 2)
  AGENT_VIDEO_GEN_WIDTH / _HEIGHT / _FPS  every image's and the video's shape (default 832x480 at 16 fps)
  AGENT_VIDEO_GEN_TIMEOUT_S        a generation slower than this is an outage (default 600)
  AGENT_VIDEO_GEN_VIDEO_MODE       ``preview``, ``i2v`` or ``reference`` (default preview)
  AGENT_VIDEO_GEN_IMAGE_MODEL / _VIDEO_MODEL  the generators' names, recorded on each episode (default: unrecorded)
"""

import asyncio
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from examples.experimental.video_gen_agent.env import (
    _ENV_PREFIX,
    _keep,
    _media_part,
    _user_request,
    options_from_env,
    parse_arguments,
    post,
)
from examples.experimental.video_gen_agent.generator import (
    ClipGenerator,
    ClipRequest,
    GeneratorUnavailable,
    ImageGenerator,
    ImageRequest,
)
from miles.rollout.agentic.agent_function import InfraAbort
from miles.utils.function_registry import load_function

_IMAGES_LIMIT = 4
_VIDEO_MODES = ("preview", "i2v", "reference")
_VIDEO_ID = "video_1"

_SHARED_PARAMETERS = {
    "negative_prompt": {"type": "string", "description": "What it must not show."},
    "seed": {"type": "integer", "description": "Random seed; the same arguments give the same result."},
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate a keyframe image from a text prompt. Returns the image and its id.",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string", "description": "What the image shows, in detail."}}
                | _SHARED_PARAMETERS,
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_video",
            "description": "Generate the final video from one of your images. Submits it and ends the episode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_id": {"type": "string", "description": "The id generate_image returned."},
                    "prompt": {"type": "string", "description": "What moves in the video and how the camera moves."},
                }
                | _SHARED_PARAMETERS,
                "required": ["image_id", "prompt"],
            },
        },
    },
]

# Reference mode's images are assets, and its generate_video takes the ids of several of them in place of one.
REFERENCE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Generate one asset image: a single subject or object isolated on a plain background, or the setting "
                "with nothing in it. Returns the image and its id."
            ),
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string", "description": "The one asset this image shows, in detail."}}
                | _SHARED_PARAMETERS,
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_video",
            "description": "Generate the final video from your assets. Submits it and ends the episode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The ids generate_image returned for the assets the video shows.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The whole video: where each asset is, what it does, and how the camera moves.",
                    },
                }
                | _SHARED_PARAMETERS,
                "required": ["image_ids", "prompt"],
            },
        },
    },
]


@dataclass(frozen=True)
class KeyframeEnvOptions:
    generator: str = ""
    image_generator: str = ""
    submitted_dir: str = ""
    max_images: int = 3
    max_turns: int = 6
    max_prompt_words: int = 0
    seconds: float = 2.0
    width: int = 832
    height: int = 480
    fps: int = 16
    timeout_s: float = 600.0
    video_mode: str = "preview"
    image_model: str = ""
    video_model: str = ""

    def __post_init__(self) -> None:
        for name in ("generator", "image_generator", "submitted_dir"):
            if not getattr(self, name):  # config-access-exempt: iterates the options dataclass's own fields
                raise ValueError(f"{_ENV_PREFIX}{name.upper()} is required")
        for name in ("max_images", "max_turns", "seconds", "width", "height", "fps", "timeout_s"):
            if getattr(self, name) <= 0:  # config-access-exempt: iterates the options dataclass's own fields
                raise ValueError(
                    f"{_ENV_PREFIX}{name.upper()} must be positive, got {getattr(self, name)}"
                )  # config-access-exempt: iterates the options dataclass's own fields
        if self.max_prompt_words < 0:
            raise ValueError(f"{_ENV_PREFIX}MAX_PROMPT_WORDS is 0 (no cap) or more, got {self.max_prompt_words}")
        if self.max_images > _IMAGES_LIMIT:
            raise ValueError(f"{_ENV_PREFIX}MAX_IMAGES is at most {_IMAGES_LIMIT}, got {self.max_images}")
        if self.video_mode not in _VIDEO_MODES:
            raise ValueError(f"{_ENV_PREFIX}VIDEO_MODE is one of {', '.join(_VIDEO_MODES)}, got {self.video_mode!r}")

    @classmethod
    def from_env(cls) -> "KeyframeEnvOptions":
        return options_from_env(cls)


@dataclass
class _GeneratedImage:
    image_id: str
    request: ImageRequest
    path: str
    # True once a planner request has carried the image, so a video only ever comes from a watched image.
    shown: bool = False


@dataclass
class _Episode:
    options: KeyframeEnvOptions
    image_generator: ImageGenerator
    video_generator: ClipGenerator
    images: dict[str, _GeneratedImage] = field(default_factory=dict)
    video: dict[str, Any] | None = None
    submitted: dict[str, Any] | None = None
    turns: int = 0
    turn_stats: list[dict] = field(default_factory=list)
    generation_time: float = 0.0

    @property
    def images_left(self) -> int:
        return self.options.max_images - len(self.images)

    def budget(self) -> str:
        return f"Images left: {self.images_left}. Turns left: {self.options.max_turns - self.turns}."


async def run_agent(base_url: str, prompt: list[dict], request_kwargs: dict, metadata: dict, **kwargs) -> dict:
    try:
        options = KeyframeEnvOptions.from_env()
        episode = _Episode(
            options=options,
            image_generator=load_function(options.image_generator)(),
            video_generator=load_function(options.generator)(),
        )
        messages = [{"role": "system", "content": _system_prompt(options)}, *prompt]
        await _play(episode, messages, base_url, request_kwargs)
    except InfraAbort:
        raise
    except Exception as e:
        # agentic_tool_call trains a failed agent's turns at reward 0, but an env fault is not the policy's doing.
        raise InfraAbort("EnvError", f"the keyframe env failed: {e!r}") from e
    return {
        "video_gen": {
            "request": _user_request(prompt),
            # So a comparison never mixes episodes of different generators.
            "video_mode": episode.options.video_mode,
            "image_model": episode.options.image_model,
            "video_model": episode.options.video_model,
            "images": [{"image_id": i.image_id, "path": i.path, **asdict(i.request)} for i in episode.images.values()],
            "video": episode.video,
            "submitted": episode.submitted,
            "turns": episode.turns,
            "turn_stats": episode.turn_stats,
        },
        "agent_metrics": {"total_tool_time": episode.generation_time},
    }


async def _play(episode: _Episode, messages: list[dict], base_url: str, request_kwargs: dict) -> None:
    while episode.submitted is None and episode.turns < episode.options.max_turns:
        tools = REFERENCE_TOOLS if episode.options.video_mode == "reference" else TOOLS
        payload = {**request_kwargs, "model": "default", "messages": messages, "tools": tools}
        for image in episode.images.values():
            image.shown = True
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
            # Session v2 refuses to extend a length-truncated turn (409), so the episode ends here without a video.
            return
        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:
            messages.append({"role": "user", "content": f"Call generate_image or generate_video. {episode.budget()}"})
        for call in tool_calls:
            if episode.submitted is not None:
                break
            content = await _call_tool(episode, call["function"]["name"], call["function"].get("arguments"))
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": content})


def _system_prompt(options: KeyframeEnvOptions) -> str:
    if options.video_mode == "reference":
        return _reference_system_prompt(options)
    if options.video_mode == "i2v":
        video = (
            "generate_video with the id of the image the video should come from and a prompt: it makes one clip "
            "that starts from that image, and the prompt says what moves and how the camera moves"
        )
    else:
        video = (
            "generate_video with the id of the image the video should come from and a prompt: it makes one clip from "
            "text, the chosen image's prompt followed by yours, so the clip looks like the image without starting "
            "from it; your prompt says what moves and how the camera moves"
        )
    return (
        "You turn a user's video request into a video, using an image generator and a video generator through "
        f"tools. Call generate_image to make keyframe images; you see each one. Then call {video}. That clip is your "
        "answer, and the "
        f"episode ends. Images and the video are {options.width}x{options.height}; the video is "
        f"{options.seconds:g} seconds at {options.fps} fps. You have {options.max_images} images, 1 video and "
        f"{options.max_turns} turns. {_prompt_cap(options)}An episode that ends without a video fails."
    )


def _reference_system_prompt(options: KeyframeEnvOptions) -> str:
    return (
        "You turn a user's video request into a video, using an image generator and a video generator through "
        "tools. First call generate_image to make the video's assets, and you see each one: one image for each "
        "subject or object in the video, alone and isolated on a plain background, and one image of the setting with "
        "nothing in it. Do not make whole scenes or variants of one scene. Then call generate_video with the ids of "
        "the assets the video shows and a prompt that composes them: where each one is, what it does, and how the "
        "camera moves. The video generator uses the assets as references, and reads only your prompt as text, so the "
        "prompt names everything the video shows. That clip is your answer, and the episode ends. Images and the "
        f"video are {options.width}x{options.height}; the video is {options.seconds:g} seconds at {options.fps} fps. "
        f"You have {options.max_images} images, 1 video and {options.max_turns} turns. {_prompt_cap(options)}An "
        "episode that ends without a video fails."
    )


def _prompt_cap(options: KeyframeEnvOptions) -> str:
    if not options.max_prompt_words:
        return ""
    if options.video_mode != "preview":
        return f"Each prompt is at most {options.max_prompt_words} words. "
    return (
        f"Each prompt is at most {options.max_prompt_words} words, and the chosen image's prompt and your video prompt "
        "together too. "
    )


async def _call_tool(episode: _Episode, name: str, raw_arguments: str | dict | None) -> str | list[dict]:
    arguments, error = parse_arguments(name, raw_arguments)
    if error:
        return f"Error: {error}. {episode.budget()}"
    if name == "generate_image":
        return await _generate_image(episode, arguments)
    if name == "generate_video":
        return await _generate_video(episode, arguments)
    return f"Error: unknown tool {name!r}; the tools are generate_image and generate_video. {episode.budget()}"


async def _generate_image(episode: _Episode, arguments: dict) -> str | list[dict]:
    if episode.images_left <= 0:
        return (
            f"Error: no images left. Call generate_video with one of {', '.join(episode.images)}. {episode.budget()}"
        )
    prompted, error = _prompted(
        arguments, tool="generate_image", default_seed=len(episode.images), max_words=episode.options.max_prompt_words
    )
    if error:
        return f"Error: {error}. {episode.budget()}"
    options = episode.options
    request = ImageRequest(**prompted, width=options.width, height=options.height)
    image = await _generated(episode, episode.image_generator.generate(request))
    image_id = f"image_{len(episode.images) + 1}"
    episode.images[image_id] = _GeneratedImage(image_id=image_id, request=request, path=image.path)
    receipt = f"Generated {image_id} from seed {request.seed}. {episode.budget()}"
    return [{"type": "text", "text": receipt}, _media_part("image_url", image.path)]


async def _generate_video(episode: _Episode, arguments: dict) -> str:
    options = episode.options
    reference = options.video_mode == "reference"
    images, error = _chosen_images(episode, arguments, several=reference)
    if error:
        return f"Error: {error}. {episode.budget()}"
    prompted, error = _prompted(
        arguments, tool="generate_video", default_seed=0, max_words=episode.options.max_prompt_words
    )
    if error:
        return f"Error: {error}. {episode.budget()}"
    request = ClipRequest(
        **prompted,
        seconds=options.seconds,
        width=options.width,
        height=options.height,
        fps=options.fps,
    )
    if options.video_mode == "i2v":
        request = replace(request, image_path=images[0].path)
    elif reference:
        # An asset's prompt describes it isolated on a plain background, so only the planner's prompt reaches the video.
        request = replace(request, reference_paths=tuple(image.path for image in images))
    else:
        request = replace(request, prompt=f"{images[0].request.prompt} {request.prompt}")
        # The cap bounds what the generator reads, here the chosen image's prompt and this one together.
        if options.max_prompt_words and (words := len(request.prompt.split())) > options.max_prompt_words:
            return (
                f"Error: {images[0].image_id}'s prompt and the generate_video prompt have {words} words together; the "
                f"limit is {options.max_prompt_words}. {episode.budget()}"
            )
    clip = await _generated(episode, episode.video_generator.generate(request))
    if reference:
        chosen = {"image_ids": [image.image_id for image in images]}
        chosen_prompts = {"image_prompts": [image.request.prompt for image in images]}
    else:
        chosen = {"image_id": images[0].image_id}
        chosen_prompts = {"image_prompt": images[0].request.prompt}
    episode.video = {"clip_id": _VIDEO_ID, "path": clip.path, **chosen, **asdict(request)}
    episode.submitted = {
        "clip_id": _VIDEO_ID,
        "path": _keep(clip.path, options.submitted_dir),
        "prompt": request.prompt,
        "prompt_words": len(request.prompt.split()),
        **chosen,
        **chosen_prompts,
    }
    return f"Submitted {_VIDEO_ID}."


def _chosen_images(
    episode: _Episode, arguments: dict, *, several: bool
) -> tuple[list[_GeneratedImage] | None, str | None]:
    """The watched images a generate_video call names, ``image_ids`` when it takes several, or what is wrong."""
    if several:
        image_ids = arguments.get("image_ids")
        if not isinstance(image_ids, list) or not image_ids or not all(isinstance(i, str) for i in image_ids):
            return None, "image_ids must be a non-empty list of the ids generate_image returned"
        if len(set(image_ids)) != len(image_ids):
            return None, f"image_ids names an image twice: {image_ids}"
    else:
        image_ids = [arguments.get("image_id")]
    images = []
    for image_id in image_ids:
        image = episode.images.get(image_id) if isinstance(image_id, str) else None
        if image is None:
            known = ", ".join(episode.images) or "none yet; call generate_image first"
            name = "image_ids entry" if several else "image_id"
            return None, f"unknown {name} {image_id!r}; generated images: {known}"
        if not image.shown:
            return None, f"you have not seen {image.image_id} yet; look at it, then call generate_video"
        images.append(image)
    return images, None



async def _generated(episode: _Episode, generation):
    started = time.monotonic()
    try:
        return await asyncio.wait_for(generation, timeout=episode.options.timeout_s)
    except (GeneratorUnavailable, TimeoutError) as e:
        raise InfraAbort("GeneratorUnavailable", f"generating failed: {e!r}") from e
    finally:
        episode.generation_time += time.monotonic() - started


def _prompted(arguments: dict, *, tool: str, default_seed: int, max_words: int) -> tuple[dict | None, str | None]:
    """The prompt, negative prompt and seed of a call, or what is wrong with them."""
    prompt = arguments.get("prompt")
    negative_prompt = arguments.get("negative_prompt") or ""
    seed = arguments.get("seed", default_seed)
    if not isinstance(prompt, str) or not prompt.strip():
        return None, f"{tool} needs a non-empty prompt"
    if max_words and (words := len(prompt.split())) > max_words:
        return None, f"the {tool} prompt has {words} words; the limit is {max_words}"
    if not isinstance(negative_prompt, str):
        return None, "negative_prompt must be a string"
    if not isinstance(seed, int) or isinstance(seed, bool):
        return None, "seed must be an integer"
    return {"prompt": prompt, "negative_prompt": negative_prompt, "seed": seed}, None
