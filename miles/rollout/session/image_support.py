"""The session server's image capability, constructed once per process.

``build_image_support`` returns ``None`` unless ``--session-image-support`` names
a family, so nothing here is loaded — and no processor is read off disk — for a
text-only run. The returned object is read-only and shared by every session; the
per-session half is ``SessionMedia``, which owns that session's interned bytes
and dies with it.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import orjson

from miles.rollout.session.config import SESSION_IMAGE_SUPPORT_OFF, SessionServerConfig
from miles.rollout.session.errors import EngineNotReadyError, EngineUnsupportedError, SessionError
from miles.rollout.session.generate_transport import CompletionParsers
from miles.rollout.session.image_processing import (
    ImageTokenCounter,
    ProcessedImage,
    ProcessorFingerprint,
    build_fingerprint,
    process_images,
)
from miles.rollout.session.media import EMPTY_MEDIA, MediaArena, RequestMedia, extract_request_media

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineImageConfig:
    """How the engine says it will preprocess images, read from the engine itself.

    Matching token ids prove the two sides agree on the image *grid*, because the
    expanded length is derived from it — they do not prove they agree on the pixel
    values behind those tokens. A decode or normalization override changes the
    pixels while leaving the grid, and therefore every token id, identical. So the
    engine's own configuration decides whether this run may serve images, and a
    reply that does not state it is refused rather than defaulted.
    """

    model_path: str
    revision: str | None
    mm_process_config: dict[str, Any]
    image_processor_backend: str
    sampling_defaults: str
    tool_call_parser: str | None
    reasoning_parser: str | None

    # The fields a reply must actually carry. Absent is refused, never defaulted: a
    # router-manager reply, an error body and `{}` all used to read as "no override".
    REQUIRED = ("model_path", "mm_process_config", "image_processor_backend", "sampling_defaults")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "revision": self.revision,
            "mm_process_config": self.mm_process_config,
            "image_processor_backend": self.image_processor_backend,
            "sampling_defaults": self.sampling_defaults,
            "tool_call_parser": self.tool_call_parser,
            "reasoning_parser": self.reasoning_parser,
        }

    def model_sampling_defaults(self, generation_defaults: dict[str, Any]) -> dict[str, Any]:
        """The defaults the engine applies to a field the request omits.

        Mirrors `ModelConfig.get_default_sampling_params`, which returns the
        checkpoint's generation config only when `sampling_defaults == "model"`.
        Without this an omitted field would take SGLang's own fallback on the image
        route and the checkpoint's on the chat route, so one session's two routes
        would sample differently.
        """
        return dict(generation_defaults) if self.sampling_defaults == "model" else {}


def EngineConfigError(message: str, *, retryable: bool) -> SessionError:
    """The engine's image configuration is unusable or unsupported.

    Two session errors rather than one, because the HTTP status the agent sees has
    to say whether trying again can help: readiness is 503, a launch flag is 500.
    """
    return EngineNotReadyError(message) if retryable else EngineUnsupportedError(message)


class EngineBinding:
    """Resolves and checks the engine's image configuration once, on first use.

    Not at construction: Miles waits for this server's listener *before* it launches
    any engine (`placement_group` waits for the session server, then initializes the
    inference controller), so a constructor-time lookup has nothing to ask and
    deadlocks the run. A readiness or transport failure is retried on the next image
    request; a configuration this path does not support is cached and refused for
    good, because retrying cannot change it.
    """

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self._lock = asyncio.Lock()
        self._resolved: EngineImageConfig | None = None
        self._rejected: SessionError | None = None

    @property
    def resolved(self) -> EngineImageConfig | None:
        return self._resolved

    async def get(self, backend: Any) -> EngineImageConfig:
        if self._resolved is not None:
            return self._resolved
        if self._rejected is not None:
            raise self._rejected
        async with self._lock:
            if self._resolved is not None:
                return self._resolved
            if self._rejected is not None:
                raise self._rejected
            payload = await _fetch_server_info(backend)
            try:
                engine = parse_engine_image_config(payload)
                assert_engine_preprocessing_supported(engine, checkpoint=self.checkpoint)
            except SessionError as exc:
                if isinstance(exc, EngineUnsupportedError):
                    self._rejected = exc
                raise
            logger.info("[session] Engine image configuration accepted: %s", engine.to_dict())
            self._resolved = engine
            return engine


async def _fetch_server_info(backend: Any) -> dict[str, Any]:
    """GET the engine's server info through this server's own proxy client."""
    from miles.rollout.session.core import ProxyRequest

    result = await backend.do_proxy(ProxyRequest(method="GET"), "get_server_info", body=b"", headers={})
    status = result.get("status_code")
    if status != 200:
        raise EngineConfigError(
            f"the engine's /get_server_info answered {status}, so its image configuration is unknown; "
            "an image request cannot be served until it does",
            retryable=True,
        )
    try:
        payload = orjson.loads(result["response_body"])
    except orjson.JSONDecodeError as exc:
        raise EngineConfigError(f"the engine's /get_server_info is not JSON: {exc}", retryable=True) from exc
    if not isinstance(payload, dict):
        raise EngineConfigError(
            f"the engine's /get_server_info returned {type(payload).__name__}, not an object", retryable=True
        )
    return payload


def parse_engine_image_config(payload: dict[str, Any]) -> EngineImageConfig:
    """Read the fields that decide pixel agreement, refusing a reply that omits any."""
    missing = [key for key in EngineImageConfig.REQUIRED if key not in payload]
    if missing:
        raise EngineConfigError(
            f"the engine's /get_server_info does not state {', '.join(missing)}, so it cannot be checked. "
            f"It answered with keys {sorted(payload)[:12]}. A router or gateway that aggregates workers "
            "answers for none of them: point --use-session-server at the engine whose images it will serve.",
            # A router that is still electing a worker can start answering for one.
            retryable=True,
        )
    return EngineImageConfig(
        model_path=str(payload["model_path"] or ""),
        revision=payload.get("revision"),
        mm_process_config=payload["mm_process_config"] or {},
        image_processor_backend=str(payload["image_processor_backend"] or ""),
        sampling_defaults=str(payload["sampling_defaults"] or ""),
        tool_call_parser=payload.get("tool_call_parser"),
        reasoning_parser=payload.get("reasoning_parser"),
    )


def assert_engine_preprocessing_supported(engine: EngineImageConfig, *, checkpoint: str) -> None:
    """Require the one engine image configuration whose pixels this slice has verified."""
    if not engine.model_path:
        raise EngineConfigError(
            "the engine states an empty model_path, so the checkpoint it serves cannot be established",
            retryable=False,
        )
    if engine.model_path != checkpoint:
        raise EngineConfigError(
            f"the engine serves {engine.model_path!r} but --hf-checkpoint is {checkpoint!r}; "
            "the two processors would disagree about every image",
            retryable=False,
        )
    if engine.revision is not None:
        raise EngineConfigError(
            f"the engine pins revision {engine.revision!r} while this server loads the processor unpinned, "
            "so the two could resize or normalize differently with every token id still matching; "
            "serve the checkpoint's default revision",
            retryable=False,
        )
    if engine.mm_process_config:
        raise EngineConfigError(
            f"the engine was launched with --mm-process-config {engine.mm_process_config!r}, which changes "
            "how it decodes or normalizes images; its pixels would then differ from the trainer's while "
            "every token id stayed identical. Launch it with the checkpoint's default image preprocessing.",
            retryable=False,
        )
    if engine.image_processor_backend != "auto":
        raise EngineConfigError(
            f"the engine's --image-processor-backend is {engine.image_processor_backend!r}; this path has "
            "only verified pixel agreement for the default 'auto' backend",
            retryable=False,
        )


@dataclass(frozen=True)
class SessionImageSupport:
    """What every image-bearing turn in this process shares.

    The processor and its fingerprint are known at construction. The engine's
    configuration is not: it is resolved on first use through ``engine``, because
    no engine exists when this server is built.
    """

    processor: Any
    fingerprint: ProcessorFingerprint
    tokenizer: Any
    text_only_leaf: str  # "drop" | "keep"
    binding: EngineBinding
    generation_defaults: dict[str, Any]

    @property
    def placeholder_id(self) -> int:
        return self.fingerprint.image_token_id

    def new_session_media(self) -> "SessionMedia":
        return SessionMedia(counter=ImageTokenCounter(self.processor, self.fingerprint))

    async def engine(self, backend: Any) -> EngineImageConfig:
        return await self.binding.get(backend)

    def parsers(self, engine: EngineImageConfig) -> CompletionParsers:
        """The parsers the engine applies, so both routes in one session agree."""
        return CompletionParsers(
            reasoning_parser=engine.reasoning_parser,
            tool_call_parser=engine.tool_call_parser,
            tokenizer=self.tokenizer,
        )

    def sampling_defaults(self, engine: EngineImageConfig) -> dict[str, Any]:
        return engine.model_sampling_defaults(self.generation_defaults)

    def provenance(self) -> dict[str, Any]:
        """What bound this run's pixels, recorded with the session and every sample.

        The engine half is present once it has been resolved, which is on the first
        image request; a text-only session never resolves it and records none.
        """
        engine = self.binding.resolved
        return {
            "processor": self.fingerprint.to_dict(),
            "engine": engine.to_dict() if engine is not None else None,
        }

    def process(self, asset_ids: list[str], counts: list[int], media: "SessionMedia") -> list[ProcessedImage]:
        assets = [media.arena.get(asset_id) for asset_id in asset_ids]
        return process_images(self.processor, assets, self.fingerprint, counts)


@dataclass
class SessionMedia:
    """One session's immutable image bytes and their expanded token counts."""

    counter: ImageTokenCounter
    arena: MediaArena = field(default_factory=MediaArena)

    def ingest(self, messages: list[dict[str, Any]]) -> tuple[RequestMedia, list[int]]:
        """Snapshot a request's images and say how many tokens each expands into."""
        media = extract_request_media(messages, self.arena)
        if not media.occurrences:
            return EMPTY_MEDIA, []
        return media, self.counts_for(media)

    def counts_for(self, media: RequestMedia) -> list[int]:
        return self.counter.count_all([self.arena.get(asset_id) for asset_id in media.asset_ids])


# The fields SGLang's ModelConfig.get_default_sampling_params reads from a checkpoint.
_GENERATION_DEFAULT_KEYS = ("repetition_penalty", "temperature", "top_k", "top_p", "min_p")


def build_image_support(config: SessionServerConfig, tokenizer: Any) -> SessionImageSupport | None:
    """Load the processor for an image-enabled run, or ``None`` for text.

    Nothing here touches the engine: it does not exist yet when this runs.
    """
    if config.session_image_support == SESSION_IMAGE_SUPPORT_OFF:
        return None

    from miles.utils.processing_utils import load_processor

    processor = load_processor(config.hf_checkpoint, trust_remote_code=True)
    if processor is None:
        raise ValueError(
            f"--session-image-support={config.session_image_support} needs an image processor, but "
            f"{config.hf_checkpoint!r} provides none"
        )
    if not hasattr(processor, "_get_num_multimodal_tokens"):
        raise ValueError(
            f"{config.hf_checkpoint!r}'s processor has no _get_num_multimodal_tokens, so per-image token "
            "counts cannot be derived without processing every image on every turn"
        )
    fingerprint = build_fingerprint(processor, tokenizer, checkpoint=config.hf_checkpoint)
    logger.info(
        "[session] Image support=%s fingerprint=%s text_only_leaf=%s; the engine's own image "
        "configuration is checked on the first image request",
        config.session_image_support,
        fingerprint.to_dict(),
        config.session_image_text_only_leaf,
    )
    return SessionImageSupport(
        processor=processor,
        fingerprint=fingerprint,
        tokenizer=tokenizer,
        text_only_leaf=config.session_image_text_only_leaf,
        binding=EngineBinding(config.hf_checkpoint),
        generation_defaults=read_generation_defaults(config.hf_checkpoint),
    )


def read_generation_defaults(checkpoint: str) -> dict[str, Any]:
    """The checkpoint's own sampling defaults, for a field the request omits."""
    from transformers import GenerationConfig

    try:
        config = GenerationConfig.from_pretrained(checkpoint).to_dict()
    except Exception as exc:
        logger.warning("[session] No generation config for %s (%s); omitted sampling fields "
                       "will take the engine's own fallbacks", checkpoint, exc)
        return {}
    return {key: config[key] for key in _GENERATION_DEFAULT_KEYS if config.get(key) is not None}
