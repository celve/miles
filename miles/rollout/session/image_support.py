"""The session server's image capability, constructed once per process.

``build_image_support`` returns ``None`` unless ``--session-image-support`` names
a family, so nothing here is loaded — and no processor is read off disk — for a
text-only run. The returned object is read-only and shared by every session; the
per-session half is ``SessionMedia``, which owns that session's interned bytes
and dies with it.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from miles.rollout.session.config import SESSION_IMAGE_SUPPORT_OFF, SessionServerConfig
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
    """How the engine says it will preprocess images, read from it once at startup.

    Matching token ids proves the two sides agree on the image *grid*, because the
    expanded length is derived from it — it does not prove they agree on the pixel
    values behind those tokens. A decode or normalization override changes the
    pixels while leaving the grid, and therefore every token id, identical. So the
    engine's own configuration is read and required to be the one this slice
    supports, rather than inferred from the ids.
    """

    model_path: str
    revision: str | None
    mm_process_config: dict[str, Any]
    image_processor_backend: str
    tool_call_parser: str | None
    reasoning_parser: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "revision": self.revision,
            "mm_process_config": self.mm_process_config,
            "image_processor_backend": self.image_processor_backend,
            "tool_call_parser": self.tool_call_parser,
            "reasoning_parser": self.reasoning_parser,
        }


@dataclass(frozen=True)
class SessionImageSupport:
    """What every image-bearing turn in this process shares."""

    processor: Any
    fingerprint: ProcessorFingerprint
    parsers: CompletionParsers
    text_only_leaf: str  # "drop" | "keep"
    engine: EngineImageConfig

    @property
    def placeholder_id(self) -> int:
        return self.fingerprint.image_token_id

    def provenance(self) -> dict[str, Any]:
        """What bound this run's pixels, recorded with the session and every sample."""
        return {"processor": self.fingerprint.to_dict(), "engine": self.engine.to_dict()}

    def new_session_media(self) -> "SessionMedia":
        return SessionMedia(counter=ImageTokenCounter(self.processor, self.fingerprint))

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


def build_image_support(config: SessionServerConfig, tokenizer: Any) -> SessionImageSupport | None:
    """Load the processor and parsers for an image-enabled run, or ``None`` for text."""
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
    engine = fetch_engine_image_config(config.backend_url)
    assert_engine_preprocessing_supported(engine, config)
    # The engine decides which parsers it applies on its own chat route, so taking
    # them from it is what keeps a text turn and an image turn in one session
    # agreeing about reasoning and tool calls.
    logger.info(
        "[session] Image support=%s fingerprint=%s engine=%s text_only_leaf=%s",
        config.session_image_support,
        fingerprint.to_dict(),
        engine.to_dict(),
        config.session_image_text_only_leaf,
    )
    return SessionImageSupport(
        processor=processor,
        fingerprint=fingerprint,
        parsers=CompletionParsers(
            reasoning_parser=engine.reasoning_parser,
            tool_call_parser=engine.tool_call_parser,
            tokenizer=tokenizer,
        ),
        text_only_leaf=config.session_image_text_only_leaf,
        engine=engine,
    )


def fetch_engine_image_config(backend_url: str) -> EngineImageConfig:
    """Read the engine's own image settings; a run that cannot read them does not start."""
    import requests

    try:
        info = requests.get(f"{backend_url.rstrip('/')}/get_server_info", timeout=60).json()
    except Exception as exc:
        raise ValueError(
            f"--session-image-support needs the engine's image configuration from "
            f"{backend_url}/get_server_info, which could not be read: {exc}"
        ) from exc
    return EngineImageConfig(
        model_path=info.get("model_path") or "",
        revision=info.get("revision"),
        mm_process_config=info.get("mm_process_config") or {},
        image_processor_backend=info.get("image_processor_backend") or "auto",
        tool_call_parser=info.get("tool_call_parser"),
        reasoning_parser=info.get("reasoning_parser"),
    )


def assert_engine_preprocessing_supported(engine: EngineImageConfig, config: SessionServerConfig) -> None:
    """Require the one engine image configuration whose pixels this slice has verified."""
    if engine.model_path and engine.model_path != config.hf_checkpoint:
        raise ValueError(
            f"the engine serves {engine.model_path!r} but --hf-checkpoint is {config.hf_checkpoint!r}; "
            "the two processors would disagree about every image"
        )
    if engine.mm_process_config:
        raise ValueError(
            f"the engine was launched with --mm-process-config {engine.mm_process_config!r}, which changes "
            "how it decodes or normalizes images; its pixels would then differ from the trainer's while "
            "every token id stayed identical, so the session image path refuses it. Launch the engine with "
            "the checkpoint's default image preprocessing."
        )
    if engine.image_processor_backend != "auto":
        raise ValueError(
            f"the engine's --image-processor-backend is {engine.image_processor_backend!r}; this path has "
            "only verified pixel agreement for the default 'auto' backend"
        )
