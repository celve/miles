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

from miles.rollout.session.config import SESSION_IMAGE_SUPPORT_OFF, SessionServerConfig, check_session_image_engine
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
    """How the policy engines preprocess images and parse replies, from the args Miles launches them with.

    Matching token ids prove the two sides agree on the image *grid*, because the
    expanded length is derived from it — they do not prove they agree on the pixel
    values behind those tokens. A decode or normalization override changes the
    pixels while leaving the grid, and therefore every token id, identical. So these
    launch settings decide whether this run may serve images at all, and they are
    checked before the server starts (`check_session_image_engine`).
    """

    model_path: str
    revision: str | None
    mm_process_config: dict[str, Any] | None
    image_processor_backend: str
    sampling_defaults: str
    tool_call_parser: str | None
    reasoning_parser: str | None

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


@dataclass(frozen=True)
class SessionImageSupport:
    """What every image-bearing turn in this process shares, all of it known at construction."""

    processor: Any
    fingerprint: ProcessorFingerprint
    tokenizer: Any
    text_only_leaf: str  # "drop" | "keep"
    engine: EngineImageConfig
    generation_defaults: dict[str, Any]

    @property
    def placeholder_id(self) -> int:
        return self.fingerprint.image_token_id

    def new_session_media(self) -> "SessionMedia":
        return SessionMedia(counter=ImageTokenCounter(self.processor, self.fingerprint))

    def parsers(self) -> CompletionParsers:
        """The parsers the engines apply on the chat route, so both routes in one session agree."""
        return CompletionParsers(
            reasoning_parser=self.engine.reasoning_parser,
            tool_call_parser=self.engine.tool_call_parser,
            tokenizer=self.tokenizer,
        )

    def sampling_defaults(self) -> dict[str, Any]:
        return self.engine.model_sampling_defaults(self.generation_defaults)

    def provenance(self) -> dict[str, Any]:
        """What bound this run's pixels, recorded with the session and every sample."""
        return {"processor": self.fingerprint.to_dict(), "engine": self.engine.to_dict()}

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

    Nothing here touches an engine: none exists yet when this runs, and the engines'
    image settings arrive in the config, read from the args Miles launches them with.
    """
    if config.session_image_support == SESSION_IMAGE_SUPPORT_OFF:
        return None
    if config.session_image_engine is None:
        raise ValueError(
            f"--session-image-support={config.session_image_support} needs the policy engines' image "
            "settings, which validate_session_image_args reads from the launch args"
        )
    # Checked again here because a config can also be built by hand, as scripts and tests do.
    check_session_image_engine(config.session_image_engine, checkpoint=config.hf_checkpoint)

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
    engine = EngineImageConfig(**config.session_image_engine)
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
        tokenizer=tokenizer,
        text_only_leaf=config.session_image_text_only_leaf,
        engine=engine,
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
