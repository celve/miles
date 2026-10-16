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
class SessionImageSupport:
    """What every image-bearing turn in this process shares."""

    processor: Any
    fingerprint: ProcessorFingerprint
    parsers: CompletionParsers
    text_only_leaf: str  # "drop" | "keep"

    @property
    def placeholder_id(self) -> int:
        return self.fingerprint.image_token_id

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

    from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser
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
    reasoning_parser, tool_call_parser = resolve_reasoning_and_tool_call_parser(config.tito_model)
    logger.info(
        "[session] Image support=%s fingerprint=%s reasoning_parser=%s tool_call_parser=%s text_only_leaf=%s",
        config.session_image_support,
        fingerprint.to_dict(),
        reasoning_parser,
        tool_call_parser,
        config.session_image_text_only_leaf,
    )
    return SessionImageSupport(
        processor=processor,
        fingerprint=fingerprint,
        parsers=CompletionParsers(
            reasoning_parser=reasoning_parser, tool_call_parser=tool_call_parser, tokenizer=tokenizer
        ),
        text_only_leaf=config.session_image_text_only_leaf,
    )
