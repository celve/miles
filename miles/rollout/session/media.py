"""Immutable image content behind a session's requests.

A screenshot harness replays its whole history every turn, so the same picture
arrives many times and two different pictures can share a size. Content, not
position or URL, is therefore the identity: every image is snapshotted to bytes
on arrival and interned under the SHA-256 of those bytes, and each *appearance*
stays a separate ordered occurrence so repeats and reorderings remain visible.

The arena lives on the session and dies with it. Extraction reads the OpenAI
chat content parts and never mutates them: ``messages`` is shared read-only with
the recorded request.
"""

import base64
import binascii
import hashlib
import re
from dataclasses import dataclass
from typing import Any

from miles.rollout.session.errors import MessageValidationError

# Content part types that carry a picture. ``image_url`` is the OpenAI spelling;
# ``image`` is the alias the Qwen and Kimi chat templates also accept.
IMAGE_PART_TYPES = ("image_url", "image")

_DATA_URL = re.compile(r"^data:(?P<mime>image/[a-zA-Z0-9.+-]+);base64,(?P<payload>.*)$", re.DOTALL)
_SUPPORTED_MIME_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp")


@dataclass(frozen=True)
class MediaAsset:
    """One image's immutable bytes and the identity derived from them."""

    asset_id: str  # "sha256:<64 hex>", the spelling SGLang's mm_content_hashes takes
    data: bytes
    mime_type: str

    def to_data_url(self) -> str:
        return f"data:{self.mime_type};base64,{base64.b64encode(self.data).decode('ascii')}"


@dataclass(frozen=True)
class MediaOccurrence:
    """One appearance of an asset in a request, located for diagnostics."""

    asset_id: str
    message_index: int
    part_index: int


@dataclass(frozen=True)
class RequestMedia:
    """A request's image occurrences in prompt order."""

    occurrences: tuple[MediaOccurrence, ...] = ()

    def __len__(self) -> int:
        return len(self.occurrences)

    @property
    def asset_ids(self) -> tuple[str, ...]:
        return tuple(occurrence.asset_id for occurrence in self.occurrences)

    def is_prefix_of(self, other: "RequestMedia") -> bool:
        return other.asset_ids[: len(self.asset_ids)] == self.asset_ids


EMPTY_MEDIA = RequestMedia()


class MediaArena:
    """Session-scoped interning store: one entry per distinct image content."""

    def __init__(self) -> None:
        self._assets: dict[str, MediaAsset] = {}

    def intern(self, data: bytes, mime_type: str) -> MediaAsset:
        asset_id = f"sha256:{hashlib.sha256(data).hexdigest()}"
        existing = self._assets.get(asset_id)
        if existing is not None:
            return existing
        asset = MediaAsset(asset_id=asset_id, data=data, mime_type=mime_type)
        self._assets[asset_id] = asset
        return asset

    def get(self, asset_id: str) -> MediaAsset:
        asset = self._assets.get(asset_id)
        if asset is None:
            raise KeyError(f"unknown media asset {asset_id!r}")
        return asset

    def __len__(self) -> int:
        return len(self._assets)

    @property
    def total_bytes(self) -> int:
        return sum(len(asset.data) for asset in self._assets.values())


def extract_request_media(messages: list[dict[str, Any]], arena: MediaArena) -> RequestMedia:
    """Snapshot every image in *messages* into *arena*, in prompt order."""
    occurrences: list[MediaOccurrence] = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type not in IMAGE_PART_TYPES:
                _reject_unsupported_media_part(part_type, message_index, part_index)
                continue
            asset = arena.intern(*_resolve_image_part(part, message_index, part_index))
            occurrences.append(
                MediaOccurrence(asset_id=asset.asset_id, message_index=message_index, part_index=part_index)
            )
    return RequestMedia(occurrences=tuple(occurrences))


def _reject_unsupported_media_part(part_type: Any, message_index: int, part_index: int) -> None:
    """Refuse the media kinds this path does not carry, rather than dropping them silently."""
    if part_type in ("video_url", "video", "input_audio", "audio_url", "audio"):
        raise MessageValidationError(
            f"messages[{message_index}].content[{part_index}].type={part_type!r} is not supported: "
            "the session server carries images only"
        )


def _resolve_image_part(part: dict[str, Any], message_index: int, part_index: int) -> tuple[bytes, str]:
    """Decode one image content part to ``(bytes, mime_type)``."""
    where = f"messages[{message_index}].content[{part_index}]"
    url = part.get("image_url") if part.get("type") == "image_url" else part.get("image")
    if isinstance(url, dict):
        url = url.get("url")
    if not isinstance(url, str) or not url:
        raise MessageValidationError(f"{where} carries no image url string")
    match = _DATA_URL.match(url)
    if match is None:
        raise MessageValidationError(
            f"{where} must be an inline base64 data URL (data:image/<kind>;base64,...): "
            "the session server snapshots image bytes as the training identity, and a remote "
            "URL is not immutable content"
        )
    mime_type = match.group("mime").lower()
    if mime_type not in _SUPPORTED_MIME_TYPES:
        raise MessageValidationError(f"{where} has unsupported image type {mime_type!r} (want one of {_SUPPORTED_MIME_TYPES})")
    try:
        data = base64.b64decode(match.group("payload"), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MessageValidationError(f"{where} has an undecodable base64 payload: {exc}") from exc
    if not data:
        raise MessageValidationError(f"{where} decodes to zero bytes")
    return data, mime_type
