"""Immutable image content behind a session's requests.

A screenshot harness replays its whole history every turn, so the same picture
arrives many times and two different pictures can share a size. Content, not
position or URL, is therefore the identity: every image is snapshotted to bytes
on arrival and stored under the SHA-256 of those bytes, and each *appearance*
stays a separate ordered occurrence so repeats and reorderings remain visible.

The store lives on the session and dies with it. Extraction reads the OpenAI
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

# The one image spelling SGLang's chat request accepts, as OpenAI's does.
IMAGE_PART_TYPES = ("image_url",)

_DATA_URL = re.compile(r"^data:(?P<mime>image/[a-zA-Z0-9.+-]+);base64,(?P<payload>.*)$", re.DOTALL)
_SUPPORTED_MIME_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif", "image/bmp")


@dataclass(frozen=True)
class StoredImage:
    """One image's immutable bytes and the identity derived from them."""

    image_id: str  # "sha256:<64 hex>", the spelling SGLang's content_hash takes
    data: bytes
    mime_type: str

    def to_data_url(self) -> str:
        return f"data:{self.mime_type};base64,{base64.b64encode(self.data).decode('ascii')}"


@dataclass(frozen=True)
class ImageOccurrence:
    """One appearance of a stored image in a request, located for diagnostics."""

    image_id: str
    message_index: int
    part_index: int


@dataclass(frozen=True)
class PromptImages:
    """A request's image occurrences in prompt order."""

    occurrences: tuple[ImageOccurrence, ...] = ()

    def __len__(self) -> int:
        return len(self.occurrences)

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(occurrence.image_id for occurrence in self.occurrences)

    def is_prefix_of(self, other: "PromptImages") -> bool:
        return other.image_ids[: len(self.image_ids)] == self.image_ids


NO_IMAGES = PromptImages()


class ImageStore:
    """Session-scoped store: one entry per distinct image content."""

    def __init__(self) -> None:
        self._images: dict[str, StoredImage] = {}

    def add(self, data: bytes, mime_type: str) -> StoredImage:
        image_id = f"sha256:{hashlib.sha256(data).hexdigest()}"
        existing = self._images.get(image_id)
        if existing is not None:
            return existing
        stored = StoredImage(image_id=image_id, data=data, mime_type=mime_type)
        self._images[image_id] = stored
        return stored

    def get(self, image_id: str) -> StoredImage:
        stored = self._images.get(image_id)
        if stored is None:
            raise KeyError(f"no stored image {image_id!r}")
        return stored

    def __len__(self) -> int:
        return len(self._images)

    @property
    def total_bytes(self) -> int:
        return sum(len(stored.data) for stored in self._images.values())


def extract_prompt_images(messages: list[dict[str, Any]], store: ImageStore) -> PromptImages:
    """Snapshot every image in *messages* into *store*, in prompt order."""
    occurrences: list[ImageOccurrence] = []
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
            stored = store.add(*_resolve_image_part(part, message_index, part_index))
            _check_declared_hash(part, stored.image_id, message_index, part_index)
            occurrences.append(
                ImageOccurrence(image_id=stored.image_id, message_index=message_index, part_index=part_index)
            )
    return PromptImages(occurrences=tuple(occurrences))


def _reject_unsupported_media_part(part_type: Any, message_index: int, part_index: int) -> None:
    """Refuse the media kinds this path does not carry, rather than dropping them silently."""
    where = f"messages[{message_index}].content[{part_index}].type={part_type!r}"
    if part_type in ("image", "input_image"):
        raise MessageValidationError(f"{where} is not supported: send an image as an image_url part")
    if part_type in ("video_url", "video", "input_audio", "audio_url", "audio"):
        raise MessageValidationError(f"{where} is not supported: the session server carries images only")


def _check_declared_hash(part: dict[str, Any], image_id: str, message_index: int, part_index: int) -> None:
    # An engine that trusts caller hashes would serve another image's cached features for a wrong one.
    declared = part["image_url"].get("content_hash")
    if declared is not None and str(declared).lower() != image_id:
        raise MessageValidationError(
            f"messages[{message_index}].content[{part_index}].image_url.content_hash={declared!r} is not the "
            f"SHA-256 of the image's bytes ({image_id})"
        )


def _resolve_image_part(part: dict[str, Any], message_index: int, part_index: int) -> tuple[bytes, str]:
    """Decode one image content part to ``(bytes, mime_type)``."""
    where = f"messages[{message_index}].content[{part_index}]"
    image_url = part.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else None
    if not isinstance(url, str) or not url:
        raise MessageValidationError(f"{where} carries no image_url object with a url string")
    match = _DATA_URL.match(url)
    if match is None:
        raise MessageValidationError(
            f"{where} must be an inline base64 data URL (data:image/<kind>;base64,...): "
            "the session server snapshots image bytes as the training identity, and a remote "
            "URL is not immutable content"
        )
    mime_type = match.group("mime").lower()
    if mime_type not in _SUPPORTED_MIME_TYPES:
        raise MessageValidationError(
            f"{where} has unsupported image type {mime_type!r} (want one of {_SUPPORTED_MIME_TYPES})"
        )
    try:
        data = base64.b64decode(match.group("payload"), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MessageValidationError(f"{where} has an undecodable base64 payload: {exc}") from exc
    if not data:
        raise MessageValidationError(f"{where} decodes to zero bytes")
    return data, mime_type
