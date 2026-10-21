"""Immutable image content behind a session's requests.

A screenshot harness replays its whole history every turn, so the same picture
arrives many times and two different pictures can share a size. Content, not
position or URL, is therefore the identity: every image is snapshotted to bytes
on arrival and stored under the SHA-256 of those bytes, and each *appearance*
stays a separate ordered occurrence so repeats and reorderings remain visible.

The store lives on the session and dies with it. Interning reads the OpenAI
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
class ImageBlob:
    """One image's immutable bytes and the identity derived from them."""

    image_id: str  # "sha256:<64 hex>", the spelling SGLang's content_hash takes
    data: bytes


@dataclass(frozen=True)
class ImageOccurrences:
    """A prompt's images in order, by content id; the same image twice is two occurrences."""

    image_ids: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.image_ids)

    def is_prefix_of(self, other: "ImageOccurrences") -> bool:
        return other.image_ids[: len(self.image_ids)] == self.image_ids


NO_IMAGES = ImageOccurrences()


class ImageStore:
    """Session-scoped store: one blob per distinct image content."""

    def __init__(self) -> None:
        self._blobs: dict[str, ImageBlob] = {}

    def add(self, data: bytes) -> ImageBlob:
        image_id = f"sha256:{hashlib.sha256(data).hexdigest()}"
        existing = self._blobs.get(image_id)
        if existing is not None:
            return existing
        blob = ImageBlob(image_id=image_id, data=data)
        self._blobs[image_id] = blob
        return blob

    def get(self, image_id: str) -> ImageBlob:
        blob = self._blobs.get(image_id)
        if blob is None:
            raise KeyError(f"no image blob {image_id!r}")
        return blob

    def __len__(self) -> int:
        return len(self._blobs)


def intern_images(messages: list[dict[str, Any]], store: ImageStore) -> ImageOccurrences:
    """Store every image in *messages* once by content, and return them in prompt order."""
    image_ids: list[str] = []
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
            blob = store.add(_resolve_image_part(part, message_index, part_index))
            _check_declared_hash(part, blob.image_id, message_index, part_index)
            image_ids.append(blob.image_id)
    return ImageOccurrences(image_ids=tuple(image_ids))


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


def _resolve_image_part(part: dict[str, Any], message_index: int, part_index: int) -> bytes:
    """Decode one image content part to its bytes."""
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
    return data
