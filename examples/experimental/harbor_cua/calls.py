"""What one policy call to the recorder was, and the media it carried.

These types are the only input the segment converter takes, and they deliberately
name nothing from Harbor or from Miles: a recorded call is replayable from a
fixture on a machine where neither is installed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MediaOccurrence:
    """One ``image_url`` part at one position of one request."""

    asset_id: str
    media_type: str


@dataclass
class AssetArena:
    """The immutable bytes behind every occurrence of one rollout.

    A screenshot appears in every later prompt that still carries it, so the
    arena holds each distinct image once while occurrences stay ordered and
    repeated.
    """

    assets: dict[str, bytes] = field(default_factory=dict)

    def put(self, data: bytes, media_type: str) -> MediaOccurrence:
        asset_id = f"sha256:{hashlib.sha256(data).hexdigest()}"
        self.assets.setdefault(asset_id, data)
        return MediaOccurrence(asset_id=asset_id, media_type=media_type)

    def get(self, asset_id: str) -> bytes:
        return self.assets[asset_id]

    def total_bytes(self) -> int:
        return sum(len(data) for data in self.assets.values())


class CallEvidenceError(ValueError):
    """A call whose token evidence is missing or self-contradictory."""


@dataclass(frozen=True)
class RecordedCall:
    """One request/response pair, as it was actually sent and returned.

    ``prompt_token_ids`` is what inference consumed — for a multimodal chat
    request the engine returns it already expanded, one run of image-pad tokens
    per image — so nothing here is re-tokenized or reconstructed.
    """

    index: int
    auxiliary: bool
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    output_logprobs: tuple[float, ...]
    media: tuple[MediaOccurrence, ...]
    finish_reason: str | None = None
    meta_info: dict = field(default_factory=dict)
    model: str | None = None
    # The request as sent, with image bytes replaced by their asset ids: enough to
    # replay the call or publish it as a fixture, without the base64.
    request: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise CallEvidenceError(f"call {self.index} has no prompt token ids")
        if not self.output_token_ids:
            raise CallEvidenceError(f"call {self.index} has no sampled token ids")
        if len(self.output_logprobs) != len(self.output_token_ids):
            raise CallEvidenceError(
                f"call {self.index} has {len(self.output_logprobs)} logprobs for "
                f"{len(self.output_token_ids)} sampled tokens"
            )


@dataclass
class RolloutRecord:
    """Every call one episode made, in the order it made them."""

    arena: AssetArena = field(default_factory=AssetArena)
    calls: list[RecordedCall] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    # Refused policy calls whose responses the agent still acted on; rollout.py writes such an episode off.
    unrecorded_policy_calls: list[str] = field(default_factory=list)

    def trained_calls(self) -> list[RecordedCall]:
        """Every policy call except the compactor's own, which are context, not actions."""
        return [call for call in self.calls if not call.auxiliary]

    def compactor_calls(self) -> list[RecordedCall]:
        return [call for call in self.calls if call.auxiliary]
