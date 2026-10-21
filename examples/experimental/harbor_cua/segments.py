"""Recorded calls to training segments.

A segment is a run of calls whose prompts strictly extend one another, so its
token sequence is exactly the context inference saw for every completion in it.
Where Harbor rewrites its history -- dropping old screenshots, or replacing the
history with a summary -- the prompt stops extending and a segment closes; the
next call opens another instead of the episode ending there.

Pure: nothing here imports Miles, Harbor or torch, so the whole segmentation
contract is testable from a fixture.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from examples.experimental.harbor_cua.calls import MediaOccurrence, RecordedCall

BOUNDARY_FIRST_CALL = "first_call"
BOUNDARY_TOKENS = "prompt_not_append_compatible"
BOUNDARY_MEDIA = "media_not_append_compatible"
BOUNDARY_RETRY = "retry_on_the_same_prompt"
BOUNDARY_AFTER_SKIP = "after_a_skipped_call"


@dataclass
class Segment:
    """One training sample's worth of tokens, before Miles types are involved."""

    tokens: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    response_start: int = 0
    media: tuple[MediaOccurrence, ...] = ()
    call_indices: list[int] = field(default_factory=list)
    weight_version_calls: list[tuple[int, dict, bool]] = field(default_factory=list)
    last_prompt: tuple[int, ...] = ()

    @property
    def response_length(self) -> int:
        return len(self.tokens) - self.response_start

    @property
    def sampled_token_count(self) -> int:
        return sum(self.loss_mask)


@dataclass(frozen=True)
class BoundaryEvent:
    """Why a segment closed, and what the rewrite did to its media.

    ``shared_media_prefix`` is what the next segment could have inherited in
    order, which a drop from the front of the history takes to zero even when
    only one screenshot went away -- hence the before/after counts beside it.
    """

    call_index: int
    reason: str
    media_before: int
    media_after: int
    shared_media_prefix: int


@dataclass
class Segmentation:
    segments: list[Segment] = field(default_factory=list)
    boundaries: list[BoundaryEvent] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


def segment_calls(calls: Sequence[RecordedCall], *, max_sample_tokens: int | None = None) -> Segmentation:
    """Group ``calls`` into segments, in call order.

    A segment's length is its last call's prompt plus output, whether that call
    opened the segment or extended it, so splitting never shortens a sample:
    sample length is set by the harness's own context policy, and
    ``max_sample_tokens`` can only exclude a call that is too long to train at
    all. Excluding beats truncating, which would detach the retained tokens from
    the images they refer to.
    """
    result = Segmentation()
    current: Segment | None = None
    skipped_since_boundary = False

    for call in calls:
        if max_sample_tokens is not None and _call_length(call) > max_sample_tokens:
            result.rejected.append(
                f"call {call.index} alone needs {_call_length(call)} tokens, over the "
                f"{max_sample_tokens}-token sample budget"
            )
            if current is not None:
                result.segments.append(current)
                current = None
            skipped_since_boundary = True
            continue

        reason = BOUNDARY_AFTER_SKIP if skipped_since_boundary else _why_not_extendable(current, call)
        if reason is None:
            assert current is not None
            _extend(current, call)
            continue

        if current is not None:
            result.segments.append(current)
        result.boundaries.append(
            BoundaryEvent(
                call_index=call.index,
                reason=reason,
                media_before=len(current.media) if current is not None else 0,
                media_after=len(call.media),
                shared_media_prefix=_shared_media_prefix(current, call),
            )
        )
        current = _open(call)
        skipped_since_boundary = False

    if current is not None:
        result.segments.append(current)
    _assert_completions_trained_once(result.segments)
    return result


def _call_length(call: RecordedCall) -> int:
    return len(call.prompt_token_ids) + len(call.output_token_ids)


def _why_not_extendable(current: Segment | None, call: RecordedCall) -> str | None:
    if current is None:
        return BOUNDARY_FIRST_CALL
    if call.prompt_token_ids == current.last_prompt and call.media == current.media:
        # The same sampled context produced a second completion, so the second was
        # never conditioned on the first and cannot follow it in one sequence.
        return BOUNDARY_RETRY
    held = tuple(current.tokens)
    if call.prompt_token_ids[: len(held)] != held or len(call.prompt_token_ids) <= len(held):
        return BOUNDARY_TOKENS
    if call.media[: len(current.media)] != current.media:
        # Two screenshots with equal processed grids expand to identical placeholder runs,
        # so the token test above cannot see a replaced image.
        return BOUNDARY_MEDIA
    return None


def _open(call: RecordedCall) -> Segment:
    prompt = list(call.prompt_token_ids)
    output = list(call.output_token_ids)
    return Segment(
        tokens=prompt + output,
        loss_mask=[1] * len(output),
        logprobs=list(call.output_logprobs),
        response_start=len(prompt),
        media=call.media,
        call_indices=[call.index],
        weight_version_calls=[(len(prompt) + len(output), dict(call.meta_info), True)],
        last_prompt=call.prompt_token_ids,
    )


def _extend(segment: Segment, call: RecordedCall) -> None:
    delta = list(call.prompt_token_ids[len(segment.tokens) :])
    output = list(call.output_token_ids)
    segment.tokens.extend(delta)
    segment.loss_mask.extend([0] * len(delta))
    segment.logprobs.extend([0.0] * len(delta))
    segment.tokens.extend(output)
    segment.loss_mask.extend([1] * len(output))
    segment.logprobs.extend(call.output_logprobs)
    segment.media = call.media
    segment.last_prompt = call.prompt_token_ids
    segment.call_indices.append(call.index)
    segment.weight_version_calls.append((len(segment.tokens), dict(call.meta_info), False))


def _shared_media_prefix(current: Segment | None, call: RecordedCall) -> int:
    if current is None:
        return 0
    shared = 0
    for held, incoming in zip(current.media, call.media, strict=False):
        if held != incoming:
            break
        shared += 1
    return shared


def _assert_completions_trained_once(segments: Sequence[Segment]) -> None:
    seen: set[int] = set()
    for segment in segments:
        for index in segment.call_indices:
            if index in seen:
                raise ValueError(f"call {index} was placed in more than one segment")
            seen.add(index)
        if len(segment.loss_mask) != segment.response_length:
            raise ValueError(
                f"segment over calls {segment.call_indices} has a {len(segment.loss_mask)}-long mask "
                f"for a {segment.response_length}-long response"
            )
        if len(segment.logprobs) != segment.response_length:
            raise ValueError(
                f"segment over calls {segment.call_indices} has {len(segment.logprobs)} logprobs "
                f"for a {segment.response_length}-long response"
            )
