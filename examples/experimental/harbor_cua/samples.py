"""Segments to Miles ``Sample``s, with the checks that must pass first.

Every sibling of one episode carries the same ``rollout_id`` and the same reward
value: Miles normalizes one reward per rollout and raises if siblings disagree,
and it sums the loss mask over the whole rollout so splitting an episode adds no
weight.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from examples.experimental.harbor_cua.calls import RolloutRecord
from examples.experimental.harbor_cua.media import (
    MediaProcessor,
    build_train_inputs,
    verify_occurrence_runs,
)
from examples.experimental.harbor_cua.segments import BoundaryEvent, Segment, Segmentation
from miles.utils.types import Sample, WeightVersionsPerCall


@dataclass
class ConversionStats:
    """What the converter did with one episode, for the run report."""

    action_calls: int = 0
    auxiliary_calls: int = 0
    segments: int = 0
    kept_segments: int = 0
    dropped_text_only: int = 0
    trained_tokens: int = 0
    image_occurrences: int = 0
    distinct_images: int = 0
    boundaries: list[BoundaryEvent] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    def as_metadata(self) -> dict:
        return {
            "action_calls": self.action_calls,
            "auxiliary_calls_excluded": self.auxiliary_calls,
            "segments": self.segments,
            "kept_segments": self.kept_segments,
            "dropped_text_only_segments": self.dropped_text_only,
            "trained_tokens": self.trained_tokens,
            "image_occurrences": self.image_occurrences,
            "distinct_images": self.distinct_images,
            "boundary_reasons": [event.reason for event in self.boundaries],
            "rejected_calls": self.rejected,
        }


def build_samples(
    *,
    template: Sample,
    record: RolloutRecord,
    segmentation: Segmentation,
    media: MediaProcessor,
    reward: float,
    rollout_id: int,
    tokenizer,
) -> tuple[list[Sample], ConversionStats]:
    """One ``Sample`` per segment that retained at least one image."""
    stats = ConversionStats(
        action_calls=len(record.action_calls()),
        auxiliary_calls=len(record.auxiliary_calls()),
        segments=len(segmentation.segments),
        boundaries=list(segmentation.boundaries),
        rejected=list(segmentation.rejected) + list(record.rejected),
        distinct_images=len(record.arena.assets),
    )

    samples: list[Sample] = []
    for segment in segmentation.segments:
        assets = media.process_all(record.arena, segment.media)
        verify_occurrence_runs(
            segment.tokens,
            media.image_pad_id,
            assets,
            where=f"segment over calls {segment.call_indices}",
        )
        train_inputs = build_train_inputs(assets)
        if train_inputs is None:
            # Under FSDP the vision tower's weight all-gathers are collectives, so a
            # text-only sample stalls every rank that issued them; a real sampled
            # context never gets a placeholder image added to keep it company.
            stats.dropped_text_only += 1
            continue
        samples.append(
            _sample_from_segment(
                template=template,
                segment=segment,
                train_inputs=train_inputs,
                reward=reward,
                rollout_id=rollout_id,
                tokenizer=tokenizer,
            )
        )
        stats.kept_segments += 1
        stats.trained_tokens += segment.sampled_token_count
        stats.image_occurrences += len(segment.media)

    return samples, stats


def _sample_from_segment(
    *,
    template: Sample,
    segment: Segment,
    train_inputs: dict,
    reward: float,
    rollout_id: int,
    tokenizer,
) -> Sample:
    sample = copy.deepcopy(template)
    sample.tokens = list(segment.tokens)
    sample.response_length = segment.response_length
    sample.loss_mask = list(segment.loss_mask)
    sample.rollout_log_probs = list(segment.logprobs)
    sample.multimodal_train_inputs = train_inputs
    sample.multimodal_inputs = None
    sample.weight_versions = _weight_versions(segment)
    sample.response = tokenizer.decode(sample.tokens[segment.response_start :], skip_special_tokens=False)
    sample.reward = reward
    sample.rollout_id = rollout_id
    sample.status = Sample.Status.COMPLETED
    return sample


def _weight_versions(segment: Segment) -> list[WeightVersionsPerCall]:
    """One entry per call, with output spans at their absolute positions.

    Prefill spans survive only for a segment's first call: the engine reports them
    relative to that call's own prompt, which coincides with the sample's tokens
    only while the prompt starts at zero.
    """
    calls: list[WeightVersionsPerCall] = []
    for output_end, meta_info, is_first in segment.weight_version_calls:
        per_call = WeightVersionsPerCall.from_meta_info(meta_info, output_end)
        if not is_first:
            per_call.prefill_spans = []
            per_call.prompt_tokens = None
        calls.append(per_call)
    return calls
