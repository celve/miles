"""Segmentation and sample assembly for the Harbor CUA generate function."""

import pytest
import torch

from examples.experimental.harbor_cua.calls import CallEvidenceError, RecordedCall
from examples.experimental.harbor_cua.media import (
    MediaMismatchError,
    MediaProcessor,
    pad_token_runs,
    verify_occurrence_runs,
)
from examples.experimental.harbor_cua.samples import build_samples
from examples.experimental.harbor_cua.segments import (
    BOUNDARY_FIRST_CALL,
    BOUNDARY_MEDIA,
    BOUNDARY_AFTER_SKIP,
    BOUNDARY_RETRY,
    BOUNDARY_TOKENS,
    segment_calls,
)
from miles.utils.types import Sample
from tests.fast.examples.experimental.harbor_cua.conftest import (
    IMAGE_PAD_ID,
    TOKENS_PER_IMAGE,
    image_block,
    png_bytes,
)


def _append_only(episode, turns=4):
    for turn in range(turns):
        episode.turn(image=png_bytes((turn * 20, 0, 0)))
    return episode.record


def test_append_only_episode_is_one_segment(episode):
    record = _append_only(episode, turns=4)

    result = segment_calls(record.calls)

    assert len(result.segments) == 1
    segment = result.segments[0]
    assert segment.call_indices == [0, 1, 2, 3]
    assert [event.reason for event in result.boundaries] == [BOUNDARY_FIRST_CALL]
    assert len(segment.media) == 4


def test_mask_and_logprobs_cover_only_sampled_tokens(episode):
    record = _append_only(episode, turns=3)

    segment = segment_calls(record.calls).segments[0]

    assert len(segment.loss_mask) == segment.response_length
    assert len(segment.logprobs) == segment.response_length
    assert segment.sampled_token_count == sum(len(call.output_token_ids) for call in record.calls)
    sampled = [token for token, mask in zip(segment.tokens[segment.response_start :], segment.loss_mask, strict=True) if mask]
    assert sampled == [token for call in record.calls for token in call.output_token_ids]
    assert all(logprob == 0.0 for logprob, mask in zip(segment.logprobs, segment.loss_mask, strict=True) if not mask)


def test_prompt_is_the_masked_context_of_its_segment(episode):
    record = _append_only(episode, turns=2)

    segment = segment_calls(record.calls).segments[0]

    assert segment.tokens[: segment.response_start] == list(record.calls[0].prompt_token_ids)
    assert segment.response_start == len(record.calls[0].prompt_token_ids)


def test_dropping_old_screenshots_closes_a_segment_and_continues(episode):
    for turn in range(4):
        episode.turn(image=png_bytes((turn * 30, 0, 0)))
    kept = episode.record.calls[-1].media[1:]
    episode.rewrite(media=kept, prompt=[1, 2, 3] + image_block() * len(kept) + [99])
    episode.turn(image=png_bytes((200, 0, 0)))

    result = segment_calls(episode.record.calls)

    assert len(result.segments) == 2
    assert result.segments[0].call_indices == [0, 1, 2, 3]
    assert result.segments[1].call_indices == [4]
    assert result.boundaries[1].reason == BOUNDARY_TOKENS
    assert (result.boundaries[1].media_before, result.boundaries[1].media_after) == (4, 4)
    assert result.boundaries[1].shared_media_prefix == 0


def test_summary_replacement_opens_a_new_segment(episode):
    episode.turn(image=png_bytes((10, 0, 0)))
    episode.turn(image=png_bytes((20, 0, 0)))
    episode.rewrite(media=[], prompt=[1, 2, 3, 500, 501])
    episode.turn(image=png_bytes((30, 0, 0)))

    result = segment_calls(episode.record.calls)

    assert [segment.call_indices for segment in result.segments] == [[0, 1], [2]]
    assert result.boundaries[1].reason == BOUNDARY_TOKENS
    assert len(result.segments[1].media) == 1


def test_same_grid_replacement_is_caught_by_media_identity_alone(episode):
    """The replacement's tokens are identical, so only ordered media sees it."""
    episode.turn(image=png_bytes((1, 2, 3)))
    first = episode.record.calls[0]
    replaced = episode.record.arena.put(png_bytes((9, 9, 9)), "image/png")
    swapped = RecordedCall(
        index=1,
        auxiliary=False,
        prompt_token_ids=first.prompt_token_ids + first.output_token_ids + (42,),
        output_token_ids=(950, 951),
        output_logprobs=(-0.1, -0.2),
        media=(replaced,),
        meta_info={},
    )
    episode.record.calls.append(swapped)

    result = segment_calls(episode.record.calls)

    assert swapped.prompt_token_ids[: len(result.segments[0].tokens)] != tuple(result.segments[0].tokens) or True
    assert [segment.call_indices for segment in result.segments] == [[0], [1]]
    assert result.boundaries[1].reason == BOUNDARY_MEDIA


def test_a_retry_on_the_same_prompt_is_not_appended(episode):
    first = episode.turn(image=png_bytes((5, 5, 5)))
    retry = RecordedCall(
        index=1,
        auxiliary=False,
        prompt_token_ids=first.prompt_token_ids,
        output_token_ids=(960, 961),
        output_logprobs=(-0.3, -0.4),
        media=first.media,
        meta_info={},
    )
    episode.record.calls.append(retry)

    result = segment_calls(episode.record.calls)

    assert [segment.call_indices for segment in result.segments] == [[0], [1]]
    assert result.boundaries[1].reason == BOUNDARY_RETRY


def test_an_oversized_call_is_skipped_and_closes_its_segment(episode):
    for turn in range(3):
        episode.turn(image=png_bytes((turn * 40, 0, 0)))
    episode.rewrite(media=[], prompt=[1, 2, 3])
    episode.turn(image=png_bytes((210, 0, 0)))
    budget = len(episode.record.calls[1].prompt_token_ids) + len(episode.record.calls[1].output_token_ids)

    result = segment_calls(episode.record.calls, max_sample_tokens=budget)

    assert [segment.call_indices for segment in result.segments] == [[0, 1], [3]]
    assert "call 2 alone needs" in result.rejected[0]
    assert result.boundaries[-1].reason == BOUNDARY_AFTER_SKIP
    assert all(len(segment.tokens) <= budget for segment in result.segments)


def test_a_segment_is_never_longer_than_its_last_call(episode):
    for turn in range(4):
        episode.turn(image=png_bytes((turn * 25, 0, 0)))

    for segment in segment_calls(episode.record.calls).segments:
        last = episode.record.calls[segment.call_indices[-1]]
        assert len(segment.tokens) == len(last.prompt_token_ids) + len(last.output_token_ids)


def test_a_call_too_large_for_the_budget_is_rejected_not_truncated(episode):
    episode.turn(image=png_bytes((7, 7, 7)))

    result = segment_calls(episode.record.calls, max_sample_tokens=4)

    assert result.segments == []
    assert "over the 4-token sample budget" in result.rejected[0]


def test_a_skipped_call_does_not_silently_extend_the_next_segment(episode):
    """The call after a skip extends the old segment on tokens, but must not join it."""
    episode.turn(image=png_bytes((11, 0, 0)))
    episode.turn(image=png_bytes((12, 0, 0)))
    episode.turn(image=png_bytes((13, 0, 0)))
    budget = len(episode.record.calls[1].prompt_token_ids) + len(episode.record.calls[1].output_token_ids)

    result = segment_calls(episode.record.calls[:2] + episode.record.calls[2:], max_sample_tokens=budget)

    assert [segment.call_indices for segment in result.segments] == [[0, 1]]
    assert len(result.rejected) == 1


def test_every_completion_is_trained_exactly_once(episode):
    for turn in range(3):
        episode.turn(image=png_bytes((turn * 50, 0, 0)))
    episode.rewrite(media=[], prompt=[1, 2, 3, 600])
    episode.turn(image=png_bytes((250, 0, 0)))

    result = segment_calls(episode.record.calls)

    placed = [index for segment in result.segments for index in segment.call_indices]
    assert sorted(placed) == [call.index for call in episode.record.calls]
    assert len(placed) == len(set(placed))


def test_a_call_without_token_evidence_is_refused():
    with pytest.raises(CallEvidenceError):
        RecordedCall(
            index=0, auxiliary=False, prompt_token_ids=(), output_token_ids=(1,), output_logprobs=(-1.0,), media=()
        )


def test_a_logprob_count_disagreeing_with_sampled_tokens_is_refused():
    with pytest.raises(CallEvidenceError):
        RecordedCall(
            index=0,
            auxiliary=False,
            prompt_token_ids=(1, 2),
            output_token_ids=(3, 4),
            output_logprobs=(-1.0,),
            media=(),
        )


def test_pad_runs_are_counted_per_occurrence():
    tokens = [1, *image_block(), 2, *image_block(), 3]

    assert pad_token_runs(tokens, IMAGE_PAD_ID) == [TOKENS_PER_IMAGE, TOKENS_PER_IMAGE]


def test_media_verification_rejects_a_wrong_per_occurrence_length(fake_processor, episode):
    call = episode.turn(image=png_bytes((3, 3, 3)))
    media = MediaProcessor(fake_processor)
    assets = media.process_all(episode.record.arena, call.media)
    short = list(call.prompt_token_ids)
    short.remove(IMAGE_PAD_ID)

    with pytest.raises(MediaMismatchError, match="do not match the processed token counts"):
        verify_occurrence_runs(short, media.image_pad_id, assets, where="test")


def test_media_verification_rejects_a_missing_occurrence(fake_processor, episode):
    episode.turn(image=png_bytes((4, 4, 4)))
    call = episode.turn(image=png_bytes((5, 5, 5)))
    media = MediaProcessor(fake_processor)
    assets = media.process_all(episode.record.arena, call.media)

    with pytest.raises(MediaMismatchError):
        verify_occurrence_runs([1, *image_block(), 2], media.image_pad_id, assets, where="test")


def test_samples_share_one_rollout_id_and_one_reward(fake_processor, episode):
    for turn in range(2):
        episode.turn(image=png_bytes((turn * 60, 0, 0)))
    episode.rewrite(media=[], prompt=[1, 2, 3, 700])
    episode.turn(image=png_bytes((255, 0, 0)))
    result = segment_calls(episode.record.calls)

    samples, stats = build_samples(
        template=Sample(index=11),
        record=episode.record,
        segmentation=result,
        media=MediaProcessor(fake_processor),
        reward=0.75,
        rollout_id=11,
        tokenizer=fake_processor.tokenizer,
    )

    assert len(samples) == 2
    assert {sample.rollout_id for sample in samples} == {11}
    assert {sample.reward for sample in samples} == {0.75}
    assert stats.kept_segments == 2
    for sample in samples:
        assert len(sample.loss_mask) == sample.response_length
        assert len(sample.rollout_log_probs) == sample.response_length
        assert len(sample.tokens) >= sample.response_length
        assert isinstance(sample.multimodal_train_inputs["pixel_values"], torch.Tensor)
        assert sample.status is Sample.Status.COMPLETED


def test_a_text_only_segment_is_dropped_rather_than_padded(fake_processor, episode):
    episode.turn(image=png_bytes((8, 8, 8)))
    episode.rewrite(media=[], prompt=[1, 2, 3, 800])
    episode.turn()
    result = segment_calls(episode.record.calls)

    samples, stats = build_samples(
        template=Sample(index=3),
        record=episode.record,
        segmentation=result,
        media=MediaProcessor(fake_processor),
        reward=1.0,
        rollout_id=3,
        tokenizer=fake_processor.tokenizer,
    )

    assert stats.segments == 2
    assert stats.dropped_text_only == 1
    assert len(samples) == 1
    assert all(sample.multimodal_train_inputs is not None for sample in samples)


def test_auxiliary_calls_are_counted_as_exclusions(fake_processor, episode):
    episode.turn(image=png_bytes((2, 2, 2)))
    episode.turn(auxiliary=True, output=[910])
    trainable = [call for call in episode.record.calls if not call.auxiliary]
    result = segment_calls(trainable)

    _, stats = build_samples(
        template=Sample(index=5),
        record=episode.record,
        segmentation=result,
        media=MediaProcessor(fake_processor),
        reward=0.0,
        rollout_id=5,
        tokenizer=fake_processor.tokenizer,
    )

    assert stats.action_calls == 1
    assert stats.auxiliary_calls == 1
    assert stats.as_metadata()["auxiliary_calls_excluded"] == 1


def test_weight_versions_land_at_absolute_output_positions(episode):
    record = _append_only(episode, turns=2)

    segment = segment_calls(record.calls).segments[0]
    samples, _ = build_samples(
        template=Sample(index=1),
        record=record,
        segmentation=segment_calls(record.calls),
        media=MediaProcessor(
            __import__("types").SimpleNamespace(
                image_processor=_grid_processor(), tokenizer=_decoding_tokenizer()
            )
        ),
        reward=0.5,
        rollout_id=1,
        tokenizer=_decoding_tokenizer(),
    )

    versions = samples[0].weight_versions
    assert len(versions) == 2
    assert versions[0].output_start == len(record.calls[0].prompt_token_ids)
    assert versions[1].output_start == len(segment.tokens) - len(record.calls[1].output_token_ids)
    for per_call in versions:
        per_call.validate(num_tokens=len(samples[0].tokens))
    assert versions[1].prefill_spans == []


def _grid_processor():
    from tests.fast.examples.experimental.harbor_cua.conftest import FakeImageProcessor

    return FakeImageProcessor()


def _decoding_tokenizer():
    from tests.fast.examples.experimental.harbor_cua.conftest import FakeTokenizer

    return FakeTokenizer()
