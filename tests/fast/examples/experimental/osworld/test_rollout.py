"""Offline tests: no GPU, no network, no Gym, no OSWorld VM, no live model.

rollout.py's translation: one NeMo Gym exact-trace envelope -> a fixed number
of Miles samples per episode, the real segments stitched and bound to their
screenshots, the rest inert. The Gym side of the seam is faked at exactly the
boundary the real code touches.
"""

from __future__ import annotations

import torch
from types import SimpleNamespace

import pytest

from examples.experimental.osworld.rollout import (
    EpisodeConfig,
    _aborted,
    _calls,
    _media_occurrences,
    _padding,
    _segments,
    _stitch,
    _vision_inputs,
)
from examples.experimental.osworld.media_binding import MediaBindingError
from miles.utils.types import Sample

IMAGE_PAD = 151655


class _FakeImageProcessor:
    """Tokens per image come from the grid, as the real fallback path does."""

    merge_size = 2
    patch_size = 16

    def __call__(self, images, return_tensors=None):
        grid = torch.tensor([[1, 2, 2 * (index + 1)] for index, _ in enumerate(images)], dtype=torch.int64)
        return {"pixel_values": torch.zeros(len(images), 4), "image_grid_thw": grid}


class _FakeProcessor:
    def __init__(self):
        self.image_processor = _FakeImageProcessor()
        self.tokenizer = SimpleNamespace(
            convert_tokens_to_ids=lambda token: {
                "<|image_pad|>": IMAGE_PAD,
                "<|vision_start|>": 151652,
                "<|vision_end|>": 151653,
            }[token]
        )


def _state():
    return SimpleNamespace(processor=_FakeProcessor(), tokenizer=SimpleNamespace(decode=lambda ids, **kw: "decoded"))


def _call(prompt, sampled, segment_id="s0", segment_index=0, media_ids=(), eligible=True):
    """One call in the shape `_calls` produces, which is what the rest consumes."""
    return {
        "prompt_token_ids": prompt,
        "sampled_token_ids": sampled,
        "sampled_logprobs": [-0.1] * len(sampled),
        "segment_id": segment_id,
        "segment_index": segment_index,
        "media_ids": list(media_ids),
        "eligible": eligible,
    }


def _envelope(calls):
    """Put calls back into the two arrays Gym's transport v3 splits them across."""
    return {
        "output": [
            {
                "prompt_token_ids": c["prompt_token_ids"],
                "generation_token_ids": c["sampled_token_ids"],
                "generation_log_probs": c["sampled_logprobs"],
            }
            for c in calls
        ],
        "model_call_metadata": [
            {
                "segment_id": c["segment_id"],
                "segment_index": c["segment_index"],
                "media_ids": c["media_ids"],
                "eligible": c["eligible"],
            }
            for c in calls
        ],
    }


def test_calls_joins_the_two_arrays_transport_v3_splits():
    calls = _calls(_envelope([_call([1, 2], [10]), _call([1, 2, 10, 3], [11])]))
    assert [c["sampled_token_ids"] for c in calls] == [[10], [11]]
    assert [c["segment_id"] for c in calls] == ["s0", "s0"]


def test_calls_rejects_a_trace_whose_arrays_disagree_in_length():
    envelope = _envelope([_call([1], [10])])
    envelope["model_call_metadata"] = []
    with pytest.raises(ValueError, match="exact trace is inconsistent"):
        _calls(envelope)


def test_segments_group_by_id_and_keep_boundary_order():
    calls = _calls(
        _envelope(
            [
                _call([1], [10], segment_id="b", segment_index=1),
                _call([2], [11], segment_id="a", segment_index=0),
                _call([2, 11, 3], [12], segment_id="a", segment_index=0),
            ]
        )
    )
    segments = _segments(calls)
    assert [len(s) for s in segments] == [2, 1]
    assert segments[0][0]["segment_id"] == "a"


def test_segments_drop_calls_gym_marked_ineligible():
    calls = _calls(_envelope([_call([1], [10], eligible=False)]))
    assert _segments(calls) == []


def test_stitch_masks_only_the_sampled_tokens():
    tokens, mask, logprobs, start = _stitch(
        [_call([1, 2, 3], [10, 11]), _call([1, 2, 3, 10, 11, 4], [12])]
    )
    assert tokens == [1, 2, 3, 10, 11, 4, 12]
    assert start == 3
    assert mask == [1, 1, 0, 1]
    assert logprobs == [-0.1, -0.1, 0.0, -0.1]
    assert len(mask) == len(logprobs) == len(tokens) - start


def test_stitch_refuses_a_call_that_does_not_extend_its_predecessor():
    """Gym promises append-compatibility only inside a segment, so a violation
    means the trace and this reader disagree and nothing here is trainable."""
    assert _stitch([_call([1, 2], [10]), _call([9, 9, 9], [11])]) is None


def test_stitch_refuses_a_call_whose_logprobs_do_not_cover_its_tokens():
    call = _call([1], [10, 11])
    call["sampled_logprobs"] = [-0.1]
    assert _stitch([call]) is None


def test_vision_inputs_verify_each_image_against_its_own_run():
    """The fake processor gives image i 1*2*2(i+1)/4 = (i+1) tokens."""
    state = _state()
    tokens = [1, IMAGE_PAD, 2, IMAGE_PAD, IMAGE_PAD, 3]
    inputs = _vision_inputs(state, [object(), object()], tokens)
    assert inputs["image_grid_thw"].shape[0] == 2


def test_vision_inputs_reject_a_sequence_missing_one_image_run():
    """Cutting the images to whichever prefix the runs total would accept this
    as a correct one-image segment and bind the wrong pixels everywhere after."""
    state = _state()
    with pytest.raises(MediaBindingError, match="1 image runs but 2 screenshots"):
        _vision_inputs(state, [object(), object()], [1, IMAGE_PAD, 2])


def test_media_occurrences_take_the_final_prompt_and_check_the_earlier_ones():
    segment = [
        _call([1], [10], media_ids=["a"]),
        _call([1, 10, 2], [11], media_ids=["a", "b"]),
    ]
    assert _media_occurrences(segment) == ["a", "b"]


def test_media_occurrences_reject_a_screenshot_swapped_inside_a_segment():
    """Two calls can swap one screenshot for another of the same size without
    disturbing the token prefix at all, and both digests stay valid -- so only
    the media prefix catches it."""
    segment = [
        _call([1], [10], media_ids=["a"]),
        _call([1, 10, 2], [11], media_ids=["b"]),
    ]
    with pytest.raises(MediaBindingError, match="not an ordered prefix"):
        _media_occurrences(segment)


def test_media_occurrences_keep_a_screenshot_referenced_twice():
    """The arena is content-addressed, so one prompt showing the same
    screenshot twice lists its id twice and owes two token runs. Collapsing
    them rejected a legitimate episode, and then aborted it as an
    infrastructure failure."""
    assert _media_occurrences([_call([1], [10], media_ids=["a", "a"])]) == ["a", "a"]


def test_vision_inputs_reject_a_swapped_image_order():
    state = _state()
    # Runs of 2 then 1, while the processor gives 1 then 2.
    tokens = [IMAGE_PAD, IMAGE_PAD, 1, IMAGE_PAD]
    with pytest.raises(MediaBindingError, match="image 0"):
        _vision_inputs(state, [object(), object()], tokens)


def test_vision_inputs_report_a_text_only_call_as_none():
    assert _vision_inputs(_state(), [], [1, 2, 3]) is None


def test_padding_carries_the_episode_reward_so_normalization_accepts_it():
    """`_normalize_rewards_by_rollout` refuses a rollout whose siblings disagree
    on a reward, and takes one reward per rollout -- so a pad holding the
    episode's own reward moves no group statistic."""
    pad = _padding(Sample(index=7), reward=1.0, rollout_id=7, index=3, state=_state())
    assert pad.metadata["reward"] == 1.0
    assert pad.rollout_id == 7
    assert pad.remove_sample is True
    assert pad.loss_mask == [] and pad.response_length == 0
    assert pad.status is Sample.Status.COMPLETED
    assert pad.multimodal_train_inputs["pixel_values"] is not None


def test_an_infrastructure_failure_aborts_rather_than_scoring_zero():
    """ABORTED is the only status Miles acts on: `apply_aborted_filter` drops
    the group before any reward statistic exists. FAILED plus `remove_sample`
    would keep the row and feed its 0.0 into the group's mean."""
    samples = _aborted(Sample(index=1), "transport: TimeoutError", _state(), dict.fromkeys(("a", "b"), 0))
    assert len(samples) == 1
    assert samples[0].status is Sample.Status.ABORTED
    assert samples[0].metadata["osworld_abort_reason"] == "transport: TimeoutError"


def test_episode_config_refuses_an_unset_sample_count(monkeypatch):
    monkeypatch.delenv("MILES_SCRIPT_OSWORLD_SAMPLES_PER_EPISODE", raising=False)
    with pytest.raises(ValueError, match="at least 1"):
        EpisodeConfig.read(SimpleNamespace())


def test_episode_config_reads_the_environment_and_strips_the_gym_url(monkeypatch):
    monkeypatch.setenv("MILES_SCRIPT_OSWORLD_SAMPLES_PER_EPISODE", "15")
    monkeypatch.setenv("MILES_SCRIPT_GYM_URL", "http://10.0.0.1:11001/")
    config = EpisodeConfig.read(SimpleNamespace())
    assert config.samples_per_episode == 15
    assert config.gym_url == "http://10.0.0.1:11001"
