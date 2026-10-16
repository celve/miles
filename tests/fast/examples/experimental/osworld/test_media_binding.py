"""Offline tests: no GPU, no model, no processor, no trace.

media_binding.py's one claim: a screenshot's pixels sit under that
screenshot's token positions. The cases below are the corruptions a summed
total would wave through.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from examples.experimental.osworld.media_binding import (
    MediaBindingError,
    image_token_runs,
    media_digests,
    processor_identity,
    verify_image_token_layout,
)

PAD = 99


def test_runs_are_maximal_and_ordered():
    tokens = [1, PAD, PAD, 2, 3, PAD, 4, PAD, PAD, PAD]
    assert image_token_runs(tokens, PAD) == [(1, 2), (5, 1), (7, 3)]


def test_run_at_the_end_is_closed():
    assert image_token_runs([1, PAD, PAD], PAD) == [(1, 2)]


def test_no_images_is_no_runs():
    assert image_token_runs([1, 2, 3], PAD) == []


def test_exact_correspondence_passes():
    tokens = [1, PAD, PAD, 2, PAD, PAD, PAD]
    assert verify_image_token_layout(tokens, PAD, [2, 3]) is None


def test_more_screenshots_than_runs_is_rejected():
    """Cutting the images to whichever prefix the runs total would accept a
    sequence that lost an occurrence as a shorter correct one. The caller only
    ever recovers screenshots a prompt referenced, so there is no legitimate
    extra to tolerate."""
    tokens = [1, PAD, PAD, 2]
    with pytest.raises(MediaBindingError, match="1 image runs but 3 screenshots"):
        verify_image_token_layout(tokens, PAD, [2, 3, 7])


def test_wrong_per_image_split_with_the_right_total_is_rejected():
    """[2, 3] and [3, 2] both total 5. Accepting the total binds image 0's
    pixels to image 1's positions, which is the corruption that matters."""
    tokens = [1, PAD, PAD, 2, PAD, PAD, PAD]
    with pytest.raises(MediaBindingError, match="image 0: ids give it 2 tokens, the processor gives it 3"):
        verify_image_token_layout(tokens, PAD, [3, 2])


def test_swapped_image_order_is_rejected():
    tokens = [PAD, PAD, 1, PAD, PAD, PAD, PAD]
    with pytest.raises(MediaBindingError, match="image 0"):
        verify_image_token_layout(tokens, PAD, [4, 2])


def test_more_runs_than_screenshots_is_rejected():
    tokens = [PAD, 1, PAD, 2, PAD]
    with pytest.raises(MediaBindingError, match="3 image runs but 2 screenshots"):
        verify_image_token_layout(tokens, PAD, [1, 1])


def test_digests_are_ordered_and_content_addressed():
    digests = media_digests([b"first", b"second", b"first"])
    assert digests[0] == digests[2] != digests[1]
    assert all(d.startswith("sha256:") and len(d) == len("sha256:") + 64 for d in digests)


def test_processor_identity_records_both_resize_spellings():
    """Qwen3-VL-4B-Instruct pins its budget as `size`; older checkpoints use
    min/max_pixels. A run is only reproducible if whichever one applied is
    written down."""
    modern = processor_identity(
        SimpleNamespace(merge_size=2, patch_size=16, size={"longest_edge": 16777216, "shortest_edge": 65536})
    )
    assert modern == {
        "class": "SimpleNamespace",
        "merge_size": 2,
        "patch_size": 16,
        "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    }

    legacy = processor_identity(SimpleNamespace(merge_size=2, min_pixels=256, max_pixels=1024))
    assert legacy["min_pixels"] == 256 and legacy["max_pixels"] == 1024 and "size" not in legacy
