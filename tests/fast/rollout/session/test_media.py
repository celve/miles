"""Media interning, identity, and the token expansion the engine performs."""

import base64
import hashlib
import io

import pytest
import torch
from PIL import Image

from miles.rollout.session.errors import MessageValidationError
from miles.rollout.session.image_processing import (
    ProcessedImage,
    expand_input_ids,
    merge_train_inputs,
    placeholder_runs,
    retained_image_count,
)
from miles.rollout.session.media import EMPTY_MEDIA, MediaArena, RequestMedia, extract_request_media

PAD = 151655


def png_bytes(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def data_url(payload: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def image_part(payload: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url(payload)}}


def user(*parts) -> dict:
    return {"role": "user", "content": list(parts)}


class TestMediaArena:
    def test_identity_is_the_content_hash(self):
        payload = png_bytes(8, 8, (1, 2, 3))
        asset = MediaArena().intern(payload, "image/png")
        assert asset.asset_id == f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def test_identical_bytes_intern_once(self):
        arena = MediaArena()
        payload = png_bytes(8, 8, (1, 2, 3))
        assert arena.intern(payload, "image/png") is arena.intern(payload, "image/png")
        assert len(arena) == 1

    def test_same_size_different_pixels_are_different_assets(self):
        arena = MediaArena()
        first = arena.intern(png_bytes(40, 40, (0, 0, 0)), "image/png")
        second = arena.intern(png_bytes(40, 40, (255, 255, 255)), "image/png")
        assert first.asset_id != second.asset_id
        assert len(arena) == 2

    def test_data_url_round_trips(self):
        payload = png_bytes(8, 8, (4, 5, 6))
        asset = MediaArena().intern(payload, "image/png")
        assert asset.to_data_url() == data_url(payload)


class TestExtraction:
    def test_occurrences_keep_prompt_order_and_location(self):
        arena = MediaArena()
        first, second = png_bytes(8, 8, (1, 1, 1)), png_bytes(8, 8, (2, 2, 2))
        media = extract_request_media(
            [
                {"role": "system", "content": "be helpful"},
                user(image_part(first), {"type": "text", "text": "and"}, image_part(second)),
            ],
            arena,
        )
        assert [(o.message_index, o.part_index) for o in media.occurrences] == [(1, 0), (1, 2)]
        assert media.asset_ids == (
            f"sha256:{hashlib.sha256(first).hexdigest()}",
            f"sha256:{hashlib.sha256(second).hexdigest()}",
        )

    def test_a_repeated_image_stays_two_occurrences_of_one_asset(self):
        arena = MediaArena()
        payload = png_bytes(8, 8, (7, 7, 7))
        media = extract_request_media([user(image_part(payload), image_part(payload))], arena)
        assert len(media) == 2
        assert media.asset_ids[0] == media.asset_ids[1]
        assert len(arena) == 1

    def test_reordering_changes_the_occurrence_sequence(self):
        arena = MediaArena()
        first, second = png_bytes(8, 8, (1, 1, 1)), png_bytes(8, 8, (2, 2, 2))
        forward = extract_request_media([user(image_part(first), image_part(second))], arena)
        reverse = extract_request_media([user(image_part(second), image_part(first))], arena)
        assert forward.asset_ids != reverse.asset_ids

    def test_string_content_and_text_parts_yield_no_media(self):
        arena = MediaArena()
        messages = [{"role": "user", "content": "plain"}, user({"type": "text", "text": "parts"})]
        assert extract_request_media(messages, arena) == EMPTY_MEDIA

    def test_a_remote_url_is_refused_because_it_is_not_immutable_content(self):
        with pytest.raises(MessageValidationError, match="inline base64 data URL"):
            extract_request_media(
                [user({"type": "image_url", "image_url": {"url": "https://example.test/a.png"}})], MediaArena()
            )

    def test_video_and_audio_parts_are_refused_rather_than_dropped(self):
        for part_type in ("video_url", "input_audio"):
            with pytest.raises(MessageValidationError, match="images only"):
                extract_request_media([user({"type": part_type})], MediaArena())

    def test_undecodable_payload_is_refused(self):
        with pytest.raises(MessageValidationError, match="base64"):
            extract_request_media(
                [user({"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!"}})], MediaArena()
            )

    def test_unsupported_mime_type_is_refused(self):
        with pytest.raises(MessageValidationError, match="unsupported image type"):
            extract_request_media(
                [user({"type": "image_url", "image_url": {"url": "data:image/svg+xml;base64,AAAA"}})], MediaArena()
            )


class TestMediaPrefix:
    def test_a_shorter_identical_sequence_is_a_prefix(self):
        parent = RequestMedia(occurrences=())
        assert parent.is_prefix_of(RequestMedia(occurrences=()))

    def test_appending_keeps_the_prefix_relation(self):
        arena = MediaArena()
        first, second = png_bytes(8, 8, (1, 1, 1)), png_bytes(8, 8, (2, 2, 2))
        parent = extract_request_media([user(image_part(first))], arena)
        child = extract_request_media([user(image_part(first), image_part(second))], arena)
        assert parent.is_prefix_of(child)
        assert not child.is_prefix_of(parent)

    def test_replacing_an_earlier_image_breaks_the_prefix_relation(self):
        arena = MediaArena()
        first, second = png_bytes(8, 8, (1, 1, 1)), png_bytes(8, 8, (2, 2, 2))
        parent = extract_request_media([user(image_part(first))], arena)
        rewritten = extract_request_media([user(image_part(second))], arena)
        assert not parent.is_prefix_of(rewritten)


class TestExpandInputIds:
    def test_each_placeholder_grows_and_other_tokens_stay_verbatim(self):
        assert expand_input_ids([1, PAD, 2, PAD, 3], [3, 2], PAD) == [1, PAD, PAD, PAD, 2, PAD, PAD, 3]

    def test_no_images_is_the_identity(self):
        assert expand_input_ids([1, 2, 3], [], PAD) == [1, 2, 3]

    def test_a_count_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="1 image placeholder"):
            expand_input_ids([1, PAD, 2], [4, 4], PAD)
        with pytest.raises(ValueError, match="2 image placeholder"):
            expand_input_ids([PAD, PAD], [4], PAD)

    def test_expansion_matches_the_engines_own_helper(self):
        """The two implementations must agree or we train on ids inference never saw."""
        sglang_expand = pytest.importorskip(
            "sglang.srt.multimodal.processors.base_processor"
        ).BaseMultimodalProcessor._expand_input_ids
        ids, counts = [5, PAD, 6, 7, PAD, 8], [4, 9]
        assert expand_input_ids(ids, counts, PAD) == sglang_expand(ids, counts, PAD)


class TestPlaceholderRuns:
    def test_runs_are_measured_in_order(self):
        assert placeholder_runs([1, PAD, PAD, 2, PAD, 3], PAD) == [2, 1]

    def test_a_trailing_run_is_counted(self):
        assert placeholder_runs([1, PAD, PAD], PAD) == [2]

    def test_no_placeholders_is_empty(self):
        assert placeholder_runs([1, 2, 3], PAD) == []


class TestRetainedImageCount:
    def test_every_whole_run_is_retained(self):
        expanded = expand_input_ids([1, PAD, 2, PAD], [3, 2], PAD)
        assert retained_image_count(expanded, [3, 2], PAD) == 2

    def test_truncation_mid_image_drops_that_image(self):
        expanded = expand_input_ids([1, PAD, 2, PAD], [3, 5], PAD)
        assert retained_image_count(expanded[:-2], [3, 5], PAD) == 1

    def test_truncation_before_an_image_drops_it(self):
        expanded = expand_input_ids([1, PAD, 2, PAD], [3, 5], PAD)
        cut = expanded.index(2) + 1
        assert retained_image_count(expanded[:cut], [3, 5], PAD) == 1

    def test_a_short_interior_run_is_a_disagreement_not_a_truncation(self):
        with pytest.raises(ValueError, match="disagree"):
            retained_image_count([PAD, PAD, 1, PAD, PAD, PAD], [3, 3], PAD)

    def test_more_runs_than_images_is_refused(self):
        with pytest.raises(ValueError, match="only 1 image"):
            retained_image_count([PAD, 1, PAD], [1], PAD)


class TestMergeTrainInputs:
    @staticmethod
    def processed(patches: int, grid: list[int], tokens: int) -> ProcessedImage:
        return ProcessedImage(
            train_inputs={
                "pixel_values": torch.full((patches, 4), float(patches)),
                "image_grid_thw": torch.tensor([grid], dtype=torch.int64),
            },
            token_count=tokens,
        )

    def test_none_for_no_images(self):
        assert merge_train_inputs([]) is None

    def test_concatenation_follows_occurrence_order(self):
        merged = merge_train_inputs([self.processed(2, [1, 2, 2], 1), self.processed(3, [1, 2, 6], 3)])
        assert merged["pixel_values"].shape == (5, 4)
        assert merged["pixel_values"][:, 0].tolist() == [2.0, 2.0, 3.0, 3.0, 3.0]
        assert merged["image_grid_thw"].tolist() == [[1, 2, 2], [1, 2, 6]]

    def test_disagreeing_keys_are_refused(self):
        odd = ProcessedImage(train_inputs={"pixel_values": torch.zeros(1, 4)}, token_count=1)
        with pytest.raises(ValueError, match="disagree on keys"):
            merge_train_inputs([self.processed(2, [1, 2, 2], 1), odd])
