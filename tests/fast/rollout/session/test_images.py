"""Image storage, identity, and the token expansion the engine performs."""

import base64
import hashlib
import io

import pytest
import torch
from PIL import Image

from miles.rollout.session.errors import MessageValidationError
from miles.rollout.session.images.processing import (
    ProcessedImage,
    expand_input_ids,
    merge_train_inputs,
    placeholder_runs,
    retained_image_count,
)
from miles.rollout.session.images.store import NO_IMAGES, ImageOccurrences, ImageStore, intern_images

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


class TestImageStore:
    def test_identity_is_the_content_hash(self):
        payload = png_bytes(8, 8, (1, 2, 3))
        blob = ImageStore().add(payload)
        assert blob.image_id == f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def test_identical_bytes_intern_once(self):
        store = ImageStore()
        payload = png_bytes(8, 8, (1, 2, 3))
        assert store.add(payload) is store.add(payload)
        assert len(store) == 1

    def test_same_size_different_pixels_are_different_images(self):
        store = ImageStore()
        first = store.add(png_bytes(40, 40, (0, 0, 0)))
        second = store.add(png_bytes(40, 40, (255, 255, 255)))
        assert first.image_id != second.image_id
        assert len(store) == 2


class TestInterning:
    def test_occurrences_keep_prompt_order(self):
        store = ImageStore()
        first, second = png_bytes(8, 8, (1, 1, 1)), png_bytes(8, 8, (2, 2, 2))
        images = intern_images(
            [
                {"role": "system", "content": "be helpful"},
                user(image_part(first), {"type": "text", "text": "and"}, image_part(second)),
            ],
            store,
        )
        assert images.image_ids == (
            f"sha256:{hashlib.sha256(first).hexdigest()}",
            f"sha256:{hashlib.sha256(second).hexdigest()}",
        )

    def test_a_repeated_image_stays_two_occurrences_of_one_blob(self):
        store = ImageStore()
        payload = png_bytes(8, 8, (7, 7, 7))
        images = intern_images([user(image_part(payload), image_part(payload))], store)
        assert len(images) == 2
        assert images.image_ids[0] == images.image_ids[1]
        assert len(store) == 1

    def test_reordering_changes_the_occurrence_sequence(self):
        store = ImageStore()
        first, second = png_bytes(8, 8, (1, 1, 1)), png_bytes(8, 8, (2, 2, 2))
        forward = intern_images([user(image_part(first), image_part(second))], store)
        reverse = intern_images([user(image_part(second), image_part(first))], store)
        assert forward.image_ids != reverse.image_ids

    def test_string_content_and_text_parts_yield_no_images(self):
        store = ImageStore()
        messages = [{"role": "user", "content": "plain"}, user({"type": "text", "text": "parts"})]
        assert intern_images(messages, store) == NO_IMAGES

    def test_a_remote_url_is_refused_because_it_is_not_immutable_content(self):
        with pytest.raises(MessageValidationError, match="inline base64 data URL"):
            intern_images(
                [user({"type": "image_url", "image_url": {"url": "https://example.test/a.png"}})], ImageStore()
            )

    def test_video_and_audio_parts_are_refused_rather_than_dropped(self):
        for part_type in ("video_url", "input_audio"):
            with pytest.raises(MessageValidationError, match="images only"):
                intern_images([user({"type": part_type})], ImageStore())

    def test_image_spellings_the_chat_route_does_not_attach_are_refused(self):
        """SGLang's chat request accepts only image_url, though the chat template renders these too."""
        for part_type in ("image", "input_image"):
            with pytest.raises(MessageValidationError, match="as an image_url part"):
                intern_images(
                    [user({"type": part_type, "image_url": data_url(png_bytes(4, 4, (0, 0, 0)))})], ImageStore()
                )

    def test_a_declared_content_hash_must_be_the_hash_of_the_bytes(self):
        payload = png_bytes(8, 8, (1, 2, 3))
        part = image_part(payload)
        part["image_url"]["content_hash"] = "sha256:" + hashlib.sha256(payload).hexdigest().upper()
        assert len(intern_images([user(part)], ImageStore())) == 1
        part["image_url"]["content_hash"] = "sha256:" + hashlib.sha256(b"other").hexdigest()
        with pytest.raises(MessageValidationError, match="is not the SHA-256"):
            intern_images([user(part)], ImageStore())

    def test_undecodable_payload_is_refused(self):
        with pytest.raises(MessageValidationError, match="base64"):
            intern_images(
                [user({"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!"}})], ImageStore()
            )

    def test_unsupported_mime_type_is_refused(self):
        with pytest.raises(MessageValidationError, match="unsupported image type"):
            intern_images(
                [user({"type": "image_url", "image_url": {"url": "data:image/svg+xml;base64,AAAA"}})], ImageStore()
            )


class TestOccurrencePrefix:
    def test_a_shorter_identical_sequence_is_a_prefix(self):
        parent = ImageOccurrences()
        assert parent.is_prefix_of(ImageOccurrences())

    def test_appending_keeps_the_prefix_relation(self):
        store = ImageStore()
        first, second = png_bytes(8, 8, (1, 1, 1)), png_bytes(8, 8, (2, 2, 2))
        parent = intern_images([user(image_part(first))], store)
        child = intern_images([user(image_part(first), image_part(second))], store)
        assert parent.is_prefix_of(child)
        assert not child.is_prefix_of(parent)

    def test_replacing_an_earlier_image_breaks_the_prefix_relation(self):
        store = ImageStore()
        first, second = png_bytes(8, 8, (1, 1, 1)), png_bytes(8, 8, (2, 2, 2))
        parent = intern_images([user(image_part(first))], store)
        rewritten = intern_images([user(image_part(second))], store)
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
