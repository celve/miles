"""Synthetic recorded calls, built the way Harbor's generic loop builds prompts."""

import io
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from examples.experimental.harbor_cua.calls import AssetArena, RecordedCall, RolloutRecord

IMAGE_PAD_ID = 777
VISION_START_ID = 775
VISION_END_ID = 776
TOKENS_PER_IMAGE = 4


def png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def image_block() -> list[int]:
    return [VISION_START_ID, *([IMAGE_PAD_ID] * TOKENS_PER_IMAGE), VISION_END_ID]


class FakeImageProcessor:
    """One fixed grid for every image, so two different images share a run length."""

    merge_size = 2
    patch_size = 16
    min_pixels = 1
    max_pixels = 2

    def __call__(self, images, return_tensors="pt"):
        count = len(images)
        return {
            "pixel_values": torch.arange(count * 8, dtype=torch.float32).reshape(count * 2, 4),
            "image_grid_thw": torch.tensor([[1, 4, 4]] * count),
        }


class FakeTokenizer:
    _ids = {"<|image_pad|>": IMAGE_PAD_ID, "<|vision_start|>": VISION_START_ID, "<|vision_end|>": VISION_END_ID}

    def convert_tokens_to_ids(self, token):
        return self._ids[token]

    def decode(self, tokens, skip_special_tokens=False):
        return " ".join(str(token) for token in tokens)


@pytest.fixture
def fake_processor():
    return SimpleNamespace(image_processor=FakeImageProcessor(), tokenizer=FakeTokenizer())


class EpisodeBuilder:
    """Grows prompts the way an append-only chat harness does."""

    def __init__(self):
        self.record = RolloutRecord(arena=AssetArena())
        self._prompt: list[int] = [1, 2, 3]
        self._media: list = []
        self._next = 0

    def turn(self, *, image: bytes | None = None, output: list[int] | None = None, auxiliary: bool = False):
        if image is not None:
            self._prompt = self._prompt + image_block()
            self._media = self._media + [self.record.arena.put(image, "image/png")]
        output = output or [900 + self._next, 901 + self._next]
        call = RecordedCall(
            index=self._next,
            auxiliary=auxiliary,
            prompt_token_ids=tuple(self._prompt),
            output_token_ids=tuple(output),
            output_logprobs=tuple(-0.5 for _ in output),
            media=tuple(self._media),
            finish_reason="stop",
            meta_info={"weight_version": "7", "output_token_logprobs": [(-0.5, t, "") for t in output]},
        )
        self.record.calls.append(call)
        self._prompt = self._prompt + list(output) + [10 + self._next]
        self._next += 1
        return call

    def rewrite(self, *, media, prompt):
        """Replace the running prompt and media, as a trim or a summary does."""
        self._prompt = list(prompt)
        self._media = list(media)


@pytest.fixture
def episode():
    return EpisodeBuilder()
