"""Turn recorded image occurrences into vision inputs, and prove they line up.

The engine returns a multimodal chat prompt's token ids already expanded, so the
job here is not to lay out placeholders but to verify them: run *i* of image-pad
tokens must be exactly as long as occurrence *i*'s processed token count, in
order. An equal total proves neither the per-image assignment nor the ordering.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from PIL import Image

from examples.experimental.harbor_cua.calls import AssetArena, MediaOccurrence

IMAGE_PAD_TOKEN = "<|image_pad|>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"


class MediaMismatchError(ValueError):
    """The token sequence and the recorded occurrences disagree."""


@dataclass(frozen=True)
class ProcessedAsset:
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    token_count: int


class MediaProcessor:
    """Processes each distinct asset once, under one pinned processor identity.

    Qwen-VL resizes every image on its own and concatenates the patch rows, so a
    per-asset call then concatenated in occurrence order is byte-identical to
    processing the whole list at once — which is what makes caching sound.
    """

    def __init__(self, processor, resize_settings: dict[str, int] | None = None) -> None:
        self._image_processor = processor.image_processor
        self._tokenizer = processor.tokenizer
        self._resize_settings = dict(resize_settings or {})
        for name, value in self._resize_settings.items():
            setattr(self._image_processor, name, value)
        self._merge_size = int(getattr(self._image_processor, "merge_size", 2))
        self._cache: dict[str, ProcessedAsset] = {}

    @property
    def image_pad_id(self) -> int:
        return self._tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)

    def fingerprint(self) -> str:
        image_processor = self._image_processor
        parts = [type(image_processor).__name__, f"merge={self._merge_size}"]
        for name in ("patch_size", "temporal_patch_size", "min_pixels", "max_pixels", "size"):
            parts.append(f"{name}={getattr(image_processor, name, None)}")
        return "|".join(parts)

    def process(self, arena: AssetArena, occurrence: MediaOccurrence) -> ProcessedAsset:
        cached = self._cache.get(occurrence.asset_id)
        if cached is not None:
            return cached
        image = Image.open(io.BytesIO(arena.get(occurrence.asset_id))).convert("RGB")
        processed = self._image_processor(images=[image], return_tensors="pt")
        grid = processed["image_grid_thw"]
        token_count = int(torch.prod(grid, dim=-1).sum().item()) // (self._merge_size**2)
        asset = ProcessedAsset(
            pixel_values=processed["pixel_values"],
            image_grid_thw=grid,
            token_count=token_count,
        )
        self._cache[occurrence.asset_id] = asset
        return asset

    def process_all(self, arena: AssetArena, occurrences: Sequence[MediaOccurrence]) -> list[ProcessedAsset]:
        return [self.process(arena, occurrence) for occurrence in occurrences]

    def placeholder_tokens(self, token_count: int) -> list[int]:
        convert = self._tokenizer.convert_tokens_to_ids
        return [convert(VISION_START_TOKEN), *([convert(IMAGE_PAD_TOKEN)] * token_count), convert(VISION_END_TOKEN)]


def pad_token_runs(tokens: Sequence[int], image_pad_id: int) -> list[int]:
    """The length of each maximal run of ``image_pad_id``, in order."""
    runs: list[int] = []
    run = 0
    for token in tokens:
        if token == image_pad_id:
            run += 1
            continue
        if run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return runs


def verify_occurrence_runs(
    tokens: Sequence[int],
    image_pad_id: int,
    assets: Sequence[ProcessedAsset],
    *,
    where: str,
) -> None:
    runs = pad_token_runs(tokens, image_pad_id)
    expected = [asset.token_count for asset in assets]
    if runs != expected:
        raise MediaMismatchError(
            f"{where}: image-pad runs {runs} do not match the processed token counts {expected} "
            f"of the {len(expected)} recorded occurrences, in order"
        )


def build_train_inputs(assets: Sequence[ProcessedAsset]) -> dict[str, torch.Tensor] | None:
    """``multimodal_train_inputs`` for one sample: tensors only, occurrence order."""
    if not assets:
        return None
    return {
        "pixel_values": torch.cat([asset.pixel_values for asset in assets], dim=0),
        "image_grid_thw": torch.cat([asset.image_grid_thw for asset in assets], dim=0),
    }


def placeholder_vision_sample(processor) -> tuple[list[int], dict[str, torch.Tensor]]:
    """One minimal image's tokens and tensors, for a sample that trains on nothing.

    It exists only to keep a write-off on the same vision-tower collective
    schedule as its siblings; it is never added to a real sampled context.
    """
    image_processor = processor.image_processor
    processed = image_processor(images=[Image.new("RGB", (64, 64))], return_tensors="pt")
    merge = int(getattr(image_processor, "merge_size", 2))
    token_count = int(torch.prod(processed["image_grid_thw"], dim=-1).sum().item()) // (merge * merge)
    convert = processor.tokenizer.convert_tokens_to_ids
    tokens = [
        convert(VISION_START_TOKEN),
        *([convert(IMAGE_PAD_TOKEN)] * token_count),
        convert(VISION_END_TOKEN),
    ]
    return tokens, {"pixel_values": processed["pixel_values"], "image_grid_thw": processed["image_grid_thw"]}


def render_prompt_ids(processor, messages: list[dict], images: Sequence[Image.Image]) -> list[int]:
    """The prompt ids this machine would render for ``messages``.

    Used to compare against the engine's effective ids, which is the check that
    covers the chat template as well as image expansion. It is deliberately not
    on the conversion path: the engine's ids are what inference consumed, and a
    local render that disagreed would be the thing in doubt.
    """
    rendered = processor.apply_chat_template(
        _hf_content(messages, images),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    ids = rendered["input_ids"]
    return ids[0].tolist() if hasattr(ids, "tolist") and getattr(ids, "ndim", 1) > 1 else list(ids)


def _hf_content(messages: list[dict], images: Sequence[Image.Image]) -> list[dict]:
    """OpenAI ``image_url`` parts as the chat template's own ``image`` parts, in order."""
    remaining = iter(images)
    converted = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            converted.append(message)
            continue
        parts = [
            {"type": "image", "image": next(remaining)} if part.get("type") == "image_url" else part
            for part in content
        ]
        converted.append({**message, "content": parts})
    return converted


def first_difference(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))
