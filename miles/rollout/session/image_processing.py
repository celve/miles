"""Turn immutable image bytes into token counts and trainer tensors.

The session server renders prompts with the plain tokenizer, so an image is one
placeholder token in the rendered ids. SGLang expands the i-th placeholder into
``counts[i]`` copies and keeps every other token verbatim
(``base_processor._expand_input_ids``), so the sequence inference consumed is a
pure positional function of the rendered ids and the per-image counts. This
module is that function, plus the processor call the counts and pixels come from.

Everything here works on one image at a time and concatenates: Qwen-VL resizes
and patchifies each image independently, so per-asset outputs stacked in
occurrence order equal a single multi-image processor call.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from miles.rollout.session.media import MediaAsset
from miles.utils.processing_utils import call_processor, extract_multimodal_train_inputs

logger = logging.getLogger(__name__)

DEFAULT_MERGE_SIZE = 2
GRID_KEY = "image_grid_thw"


@dataclass(frozen=True)
class ProcessorFingerprint:
    """What a processed image depends on besides its own bytes.

    Recorded on every request and compared across turns: a run whose resize
    settings or processor revision changed cannot reuse an earlier turn's pixels.
    """

    checkpoint: str
    image_token: str
    image_token_id: int
    merge_size: int
    min_pixels: int | None
    max_pixels: int | None
    patch_size: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "image_token": self.image_token,
            "image_token_id": self.image_token_id,
            "merge_size": self.merge_size,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "patch_size": self.patch_size,
        }


@dataclass(frozen=True)
class ProcessedImage:
    """One asset's full processor output: the trainer's kwargs for that image."""

    train_inputs: dict[str, torch.Tensor]  # trainer kwargs, e.g. pixel_values + image_grid_thw
    token_count: int  # placeholder tokens this image expands into

    @property
    def grid(self) -> torch.Tensor:
        return self.train_inputs[GRID_KEY]


class ImageTokenCounter:
    """Per-image expanded token counts, from the image's size alone.

    Serving a turn needs the count, not the pixels: the count decides the token
    layout, and one 720p frame's ``pixel_values`` is 21 MB. Measured on
    Qwen3-VL-2B, counting four images from their sizes takes 0.1 ms against
    83 ms to process them, and the two agree exactly — so a session holds counts
    while it runs and pays for pixels once, at assembly, for the images that
    survived.
    """

    def __init__(self, processor: Any, fingerprint: ProcessorFingerprint) -> None:
        self.processor = processor
        self.fingerprint = fingerprint
        self._counts: dict[str, int] = {}

    def count(self, asset: MediaAsset) -> int:
        cached = self._counts.get(asset.asset_id)
        if cached is not None:
            return cached
        count = _count_from_size(self.processor, asset)
        self._counts[asset.asset_id] = count
        return count

    def count_all(self, assets: Sequence[MediaAsset]) -> list[int]:
        return [self.count(asset) for asset in assets]


def _count_from_size(processor: Any, asset: MediaAsset) -> int:
    """How many placeholder tokens this image expands into, read from its header."""
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(asset.data)) as handle:
        size = (handle.height, handle.width)
    counts = processor._get_num_multimodal_tokens(image_sizes=[size]).num_image_tokens
    count = int(counts[0])
    if count <= 0:
        raise ValueError(f"{asset.asset_id} ({size[1]}x{size[0]}) expands into {count} tokens")
    return count


def process_image(processor: Any, asset: MediaAsset, fingerprint: ProcessorFingerprint) -> ProcessedImage:
    """Run the processor over one image's bytes, with its placeholder as the text."""
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(asset.data)) as handle:
        image = handle.convert("RGB")
    output = call_processor(processor, fingerprint.image_token, {"images": [image]})
    train_inputs = extract_multimodal_train_inputs(output)
    if not train_inputs or GRID_KEY not in train_inputs:
        raise ValueError(
            f"processor produced no {GRID_KEY} for {asset.asset_id}; "
            f"keys={sorted(train_inputs or {})} — this model is not supported by the session image path"
        )
    _assert_tensor_values(train_inputs, asset.asset_id)
    grid = train_inputs[GRID_KEY]
    if tuple(grid.shape) != (1, 3):
        raise ValueError(f"expected one {GRID_KEY} row for {asset.asset_id}, got shape {tuple(grid.shape)}")
    token_count = int(torch.prod(grid, dim=-1).sum().item()) // (fingerprint.merge_size**2)
    return ProcessedImage(train_inputs=train_inputs, token_count=token_count)


def process_images(
    processor: Any, assets: Sequence[MediaAsset], fingerprint: ProcessorFingerprint, expected_counts: Sequence[int]
) -> list[ProcessedImage]:
    """Process the retained images and check each against the count serving used.

    A disagreement means the pixels no longer match the token span they are bound
    to, which is the one failure that would silently train on the wrong image.
    """
    processed = [process_image(processor, asset, fingerprint) for asset in assets]
    for index, (image, expected) in enumerate(zip(processed, expected_counts)):
        if image.token_count != expected:
            raise ValueError(
                f"image {index} ({assets[index].asset_id}) processes to {image.token_count} tokens but "
                f"serving laid out {expected}; grid={image.grid.tolist()}"
            )
    return processed


def _assert_tensor_values(train_inputs: dict[str, Any], asset_id: str) -> None:
    """The trainer calls ``.to(device)`` on every value, so none may be a list or array."""
    for key, value in train_inputs.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"processor output {key!r} for {asset_id} is {type(value).__name__}, not a torch.Tensor; "
                "the trainer moves every multimodal value to the device unconditionally"
            )


def expand_input_ids(ids: Sequence[int], counts: Sequence[int], placeholder_id: int) -> list[int]:
    """Expand the i-th placeholder in *ids* into ``counts[i]`` copies, verbatim otherwise.

    Mirrors ``sglang.srt.multimodal.processors.base_processor._expand_input_ids``;
    the two must agree or the ids we train on are not the ids inference consumed.
    """
    found = sum(1 for token_id in ids if token_id == placeholder_id)
    if found != len(counts):
        raise ValueError(
            f"prompt has {found} image placeholder token(s) but {len(counts)} image(s) were provided"
        )
    expanded: list[int] = []
    next_image = 0
    for token_id in ids:
        if token_id == placeholder_id:
            expanded.extend([placeholder_id] * counts[next_image])
            next_image += 1
        else:
            expanded.append(token_id)
    return expanded


def placeholder_runs(ids: Sequence[int], placeholder_id: int) -> list[int]:
    """Lengths of the maximal placeholder runs in *ids*, in order."""
    runs: list[int] = []
    run = 0
    for token_id in ids:
        if token_id == placeholder_id:
            run += 1
            continue
        if run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return runs


def retained_image_count(expanded_ids: Sequence[int], counts: Sequence[int], placeholder_id: int) -> int:
    """How many leading occurrences survive in an expanded, possibly truncated sequence.

    Truncation cuts at a token, so the last surviving run may be partial; a partial
    run means that image's pixels no longer match its token span, and it is dropped.
    """
    runs = placeholder_runs(expanded_ids, placeholder_id)
    if len(runs) > len(counts):
        raise ValueError(f"sequence has {len(runs)} placeholder run(s) but only {len(counts)} image(s) are known")
    retained = 0
    for index, run in enumerate(runs):
        expected = counts[index]
        if run == expected:
            retained += 1
            continue
        is_last = index == len(runs) - 1
        if is_last and run < expected:
            logger.debug("Dropping image %d: truncation left %d of %d placeholder tokens", index, run, expected)
            break
        raise ValueError(
            f"placeholder run {index} has {run} tokens but image {index} expands into {expected}; "
            "the token sequence and its media disagree"
        )
    return retained


def merge_train_inputs(processed: Sequence[ProcessedImage]) -> dict[str, torch.Tensor] | None:
    """Concatenate per-image trainer kwargs along dim 0, in occurrence order."""
    if not processed:
        return None
    keys = set(processed[0].train_inputs)
    for image in processed[1:]:
        if set(image.train_inputs) != keys:
            raise ValueError(f"processor outputs disagree on keys: {sorted(keys)} vs {sorted(image.train_inputs)}")
    return {key: torch.cat([image.train_inputs[key] for image in processed], dim=0) for key in sorted(keys)}


def build_fingerprint(processor: Any, tokenizer: Any, *, checkpoint: str) -> ProcessorFingerprint:
    """Read the resize and placeholder settings this run is pinned to."""
    image_processor = getattr(processor, "image_processor", None)
    image_token = getattr(processor, "image_token", None) or "<|image_pad|>"
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    if image_token_id is None or image_token_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"tokenizer does not know the image placeholder {image_token!r}")
    return ProcessorFingerprint(
        checkpoint=checkpoint,
        image_token=image_token,
        image_token_id=int(image_token_id),
        merge_size=int(getattr(image_processor, "merge_size", DEFAULT_MERGE_SIZE)),
        min_pixels=_optional_int(getattr(image_processor, "min_pixels", None)),
        max_pixels=_optional_int(getattr(image_processor, "max_pixels", None)),
        patch_size=_optional_int(getattr(image_processor, "patch_size", None)),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
