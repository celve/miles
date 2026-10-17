"""Independent checks that a training sample's image tokens match its screenshots."""

from __future__ import annotations

import hashlib
from dataclasses import fields as dc_fields, is_dataclass
from typing import Any


class MediaBindingError(ValueError):
    """The ids and the screenshots disagree; the sample is not safe to train."""


def image_token_runs(tokens: list[int], image_token_id: int) -> list[tuple[int, int]]:
    """``(start, length)`` of each maximal run of ``image_token_id``, in order."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, token in enumerate(tokens):
        if token == image_token_id:
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index - start))
            start = None
    if start is not None:
        runs.append((start, len(tokens) - start))
    return runs


def verify_image_token_layout(
    tokens: list[int],
    image_token_id: int,
    expected_counts: list[int],
) -> None:
    """Check each image's token run positionally, one run per image, in order."""
    runs = image_token_runs(tokens, image_token_id)
    if len(runs) != len(expected_counts):
        raise MediaBindingError(
            f"ids hold {len(runs)} image runs but {len(expected_counts)} screenshots were recovered; "
            f"runs={[length for _, length in runs]} expected={expected_counts}"
        )
    for index, ((_, length), expected) in enumerate(zip(runs, expected_counts, strict=True)):
        if length != expected:
            raise MediaBindingError(
                f"image {index}: ids give it {length} tokens, the processor gives it {expected}; "
                f"runs={[length for _, length in runs]} expected={expected_counts}"
            )


def media_digests(payloads: list[bytes]) -> list[str]:
    """``sha256:<hex>`` per screenshot, in prompt order."""
    return [f"sha256:{hashlib.sha256(payload).hexdigest()}" for payload in payloads]


def processor_identity(image_processor: Any) -> dict[str, Any]:
    """The preprocessing settings that decide an image's token count and pixels."""
    # Qwen3-VL-4B-Instruct sets size {longest_edge, shortest_edge}; older checkpoints set min/max_pixels.
    fields = (
        "merge_size",
        "patch_size",
        "temporal_patch_size",
        "size",
        "min_pixels",
        "max_pixels",
        "do_resize",
        "do_rescale",
        "do_normalize",
        "resample",
        "image_mean",
        "image_std",
    )
    identity: dict[str, Any] = {"class": type(image_processor).__name__}
    for field in fields:
        value = getattr(image_processor, field, None)
        if value is not None:
            identity[field] = _plain(value)
    return identity


def _plain(value: Any) -> Any:
    """Coerce to types msgpack can carry."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in dc_fields(value)}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    return str(value)


def expanded_token_counts(processor: Any, images: list[Any]) -> list[int]:
    """Tokens each image costs after merging, without re-running the vision tower."""
    sizer = getattr(processor, "_get_num_multimodal_tokens", None)
    if callable(sizer):
        counts = sizer(image_sizes=[(image.height, image.width) for image in images])["num_image_tokens"]
        return [int(count) for count in counts]

    image_processor = getattr(processor, "image_processor", processor)
    merge = int(getattr(image_processor, "merge_size", 2))
    grid = image_processor(images=images, return_tensors="pt")["image_grid_thw"]
    return [int(row.prod().item()) // (merge * merge) for row in grid]
