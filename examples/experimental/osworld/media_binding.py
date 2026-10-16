"""Independent checks that a training sample's image tokens match its screenshots.

A converter that rebuilds vision inputs from an agent's trace has to prove the
pixels it hands the trainer are the pixels the policy saw. The engine returns
prompt ids already expanded -- one run of the image placeholder per image, as
many tokens long as that image costs -- so the proof is positional: run *i* must
be exactly as long as the processor says image *i* costs, in order.

Checking the *total* count instead, against the summed per-image counts, accepts
two different corruptions: a wrong per-image assignment whose total happens to
agree, and a wrong image order. Both put a screenshot's pixels under another
screenshot's positions, which trains the policy on a context that never existed.

Nothing here imports NeMo Gym, Harbor or Miles, so either connector's converter
can use it, and the arithmetic is testable without a GPU, a model or a trace.
"""

from __future__ import annotations

import hashlib
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
) -> int:
    """Check each image's token run positionally and return how many ids account for.

    ``expected_counts`` may be longer than the ids: an agent's trace can carry a
    screenshot captured after the last model call, which never entered a prompt.
    It may not be shorter, and no run may disagree with its image.
    """
    runs = image_token_runs(tokens, image_token_id)
    if len(runs) > len(expected_counts):
        raise MediaBindingError(
            f"ids hold {len(runs)} image runs but only {len(expected_counts)} screenshots were recovered"
        )
    for index, ((_, length), expected) in enumerate(zip(runs, expected_counts, strict=False)):
        if length != expected:
            raise MediaBindingError(
                f"image {index}: ids give it {length} tokens, the processor gives it {expected}; "
                f"runs={[length for _, length in runs]} expected={expected_counts}"
            )
    return len(runs)


def media_digests(payloads: list[bytes]) -> list[str]:
    """``sha256:<hex>`` per screenshot, in prompt order.

    Bound to the sample so a later run with different pixels under the same
    token layout is detectable rather than silent.
    """
    return [f"sha256:{hashlib.sha256(payload).hexdigest()}" for payload in payloads]


def processor_identity(image_processor: Any) -> dict[str, Any]:
    """The preprocessing settings that decide an image's token count and pixels.

    Recorded per sample because a processor change silently rewrites the mapping
    between a screenshot and its tokens, and a checkpoint trained across such a
    change cannot be reproduced from the trace alone.
    """
    # Both resize spellings are read: Qwen3-VL-4B-Instruct pins the budget as
    # `size: {longest_edge, shortest_edge}`, older checkpoints as min/max_pixels.
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
            identity[field] = list(value) if isinstance(value, (list, tuple)) else value
    return identity


def expanded_token_counts(processor: Any, images: list[Any]) -> list[int]:
    """Tokens each image costs after merging, without re-running the vision tower.

    Takes the *processor*, because transformers puts the sizing call there and
    the image processor only underneath it. Preferring that call keeps the count
    on the same code path the engine expanded the prompt with; the grid is the
    fallback for a processor too old to expose it.
    """
    sizer = getattr(processor, "_get_num_multimodal_tokens", None)
    if callable(sizer):
        counts = sizer(image_sizes=[(image.height, image.width) for image in images])["num_image_tokens"]
        return [int(count) for count in counts]

    image_processor = getattr(processor, "image_processor", processor)
    merge = int(getattr(image_processor, "merge_size", 2))
    grid = image_processor(images=images, return_tensors="pt")["image_grid_thw"]
    return [int(row.prod().item()) // (merge * merge) for row in grid]
