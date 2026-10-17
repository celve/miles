"""Turn immutable image bytes into token counts and trainer tensors.

The session server renders prompts with the plain tokenizer, so an image is one
placeholder token in the rendered ids. SGLang expands the i-th placeholder into
``counts[i]`` copies and keeps every other token verbatim
(``base_processor._expand_input_ids``), so the sequence inference consumed is a
pure positional function of the rendered ids and the per-image counts. This
module is that function, the processor call the counts and pixels come from, and
``SessionImageProcessor``, which holds the processor and the engines' launch settings.

Everything here works on one image at a time and concatenates: Qwen-VL resizes
and patchifies each image independently, so per-image outputs stacked in
occurrence order equal a single multi-image processor call.
"""

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from miles.rollout.session.config import SESSION_IMAGE_SUPPORT_OFF, SessionServerConfig
from miles.rollout.session.errors import UpstreamResponseError
from miles.rollout.session.images.launch_settings import check_engine_launch_settings
from miles.rollout.session.images.store import NO_IMAGES, ImageBlob, ImageOccurrences, ImageStore, intern_images
from miles.utils.processing_utils import call_processor, extract_multimodal_train_inputs, load_processor

logger = logging.getLogger(__name__)

DEFAULT_MERGE_SIZE = 2
GRID_KEY = "image_grid_thw"


@dataclass(frozen=True)
class ImageProcessorSettings:
    """The processor's placeholder token and resize settings, fixed for the process and recorded with every sample."""

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
    """One image's full processor output: the trainer's kwargs for that image."""

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

    def __init__(self, processor: Any, processor_settings: ImageProcessorSettings) -> None:
        self.processor = processor
        self.processor_settings = processor_settings
        self._counts: dict[str, int] = {}

    def count(self, blob: ImageBlob) -> int:
        cached = self._counts.get(blob.image_id)
        if cached is not None:
            return cached
        count = _count_from_size(self.processor, blob)
        self._counts[blob.image_id] = count
        return count

    def count_all(self, blobs: Sequence[ImageBlob]) -> list[int]:
        return [self.count(blob) for blob in blobs]


def _count_from_size(processor: Any, blob: ImageBlob) -> int:
    """How many placeholder tokens this image expands into, read from its header."""
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(blob.data)) as handle:
        size = (handle.height, handle.width)
    counts = processor._get_num_multimodal_tokens(image_sizes=[size]).num_image_tokens
    count = int(counts[0])
    if count <= 0:
        raise ValueError(f"{blob.image_id} ({size[1]}x{size[0]}) expands into {count} tokens")
    return count


def process_image(processor: Any, blob: ImageBlob, processor_settings: ImageProcessorSettings) -> ProcessedImage:
    """Run the processor over one image's bytes, with its placeholder as the text."""
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(blob.data)) as handle:
        image = handle.convert("RGB")
    output = call_processor(processor, processor_settings.image_token, {"images": [image]})
    train_inputs = extract_multimodal_train_inputs(output)
    if not train_inputs or GRID_KEY not in train_inputs:
        raise ValueError(
            f"processor produced no {GRID_KEY} for {blob.image_id}; "
            f"keys={sorted(train_inputs or {})} — this model is not supported by the session image path"
        )
    _assert_tensor_values(train_inputs, blob.image_id)
    grid = train_inputs[GRID_KEY]
    if tuple(grid.shape) != (1, 3):
        raise ValueError(f"expected one {GRID_KEY} row for {blob.image_id}, got shape {tuple(grid.shape)}")
    token_count = int(torch.prod(grid, dim=-1).sum().item()) // (processor_settings.merge_size**2)
    return ProcessedImage(train_inputs=train_inputs, token_count=token_count)


def process_images(
    processor: Any,
    blobs: Sequence[ImageBlob],
    processor_settings: ImageProcessorSettings,
    expected_counts: Sequence[int],
) -> list[ProcessedImage]:
    """Process the retained images and check each against the count serving used.

    A disagreement means the pixels no longer match the token span they are bound
    to, which is the one failure that would silently train on the wrong image.
    """
    processed = [process_image(processor, blob, processor_settings) for blob in blobs]
    for index, (image, expected) in enumerate(zip(processed, expected_counts, strict=True)):
        if image.token_count != expected:
            raise ValueError(
                f"image {index} ({blobs[index].image_id}) processes to {image.token_count} tokens but "
                f"serving laid out {expected}; grid={image.grid.tolist()}"
            )
    return processed


def _assert_tensor_values(train_inputs: dict[str, Any], image_id: str) -> None:
    """The trainer calls ``.to(device)`` on every value, so none may be a list or array."""
    for key, value in train_inputs.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"processor output {key!r} for {image_id} is {type(value).__name__}, not a torch.Tensor; "
                "the trainer moves every multimodal value to the device unconditionally"
            )


def expand_input_ids(ids: Sequence[int], counts: Sequence[int], placeholder_id: int) -> list[int]:
    """Expand the i-th placeholder in *ids* into ``counts[i]`` copies, verbatim otherwise.

    Mirrors ``sglang.srt.multimodal.processors.base_processor._expand_input_ids``;
    the two must agree or the ids we train on are not the ids inference consumed.
    """
    found = sum(1 for token_id in ids if token_id == placeholder_id)
    if found != len(counts):
        raise ValueError(f"prompt has {found} image placeholder token(s) but {len(counts)} image(s) were provided")
    expanded: list[int] = []
    next_image = 0
    for token_id in ids:
        if token_id == placeholder_id:
            expanded.extend([placeholder_id] * counts[next_image])
            next_image += 1
        else:
            expanded.append(token_id)
    return expanded


def assert_engine_expansion_matches(
    effective: Sequence[int] | None, expected_ids: list[int], images: ImageOccurrences, counts: list[int]
) -> None:
    """Proves the token layout and grid inference consumed, not its pixels, which the startup check covers."""
    if effective is None:
        raise UpstreamResponseError(
            "chat response carries no prompt_token_ids; the session image path needs "
            "return_prompt_token_ids to prove the ids inference consumed"
        )
    if list(effective) == expected_ids:
        return
    first_diff = next(
        (i for i, (a, b) in enumerate(zip(effective, expected_ids, strict=False)) if a != b),
        min(len(effective), len(expected_ids)),
    )
    raise UpstreamResponseError(
        f"effective prompt ids differ from the rendered expansion: engine has {len(effective)} tokens, "
        f"the session server expected {len(expected_ids)} (first difference at index {first_diff}); "
        f"images={list(images.image_ids)} counts={counts} — the engine's processor settings or image "
        "decoding disagree with this run's pinned processor, or its chat route dropped the images "
        "(an SGLang whose pretokenized chat branch does not keep message images)"
    )


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
            "the token sequence and its images disagree"
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


def read_processor_settings(processor: Any, tokenizer: Any, *, checkpoint: str) -> ImageProcessorSettings:
    """Read the resize and placeholder settings this run is pinned to."""
    image_processor = getattr(processor, "image_processor", None)
    image_token = getattr(processor, "image_token", None) or "<|image_pad|>"
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    if image_token_id is None or image_token_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"tokenizer does not know the image placeholder {image_token!r}")
    return ImageProcessorSettings(
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


@dataclass(frozen=True)
class EngineLaunchSettings:
    """What the policy engines launch with that decides an image's pixels."""

    model_path: str
    revision: str | None
    tokenizer_path: str
    mm_process_config: dict[str, Any] | None
    image_processor_backend: str
    disable_fast_image_processor: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SessionImageProcessor:
    """The checkpoint's image processor and the settings every image turn in this process shares."""

    hf_processor: Any
    processor_settings: ImageProcessorSettings
    engine_settings: EngineLaunchSettings
    text_only_leaf: str  # "drop" | "keep"

    @property
    def placeholder_id(self) -> int:
        return self.processor_settings.image_token_id

    def new_session_images(self) -> "SessionImages":
        return SessionImages(token_counter=ImageTokenCounter(self.hf_processor, self.processor_settings))

    def provenance(self) -> dict[str, Any]:
        """What bound this run's pixels, recorded with the session and every sample."""
        return {"processor": self.processor_settings.to_dict(), "engine": self.engine_settings.to_dict()}

    def compute_pixels(self, image_ids: list[str], counts: list[int], images: "SessionImages") -> list[ProcessedImage]:
        blobs = [images.store.get(image_id) for image_id in image_ids]
        return process_images(self.hf_processor, blobs, self.processor_settings, counts)


@dataclass
class SessionImages:
    """One session's image bytes, keyed by content, and their token counts."""

    token_counter: ImageTokenCounter
    store: ImageStore = field(default_factory=ImageStore)

    def ingest(self, messages: list[dict[str, Any]]) -> tuple[ImageOccurrences, list[int]]:
        """Store a request's images and say how many tokens each expands into."""
        images = intern_images(messages, self.store)
        if not images.image_ids:
            return NO_IMAGES, []
        return images, self.counts_for(images)

    def counts_for(self, images: ImageOccurrences) -> list[int]:
        return self.token_counter.count_all([self.store.get(image_id) for image_id in images.image_ids])


def build_session_image_processor(config: SessionServerConfig, tokenizer: Any) -> SessionImageProcessor | None:
    """Load the image processor for an image-enabled run, or ``None`` for text."""
    if config.session_image_support == SESSION_IMAGE_SUPPORT_OFF:
        return None
    if config.session_image_engine_settings is None:
        raise ValueError(
            f"--session-image-support={config.session_image_support} needs the policy engines' image "
            "settings, which validate_session_image_args reads from the launch args"
        )
    # Checked again here because a config can also be built by hand, as scripts and tests do.
    check_engine_launch_settings(config.session_image_engine_settings, checkpoint=config.hf_checkpoint)
    hf_processor = load_processor(config.hf_checkpoint, trust_remote_code=True)
    if hf_processor is None:
        raise ValueError(
            f"--session-image-support={config.session_image_support} needs an image processor, but "
            f"{config.hf_checkpoint!r} provides none"
        )
    if not hasattr(hf_processor, "_get_num_multimodal_tokens"):
        raise ValueError(
            f"{config.hf_checkpoint!r}'s processor has no _get_num_multimodal_tokens, so per-image token "
            "counts cannot be derived without processing every image on every turn"
        )
    processor_settings = read_processor_settings(hf_processor, tokenizer, checkpoint=config.hf_checkpoint)
    engine_settings = EngineLaunchSettings(**config.session_image_engine_settings)
    logger.info(
        "[session] Image support=%s processor=%s engine=%s text_only_leaf=%s",
        config.session_image_support,
        processor_settings.to_dict(),
        engine_settings.to_dict(),
        config.session_image_text_only_leaf,
    )
    return SessionImageProcessor(
        hf_processor=hf_processor,
        processor_settings=processor_settings,
        engine_settings=engine_settings,
        text_only_leaf=config.session_image_text_only_leaf,
    )
