import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from PIL import Image

from miles.rollout.session.config import SessionServerConfig
from miles.rollout.session.errors import UpstreamResponseError
from miles.rollout.session.images.store import NO_IMAGES, ImageBlob, ImageOccurrences, ImageStore, intern_images
from miles.utils.media_refs import materialize_media_ref
from miles.utils.processing_utils import load_processor

logger = logging.getLogger(__name__)


class ImageTokenCounter:
    def __init__(self, processor: Any) -> None:
        self.processor = processor
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
    with Image.open(BytesIO(blob.data)) as handle:
        size = (handle.height, handle.width)
    count = int(processor._get_num_multimodal_tokens(image_sizes=[size]).num_image_tokens[0])
    if count <= 0:
        raise ValueError(f"{blob.image_id} ({size[1]}x{size[0]}) expands into {count} tokens")
    return count


def expand_input_ids(ids: Sequence[int], counts: Sequence[int], placeholder_id: int) -> list[int]:
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
        f"images={list(images.image_ids)} counts={counts} — the engine's image processor disagrees with "
        "--hf-checkpoint's, or the images were dropped on the way (an engine without "
        "--sglang-enable-token-space-processor, or a router that does not forward mm_token_expansion_start_len "
        "on the chat route)"
    )


def placeholder_runs(ids: Sequence[int], placeholder_id: int) -> list[int]:
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


@dataclass(frozen=True)
class SessionImageProcessor:
    hf_processor: Any
    placeholder_id: int
    text_only_samples: str
    media_dir: str

    def new_session_images(self) -> "SessionImages":
        return SessionImages(token_counter=ImageTokenCounter(self.hf_processor))

    def write_media_refs(self, image_ids: list[str], images: "SessionImages") -> list[str]:
        return [materialize_media_ref(images.store.get(image_id).data, self.media_dir) for image_id in image_ids]


@dataclass
class SessionImages:
    token_counter: ImageTokenCounter
    store: ImageStore = field(default_factory=ImageStore)

    def ingest(self, messages: list[dict[str, Any]]) -> tuple[ImageOccurrences, list[int]]:
        images = intern_images(messages, self.store)
        if not images.image_ids:
            return NO_IMAGES, []
        return images, self.counts_for(images)

    def counts_for(self, images: ImageOccurrences) -> list[int]:
        return self.token_counter.count_all([self.store.get(image_id) for image_id in images.image_ids])


def build_session_image_processor(config: SessionServerConfig, tokenizer: Any) -> SessionImageProcessor | None:
    if not config.use_session_images:
        return None
    if config.run_media_dir is None:
        raise ValueError(
            "--use-session-images hands the trainer each retained image as a file, so it needs --shared-media-dir "
            "on a filesystem the trainers mount"
        )
    hf_processor = load_processor(config.hf_checkpoint, trust_remote_code=True)
    if hf_processor is None:
        raise ValueError(f"--use-session-images needs an image processor, but {config.hf_checkpoint!r} provides none")
    placeholder_id = _placeholder_id(hf_processor, tokenizer)
    logger.info(
        "[session] Image support on: placeholder_id=%d text_only_samples=%s media_dir=%s",
        placeholder_id,
        config.session_image_text_only_samples,
        config.run_media_dir,
    )
    return SessionImageProcessor(
        hf_processor=hf_processor,
        placeholder_id=placeholder_id,
        text_only_samples=config.session_image_text_only_samples,
        media_dir=config.run_media_dir,
    )


def _placeholder_id(processor: Any, tokenizer: Any) -> int:
    image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    if image_token_id is None or image_token_id == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer does not know the image placeholder {processor.image_token!r}")
    return int(image_token_id)
