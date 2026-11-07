"""Clips read from the file a video generator wrote."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import av
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Clip:
    # [T, C, H, W] uint8
    frames: torch.Tensor
    fps: float


def read_clip(path: str) -> Clip | None:
    """Every frame of the clip at `path`, or None for a clip that is missing, unreadable or empty."""
    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
            fps = float(stream.average_rate or stream.guessed_rate or 0)
    except (OSError, av.FFmpegError, IndexError) as error:
        logger.warning(f"Cannot read the clip {path}: {error!r}; it scores None")
        return None
    if not frames or fps <= 0:
        logger.warning(f"The clip {path} has {len(frames)} frames at {fps} fps; it scores None")
        return None
    return Clip(frames=torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2), fps=fps)


def score_clips(
    paths: list[str], requests: list[str], score_clip: Callable[[Clip, str], float]
) -> list[float | None]:
    """`score_clip(clip, request)` for each clip, one at a time, so a clip that cannot be read or whose scoring
    raises scores None while the others keep their scores."""
    scores = []
    for path, request in zip(paths, requests, strict=True):
        if (clip := read_clip(path)) is None:
            scores.append(None)
            continue
        try:
            scores.append(float(score_clip(clip, request)))
        except Exception:
            logger.warning(f"Scoring the clip {path} failed; it scores None", exc_info=True)
            scores.append(None)
    return scores
