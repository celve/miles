"""Clips read from the file a video generator wrote."""

import logging
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
