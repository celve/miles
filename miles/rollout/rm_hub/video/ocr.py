"""Unrequested text in a clip: PaddleOCR (`miles.diffusion.rollout.ocr`) reads evenly spaced frames, and the score
is how many distinct lines it reads that the request does not ask for, so a planner gains nothing by drawing the
request's words into the video.

A request asks for text only by quoting it; a line counts when its letters and digits are not in the request's
quoted text, it has at least `min_chars` of them, and PaddleOCR is at least `min_confidence` sure of it.
"""

import re

import numpy as np
import torch

from miles.diffusion.rollout.ocr import OcrScorer
from miles.diffusion.rollout.pickscore import sample_frame_indices
from miles.rollout.rm_hub.video.clip import Clip, score_clips

_QUOTED = re.compile(r"\"([^\"]*)\"|“([^”]*)”")


class VideoOcrScorer:
    def __init__(
        self, *, device: torch.device, num_frames: int = 4, min_confidence: float = 0.8, min_chars: int = 2
    ) -> None:
        self.num_frames = num_frames
        self.min_confidence = min_confidence
        self.min_chars = min_chars
        self._ocr = OcrScorer()._ocr

    def score_batch(self, outputs: list[str], prompts: list[str]) -> list[float | None]:
        return score_clips(outputs, prompts, self._score)

    def _score(self, clip: Clip, request: str) -> float:
        lines = [(text, confidence) for _, text, confidence in self.read(clip)]
        return float(
            len(unrequested_lines(lines, request, min_confidence=self.min_confidence, min_chars=self.min_chars))
        )

    def read(self, clip: Clip) -> list[tuple[int, str, float]]:
        """Every line PaddleOCR reads on the sampled frames, as (frame index, text, confidence)."""
        lines = []
        for index in sample_frame_indices(len(clip.frames), self.num_frames):
            image = np.ascontiguousarray(clip.frames[index].permute(1, 2, 0).numpy())
            result = self._ocr.ocr(image, cls=False)
            lines.extend((index, text, float(confidence)) for _, (text, confidence) in result[0] or [])
        return lines


def unrequested_lines(
    lines: list[tuple[str, float]], request: str, *, min_confidence: float, min_chars: int
) -> set[str]:
    """The distinct normalized lines of `lines` that count as text `request` does not ask for."""
    requested = _normalized(" ".join(a or b for a, b in _QUOTED.findall(request)))
    found = set()
    for text, confidence in lines:
        line = _normalized(text)
        if confidence >= min_confidence and len(line) >= min_chars and line not in requested:
            found.add(line)
    return found


def _normalized(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum())
