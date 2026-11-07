"""PickScore of a clip read from its file: the mean over evenly spaced frames, as `--rm-type pickscore` scores a
diffusion run's generated tensor."""

import torch
from PIL import Image

from miles.diffusion.rollout.pickscore import PickScoreScorer, sample_frame_indices
from miles.rollout.rm_hub.video.clip import Clip, score_clips


class VideoPickScoreScorer:
    def __init__(
        self,
        *,
        device: torch.device,
        processor_path: str,
        model_path: str,
        num_frames: int | None = 8,
        frames_per_forward: int = 32,
    ) -> None:
        self.num_frames = num_frames
        self.frames_per_forward = frames_per_forward
        self.scorer = PickScoreScorer(device=device, processor_path=processor_path, model_path=model_path)

    def to(self, device: torch.device) -> None:
        self.scorer.model.to(device)
        self.scorer.device = device

    def score_batch(self, outputs: list[str], prompts: list[str]) -> list[float | None]:
        return score_clips(outputs, prompts, self._score)

    def _score(self, clip: Clip, request: str) -> float:
        images = [
            Image.fromarray(clip.frames[index].permute(1, 2, 0).numpy())
            for index in sample_frame_indices(len(clip.frames), self.num_frames)
        ]
        scores = []
        for start in range(0, len(images), self.frames_per_forward):
            batch = images[start : start + self.frames_per_forward]
            scores.extend(self.scorer([request] * len(batch), batch))
        return sum(scores) / len(scores)
