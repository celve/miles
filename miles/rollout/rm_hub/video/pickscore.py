"""PickScore of a clip read from its file: the mean over evenly spaced frames, as `--rm-type pickscore` scores a
diffusion run's generated tensor."""

import torch
from PIL import Image

from miles.diffusion.rollout.pickscore import PickScoreScorer, sample_frame_indices
from miles.rollout.rm_hub.video.clip import read_clip


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
        images, owners = [], []
        for index, path in enumerate(outputs):
            if (clip := read_clip(path)) is None:
                continue
            for frame_index in sample_frame_indices(len(clip.frames), self.num_frames):
                images.append(Image.fromarray(clip.frames[frame_index].permute(1, 2, 0).numpy()))
                owners.append(index)
        frame_scores = []
        for start in range(0, len(images), self.frames_per_forward):
            end = start + self.frames_per_forward
            frame_scores.extend(self.scorer([prompts[owner] for owner in owners[start:end]], images[start:end]))
        sums, counts = [0.0] * len(outputs), [0] * len(outputs)
        for owner, score in zip(owners, frame_scores, strict=True):
            sums[owner] += score
            counts[owner] += 1
        return [total / count if count else None for total, count in zip(sums, counts, strict=True)]
