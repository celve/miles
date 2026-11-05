"""PickScore reward of diffusion rollouts: CLIP-H PickScore on the workers of the `pickscore` reward pool, a video's
frames averaged.

Ported from radixark/miles_diffusion (miles/rollout/rm_hub/pickscore.py and core.py). Batching is part of the
numerics: each worker call takes --diffusion-pickscore-batch-size samples, and each CLIP forward that many frames.
"""

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image

from miles.utils.types import Sample

# flow_grpo's scale of the raw PickScore logits (about 0-26) to about 0-1.
_LOGIT_SCALE_DIVISOR = 26.0


def sample_frame_indices(num_total_frames: int, num_frames: int | None) -> list[int]:
    """`num_frames` evenly spaced frames, the middle one for 1, every frame for None or a short video."""
    if num_frames is None or num_total_frames <= num_frames:
        return list(range(num_total_frames))
    if num_frames == 1:
        return [num_total_frames // 2]
    step = (num_total_frames - 1) / (num_frames - 1)
    return [int(round(i * step)) for i in range(num_frames)]


def video_to_uint8_frames(cfhw: torch.Tensor) -> list[np.ndarray]:
    """Each frame of a ``[C, F, H, W]`` output in [0, 1] as HWC uint8, truncated as the fork's PickScore reads it."""
    frames = cfhw.detach().cpu().float()
    if float(frames.max()) <= 1.0 + 1e-3:
        frames = frames * 255.0
    fhwc = frames.clamp(0, 255).to(torch.uint8).permute(1, 2, 3, 0).contiguous()
    return [np.ascontiguousarray(frame.numpy()) for frame in fhwc]


class PickScoreScorer(torch.nn.Module):
    def __init__(self, *, device: torch.device, processor_path: str, model_path: str) -> None:
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def forward(self, prompts: Sequence[str], images: Sequence[Image.Image]) -> list[float]:
        image_inputs = self.processor(images=list(images), return_tensors="pt", padding=True)
        image_inputs = {key: value.to(device=self.device) for key, value in image_inputs.items()}
        image_inputs["pixel_values"] = image_inputs["pixel_values"].float()
        text_inputs = self.processor(
            text=list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}

        image_embs = _embedding(self.model.get_image_features(**image_inputs))
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        text_embs = _embedding(self.model.get_text_features(**text_inputs))
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        scores = self.model.logit_scale.exp() * (text_embs * image_embs).sum(dim=-1)
        return [float(score) for score in (scores.float() / _LOGIT_SCALE_DIVISOR).cpu()]


def _embedding(features) -> torch.Tensor:
    # transformers 5 returns a BaseModelOutputWithPooling where 4.x returned the tensor.
    return features if isinstance(features, torch.Tensor) else features.pooler_output


class PickScoreBatchScorer:
    """A reward pool worker's scorer: each sample's mean over the frames it arrives with."""

    def __init__(self, *, device: torch.device, processor_path: str, model_path: str, frames_per_forward: int) -> None:
        self.frames_per_forward = frames_per_forward
        self.scorer = PickScoreScorer(device=device, processor_path=processor_path, model_path=model_path)

    def score_batch(self, outputs: list[torch.Tensor], prompts: list[str]) -> list[float]:
        images, frame_counts = [], []
        for output in outputs:
            frames = video_to_uint8_frames(output)
            images.extend(Image.fromarray(frame) for frame in frames)
            frame_counts.append(len(frames))
        flat_prompts = [prompt for prompt, count in zip(prompts, frame_counts, strict=True) for _ in range(count)]
        flat_scores = []
        for start in range(0, len(images), self.frames_per_forward):
            end = start + self.frames_per_forward
            flat_scores.extend(self.scorer(flat_prompts[start:end], images[start:end]))
        scores, offset = [], 0
        for count in frame_counts:
            scores.append(float(sum(flat_scores[offset : offset + count]) / count))
            offset += count
        return scores


async def pickscore_rm(args, samples: Sequence[Sample], *, pickscore_pool, **_other_pools) -> list[float]:
    """`--rm-type pickscore`: one reward per sample, its --diffusion-pickscore-num-frames frames averaged."""
    outputs = []
    for sample in samples:
        assert sample.diffusion is not None, "--rm-type pickscore scores diffusion samples only"
        output = sample.diffusion.generated_output
        outputs.append(output[:, sample_frame_indices(output.shape[1], args.diffusion_pickscore_num_frames)])
    return await pickscore_pool.score(outputs, [sample.prompt for sample in samples])
