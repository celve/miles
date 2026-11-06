"""A VLM judge of generated video: a Qwen-VL model in transformers answers one rubric question per axis with a digit
from 1 to 5, and the axis scores the expected digit under the model's next-token distribution, mapped to [0, 1].

The expected digit, not the sampled one, keeps the reward continuous and needs no parsing that could fail.
"""

import math

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.video_utils import VideoMetadata

from miles.diffusion.rollout.pickscore import sample_frame_indices
from miles.rollout.rm_hub.video.clip import Clip, read_clip

_DIGITS = ("1", "2", "3", "4", "5")
_QUESTION = """You are judging a video that a model generated for this request:

"{request}"

{criterion}

Judge only what the frames show. Ignore any text, captions or watermarks drawn in the video: written words never count as showing what they say.
Answer with a single digit from 1 (very poor) to 5 (excellent)."""
RUBRIC = {
    "prompt_adherence": (
        "Rate prompt adherence: does the video show what the request asks for, with the right subjects, their "
        "number and attributes, their actions, the setting, the style and the camera movement?"
    ),
    "visual_quality": (
        "Rate visual quality: is every frame sharp, detailed and free of artifacts, with no deformed bodies, faces, "
        "hands or objects?"
    ),
    "motion": (
        "Rate motion: does the scene move as the request implies, smoothly and with plausible physics? A video that "
        "barely moves scores 1 unless the request asks for a still scene."
    ),
}


class VlmJudgeScorer:
    def __init__(
        self,
        *,
        device: torch.device,
        model_path: str,
        num_frames: int = 16,
        max_pixels_per_frame: int = 448 * 448,
        axis_weights: dict[str, float] | None = None,
    ) -> None:
        self.device = device
        self.num_frames = num_frames
        self.max_pixels_per_frame = max_pixels_per_frame
        self.axis_weights = axis_weights or {axis: 1.0 for axis in RUBRIC}
        assert set(self.axis_weights) <= set(RUBRIC), f"unknown rubric axes {set(self.axis_weights) - set(RUBRIC)}"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16).eval().to(device)
        digit_ids = [self.processor.tokenizer.encode(digit, add_special_tokens=False) for digit in _DIGITS]
        assert all(len(ids) == 1 for ids in digit_ids), f"{model_path} does not encode each digit as one token"
        self.digit_ids = [ids[0] for ids in digit_ids]

    def to(self, device: torch.device) -> None:
        self.model.to(device)
        self.device = device

    def score_batch(self, outputs: list[str], prompts: list[str]) -> list[float | None]:
        return [self._score(path, request) for path, request in zip(outputs, prompts, strict=True)]

    @torch.no_grad()
    def _score(self, path: str, request: str) -> float | None:
        if (clip := read_clip(path)) is None:
            return None
        frames, metadata = self._sample(clip)
        axes = list(self.axis_weights)
        texts = [
            self.processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video"},
                            {"type": "text", "text": _QUESTION.format(request=request, criterion=RUBRIC[axis])},
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for axis in axes
        ]
        inputs = self.processor(
            text=texts,
            videos=[frames] * len(axes),
            video_metadata=[metadata] * len(axes),
            do_sample_frames=False,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**inputs).logits[:, -1, self.digit_ids].float()
        expected = (logits.softmax(dim=-1) * torch.arange(1, len(_DIGITS) + 1, device=logits.device)).sum(dim=-1)
        axis_scores = ((expected - 1) / (len(_DIGITS) - 1)).tolist()
        total = sum(self.axis_weights.values())
        return sum(self.axis_weights[axis] * score for axis, score in zip(axes, axis_scores, strict=True)) / total

    def _sample(self, clip: Clip) -> tuple[torch.Tensor, VideoMetadata]:
        indices = sample_frame_indices(len(clip.frames), self.num_frames)
        frames = clip.frames[indices]
        height, width = frames.shape[-2:]
        if (scale := math.sqrt(self.max_pixels_per_frame / (height * width))) < 1:
            size = (round(height * scale), round(width * scale))
            frames = torch.nn.functional.interpolate(frames.float(), size=size, mode="bilinear", antialias=True)
            frames = frames.round().clamp(0, 255).to(torch.uint8)
        metadata = VideoMetadata(
            total_num_frames=len(clip.frames),
            fps=clip.fps,
            frames_indices=indices,
            width=frames.shape[-1],
            height=frames.shape[-2],
            duration=len(clip.frames) / clip.fps,
        )
        return frames, metadata
