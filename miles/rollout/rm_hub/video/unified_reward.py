"""UnifiedReward-2.0 (CodeGoat24/UnifiedReward-2.0-qwen3vl-8b) in its pointwise video-generation mode: 16 frames as
images and the caption, answered greedily with alignment, physics and style scores from 1 to 5.

Follows the repo's `inference_qwen/UnifiedReward-2.0-inference/point_score_APS_video_generation.py`: its frames
are JPEG files written by OpenCV and re-encoded by its client, both at quality 95, so each frame is encoded twice
here too. The scorer returns one axis (alignment by default) as the model writes it; a reply without it scores None.
"""

import io
import re

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from miles.rollout.rm_hub.video.clip import Clip, score_clips

AXES = ("Alignment", "Physics", "Style")
_PROMPT = (
    "You are presented with a generated video and its associated text caption. "
    "Your task is to analyze the video across multiple dimensions in relation to the caption. Specifically:\n"
    "Provide overall assessments for the video along the following axes (each rated from 1 to 5):\n"
    "- Alignment Score: How well the video matches the caption in terms of content.\n"
    "- Physics Score: How well the gravity, movements, collisions, and interactions make physical sense.\n"
    "- Style Score: How visually appealing the video looks, regardless of caption accuracy.\n\n"
    "Output your evaluation using the format below:\n\n"
    "Alignment Score (1-5): X\n"
    "Physics Score (1-5): Y\n"
    "Style Score (1-5): Z\n\n"
    "Your task is provided as follows:\n"
    "Text Caption: [{request}]"
)


class UnifiedRewardScorer:
    def __init__(
        self, *, device: torch.device, model_path: str, axis: str = "Alignment", num_frames: int = 16
    ) -> None:
        assert axis in AXES, f"UnifiedReward scores {AXES}, got {axis!r}"
        self.device = device
        self.axis = axis
        self.num_frames = num_frames
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16).eval().to(device)

    def to(self, device: torch.device) -> None:
        self.model.to(device)
        self.device = device

    def score_batch(self, outputs: list[str], prompts: list[str]) -> list[float | None]:
        return score_clips(outputs, prompts, self._score)

    def _score(self, clip: Clip, request: str) -> float | None:
        return parse_score(self.reply(clip, request), self.axis)

    @torch.no_grad()
    def reply(self, clip: Clip, request: str) -> str:
        """The model's greedy answer for the clip."""
        total = len(clip.frames)
        indices = [i * total // self.num_frames for i in range(min(self.num_frames, total))]
        images = [_jpeg(_jpeg(Image.fromarray(clip.frames[i].permute(1, 2, 0).numpy()))) for i in indices]
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": _PROMPT.format(request=request)}]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
        generated = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        return self.processor.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def parse_score(reply: str, axis: str) -> float | None:
    """The score the reply gives `axis`, as its prompt's output format writes it."""
    match = re.search(rf"{axis} Score \(1-5\):\s*([0-9]+(?:\.[0-9]+)?)", reply)
    return float(match.group(1)) if match else None


def _jpeg(image: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
