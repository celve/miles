"""A pairwise VLM judge of generated video: the probability that a clip fulfils the request better than an anchor clip
made for the same request, from a Qwen-VL model shown both clips, read from its next-token distribution over the two
answers and averaged over both orders, which cancels the judge's preference for a position.

The anchor is a one-shot baseline's clip (the keyframe agent's B0-ref), found in ``anchor_dir`` under
``anchor_name(request)``; a request without one scores None. Pref-GRPO (Wang et al., 2025) pairs a group's clips with
each other instead; a fixed anchor keeps the score pointwise, so a group's clips are judged against the same clip.
"""

import hashlib
import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from miles.rollout.rm_hub.video.clip import Clip, read_clip, score_clips
from miles.rollout.rm_hub.video.vlm_judge import sample_frames

QUESTION = """Two videos were generated for this request:

"{request}"

Which video better fulfils the request? Weigh, in this order: whether it shows what the request asks for (the right subjects, their number and attributes, their actions and spatial relations, the setting and style), whether it moves as the request implies with plausible physics, and its visual quality. A video that barely moves is worse unless the request asks for a still scene. Ignore any text, captions or watermarks drawn in the videos: written words never count as showing what they say.
Answer with a single digit: 1 for Video 1, 2 for Video 2."""


def anchor_name(request: str) -> str:
    return f"{hashlib.sha256(request.encode()).hexdigest()[:16]}.mp4"


class VideoPairwiseScorer:
    def __init__(
        self,
        *,
        device: torch.device,
        model_path: str,
        anchor_dir: str,
        num_frames: int = 16,
        max_pixels_per_frame: int = 448 * 448,
    ) -> None:
        self.device = device
        self.anchor_dir = anchor_dir
        self.num_frames = num_frames
        self.max_pixels_per_frame = max_pixels_per_frame
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16).eval().to(device)
        answer_ids = [self.processor.tokenizer.encode(digit, add_special_tokens=False) for digit in ("1", "2")]
        assert all(len(ids) == 1 for ids in answer_ids), f"{model_path} does not encode each answer as one token"
        self.answer_ids = [ids[0] for ids in answer_ids]

    def to(self, device: torch.device) -> None:
        self.model.to(device)
        self.device = device

    def score_batch(self, outputs: list[str], prompts: list[str]) -> list[float | None]:
        return score_clips(outputs, prompts, self._score)

    def _score(self, clip: Clip, request: str) -> float:
        path = os.path.join(self.anchor_dir, anchor_name(request))
        if (anchor := read_clip(path)) is None:
            raise FileNotFoundError(f"no readable anchor clip {path} for {request!r}")
        return (self.first_better(clip, anchor, request) + 1 - self.first_better(anchor, clip, request)) / 2

    @torch.no_grad()
    def first_better(self, first: Clip, second: Clip, request: str) -> float:
        """P(Video 1 is the better one), renormalized over the two answers."""
        sampled = [sample_frames(clip, self.num_frames, self.max_pixels_per_frame) for clip in (first, second)]
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Video 1:"},
                    {"type": "video"},
                    {"type": "text", "text": "Video 2:"},
                    {"type": "video"},
                    {"type": "text", "text": QUESTION.format(request=request)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            videos=[frames for frames, _ in sampled],
            video_metadata=[metadata for _, metadata in sampled],
            do_sample_frames=False,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**inputs).logits[0, -1, self.answer_ids].float()
        return logits.softmax(dim=-1)[0].item()
