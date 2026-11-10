"""VQAScore (Lin et al., 2024) of a clip: a Qwen-VL model's probability of answering yes, in any spelling, to whether
the video shows the request, read from its next-token distribution, with t2v_metrics' video question.

One yes/no question on the request itself, so restating the request cannot add items for the planner to satisfy; the
frames are sampled and scaled as the VLM judge samples them.
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from miles.rollout.rm_hub.video.clip import Clip, score_clips
from miles.rollout.rm_hub.video.vlm_judge import sample_frames

QUESTION = 'Does this video show "{request}"? Please answer yes or no.'
_YES = ("yes", "Yes", "YES", " yes", " Yes")


class VideoVqaScorer:
    def __init__(
        self, *, device: torch.device, model_path: str, num_frames: int = 16, max_pixels_per_frame: int = 448 * 448
    ) -> None:
        self.device = device
        self.num_frames = num_frames
        self.max_pixels_per_frame = max_pixels_per_frame
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16).eval().to(device)
        # Qwen3-VL answers "yes" in lower case; each spelling is one token.
        self.yes_ids = [self.processor.tokenizer.encode(yes, add_special_tokens=False)[0] for yes in _YES]

    def to(self, device: torch.device) -> None:
        self.model.to(device)
        self.device = device

    def score_batch(self, outputs: list[str], prompts: list[str]) -> list[float | None]:
        return score_clips(outputs, prompts, self._score)

    @torch.no_grad()
    def _score(self, clip: Clip, request: str) -> float:
        frames, metadata = sample_frames(clip, self.num_frames, self.max_pixels_per_frame)
        message = [
            {
                "role": "user",
                "content": [{"type": "video"}, {"type": "text", "text": QUESTION.format(request=request)}],
            }
        ]
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            videos=[frames],
            video_metadata=[metadata],
            do_sample_frames=False,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**inputs).logits[0, -1].float()
        return logits.softmax(dim=-1)[self.yes_ids].sum().item()
