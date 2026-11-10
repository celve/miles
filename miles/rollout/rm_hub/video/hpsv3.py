"""HPSv3 (MizzenAI/HPSv3) of a clip: a Qwen2-VL-7B image preference model whose ranknet head reads one special
token, scored on evenly spaced frames and reduced to the mean of the best `top_fraction` of them (all by default).

Ported from the repo's `inference.py`, which pins transformers 4.45: the forward keeps theirs, including plain 1D
positions in place of Qwen2-VL's 3D rope ones, and the score is the head's mean (mu), not its sigma.
"""

import math

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

from miles.diffusion.rollout.pickscore import sample_frame_indices
from miles.rollout.rm_hub.video.clip import Clip, score_clips
from miles.rollout.rm_hub.video.videoalign import _smart_resize
from miles.utils.hf_utils.config import load_hf_config

_SPECIAL_TOKEN = "<|Reward|>"
# HPSv3_7B.yaml's min_pixels and max_pixels, both 256 * 28 * 28.
_PIXELS = 256 * 28 * 28

# HPSv3's INSTRUCTION followed by prompt_with_special_token, verbatim; \x20 keeps the trailing spaces it was trained on.
_PROMPT = """
You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best.\x20

**Visual Quality:**\x20\x20
Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:
- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible.\x20

**Text Alignment:**\x20\x20
Assess how well the image matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.
- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.
- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.
- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.
Textual prompt - {text_prompt}



Please provide the overall ratings of this image: <|Reward|>

END
"""


class VideoHpsv3Scorer:
    def __init__(
        self,
        *,
        device: torch.device,
        checkpoint_path: str,
        base_model_path: str,
        num_frames: int | None = 8,
        frames_per_forward: int = 8,
        top_fraction: float = 1.0,
    ) -> None:
        assert 0 < top_fraction <= 1, f"top_fraction is in (0, 1], got {top_fraction}"
        self.top_fraction = top_fraction
        self.device = device
        self.num_frames = num_frames
        self.frames_per_forward = frames_per_forward
        self.processor = AutoProcessor.from_pretrained(base_model_path, padding_side="right")
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": [_SPECIAL_TOKEN]})
        self.special_token_id = self.processor.tokenizer.convert_tokens_to_ids(_SPECIAL_TOKEN)

        state = _renamed_state_dict(load_file(checkpoint_path))
        model_config = load_hf_config(base_model_path)
        model_config.text_config.vocab_size = state["language_model.embed_tokens.weight"].shape[0]
        self.model = Qwen2VLModel._from_config(model_config, dtype=torch.bfloat16)
        head = {name[len("rm_head.") :]: state.pop(name) for name in list(state) if name.startswith("rm_head.")}
        self.model.load_state_dict(state, strict=True)
        # HPSv3's ranknet head without rm_head_kwargs; its dropout is inert in eval.
        self.rm_head = torch.nn.Sequential(
            torch.nn.Linear(head["0.weight"].shape[1], head["0.weight"].shape[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(head["3.weight"].shape[1], head["3.weight"].shape[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(head["5.weight"].shape[1], head["5.weight"].shape[0]),
        )
        self.rm_head.load_state_dict(head, strict=True)
        self.model.eval().to(device=device)
        self.rm_head.eval().to(device=device, dtype=torch.float32)

    def to(self, device: torch.device) -> None:
        self.model.to(device)
        self.rm_head.to(device)
        self.device = device

    def score_batch(self, outputs: list[str], prompts: list[str]) -> list[float | None]:
        return score_clips(outputs, prompts, self._score)

    def _score(self, clip: Clip, request: str) -> float:
        images = [
            Image.fromarray(clip.frames[index].permute(1, 2, 0).numpy())
            for index in sample_frame_indices(len(clip.frames), self.num_frames)
        ]
        scores = []
        for start in range(0, len(images), self.frames_per_forward):
            scores.extend(self.score_images(images[start : start + self.frames_per_forward], request))
        return top_mean(scores, self.top_fraction)

    @torch.no_grad()
    def score_images(self, images: list[Image.Image], request: str) -> list[float]:
        """Each image's HPSv3 mu against `request`."""
        height, width = _smart_resize(images[0].height, images[0].width, min_pixels=_PIXELS, max_pixels=_PIXELS)
        images = [image.convert("RGB").resize((width, height), Image.BICUBIC) for image in images]
        message = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": _PROMPT.format(text_prompt=request)}],
            }
        ]
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text] * len(images), images=images, padding=True, return_tensors="pt").to(
            self.device
        )
        input_ids = inputs["input_ids"]
        embeds = self.model.get_input_embeddings()(input_ids)
        image_embeds = self.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
        image_embeds = torch.cat(image_embeds.pooler_output).to(embeds.dtype)
        image_mask = (input_ids == self.model.config.image_token_id).unsqueeze(-1).expand_as(embeds)
        embeds = embeds.masked_scatter(image_mask, image_embeds)
        # No position_ids: transformers 4.45, which HPSv3 trained on, gave its text model 1D positions here.
        hidden = self.model.language_model(inputs_embeds=embeds, attention_mask=inputs["attention_mask"])
        logits = self.rm_head(hidden.last_hidden_state[input_ids == self.special_token_id].float())
        return logits[:, 0].tolist()


def top_mean(scores: list[float], fraction: float) -> float:
    """The mean of the highest `fraction` of `scores`, at least one, rounding the count up."""
    best = sorted(scores, reverse=True)[: max(1, math.ceil(fraction * len(scores)))]
    return sum(best) / len(best)


def _renamed_state_dict(checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The transformers 4.45 checkpoint as 5.x Qwen2VLModel names (`model.` to `language_model.`), plus `rm_head.*`,
    with the unused `lm_head` dropped."""
    state = {}
    for name, tensor in checkpoint.items():
        if name == "lm_head.weight":
            continue
        if name.startswith("model."):
            name = "language_model." + name[len("model.") :]
        state[name] = tensor
    return state
