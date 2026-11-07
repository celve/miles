"""VideoAlign's VideoReward (KwaiVGI/VideoAlign, checkpoint KwaiVGI/VideoReward): a Qwen2-VL-2B reward model whose
reward head reads three special tokens in its prompt, for visual quality (VQ), motion quality (MQ) and text alignment
(TA).

Ported from the repo's `inference.py`, which pins transformers 4.45: the LoRA is merged into the checkpoint's weights
at load, and the forward keeps theirs, including plain 1D positions in place of Qwen2-VL's 3D rope ones.
"""

import json
import math
from pathlib import Path

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from transformers import AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

from miles.rollout.rm_hub.video.clip import Clip, score_clips
from miles.utils.hf_utils.config import load_hf_config

HEADS = ("VQ", "MQ", "TA")
_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
_SPECIAL_TOKENS = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
_CHECKPOINT_PREFIX = "base_model.model."
# Frame sampling and sizing of VideoAlign's vision_process.py, at the checkpoint's data config.
_FRAME_FACTOR = 2
_SIZE_FACTOR = 28
_MIN_FRAMES = 4
_MAX_FRAMES = 768
_MIN_PIXELS = 128 * 28 * 28

# VideoAlign's DETAILED_PROMPT_WITH_SPECIAL_TOKEN verbatim; \x20 keeps its trailing spaces, which it was trained on.
_PROMPT = """
You are tasked with evaluating a generated video based on three distinct criteria: Visual Quality, Motion Quality, and Text Alignment. Please provide a rating from 0 to 10 for each of the three categories, with 0 being the worst and 10 being the best. Each evaluation should be independent of the others.

**Visual Quality:**\x20\x20
Evaluate the overall visual quality of the video, with a focus on static factors. The following sub-dimensions should be considered:
- **Reasonableness:** The video should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the video. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the video, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The video should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible.\x20

Please provide the ratings of Visual Quality: <|VQ_reward|>
END

**Motion Quality:**\x20\x20
Assess the dynamic aspects of the video, with a focus on dynamic factors. Consider the following sub-dimensions:
- **Stability:** Evaluate the continuity and stability between frames. There should be no sudden, unnatural jumps, and the video should maintain stable attributes (e.g., no fluctuating colors, textures, or missing body parts).
- **Naturalness:** The movement should align with physical laws and be realistic. For example, clothing should flow naturally with motion, and facial expressions should change appropriately (e.g., blinking, mouth movements).
- **Aesthetic Quality:** The movement should be smooth and fluid. The transitions between different motions or camera angles should be seamless, and the overall dynamic feel should be visually pleasing.
- **Fusion:** Ensure that elements in motion (e.g., edges of the subject, hair, clothing) blend naturally with the background, without obvious artifacts or the feeling of cut-and-paste effects.
- **Clarity of Motion:** The video should be clear and smooth in motion. Pay attention to any areas where the video might have blurry or unsteady sections that hinder visual continuity.
- **Amplitude:** If the video is largely static or has little movement, assign a low score for motion quality.

Please provide the ratings of Motion Quality: <|MQ_reward|>
END

**Text Alignment:**\x20\x20
Assess how well the video matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the video (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Motion Relevance:** Evaluate if the dynamic actions (e.g., gestures, posture, facial expressions like talking or blinking) align with the described prompt. The motion should match the prompt in terms of type, scale, and direction.
- **Environment Relevance:** Assess whether the background and scene fit the prompt. This includes checking if real-world locations or scenes are accurately represented, though some stylistic adaptation is acceptable.\x20\x20
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the video adheres to this style.
- **Camera Movement Relevance:** Check if the camera movements (e.g., following the subject, focus shifts) are consistent with the expected behavior from the prompt.

Textual prompt - {text_prompt}
Please provide the ratings of Text Alignment: <|TA_reward|>
END
"""


class VideoAlignScorer:
    """Scores each clip as the sum of the heads in `heads`, each normalized by the checkpoint's mean and std, as
    VideoAlign's `Overall` is for all three."""

    def __init__(
        self,
        *,
        device: torch.device,
        checkpoint_path: str,
        base_model_path: str,
        heads: tuple[str, ...] = HEADS,
        use_norm: bool = True,
        # bfloat16 moves this model's normalized heads by up to 0.3 against float32, which matches VideoAlign's own code.
        dtype: str = "float32",
    ) -> None:
        assert set(heads) <= set(HEADS), f"VideoAlign has heads {HEADS}, got {heads}"
        self.device = device
        self.heads = heads
        config = json.loads((Path(checkpoint_path) / "model_config.json").read_text())
        self.fps = config["data_config"]["fps"]
        self.max_pixels = config["data_config"]["max_frame_pixels"]
        self.norm = config["inference_config"] if use_norm else None
        lora = config["peft_lora_config"]

        self.processor = AutoProcessor.from_pretrained(base_model_path, padding_side="right")
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": _SPECIAL_TOKENS})
        self.special_token_ids = self.processor.tokenizer.convert_tokens_to_ids(_SPECIAL_TOKENS)

        state = _merged_state_dict(_load_checkpoint(checkpoint_path), lora_scaling=lora["lora_alpha"] / lora["lora_r"])
        model_config = load_hf_config(base_model_path)
        model_config.text_config.vocab_size = state["language_model.embed_tokens.weight"].shape[0]
        self.model = Qwen2VLModel._from_config(model_config, dtype=_DTYPES[dtype])
        rm_head_weight = state.pop("rm_head.weight")
        self.model.load_state_dict(state, strict=True)
        self.rm_head = torch.nn.Linear(rm_head_weight.shape[1], rm_head_weight.shape[0], bias=False)
        self.rm_head.weight.data.copy_(rm_head_weight)
        self.model.eval().to(device=device, dtype=_DTYPES[dtype])
        self.rm_head.eval().to(device=device, dtype=_DTYPES[dtype])

    def to(self, device: torch.device) -> None:
        self.model.to(device)
        self.rm_head.to(device)
        self.device = device

    def score_batch(self, outputs: list[str], prompts: list[str]) -> list[float | None]:
        return score_clips(outputs, prompts, self._score)

    def _score(self, clip: Clip, request: str) -> float:
        heads = self.reward_heads(clip, request)
        return sum(heads[head] for head in self.heads)

    @torch.no_grad()
    def reward_heads(self, clip: Clip, request: str) -> dict[str, float]:
        """Each head's reward, normalized when the scorer normalizes."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "video"}, {"type": "text", "text": _PROMPT.format(text_prompt=request)}],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            videos=[self._frames(clip)],
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True, "do_sample_frames": False},
        ).to(self.device)
        input_ids = inputs["input_ids"]
        embeds = self.model.get_input_embeddings()(input_ids)
        video_embeds = self.model.get_video_features(inputs["pixel_values_videos"], inputs["video_grid_thw"])
        video_embeds = torch.cat(video_embeds.pooler_output).to(embeds.dtype)
        video_mask = (input_ids == self.model.config.video_token_id).unsqueeze(-1).expand_as(embeds)
        embeds = embeds.masked_scatter(video_mask, video_embeds)
        # No position_ids: VideoAlign trained its head on the text model's default 1D positions.
        hidden = self.model.language_model(inputs_embeds=embeds, attention_mask=inputs["attention_mask"])
        logits = self.rm_head(hidden.last_hidden_state)
        special = torch.isin(input_ids, torch.tensor(self.special_token_ids, device=input_ids.device))
        rewards = dict(zip(HEADS, logits[special].float().view(-1).tolist(), strict=True))
        if self.norm is not None:
            rewards = {
                head: (value - self.norm[f"{head}_mean"]) / self.norm[f"{head}_std"] for head, value in rewards.items()
            }
        return rewards

    def _frames(self, clip: Clip) -> torch.Tensor:
        """VideoAlign's `uniform` sampling at the checkpoint's fps, resized as its `fetch_video`: [T, C, H, W] float."""
        total = len(clip.frames)
        num_frames = total / clip.fps * self.fps
        num_frames = min(
            max(num_frames, _MIN_FRAMES), math.floor(min(_MAX_FRAMES, total) / _FRAME_FACTOR) * _FRAME_FACTOR
        )
        num_frames = min(round(num_frames / _FRAME_FACTOR) * _FRAME_FACTOR, total)
        indices = torch.linspace(0, total - 1, num_frames).round().long().tolist()
        frames = clip.frames[indices]
        height, width = _smart_resize(*frames.shape[-2:], min_pixels=_MIN_PIXELS, max_pixels=self.max_pixels)
        return F.resize(frames, [height, width], interpolation=InterpolationMode.BICUBIC, antialias=True).float()


def _smart_resize(height: int, width: int, *, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Qwen2-VL's smart_resize: both sides multiples of 28, the area within [min_pixels, max_pixels]."""
    factor = _SIZE_FACTOR
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _load_checkpoint(checkpoint_path: str) -> dict[str, torch.Tensor]:
    (path,) = sorted(Path(checkpoint_path).glob("checkpoint-*/model.pth"))
    return torch.load(path, map_location="cpu", weights_only=True)


def _merged_state_dict(checkpoint: dict[str, torch.Tensor], *, lora_scaling: float) -> dict[str, torch.Tensor]:
    """The PEFT checkpoint as Qwen2VLModel's state dict, plus `rm_head.weight`: each LoRA merged into its base layer,
    the transformers 4.45 names moved to 5.x's (`model.` to `language_model.`), and the unused `lm_head` dropped."""
    state = {}
    for name, tensor in checkpoint.items():
        assert name.startswith(_CHECKPOINT_PREFIX), f"unexpected VideoAlign checkpoint key {name}"
        name = name[len(_CHECKPOINT_PREFIX) :]
        if ".lora_" in name or name == "lm_head.weight":
            continue
        if ".base_layer." in name:
            module = name.split(".base_layer.")[0]
            name = name.replace(".base_layer.", ".")
            if name.endswith(".weight"):
                lora_a = checkpoint[f"{_CHECKPOINT_PREFIX}{module}.lora_A.default.weight"].float()
                lora_b = checkpoint[f"{_CHECKPOINT_PREFIX}{module}.lora_B.default.weight"].float()
                tensor = tensor.float() + lora_scaling * lora_b @ lora_a
        if name.startswith("model."):
            name = "language_model." + name[len("model.") :]
        state[name] = tensor
    return state
