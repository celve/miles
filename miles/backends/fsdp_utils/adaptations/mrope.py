"""M-RoPE positions for the HF multimodal models whose forward builds them only when it is given no positions.

Every such model runs ``compute_3d_position_ids`` under ``if position_ids is None``, so the 1-D ramp ``get_batch``
passes silently replaces the temporal/spatial coordinates media tokens are trained on. Their text models take a
four-row ``[text, temporal, height, width]`` tensor instead: row 0 keeps driving the causal mask and the
packed-document boundaries, rows 1-3 drive M-RoPE.
"""

from collections.abc import Mapping
from functools import partial

import torch
import torch.nn as nn

from miles.backends.fsdp_utils.adaptations.arch_adapter import PositionIdsBuilder

# the modality values HF's get_rope_index groups mm_token_type_ids by
_TEXT, _IMAGE, _VIDEO = 0, 1, 2

# what get_rope_index reads besides the tokens: the grids, and the per-video timing Qwen2.5-VL's processor adds
_ROPE_INPUT_KEYS = ("image_grid_thw", "video_grid_thw", "second_per_grid_ts")


class MropePositionIds:
    """Mixed into the adapter of each architecture that builds its M-RoPE only when given no positions."""

    def position_ids_builder(self, model: nn.Module) -> PositionIdsBuilder:
        return partial(mrope_position_ids, model.model)


def mrope_position_ids(
    model: nn.Module, tokens: torch.Tensor, multimodal_inputs: Mapping[str, torch.Tensor] | None
) -> torch.Tensor:
    """One packed document's ``[text, temporal, height, width]`` positions, as ``[4, seq]``.

    ``get_rope_index`` emits one position per token, so a document whose media disagree with its placeholders
    raises there instead of training on wrong coordinates.
    """
    text_positions = torch.arange(tokens.size(0), device=tokens.device, dtype=torch.long)

    rope_inputs = {
        key: multimodal_inputs[key]
        for key in _ROPE_INPUT_KEYS
        if multimodal_inputs is not None and multimodal_inputs.get(key) is not None
    }
    if "image_grid_thw" not in rope_inputs and "video_grid_thw" not in rope_inputs:
        return text_positions.expand(4, -1)

    # HF numbers one row at a time, so a packed row would run its positions straight through the boundaries.
    mrope_positions, _ = model.get_rope_index(
        tokens.unsqueeze(0), mm_token_type_ids=_mm_token_type_ids(model.config, tokens).unsqueeze(0), **rope_inputs
    )
    return torch.cat([text_positions.unsqueeze(0), mrope_positions.squeeze(1)], dim=0)


def _mm_token_type_ids(config, tokens: torch.Tensor) -> torch.Tensor:
    """The per-token modality the processor would have emitted, rebuilt from the placeholder ids.

    Rebuilt rather than carried: ``extract_multimodal_train_inputs`` drops the processor's copy, and the dict
    that survives is splatted into Megatron's model too, where an extra kwarg has nowhere to go.
    """
    image = tokens == config.image_token_id
    mm_token_type_ids = torch.full_like(tokens, _TEXT)
    mm_token_type_ids[image] = _IMAGE
    if (video_token_id := getattr(config, "video_token_id", None)) is not None:
        mm_token_type_ids[tokens == video_token_id] = _VIDEO
    # GLM-V processors mark video frames with the image token, between a video's start and end markers
    start, end = getattr(config, "video_start_token_id", None), getattr(config, "video_end_token_id", None)
    if start is not None and end is not None:
        in_video = torch.cumsum(tokens == start, dim=0) > torch.cumsum(tokens == end, dim=0)
        mm_token_type_ids[image & in_video] = _VIDEO
    return mm_token_type_ids
