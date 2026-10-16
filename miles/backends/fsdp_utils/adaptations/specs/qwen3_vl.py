"""Qwen3-VL: the backend builds M-RoPE positions itself, because HF computes them only when it is given none.

``Qwen3VLModel.forward`` runs ``compute_3d_position_ids`` under ``if position_ids is None``, so the 1-D ramp
``get_batch`` passes silently replaces the temporal/spatial coordinates image tokens are trained on. HF's text
model accepts a four-row ``[text, temporal, height, width]`` tensor instead: row 0 keeps driving the causal mask
and the packed-document boundaries, rows 1-3 drive M-RoPE.
"""

from collections.abc import Mapping
from functools import partial

import torch
import torch.nn as nn

from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter, PositionIdsBuilder

# the modality values HF's get_rope_index groups mm_token_type_ids by
_TEXT, _IMAGE, _VIDEO = 0, 1, 2

_GRID_KEYS = {_IMAGE: "image_grid_thw", _VIDEO: "video_grid_thw"}


class Qwen3VLAdapter(ArchAdapter):
    model_types = frozenset({"qwen3_vl"})
    verified = True

    def position_ids_builder(self, model: nn.Module) -> PositionIdsBuilder:
        return partial(mrope_position_ids, model.model)


class Qwen3VLMoeAdapter(Qwen3VLAdapter):
    """The MoE sibling is generated from the same modular source and bypasses M-RoPE the same way."""

    model_types = frozenset({"qwen3_vl_moe"})
    verified = False


def mrope_position_ids(
    model: nn.Module, tokens: torch.Tensor, multimodal_inputs: Mapping[str, torch.Tensor] | None
) -> torch.Tensor:
    """One packed document's ``[text, temporal, height, width]`` positions, as ``[4, seq]``."""
    text_positions = torch.arange(tokens.size(0), device=tokens.device, dtype=torch.long)

    grids = {
        key: multimodal_inputs[key]
        for key in _GRID_KEYS.values()
        if multimodal_inputs is not None and multimodal_inputs.get(key) is not None
    }
    if not grids:
        return text_positions.expand(4, -1)

    mm_token_type_ids = _mm_token_type_ids(model.config, tokens)
    _check_grids_cover_media_tokens(model.config, mm_token_type_ids, grids)

    # HF numbers one row at a time, so a packed row would run its positions straight through the boundaries.
    mrope_positions, _ = model.get_rope_index(
        tokens.unsqueeze(0), mm_token_type_ids=mm_token_type_ids.unsqueeze(0), **grids
    )
    return torch.cat([text_positions.unsqueeze(0), mrope_positions.squeeze(1)], dim=0)


def _mm_token_type_ids(config, tokens: torch.Tensor) -> torch.Tensor:
    """The per-token modality the processor would have emitted, rebuilt from the placeholder ids.

    Rebuilt rather than carried: ``extract_multimodal_train_inputs`` drops the processor's copy, and the dict
    that survives is splatted into Megatron's model too, where an extra kwarg has nowhere to go.
    """
    mm_token_type_ids = torch.full_like(tokens, _TEXT)
    mm_token_type_ids[tokens == config.image_token_id] = _IMAGE
    mm_token_type_ids[tokens == config.video_token_id] = _VIDEO
    return mm_token_type_ids


def _check_grids_cover_media_tokens(config, mm_token_type_ids: torch.Tensor, grids: Mapping[str, torch.Tensor]):
    """A sample whose grids disagree with its placeholders would train on quietly wrong coordinates."""
    merge_size = config.vision_config.spatial_merge_size
    for modality, grid_key in _GRID_KEYS.items():
        grid_thw = grids.get(grid_key)
        if grid_thw is None:
            expected = 0
        else:
            temporal, height, width = grid_thw.unbind(-1)
            expected = int((temporal * (height // merge_size) * (width // merge_size)).sum())
        placeholders = int((mm_token_type_ids == modality).sum())
        assert placeholders == expected, (
            f"Qwen3-VL sample carries {placeholders} {grid_key.split('_')[0]} placeholder token(s) but "
            f"{grid_key}={None if grid_thw is None else grid_thw.tolist()} accounts for {expected}"
        )
