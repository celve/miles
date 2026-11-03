from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from miles.utils.replay_base import Replay


def register_replay_list_moe(replay_list, replay_data, *, models, **_kwargs):
    """Map replay streams to Megatron MoE layers using the local model layout."""
    # A router built and then discarded during construction (Megatron-Bridge's Qwen3-VL text model replaces
    # GPTModel's decoder) still holds a replay in replay_list, so record into the live routers' replays only.
    live = [
        replay
        for model in models
        for module in model.modules()
        if isinstance(replay := getattr(module, "routing_replay", None), Replay)
    ]
    layer_indices = []
    for vp_stage, model in enumerate(models):
        config = model.module.config
        num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        for layer_id in range(offset, offset + num_layers_to_build):
            if isinstance(config.moe_layer_freq, int):
                if layer_id % config.moe_layer_freq != 0:
                    continue
            elif isinstance(config.moe_layer_freq, list):
                assert len(config.moe_layer_freq) == config.num_layers
                if config.moe_layer_freq[layer_id] == 0:
                    continue
            layer_indices.append(layer_id)

    if len(live) != len(layer_indices):
        raise ValueError(
            f"{len(live)} routers hold a routing replay, but the local layout has {len(layer_indices)} MoE layers"
        )
    for replay, layer_idx in zip(live, layer_indices, strict=True):
        replay.record(replay_data[:, layer_idx])
