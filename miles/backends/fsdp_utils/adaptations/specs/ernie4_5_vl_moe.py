"""ERNIE-4.5-VL-MoE: HF builds its M-RoPE only when given no positions, so the backend builds them.

This fixes positions only: its MoE layers also route media tokens by the per-token ``moe_mm_token_type_ids``,
which Miles's packed batches cannot carry.
"""

from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter
from miles.backends.fsdp_utils.adaptations.mrope import MropePositionIds


class Ernie45VLMoeAdapter(MropePositionIds, ArchAdapter):
    model_types = frozenset({"ernie4_5_vl_moe"})
