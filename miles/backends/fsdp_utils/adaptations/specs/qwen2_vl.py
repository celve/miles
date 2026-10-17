"""Qwen2-VL and Qwen2.5-VL: HF builds their M-RoPE only when given no positions, so the backend builds them."""

from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter
from miles.backends.fsdp_utils.adaptations.mrope import MropePositionIds


class Qwen2VLAdapter(MropePositionIds, ArchAdapter):
    model_types = frozenset({"qwen2_vl", "qwen2_5_vl"})
