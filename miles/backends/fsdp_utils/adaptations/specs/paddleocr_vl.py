"""PaddleOCR-VL: HF builds its M-RoPE only when given no positions, so the backend builds them."""

from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter
from miles.backends.fsdp_utils.adaptations.mrope import MropePositionIds


class PaddleOcrVLAdapter(MropePositionIds, ArchAdapter):
    model_types = frozenset({"paddleocr_vl"})
