"""GLM-4.1V, GLM-4.5V, GLM-4.6V and GLM-OCR: HF builds their M-RoPE only when given no positions, so the backend
builds them. ``glmga`` is a further model type HF maps to GLM-4.6V's model."""

from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter
from miles.backends.fsdp_utils.adaptations.mrope import MropePositionIds


class Glm4vAdapter(MropePositionIds, ArchAdapter):
    model_types = frozenset({"glm4v", "glm4v_moe", "glm46v", "glmga", "glm_ocr"})
