from miles.backends.megatron_utils.te_fa4_window import install_fa4_causal_window_patch
from miles.backends.megatron_utils.te_fused_adam_batches import install_fused_adam_batches_patch


def install_te_training_patches() -> None:
    """Every Transformer Engine patch Megatron training needs; idempotent, and a no-op without TE."""
    install_fa4_causal_window_patch()
    install_fused_adam_batches_patch()
