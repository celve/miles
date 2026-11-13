"""The diffusion trainer's micro-batch losses by `--loss-type`, as `get_loss_function` picks the text losses."""

from collections.abc import Callable

from miles.backends.fsdp_utils.diffusion.losses import flow_grpo, nft

LOSSES: dict[str, Callable] = {"flow_grpo": flow_grpo.microbatch_loss, "nft": nft.microbatch_loss}


def get_diffusion_loss_function(loss_type: str) -> Callable:
    if loss_type not in LOSSES:
        raise ValueError(f"no diffusion loss for --loss-type {loss_type!r}; known: {sorted(LOSSES)}")
    return LOSSES[loss_type]
