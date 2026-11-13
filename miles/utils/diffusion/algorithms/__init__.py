"""Diffusion RL algorithms, selected by `--loss-type`: each owns its rollout request and train pairs."""

from miles.utils.diffusion.algorithms.base import DiffusionAlgorithm
from miles.utils.diffusion.algorithms.flow_grpo import FlowGRPO
from miles.utils.diffusion.algorithms.nft import DiffusionNFT

ALGORITHMS: dict[str, type[DiffusionAlgorithm]] = {"flow_grpo": FlowGRPO, "nft": DiffusionNFT}


def get_algorithm(loss_type: str) -> DiffusionAlgorithm:
    if loss_type not in ALGORITHMS:
        raise ValueError(f"--model-kind diffusion trains --loss-type {sorted(ALGORITHMS)}, got {loss_type!r}")
    return ALGORITHMS[loss_type]()
