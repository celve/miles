"""Diffusion RL algorithms, selected by `--loss-type`: each owns its rollout request, train pairs and loss."""

from miles.diffusion.algorithms.base import DiffusionAlgorithm
from miles.diffusion.algorithms.flow_grpo import FlowGRPO

# Miles' default `policy_loss` is Flow-GRPO, the diffusion policy-gradient objective.
ALGORITHMS: dict[str, type[DiffusionAlgorithm]] = {"policy_loss": FlowGRPO}


def get_algorithm(loss_type: str) -> DiffusionAlgorithm:
    if loss_type not in ALGORITHMS:
        raise ValueError(f"--model-kind diffusion trains --loss-type {sorted(ALGORITHMS)}, got {loss_type!r}")
    return ALGORITHMS[loss_type]()
