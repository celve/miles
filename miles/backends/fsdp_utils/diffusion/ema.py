"""EMA of the trainable weights, kept as optimizer state so Miles' offload and DCP checkpointing carry it.

Ported from radixark/miles_diffusion at ecb1363e (backends/fsdp_utils/ema.py), without the async-only previous EMA.
It has no diffusion-specific code; it lives here until a text trainer wants it.
"""

from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from miles.backends.fsdp_utils.checkpoint import OptimizerState


class EMAOptimizer(torch.optim.Optimizer):
    def __init__(self, model: torch.nn.Module, *, decay: float, uprate: float, uphold: float, flat_steps: int) -> None:
        super().__init__(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            dict(decay=decay, uprate=uprate, uphold=uphold, flat_steps=flat_steps),
        )
        self.reset_from_model()

    @torch.no_grad()
    def reset_from_model(self) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                self.state[parameter]["ema"] = parameter.detach().clone()

    @torch.no_grad()
    def step(self, optimizer_step: int) -> None:  # type: ignore[override]
        """Average in the weights after the actor's `optimizer_step`-th update; the decay ramps with that count."""
        for group in self.param_groups:
            decay = (
                group["decay"]
                if optimizer_step <= group["flat_steps"]
                else min((optimizer_step - group["flat_steps"]) * group["uprate"], group["uphold"])
            )
            ema_tensors = [_local(self.state[parameter]["ema"]) for parameter in group["params"]]
            actor_tensors = [_local(parameter.detach()) for parameter in group["params"]]
            torch._foreach_lerp_(ema_tensors, actor_tensors, 1.0 - decay)

    @contextmanager
    def use_weights(self, model: torch.nn.Module):
        """Swap the EMA into `model`'s shards for the block; gathered parameters are dropped on each swap."""
        # Resharding first: until then FSDP's root registers gathered parameters, which the EMA is not keyed by.
        _reshard(model)
        ema_weights = {
            parameter: self.state[parameter]["ema"] for parameter in model.parameters() if parameter in self.state
        }
        self._swap(ema_weights)
        try:
            yield
        finally:
            _reshard(model)
            self._swap(ema_weights)

    @torch.no_grad()
    def _swap(self, ema_weights: dict[torch.nn.Parameter, torch.Tensor]) -> None:
        for parameter, ema_weight in ema_weights.items():
            actor, ema = _local(parameter), _local(ema_weight)
            actor_copy = actor.clone()
            actor.copy_(ema)
            ema.copy_(actor_copy)


def save_ema(ema: EMAOptimizer, model: torch.nn.Module, checkpoint_dir: Path) -> None:
    dcp.save({"ema_state": OptimizerState(model, ema)}, checkpoint_id=str(checkpoint_dir / "ema"))


def load_ema(ema: EMAOptimizer, model: torch.nn.Module, checkpoint_dir: Path) -> None:
    """Restore the EMA saved with the checkpoint; one saved without it restarts the EMA from the loaded weights."""
    if (checkpoint_dir / "ema").exists():
        dcp.load({"ema_state": OptimizerState(model, ema)}, checkpoint_id=str(checkpoint_dir / "ema"))
    else:
        ema.reset_from_model()


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _reshard(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()
