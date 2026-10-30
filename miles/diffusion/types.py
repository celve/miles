"""Denoising-trajectory payload a diffusion rollout attaches to `Sample.diffusion`."""

from dataclasses import dataclass, field

import torch
from safetensors.torch import load


def decode_tensor(data: bytes) -> torch.Tensor:
    """Deserialize the engine's safetensors bytes (single key ``t``) to a CPU tensor."""
    return load(data)["t"]


@dataclass
class CondKwargs:
    encoder_hidden_states: list[torch.Tensor] | None = None
    pooled_projections: list[torch.Tensor] | None = None


@dataclass
class DenoisingEnv:
    pos_cond_kwargs: CondKwargs | None = None
    neg_cond_kwargs: CondKwargs | None = None


@dataclass
class DiTTrajectory:
    latents: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    sigmas: torch.Tensor | None = None
    # [K] original step index of each latent; None means the full 0..T trajectory.
    latent_step_indices: torch.Tensor | None = None


@dataclass
class DiffusionRollout:
    seed: int | None = None
    request_id: str | None = None
    # Image as ``[C, F, H, W]`` on CPU.
    generated_output: torch.Tensor | None = None
    # [T] per-step SDE log-probs from the engine.
    rollout_log_probs: torch.Tensor | None = None
    denoising_env: DenoisingEnv | None = None
    dit_trajectory: DiTTrajectory | None = None
    sde_step_indices: list[int] = field(default_factory=list)
    inference_time_s: float | None = None
