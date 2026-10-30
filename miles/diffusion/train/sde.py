"""Scores a recorded (x_t -> x_t+1) transition under the flow-matching SDE the engine sampled it with.

Mirrors sglang.multimodal_gen's `flow_sde_sampling` with `rollout_sde_type="sde"`. Every transition carries
the sigmas the engine stepped with (sigma, next_sigma, sigma_max), so scoring needs no scheduler.
"""

import math

import torch


def flow_sde_prev_mean_and_std(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_prev: torch.Tensor,
    *,
    sigma_max: torch.Tensor,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(prev_mean, noise_std, std_dev_t)``; ``std_dev_t`` is the KL denominator's diffusion scale."""
    dt = sigma_prev - sigma
    # sigma_max (the grid's second sigma) stands in at sigma == 1, where sigma / (1 - sigma) diverges;
    # isclose, not ==, is the engine's test.
    at_one = torch.isclose(sigma, sigma.new_tensor(1.0))
    std_dev_t = torch.sqrt(sigma / (1 - torch.where(at_one, sigma_max, sigma))) * noise_level
    prev_mean = (
        sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )
    return prev_mean, std_dev_t * torch.sqrt(-1 * dt), std_dev_t


def gaussian_log_prob(prev_sample: torch.Tensor, prev_mean: torch.Tensor, noise_std: torch.Tensor) -> torch.Tensor:
    """Gaussian log-density of the recorded transition, averaged over non-batch dims."""
    log_prob = (
        -((prev_sample.detach() - prev_mean) ** 2) / (2 * noise_std**2)
        - torch.log(noise_std)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    return log_prob.mean(dim=tuple(range(1, log_prob.ndim)))


def sde_step_log_prob(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    *,
    prev_sample: torch.Tensor,
    noise_level: float,
    sigmas: torch.Tensor,
    next_sigmas: torch.Tensor,
    sigma_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(log_prob, prev_mean, std_dev_t)`` per batch row."""
    model_output, sample, prev_sample = model_output.float(), sample.float(), prev_sample.float()
    view = (-1, *([1] * (sample.ndim - 1)))
    prev_mean, noise_std, std_dev_t = flow_sde_prev_mean_and_std(
        model_output,
        sample,
        sigmas.view(view),
        next_sigmas.view(view),
        sigma_max=sigma_max.view(view),
        noise_level=noise_level,
    )
    return gaussian_log_prob(prev_sample, prev_mean, noise_std), prev_mean, std_dev_t
