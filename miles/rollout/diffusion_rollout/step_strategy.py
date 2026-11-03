"""Functions choosing which denoising steps of one rollout request are SDE steps that enter training.

Each has signature ``(args, sample, num_steps, seed) -> (sde_steps, returned_latents)``; ``returned_latents``
None asks the engine for every latent. Point ``--diffusion-step-strategy-path`` at one.
"""

from argparse import Namespace

import numpy as np
import torch

from miles.utils.types import Sample


def sde_window(args: Namespace, sample: Sample, num_steps: int, seed: int) -> tuple[list[int], list[int]]:
    """flow_grpo's random contiguous SDE window; the engine returns every latent the window's steps touch."""
    window_size = args.diffusion_num_sde_steps
    lo, hi = _window_range(args.diffusion_sde_window_range, num_steps=num_steps)
    if window_size <= 0 or window_size > hi - lo:
        raise ValueError(f"--diffusion-num-sde-steps {window_size} does not fit in window range [{lo},{hi})")
    start = int(np.random.default_rng(seed).integers(lo, hi - window_size + 1))
    steps = list(range(start, start + window_size))
    return steps, sorted(set(steps) | {step + 1 for step in steps})


def epoch_global_random_choice(
    args: Namespace, sample: Sample, num_steps: int, seed: int
) -> tuple[list[int], list[int]]:
    """--diffusion-num-sde-steps steps drawn from --diffusion-sde-candidate-steps once per rollout (every
    --rollout-batch-size prompt groups), so each rollout trains the same steps on every image."""
    candidates = _candidate_steps(args.diffusion_sde_candidate_steps, num_steps=num_steps)
    if args.diffusion_num_sde_steps >= len(candidates):
        steps = sorted(candidates)
    else:
        epoch = int(sample.group_index or 0) // args.rollout_batch_size
        generator = torch.Generator().manual_seed(epoch + args.rollout_seed)
        selected = torch.randperm(len(candidates), generator=generator)[: args.diffusion_num_sde_steps]
        steps = sorted(candidates[index] for index in selected.tolist())
    return steps, sorted(set(steps) | {step + 1 for step in steps})


def ode_and_return_last(args: Namespace, sample: Sample, num_steps: int, seed: int) -> tuple[None, list[int]]:
    """DiffusionNFT: a pure ODE rollout that returns only the final, clean latent."""
    return None, [num_steps]


def _candidate_steps(raw: str | None, *, num_steps: int) -> list[int]:
    if raw is None:
        raise ValueError("epoch_global_random_choice needs --diffusion-sde-candidate-steps, e.g. '1,2,3'")
    candidates = [int(step) for step in raw.split(",")]
    if out_of_range := [step for step in candidates if not 0 <= step < num_steps]:
        raise ValueError(
            f"--diffusion-sde-candidate-steps {out_of_range} out of range for a {num_steps}-step schedule"
        )
    return candidates


def _window_range(raw: str | None, *, num_steps: int) -> tuple[int, int]:
    if not raw:
        return 0, num_steps
    lo, hi = (int(part) for part in raw.split(","))
    if not 0 <= lo < hi <= num_steps:
        raise ValueError(f"--diffusion-sde-window-range [{lo},{hi}) out of range for a {num_steps}-step schedule")
    return lo, hi
