"""Functions choosing which denoising steps of one rollout request are SDE steps that enter training.

Each has signature ``(args, sample, num_steps, seed) -> (sde_steps, returned_latents)``; ``returned_latents``
None asks the engine for every latent. Point ``--diffusion-step-strategy-path`` at one.
"""

from argparse import Namespace

import numpy as np

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


def _window_range(raw: str | None, *, num_steps: int) -> tuple[int, int]:
    if not raw:
        return 0, num_steps
    lo, hi = (int(part) for part in raw.split(","))
    if not 0 <= lo < hi <= num_steps:
        raise ValueError(f"--diffusion-sde-window-range [{lo},{hi}) out of range for a {num_steps}-step schedule")
    return lo, hi
