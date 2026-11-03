"""DiffusionNFT: ODE rollouts, K (x0, sigma) train pairs per image, and a dual-policy x0 regression against the EMA.

Ported from radixark/miles_diffusion at ecb1363e (loss_hub/nft.py, data_conversion_hub/nft.py).
"""

import hashlib
from argparse import Namespace
from typing import Any

import torch

from miles.utils.diffusion.algorithms.base import DiffusionAlgorithm
from miles.utils.function_registry import load_function
from miles.utils.types import Sample


class DiffusionNFT(DiffusionAlgorithm):
    uses_ema = True
    unsupported = [
        (
            "--loss-type nft with SDE rollouts; it trains on --diffusion-sde-type ode",
            lambda a: a.diffusion_sde_type != "ode",
        ),
        (
            "--loss-type nft with --diffusion-noise-level other than 0: ODE rollouts add no noise",
            lambda a: a.diffusion_noise_level != 0,
        ),
        (
            "--loss-type nft with classifier-free guidance: it samples and trains CFG-free, so "
            "--diffusion-guidance-scale must be 1",
            lambda a: a.diffusion_guidance_scale != 1.0,
        ),
        ("--loss-type nft with a KL term (--diffusion-kl-beta)", lambda a: a.diffusion_kl_beta != 0),
        ("--diffusion-nft-beta <= 0", lambda a: a.diffusion_nft_beta <= 0),
        ("--diffusion-adv-clip-max <= 0 under --loss-type nft", lambda a: a.diffusion_adv_clip_max <= 0),
        (
            "--diffusion-nft-num-dropped-timesteps outside [0, --diffusion-num-steps)",
            lambda a: not 0 <= a.diffusion_nft_num_dropped_timesteps < a.diffusion_num_steps,
        ),
        (
            "an EMA decay outside [0, 1] or a negative --diffusion-ema-decay-ramp/-flat-steps",
            lambda a: not (
                0 <= a.diffusion_ema_decay_init <= 1
                and 0 <= a.diffusion_ema_decay_max <= 1
                and a.diffusion_ema_decay_ramp >= 0
                and a.diffusion_ema_decay_flat_steps >= 0
            ),
        ),
    ]
    unread_flags = (
        "eps_clip",
        "diffusion_num_sde_steps",
        "diffusion_sde_window_range",
        "diffusion_sde_candidate_steps",
        "diffusion_negative_prompt",
    )

    def rollout_request(self, sde_steps: list[int] | None, returned_latents: list[int] | None) -> dict[str, Any]:
        return {"rollout_sde_step_indices": sde_steps, "rollout_return_step_indices": returned_latents}

    def build_train_pairs(self, args: Namespace, sample: Sample, *, advantage: float) -> list[dict[str, Any]]:
        rollout = sample.diffusion
        x0 = _clean_x0(sample)
        # The grid ends with the terminal sigma 0; the sigmas before it are the sampling steps.
        step_sigmas = rollout.dit_trajectory.sigmas.detach().float()[:-1]
        # Keyed on the run-wide sample index, so each image draws its own steps and the run reproduces.
        generator = torch.Generator().manual_seed(stable_hash("nft_sigma_shuffle", int(args.seed), int(sample.index)))
        select_steps = load_function(args.diffusion_nft_timestep_strategy_path)
        train_steps = torch.tensor(select_steps(args, len(step_sigmas), generator), dtype=torch.long)
        num_timesteps = len(train_steps)
        return [
            {
                "sample_index": sample.index,
                "advantage": float(advantage),
                "denoising_env": rollout.denoising_env,
                "x0": x0,
                "timestep": float(sigma),
                "num_timesteps": num_timesteps,
            }
            for sigma in step_sigmas[train_steps].tolist()
        ]


# Timestep strategies: (args, num_steps, generator) -> the sampling steps an image trains on, in training order.


def drop_random_steps(args: Namespace, num_steps: int, generator: torch.Generator) -> list[int]:
    """All steps but `--diffusion-nft-num-dropped-timesteps` drawn per image, in a random order."""
    kept = torch.randperm(num_steps, generator=generator)[: num_steps - args.diffusion_nft_num_dropped_timesteps]
    return kept.tolist()


def drop_final_steps(args: Namespace, num_steps: int, generator: torch.Generator) -> list[int]:
    """All steps but the final (lowest-noise) `--diffusion-nft-num-dropped-timesteps`, in schedule order."""
    return list(range(num_steps - args.diffusion_nft_num_dropped_timesteps))


def stable_hash(*parts: object) -> int:
    """A 63-bit hash of `parts` that every process and run agrees on, unlike `hash()`."""
    digest = hashlib.blake2b(":".join(str(part) for part in parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & (2**63 - 1)


def _clean_x0(sample: Sample) -> torch.Tensor:
    traj = sample.diffusion.dit_trajectory
    assert (
        traj is not None and traj.latents is not None and traj.sigmas is not None
    ), f"sample {sample.index}: NFT needs the trajectory's final latent and its sigma grid"
    if traj.latent_step_indices is not None:
        final_step = int(traj.latent_step_indices[-1])
        assert final_step == traj.sigmas.shape[0] - 1, (
            f"sample {sample.index}: trajectory ends at step {final_step}, not the clean x0; "
            "NFT rollouts use --diffusion-step-strategy-path ...ode_and_return_last"
        )
    return traj.latents[-1].detach().cpu().float()
