"""DiffusionNFT: ODE rollouts, K (x0, sigma) train pairs per image, and a dual-policy x0 regression against the EMA.

Ported from radixark/miles_diffusion at ecb1363e (loss_hub/nft.py, data_conversion_hub/nft.py).
"""

import hashlib
from argparse import Namespace
from dataclasses import dataclass
from typing import Any

import torch

from miles.diffusion.algorithms.base import DiffusionAlgorithm
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

    def loss(self, actor, pairs: list[dict], *, rollout_id: int, microbatch_id: int, metrics) -> torch.Tensor:
        args = actor.args
        device = torch.device("cuda", torch.cuda.current_device())
        batch = collate_nft_pairs(pairs, config=actor.pipeline_config, device=device)
        generator = torch.Generator(device=device).manual_seed(
            stable_hash("nft_corrupt", int(args.backend.seed), rollout_id, microbatch_id, actor.dp_rank)
        )
        eps = torch.randn(batch.x0.shape, device=device, dtype=batch.x0.dtype, generator=generator)
        xt = corrupt(batch.x0, batch.t, eps)
        model_t = actor.pipeline_config.process_sigma_as_timesteps_input(batch.t)
        component = actor.pipeline_config.component_for_timesteps(model_t)
        new_pred = actor.noise_pred(component, xt, model_t, batch.pos_cond, None)
        # pi_old is the EMA, which is also what the engines sample with.
        with torch.no_grad(), actor.ema.use_weights(actor.model):
            old_pred = actor.noise_pred(component, xt, model_t, batch.pos_cond, None)
        return nft_loss(
            batch,
            xt=xt,
            new_pred=new_pred,
            old_pred=old_pred,
            beta=args.diffusion_nft_beta,
            adv_clip_max=args.diffusion_adv_clip_max,
            use_adaptive=args.diffusion_nft_adaptive_weight,
            num_timesteps=pairs[0]["num_timesteps"],
            metrics=metrics,
        )


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


@dataclass
class NFTBatch:
    x0: torch.Tensor
    t: torch.Tensor
    advantage: torch.Tensor
    pos_cond: dict

    def __len__(self) -> int:
        return self.x0.shape[0]


def collate_nft_pairs(pairs: list[dict], *, config, device: torch.device) -> NFTBatch:
    return NFTBatch(
        x0=torch.stack([pair["x0"] for pair in pairs]).to(device=device, dtype=torch.float32),
        t=torch.tensor([pair["timestep"] for pair in pairs], device=device, dtype=torch.float32),
        advantage=torch.tensor([pair["advantage"] for pair in pairs], device=device, dtype=torch.float32),
        pos_cond=config.collate_cond(
            [config.prepare_cond_kwargs(pair["denoising_env"].pos_cond_kwargs, device) for pair in pairs], device
        ),
    )


def corrupt(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """The linear flow ``x_t = (1 - t) x0 + t eps``."""
    t = t.view(-1, *([1] * (x0.ndim - 1)))
    return (1.0 - t) * x0 + t * eps


def nft_loss(
    batch: NFTBatch,
    *,
    xt: torch.Tensor,
    new_pred: torch.Tensor,
    old_pred: torch.Tensor,
    beta: float,
    adv_clip_max: float,
    use_adaptive: bool,
    num_timesteps: int,
    metrics,
) -> torch.Tensor:
    """Summed loss: the advantage mixes a positive policy ``beta v_new + (1-beta) v_old`` and its mirror
    ``(1+beta) v_old - beta v_new``, each regressed onto x0."""
    r = torch.clamp(torch.clamp(batch.advantage, -adv_clip_max, adv_clip_max) / adv_clip_max / 2.0 + 0.5, 0.0, 1.0)
    t = batch.t.view(len(batch), *([1] * (batch.x0.ndim - 1)))
    pos_loss, neg_loss = _branch_losses(
        batch.x0, xt, t, new_pred=new_pred, old_pred=old_pred, beta=beta, use_adaptive=use_adaptive
    )
    r_b = r.to(dtype=pos_loss.dtype)
    per_pair = (r_b * pos_loss / beta + (1.0 - r_b) * neg_loss / beta) * adv_clip_max
    loss_sum = per_pair.sum()

    with torch.no_grad():
        n = len(batch)
        total = per_pair.sum()
        metrics.emit_mean("loss", total=total * num_timesteps, count=n)
        metrics.emit_mean("nft_loss", total=total * num_timesteps, count=n)
        metrics.emit_mean("nft_loss_per_pair", total=total, count=n)
        metrics.emit_mean("nft_r_mean", total=r.sum(), count=n)
        metrics.emit_mean("nft_pos_loss", total=pos_loss.sum(), count=n)
        metrics.emit_mean("nft_neg_loss", total=neg_loss.sum(), count=n)
        metrics.emit_mean("nft_adv_mean", total=batch.advantage.sum(), count=n)
        metrics.emit_mean("nft_t_mean", total=batch.t.sum(), count=n)
        metrics.emit_mean("nft_num_timesteps", total=torch.tensor(float(num_timesteps)), count=1)
        metrics.emit_mean("adv_abs_mean", total=batch.advantage.abs().sum(), count=n)
    return loss_sum


def _branch_losses(x0, xt, t, *, new_pred, old_pred, beta: float, use_adaptive: bool):
    dims = tuple(range(1, x0.ndim))
    dtype = new_pred.dtype
    target = x0.to(dtype=dtype)
    x0_pos = xt.to(dtype=dtype) - t.to(dtype=dtype) * (beta * new_pred + (1.0 - beta) * old_pred)
    x0_neg = xt.to(dtype=dtype) - t.to(dtype=dtype) * ((1.0 + beta) * old_pred - beta * new_pred)

    def error(x0_hat: torch.Tensor) -> torch.Tensor:
        squared = (x0_hat - target) ** 2
        if not use_adaptive:
            return squared.mean(dim=dims)
        with torch.no_grad():
            # Computed in fp64, as the fork does, so the weight does not depend on the forward dtype.
            weight = (x0_hat.detach().double() - target.double()).abs().mean(dim=dims, keepdim=True).clamp(min=1e-5)
        return (squared / weight.to(dtype=dtype)).mean(dim=dims)

    return error(x0_pos), error(x0_neg)
