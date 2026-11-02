"""Flow-GRPO: SDE rollouts, one (x_t, x_t+1) train pair per SDE step, and a PPO-clipped log-prob ratio."""

from argparse import Namespace
from typing import Any

import torch

from miles.backends.fsdp_utils.diffusion.losses.flow_grpo import collate_pairs, flow_grpo_loss
from miles.utils.args.configs.algo import AlgoConfig
from miles.utils.diffusion.algorithms.base import DiffusionAlgorithm
from miles.utils.types import Sample


class FlowGRPO(DiffusionAlgorithm):
    unsupported = [
        (
            # Equality with the parser default is all args can tell; Flow-GRPO clips near 1e-4, never at text PPO's 0.2.
            "Miles' text PPO clip default (--eps-clip 0.2); pass the ratio clip explicitly, e.g. --eps-clip 1e-4",
            lambda a: a.eps_clip == AlgoConfig.model_fields["eps_clip"].default,
        ),
    ]

    def rollout_request(self, sde_steps: list[int] | None, returned_latents: list[int] | None) -> dict[str, Any]:
        # The trainer slices the full trajectory itself, so the engine returns every latent.
        return {"rollout_sde_step_indices": sde_steps, "rollout_return_step_indices": None}

    def build_train_pairs(self, args: Namespace, sample: Sample, *, advantage: float) -> list[dict[str, Any]]:
        rollout = sample.diffusion
        traj = rollout.dit_trajectory
        steps = rollout.sde_step_indices
        assert steps, f"sample {sample.index} has no SDE steps to train on"
        position = _latent_positions(traj)
        missing = sorted({s for step in steps for s in (step, step + 1)} - position.keys())
        assert not missing, f"sample {sample.index}: trajectory lacks latents for steps {missing}"

        # Cloned so a pickled pair carries its own slice, not the whole trajectory storage it views.
        return [
            {
                "sample_index": sample.index,
                "advantage": float(advantage),
                "denoising_env": rollout.denoising_env,
                "latent": traj.latents[position[step]].clone(),
                "next_latent": traj.latents[position[step + 1]].clone(),
                "timestep": traj.timesteps[step].clone(),
                "sigma": traj.sigmas[step].clone(),
                "next_sigma": traj.sigmas[step + 1].clone(),
                # The first sigma below 1 on this image's grid; the SDE kernel substitutes it at sigma == 1.
                "sigma_max": traj.sigmas[1].clone(),
                "log_prob_old": rollout.rollout_log_probs[step].clone(),
            }
            for step in steps
        ]

    def loss(self, actor, pairs: list[dict], *, metrics) -> torch.Tensor:
        args = actor.args
        batch = collate_pairs(
            pairs,
            config=actor.pipeline_config,
            use_cfg=args.diffusion_guidance_scale > 1.0,
            adv_clip_max=args.diffusion_adv_clip_max,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        new_pred = actor.noise_pred(batch.latents, batch.timesteps, batch.pos_cond, batch.neg_cond)
        ref_pred = None
        if args.diffusion_kl_beta > 0:
            ref_pred = actor.lora_base_noise_pred(batch.latents, batch.timesteps, batch.pos_cond, batch.neg_cond)
        return flow_grpo_loss(
            batch,
            new_pred=new_pred,
            ref_pred=ref_pred,
            noise_level=args.diffusion_noise_level,
            clip_range=args.eps_clip,
            kl_beta=args.diffusion_kl_beta,
            metrics=metrics,
        )


def _latent_positions(traj) -> dict[int, int]:
    if traj.latent_step_indices is None:
        return {step: step for step in range(int(traj.latents.shape[0]))}
    return {int(step): pos for pos, step in enumerate(traj.latent_step_indices.tolist())}
