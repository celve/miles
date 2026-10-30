"""Flow-GRPO train data: GRPO advantages and one (x_t, x_t+1) train pair per SDE step of each image.

Registered through `--custom-convert-samples-to-train-data-path` and `--custom-rollout-log-function-path`.
"""

import logging
from typing import Any

import torch

from miles.diffusion.metrics import log_metrics
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def convert_samples_to_train_data(args, samples: list[Sample]) -> dict[str, Any]:
    """One row per image, each holding that image's train pairs, so the DP split keeps an image on one rank."""
    advantages = compute_grpo_advantages(args, raw_rewards=[sample.reward for sample in samples])
    return {
        "diffusion_train_pairs": [
            build_train_pairs(sample, advantage=advantage)
            for sample, advantage in zip(samples, advantages, strict=True)
        ],
        "sample_indices": [sample.index for sample in samples],
        "raw_reward": [sample.reward for sample in samples],
    }


DIFFUSION_TRAIN_DATA_KEYS = frozenset({"diffusion_train_pairs", "sample_indices", "raw_reward"})


def split_train_data_by_dp(data: dict[str, Any], *, dp_size: int) -> list[dict[str, Any]]:
    """Image i goes to DP rank i % dp_size, as miles_diffusion's stride split does."""
    assert data.keys() == DIFFUSION_TRAIN_DATA_KEYS, f"diffusion train data has keys {sorted(data)}"
    num_rows = len(data["sample_indices"])
    assert all(len(data[key]) == num_rows for key in DIFFUSION_TRAIN_DATA_KEYS), "ragged diffusion train data"
    assert num_rows % dp_size == 0, f"{num_rows} images do not split evenly over {dp_size} DP ranks"
    return [{key: data[key][rank::dp_size] for key in DIFFUSION_TRAIN_DATA_KEYS} for rank in range(dp_size)]


def compute_grpo_advantages(args, *, raw_rewards: list[float]) -> list[float]:
    """Per-prompt mean baseline over `--n-samples-per-prompt` blocks, divided by the group or rollout std."""
    flat = torch.tensor(raw_rewards, dtype=torch.float)
    grouped = flat.view(-1, args.n_samples_per_prompt)
    centered = grouped - grouped.mean(dim=-1, keepdim=True)
    if args.grpo_std_normalization:
        std = flat.std() if args.diffusion_globalize_reward_std else grouped.std(dim=-1, keepdim=True)
        # flow_grpo's `+ 1e-4`
        centered = centered / (std + 1e-4)
    return centered.flatten().tolist()


def build_train_pairs(sample: Sample, *, advantage: float) -> list[dict[str, Any]]:
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


def _latent_positions(traj) -> dict[int, int]:
    if traj.latent_step_indices is None:
        return {step: step for step in range(int(traj.latents.shape[0]))}
    return {int(step): pos for pos, step in enumerate(traj.latent_step_indices.tolist())}


def log_rollout_data(rollout_id: int, args, samples: list[Sample], rollout_extra_metrics, rollout_time: float) -> bool:
    raw = torch.tensor([sample.reward for sample in samples], dtype=torch.float)
    advantages = torch.tensor(compute_grpo_advantages(args, raw_rewards=raw.tolist()), dtype=torch.float)
    grouped = raw.view(-1, args.n_samples_per_prompt)
    metrics = {
        **(rollout_extra_metrics or {}),
        **_stats(raw, prefix="rollout/reward/raw_"),
        **_stats(advantages, prefix="rollout/reward/norm_"),
        "rollout/reward/group_mean_avg": float(grouped.mean(dim=-1).mean()),
        "rollout/reward/group_std_avg": float(grouped.std(dim=-1, unbiased=False).mean()),
        "rollout/zero_std_groups": float((grouped.std(dim=-1) == 0).sum()),
        "perf/rollout_time": rollout_time,
        "rollout/step": rollout_id,
    }
    logger.info(f"rollout {rollout_id}: {metrics}")
    log_metrics(args, metrics, step_key="rollout/step")
    # The default text-rollout metrics read token fields these samples do not have.
    return True


def _stats(values: torch.Tensor, *, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}mean": float(values.mean()),
        f"{prefix}std": float(values.std(unbiased=False)),
        f"{prefix}min": float(values.min()),
        f"{prefix}max": float(values.max()),
        f"{prefix}median": float(values.median()),
        f"{prefix}num_samples": float(values.numel()),
    }
