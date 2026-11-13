"""Diffusion train data: GRPO advantages and one row per image of the `--loss-type` algorithm's train pairs.

Registered through `--custom-convert-samples-to-train-data-path`; it also logs the rollout's reward metrics, while
Miles' default rollout and perf panels log the rest.
"""

import logging
from typing import Any

import torch

from miles.utils.args.custom_view import compute_custom_function_config
from miles.utils.diffusion.algorithms import get_algorithm
from miles.utils.diffusion.metrics import log_metrics
from miles.utils.function_registry import load_function
from miles.utils.metric_utils import compute_rollout_step, namespace_metrics
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def convert_samples_to_train_data(args, samples: list[Sample]) -> dict[str, Any]:
    """One row per image, each holding that image's train pairs, so the DP split keeps an image on one rank.

    The only `post_process_rewards` call of a rollout: its rewards are logged here and trained on, never recomputed.
    """
    raw_rewards, advantages = post_process_rewards(args, samples)
    _log_reward_metrics(args, samples, raw_rewards=raw_rewards, advantages=advantages)
    algorithm = get_algorithm(args.loss_type)
    return {
        "diffusion_train_pairs": [
            algorithm.build_train_pairs(args, sample, advantage=advantage)
            for sample, advantage in zip(samples, advantages, strict=True)
        ],
        "sample_indices": [sample.index for sample in samples],
        "raw_reward": raw_rewards,
    }


def post_process_rewards(args, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """`(raw, normalized)` rewards behind the text runs' boundary: `--custom-reward-post-process-path` replaces both,
    `--reward-key` picks the raw value, and `--disable-rewards-normalization` skips Flow-GRPO's normalization."""
    if (path := args.custom_reward_post_process_path) is not None:
        return load_function(path)(compute_custom_function_config(args, path), samples)
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if not args.rewards_normalization:
        return raw_rewards, raw_rewards
    return raw_rewards, compute_grpo_advantages(args, raw_rewards=raw_rewards)


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


def _log_reward_metrics(args, samples: list[Sample], *, raw_rewards: list[float], advantages: list[float]) -> None:
    (rollout_id,) = {sample.diffusion.rollout_id for sample in samples}
    assert rollout_id is not None, "DiffusionRolloutFn stamps each sample with its rollout id"
    raw = torch.tensor(raw_rewards, dtype=torch.float)
    grouped = raw.view(-1, args.n_samples_per_prompt)
    _log_rollout_step(
        args,
        rollout_id,
        {
            **_stats(raw, prefix="rollout/reward/raw_"),
            **_stats(torch.tensor(advantages, dtype=torch.float), prefix="rollout/reward/norm_"),
            "rollout/reward/group_mean_avg": float(grouped.mean(dim=-1).mean()),
            "rollout/reward/group_std_avg": float(grouped.std(dim=-1, unbiased=False).mean()),
        },
    )


def _log_rollout_step(args, rollout_id: int, metrics: dict[str, float]) -> None:
    log_dict, step_key = namespace_metrics(
        metrics, trainer_model_id=None, step_name="rollout/step", step=compute_rollout_step(args, rollout_id)
    )
    logger.info(f"rollout {rollout_id}: {log_dict}")
    log_metrics(args, log_dict, step_key=step_key)


def _stats(values: torch.Tensor, *, prefix: str) -> dict[str, float]:
    # Not `compute_statistics`: the e2e gate records float32 means, torch's lower median and the population std.
    return {
        f"{prefix}mean": float(values.mean()),
        f"{prefix}std": float(values.std(unbiased=False)),
        f"{prefix}min": float(values.min()),
        f"{prefix}max": float(values.max()),
        f"{prefix}median": float(values.median()),
        f"{prefix}num_samples": float(values.numel()),
    }
