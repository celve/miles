"""M2PO (https://arxiv.org/abs/2510.01161, §5.2): PPO's ratio clip replaced by a second-moment mask."""

import torch

from miles.backends.training_utils.loss.hub.math_utils import (
    _safe_clamp_log_ratio,
    _safe_exp_neg_ppo_kl,
    compute_policy_loss,
)


def compute_m2po_mask(log_ratio: torch.Tensor, advantages: torch.Tensor, tau: float) -> torch.Tensor:
    """Algorithm 1: mask the trust-region tokens with the largest (log r)² until the rest average at most tau."""
    trust_region = ((advantages > 0) & (log_ratio > 0)) | ((advantages < 0) & (log_ratio < 0))
    sort_key = torch.where(trust_region, log_ratio.square(), torch.full_like(log_ratio, -1.0))
    sorted_key, order = sort_key.sort(descending=True)
    sorted_sq = sorted_key.clamp(min=0)
    position = torch.arange(sort_key.numel(), device=sort_key.device)
    num_trust_region = trust_region.sum()
    kept_mean = (sorted_sq.sum() - sorted_sq.cumsum(0) + sorted_sq) / (num_trust_region - position).clamp(min=1)
    masked_sorted = (position < num_trust_region) & (kept_mean > tau)
    return torch.ones_like(log_ratio).scatter(0, order, (~masked_sorted).to(log_ratio.dtype))


def compute_m2po_policy_loss(
    ppo_kl: torch.Tensor,
    behaviour_log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    *,
    tau: float,
    eps_clip: float,
    eps_clip_high: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Unclipped `−(π_θ/π_old)·A` under the mask of r = π_θ/π_behaviour, the masked share, and diagnostics."""
    behaviour_log_ratio = _safe_clamp_log_ratio(behaviour_log_ratio)
    keep = compute_m2po_mask(behaviour_log_ratio, advantages, tau)
    pg_losses = -_safe_exp_neg_ppo_kl(ppo_kl) * advantages * keep
    _, ppo_clipfrac = compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high)
    metrics = {"m2po_ppo_clipfrac": ppo_clipfrac.detach(), "m2po_log_ratio_sq": behaviour_log_ratio.square()}
    return pg_losses, 1 - keep, metrics
