"""Flow-GRPO objective over a micro-batch of SDE train pairs: PPO-clipped ratio plus optional KL to a reference."""

from dataclasses import dataclass

import torch

from miles.diffusion.train.metric_buffer import MetricBuffer
from miles.diffusion.train.pipeline_config import TrainPipelineConfig
from miles.diffusion.train.sde import sde_step_log_prob


@dataclass
class PairBatch:
    latents: torch.Tensor
    next_latents: torch.Tensor
    timesteps: torch.Tensor
    sigmas: torch.Tensor
    next_sigmas: torch.Tensor
    sigma_max: torch.Tensor
    log_prob_old: torch.Tensor
    advantage: torch.Tensor
    pos_cond: dict
    neg_cond: dict | None

    def __len__(self) -> int:
        return self.latents.shape[0]


def collate_pairs(
    pairs: list[dict], *, config: TrainPipelineConfig, use_cfg: bool, adv_clip_max: float, device: torch.device
) -> PairBatch:
    def stack(key: str) -> torch.Tensor:
        return torch.stack([pair[key] for pair in pairs]).to(device=device, dtype=torch.float32)

    advantage = torch.tensor([pair["advantage"] for pair in pairs], device=device, dtype=torch.float32)
    envs = [pair["denoising_env"] for pair in pairs]
    return PairBatch(
        latents=stack("latent"),
        next_latents=stack("next_latent"),
        timesteps=stack("timestep"),
        sigmas=stack("sigma"),
        next_sigmas=stack("next_sigma"),
        sigma_max=stack("sigma_max"),
        log_prob_old=stack("log_prob_old"),
        advantage=torch.clamp(advantage, -adv_clip_max, adv_clip_max),
        pos_cond=config.collate_cond([config.prepare_cond_kwargs(env.pos_cond_kwargs, device) for env in envs], device),
        neg_cond=(
            config.collate_cond([config.prepare_cond_kwargs(env.neg_cond_kwargs, device) for env in envs], device)
            if use_cfg
            else None
        ),
    )


def flow_grpo_loss(
    batch: PairBatch,
    *,
    new_pred: torch.Tensor,
    ref_pred: torch.Tensor | None,
    noise_level: float,
    clip_range: float,
    kl_beta: float,
    metrics: MetricBuffer,
) -> torch.Tensor:
    """Summed loss of the micro-batch; the caller normalizes by the optimizer step's pair count."""
    sigmas = dict(sigmas=batch.sigmas, next_sigmas=batch.next_sigmas, sigma_max=batch.sigma_max)
    log_prob_new, prev_mean_new, std_dev_t = sde_step_log_prob(
        new_pred.float(), batch.latents, prev_sample=batch.next_latents, noise_level=noise_level, **sigmas
    )
    ratio = torch.exp(log_prob_new - batch.log_prob_old)
    unclipped = -batch.advantage * ratio
    clipped = -batch.advantage * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    per_pair_loss = torch.maximum(unclipped, clipped)
    loss_sum = per_pair_loss.sum()

    kl_sum = loss_sum.new_zeros(())
    if kl_beta > 0:
        assert ref_pred is not None, "Flow-GRPO KL needs a reference forward"
        _, prev_mean_ref, _ = sde_step_log_prob(
            ref_pred.float(), batch.latents, prev_sample=batch.next_latents, noise_level=noise_level, **sigmas
        )
        kl_per_pair = ((prev_mean_new - prev_mean_ref) ** 2).mean(
            dim=tuple(range(1, prev_mean_new.ndim)), keepdim=True
        ) / (2 * std_dev_t**2)
        kl_sum = kl_per_pair.sum()
        loss_sum = loss_sum + kl_beta * kl_sum

    _record_metrics(metrics, batch, log_prob_new, ratio, per_pair_loss, loss_sum, kl_sum, clip_range=clip_range)
    return loss_sum


@torch.no_grad()
def _record_metrics(metrics, batch, log_prob_new, ratio, per_pair_loss, loss_sum, kl_sum, *, clip_range) -> None:
    n = len(batch)
    log_prob_old = batch.log_prob_old
    metrics.emit_mean("loss", total=loss_sum, count=n)
    metrics.emit_mean("policy_loss", total=per_pair_loss.sum(), count=n)
    metrics.emit_mean("kl_loss", total=kl_sum, count=n)
    metrics.emit_mean("loss_abs_mean", total=per_pair_loss.abs().sum(), count=n)
    metrics.emit_mean("adv_abs_mean", total=batch.advantage.abs().sum(), count=n)
    metrics.emit_mean("ratio_abs_minus_1", total=(ratio - 1.0).abs().sum(), count=n)
    metrics.emit_mean("approx_kl", total=0.5 * ((log_prob_new - log_prob_old) ** 2).sum(), count=n)
    metrics.emit_mean("clipfrac", total=(torch.abs(ratio - 1.0) > clip_range).float().sum(), count=n)
    metrics.emit_mean("log_prob_new_idx_0", total=log_prob_new[0], count=1)
    metrics.emit_mean("log_prob_old_idx_0", total=log_prob_old[0], count=1)
    metrics.emit_mean("log_prob_mean_abs_diff", total=torch.abs(log_prob_new - log_prob_old).sum(), count=n)
