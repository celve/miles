"""DiffusionNFT objective over a micro-batch of (x0, sigma) pairs: a dual-policy x0 regression against the EMA.

Ported from radixark/miles_diffusion at ecb1363e (loss_hub/nft.py).
"""

from dataclasses import dataclass

import torch

from miles.utils.diffusion.algorithms.nft import stable_hash


def microbatch_loss(actor, pairs: list[dict], *, rollout_id: int, microbatch_id: int, metrics) -> torch.Tensor:
    args = actor.args
    device = torch.device("cuda", torch.cuda.current_device())
    batch = collate_nft_pairs(pairs, config=actor.pipeline_config, device=device)
    generator = torch.Generator(device=device).manual_seed(
        stable_hash("nft_corrupt", int(args.backend.seed), rollout_id, microbatch_id, actor.dp_rank)
    )
    eps = torch.randn(batch.x0.shape, device=device, dtype=batch.x0.dtype, generator=generator)
    xt = corrupt(batch.x0, batch.t, eps)
    model_t = actor.pipeline_config.process_sigma_as_timesteps_input(batch.t)
    component = actor.component_for(model_t)
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
