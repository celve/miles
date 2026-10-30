import pytest
import torch

from miles.diffusion.train.loss import PairBatch, flow_grpo_loss
from miles.diffusion.train.metric_buffer import MetricBuffer
from miles.diffusion.train.schedule import tile_train_pairs


def _rows(num_images, num_steps):
    return [[(image, step) for step in range(num_steps)] for image in range(num_images)]


def test_tiles_take_the_step_chunk_outermost_and_flatten_image_major():
    steps = tile_train_pairs(_rows(4, 4), num_optim_steps=2, sample_microbatch=2, tstep_microbatch=2)

    assert len(steps) == 2
    assert steps[1] == [
        [(2, 0), (2, 1), (3, 0), (3, 1)],
        [(2, 2), (2, 3), (3, 2), (3, 3)],
    ]


def test_the_recipe_schedule_is_two_steps_of_four_80_pair_tiles():
    steps = tile_train_pairs(_rows(64, 10), num_optim_steps=2, sample_microbatch=16, tstep_microbatch=5)

    assert [[len(tile) for tile in step] for step in steps] == [[80] * 4] * 2
    assert sorted(pair for step in steps for tile in step for pair in tile) == sorted(
        pair for row in _rows(64, 10) for pair in row
    )


def test_ragged_tiles_are_rejected():
    with pytest.raises(AssertionError, match="uniform"):
        tile_train_pairs(_rows(6, 10), num_optim_steps=1, sample_microbatch=4, tstep_microbatch=5)


def _batch(advantage, log_prob_old):
    n = len(advantage)
    return PairBatch(
        latents=torch.zeros(n, 2),
        next_latents=torch.zeros(n, 2),
        timesteps=torch.full((n,), 500.0),
        sigmas=torch.full((n,), 0.5),
        next_sigmas=torch.full((n,), 0.25),
        sigma_max=torch.full((n,), 0.75),
        log_prob_old=torch.tensor(log_prob_old),
        advantage=torch.tensor(advantage),
        pos_cond={},
        neg_cond=None,
    )


def _loss(batch, *, new_pred, ref_pred=None, kl_beta=0.0, clip_range=1e-4):
    metrics = MetricBuffer(group=None, device=torch.device("cpu"))
    loss = flow_grpo_loss(
        batch,
        new_pred=new_pred,
        ref_pred=ref_pred,
        noise_level=0.7,
        clip_range=clip_range,
        kl_beta=kl_beta,
        metrics=metrics,
    )
    return loss, metrics


def _on_policy_log_prob(pred):
    from miles.diffusion.train.sde import sde_step_log_prob

    batch = _batch([0.0], [0.0])
    log_prob, _, _ = sde_step_log_prob(
        pred,
        batch.latents,
        prev_sample=batch.next_latents,
        noise_level=0.7,
        sigmas=batch.sigmas,
        next_sigmas=batch.next_sigmas,
        sigma_max=batch.sigma_max,
    )
    return log_prob.item()


def test_on_policy_the_loss_is_minus_the_advantage_sum():
    pred = torch.ones(2, 2)
    old = _on_policy_log_prob(pred[:1])
    loss, metrics = _loss(_batch([1.0, -3.0], [old, old]), new_pred=pred)

    torch.testing.assert_close(loss, torch.tensor(2.0))
    assert metrics._counts["loss"] == 2.0


def test_the_ratio_is_clipped_against_the_advantage():
    pred = torch.ones(1, 2)
    old = _on_policy_log_prob(pred) - 0.5
    loss, _ = _loss(_batch([1.0], [old]), new_pred=pred, clip_range=0.1)

    # ratio e^0.5 > 1.1: a positive advantage takes the clipped (larger) loss -1.1
    torch.testing.assert_close(loss, torch.tensor(-1.1))


def test_the_kl_term_vanishes_against_an_identical_reference():
    pred = torch.ones(1, 2)
    old = _on_policy_log_prob(pred)
    with_kl, _ = _loss(_batch([1.0], [old]), new_pred=pred, ref_pred=pred.clone(), kl_beta=0.04)
    with_shifted_ref, _ = _loss(_batch([1.0], [old]), new_pred=pred, ref_pred=pred + 1, kl_beta=0.04)

    torch.testing.assert_close(with_kl, torch.tensor(-1.0))
    assert with_shifted_ref > with_kl
