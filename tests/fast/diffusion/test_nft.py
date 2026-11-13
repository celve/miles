from argparse import Namespace

import pytest
import torch

from miles.backends.fsdp_utils.diffusion.losses import LOSSES
from miles.backends.fsdp_utils.diffusion.losses.nft import NFTBatch, corrupt, nft_loss
from miles.backends.fsdp_utils.diffusion.metric_buffer import MetricBuffer
from miles.backends.fsdp_utils.diffusion.pipeline_config import get_train_pipeline_config
from miles.rollout.diffusion_rollout.rollout_fn import build_sampling_params
from miles.rollout.diffusion_rollout.step_strategy import ode_and_return_last
from miles.utils.diffusion.algorithms import ALGORITHMS, get_algorithm
from miles.utils.diffusion.algorithms.nft import DiffusionNFT, drop_final_steps, drop_random_steps, stable_hash
from miles.utils.diffusion.types import DenoisingEnv, DiffusionRollout, DiTTrajectory
from miles.utils.types import Sample

SIGMAS = torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0])


def _args(**overrides):
    fields = dict(
        seed=1234,
        diffusion_nft_timestep_strategy_path="miles.utils.diffusion.algorithms.nft.drop_random_steps",
        diffusion_nft_num_dropped_timesteps=1,
    )
    return Namespace(**(fields | overrides))


def _sample(index, *, returned_steps=(4,)):
    traj = DiTTrajectory(
        latents=torch.full((1, 2, 2), float(index)),
        sigmas=SIGMAS.clone(),
        latent_step_indices=torch.tensor(returned_steps),
    )
    return Sample(index=index, diffusion=DiffusionRollout(denoising_env=DenoisingEnv(), dit_trajectory=traj))


def test_nft_is_selected_by_loss_type_and_asks_the_engine_for_the_final_latent_only():
    assert isinstance(get_algorithm("nft"), DiffusionNFT)
    sde_steps, returned = ode_and_return_last(None, None, 10, seed=3)

    assert DiffusionNFT().rollout_request(sde_steps, returned) == {
        "rollout_sde_step_indices": None,
        "rollout_return_step_indices": [10],
    }


@pytest.mark.parametrize("sde_type,no_const", [("ode", True), ("sde", False)])
def test_an_ode_request_asks_for_log_probs_without_constants(sde_type, no_const):
    args = Namespace(
        diffusion_width=512,
        diffusion_height=512,
        diffusion_num_steps=10,
        diffusion_guidance_scale=1.0,
        diffusion_sde_type=sde_type,
        diffusion_noise_level=0.0,
        diffusion_quality="exact",
        diffusion_negative_prompt=None,
        diffusion_output_num_frames=1,
        diffusion_guidance_scale_2=None,
    )

    params = build_sampling_params(args)

    assert params["rollout_sde_type"] == sde_type and params["rollout_log_prob_no_const"] is no_const


def test_each_image_trains_on_its_sampling_sigmas_but_the_dropped_ones_against_its_clean_latent():
    pairs = DiffusionNFT().build_train_pairs(_args(), _sample(7), advantage=0.25)

    assert len(pairs) == 3 and {p["num_timesteps"] for p in pairs} == {3}
    assert {round(p["timestep"], 6) for p in pairs} < {1.0, 0.9, 0.7, 0.4}
    assert all(p["x0"].eq(7.0).all() and p["advantage"] == 0.25 and p["sample_index"] == 7 for p in pairs)


def test_the_sigma_draw_is_a_function_of_the_seed_and_the_sample_index():
    def order(index, seed=1234):
        return [p["timestep"] for p in DiffusionNFT().build_train_pairs(_args(seed=seed), _sample(index), advantage=0)]

    assert order(3) == order(3)
    assert len({tuple(order(index)) for index in range(8)}) > 1
    assert (
        stable_hash("nft_sigma_shuffle", 1234, 3)
        == stable_hash("nft_sigma_shuffle", 1234, 3)
        != stable_hash("nft_sigma_shuffle", 1235, 3)
    )


def test_a_trajectory_that_stops_before_the_clean_latent_is_rejected():
    with pytest.raises(AssertionError, match="ode_and_return_last"):
        DiffusionNFT().build_train_pairs(_args(), _sample(0, returned_steps=(2, 3)), advantage=0)


def test_the_timestep_strategies_drop_random_or_final_steps():
    generator = torch.Generator().manual_seed(0)
    assert sorted(drop_random_steps(_args(diffusion_nft_num_dropped_timesteps=2), 6, generator)) != list(range(6))
    assert len(drop_random_steps(_args(diffusion_nft_num_dropped_timesteps=2), 6, generator)) == 4
    assert drop_final_steps(_args(diffusion_nft_num_dropped_timesteps=2), 6, generator) == [0, 1, 2, 3]


def test_corrupt_is_the_linear_flow_and_sd3_takes_sigma_times_1000():
    x0, eps = torch.ones(2, 3), torch.zeros(2, 3)

    assert corrupt(x0, torch.tensor([0.25, 1.0]), eps).tolist() == [[0.75] * 3, [0.0] * 3]
    assert get_train_pipeline_config("sd3").process_sigma_as_timesteps_input(torch.tensor([0.5])).item() == 500.0


def _loss(advantage, *, new_pred, old_pred, use_adaptive=False, beta=1.0):
    x0 = torch.zeros(1, 2)
    batch = NFTBatch(x0=x0, t=torch.tensor([0.5]), advantage=torch.tensor([advantage]), pos_cond={})
    metrics = MetricBuffer(group=None, device=torch.device("cpu"))
    loss = nft_loss(
        batch,
        xt=torch.ones(1, 2),
        new_pred=new_pred,
        old_pred=old_pred,
        beta=beta,
        adv_clip_max=5.0,
        use_adaptive=use_adaptive,
        num_timesteps=4,
        metrics=metrics,
    )
    return loss, metrics


def test_the_advantage_picks_between_the_positive_and_the_mirrored_policy():
    new, old = torch.full((1, 2), 2.0), torch.full((1, 2), 1.0)
    # xt - t * v against x0 = 0: positive v = new gives 0, mirrored 2 old - new = 0 gives 1.
    best, _ = _loss(5.0, new_pred=new, old_pred=old)
    worst, _ = _loss(-5.0, new_pred=new, old_pred=old)
    neutral, _ = _loss(0.0, new_pred=new, old_pred=old)

    assert (best.item(), worst.item(), neutral.item()) == (0.0, 5.0, 2.5)


def test_the_adaptive_weight_divides_each_branch_by_its_detached_mean_error():
    new = torch.full((1, 2), 2.0, requires_grad=True)
    old = torch.full((1, 2), 0.0)
    loss, _ = _loss(-5.0, new_pred=new, old_pred=old, use_adaptive=True)
    loss.backward()

    # Mirrored x0 is 1 - 0.5 * (-2) = 2: squared error 4 over the mean error 2, times adv_clip_max.
    assert loss.item() == pytest.approx(10.0)
    assert new.grad.tolist() == [[pytest.approx(2.5), pytest.approx(2.5)]]


def test_the_loss_metrics_scale_by_the_timesteps_per_image():
    new, old = torch.full((1, 2), 2.0), torch.full((1, 2), 1.0)
    loss, metrics = _loss(0.0, new_pred=new, old_pred=old)

    assert metrics._sums["nft_loss"].item() == pytest.approx(4 * loss.item())
    assert metrics._sums["nft_r_mean"].item() == 0.5 and metrics._sums["nft_t_mean"].item() == 0.5
    assert metrics._sums["nft_num_timesteps"].item() == 4 and metrics._counts["nft_num_timesteps"] == 1


def test_the_logged_mixing_weight_is_the_fp32_one_under_an_fp16_forward():
    new, old = torch.full((1, 2), 2.0, dtype=torch.float16), torch.full((1, 2), 1.0, dtype=torch.float16)
    _, metrics = _loss(0.3, new_pred=new, old_pred=old)

    assert metrics._sums["nft_r_mean"].item() == torch.tensor(0.3 / 5.0 / 2.0 + 0.5, dtype=torch.float32).item()


def test_every_diffusion_algorithm_has_a_trainer_loss_of_the_same_loss_type():
    assert LOSSES.keys() == ALGORITHMS.keys()
