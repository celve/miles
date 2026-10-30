from types import SimpleNamespace

import pytest
import torch

from miles.diffusion.conversion import build_train_pairs, compute_grpo_advantages, convert_samples_to_train_data
from miles.diffusion.types import DenoisingEnv, DiffusionRollout, DiTTrajectory
from miles.ray.rollout.train_data_conversion import split_train_data_by_dp
from miles.utils.types import Sample


class _Args:
    n_samples_per_prompt = 2
    grpo_std_normalization = True
    diffusion_globalize_reward_std = True


def _sample(index, *, reward, steps, latent_step_indices=None, num_latents=4):
    latents = torch.arange(num_latents, dtype=torch.float32).view(num_latents, 1) + 10 * index
    rollout = DiffusionRollout(
        rollout_log_probs=torch.arange(num_latents - 1, dtype=torch.float32) - index,
        denoising_env=DenoisingEnv(),
        dit_trajectory=DiTTrajectory(
            latents=latents,
            timesteps=torch.tensor([1000.0, 750.0, 500.0, 0.0]),
            sigmas=torch.tensor([1.0, 0.75, 0.5, 0.0]),
            latent_step_indices=latent_step_indices,
        ),
        sde_step_indices=steps,
    )
    return Sample(index=index, reward=reward, diffusion=rollout)


def test_one_pair_per_sde_step_pairs_x_t_with_x_t_plus_1():
    pairs = build_train_pairs(_sample(0, reward=1.0, steps=[0, 2]), advantage=0.5)

    assert [(p["latent"].item(), p["next_latent"].item()) for p in pairs] == [(0.0, 1.0), (2.0, 3.0)]
    assert [(p["sigma"].item(), p["next_sigma"].item()) for p in pairs] == [(1.0, 0.75), (0.5, 0.0)]
    assert [p["log_prob_old"].item() for p in pairs] == [0.0, 2.0]
    assert all(p["sigma_max"].item() == 0.75 and p["advantage"] == 0.5 for p in pairs)


def test_a_windowed_trajectory_is_indexed_by_its_echoed_step_numbers():
    sample = _sample(0, reward=1.0, steps=[1], latent_step_indices=torch.tensor([1, 2]), num_latents=2)
    sample.diffusion.rollout_log_probs = torch.tensor([7.0, 8.0, 9.0])

    (pair,) = build_train_pairs(sample, advantage=0.0)

    assert (pair["latent"].item(), pair["next_latent"].item(), pair["log_prob_old"].item()) == (0.0, 1.0, 8.0)


def test_grpo_advantages_center_per_prompt_and_scale_by_the_rollout_std():
    raw = [1.0, 0.0, 0.5, 0.5]

    advantages = compute_grpo_advantages(_Args, raw_rewards=raw)

    std = torch.tensor(raw).std().item()
    torch.testing.assert_close(torch.tensor(advantages), torch.tensor([0.5, -0.5, 0.0, 0.0]) / (std + 1e-4))


def test_rows_are_images_and_the_dp_split_strides_them():
    samples = [_sample(i, reward=float(i % 2), steps=[0, 1, 2]) for i in range(4)]

    data = convert_samples_to_train_data(_Args, samples)
    shards = split_train_data_by_dp(_Args, data, train_parallel_config=SimpleNamespace(dp_size=2))

    assert [len(row) for row in data["diffusion_train_pairs"]] == [3, 3, 3, 3]
    assert [shard["sample_indices"] for shard in shards] == [[0, 2], [1, 3]]
    assert [row[0]["sample_index"] for row in shards[1]["diffusion_train_pairs"]] == [1, 3]


def test_the_split_rejects_diffusion_rows_with_a_foreign_schema():
    with pytest.raises(AssertionError, match="keys"):
        split_train_data_by_dp(
            _Args,
            {"diffusion_train_pairs": [[]], "sample_indices": [0], "tokens": [[1]]},
            train_parallel_config=SimpleNamespace(dp_size=1),
        )
