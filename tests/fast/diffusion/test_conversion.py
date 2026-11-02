from types import SimpleNamespace

import pytest
import torch

from miles.ray.rollout.train_data_conversion import split_train_data_by_dp
from miles.rollout.diffusion_rollout import conversion
from miles.rollout.diffusion_rollout.conversion import compute_grpo_advantages, convert_samples_to_train_data
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import ImmutableNamespace
from miles.utils.diffusion.algorithms.flow_grpo import FlowGRPO
from miles.utils.diffusion.types import DenoisingEnv, DiffusionRollout, DiTTrajectory
from miles.utils.types import Sample


class _Args:
    n_samples_per_prompt = 2
    grpo_std_normalization = True
    diffusion_globalize_reward_std = True
    rewards_normalization = True
    reward_key = None
    custom_reward_post_process_path = None
    advantage_estimator = "grpo"
    wandb_always_use_train_step = False
    rollout_batch_size = 2
    global_batch_size = 2
    loss_type = "policy_loss"


def _fn_args(**overrides):
    fields = {k: v for k, v in vars(_Args).items() if not k.startswith("_")}
    return ImmutableNamespace(**(fields | overrides))


POST_PROCESS_CALLS = []


def double_raw_rewards(args, samples):
    POST_PROCESS_CALLS.append(len(samples))
    raw = [sample.reward for sample in samples]
    return raw, [2 * r for r in raw]


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
        rollout_id=3,
    )
    return Sample(index=index, reward=reward, diffusion=rollout)


@pytest.fixture(autouse=True)
def logged(monkeypatch):
    records = []
    monkeypatch.setattr(conversion, "log_metrics", lambda args, metrics, *, step_key: records.append(metrics))
    return records


def test_one_pair_per_sde_step_pairs_x_t_with_x_t_plus_1():
    pairs = FlowGRPO().build_train_pairs(_fn_args(), _sample(0, reward=1.0, steps=[0, 2]), advantage=0.5)

    assert [(p["latent"].item(), p["next_latent"].item()) for p in pairs] == [(0.0, 1.0), (2.0, 3.0)]
    assert [(p["sigma"].item(), p["next_sigma"].item()) for p in pairs] == [(1.0, 0.75), (0.5, 0.0)]
    assert [p["log_prob_old"].item() for p in pairs] == [0.0, 2.0]
    assert all(p["sigma_max"].item() == 0.75 and p["advantage"] == 0.5 for p in pairs)


def test_a_windowed_trajectory_is_indexed_by_its_echoed_step_numbers():
    sample = _sample(0, reward=1.0, steps=[1], latent_step_indices=torch.tensor([1, 2]), num_latents=2)
    sample.diffusion.rollout_log_probs = torch.tensor([7.0, 8.0, 9.0])

    (pair,) = FlowGRPO().build_train_pairs(_fn_args(), sample, advantage=0.0)

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


def _rewarded(rewards):
    return [_sample(i, reward=r, steps=[0]) for i, r in enumerate(rewards)]


def _advantages(data):
    return [row[0]["advantage"] for row in data["diffusion_train_pairs"]]


def test_reward_key_selects_the_raw_reward():
    samples = _rewarded([{"ocr": 1.0, "aux": 9.0}, {"ocr": 0.0, "aux": 9.0}])

    data = convert_samples_to_train_data(_fn_args(reward_key="ocr", grpo_std_normalization=False), samples)

    assert data["raw_reward"] == [1.0, 0.0]
    assert _advantages(data) == [0.5, -0.5]


def test_the_post_process_hook_replaces_normalization_once_per_conversion():
    POST_PROCESS_CALLS.clear()
    hook = CustomFunctionConfig(path=f"{__name__}.double_raw_rewards")

    data = convert_samples_to_train_data(_fn_args(custom_reward_post_process_path=hook), _rewarded([1.0, 0.25]))

    assert POST_PROCESS_CALLS == [2]
    assert data["raw_reward"] == [1.0, 0.25] and _advantages(data) == [2.0, 0.5]


def test_disabled_normalization_trains_on_the_raw_rewards():
    data = convert_samples_to_train_data(_fn_args(rewards_normalization=False), _rewarded([1.0, 0.25]))

    assert _advantages(data) == [1.0, 0.25]


def stateful_rewards(args, samples):
    POST_PROCESS_CALLS.append(len(samples))
    raw = [sample.reward for sample in samples]
    return raw, [len(POST_PROCESS_CALLS) * r for r in raw]


def test_the_converter_logs_the_rewards_it_trains_on_from_one_hook_call(logged):
    POST_PROCESS_CALLS.clear()
    args = _fn_args(custom_reward_post_process_path=CustomFunctionConfig(path=f"{__name__}.stateful_rewards"))
    samples = _rewarded([1.0, 0.0])

    data = convert_samples_to_train_data(args, samples)

    assert POST_PROCESS_CALLS == [2]
    (reward_record,) = [record for record in logged if "rollout/reward/norm_mean" in record]
    assert reward_record["rollout/step"] == 3
    assert reward_record["rollout/reward/norm_mean"] == sum(_advantages(data)) / 2
