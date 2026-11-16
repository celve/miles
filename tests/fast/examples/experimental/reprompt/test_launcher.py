import shlex

import pytest
from examples.experimental.reprompt.reward import RePromptLogConfig, RePromptRewardConfig
from examples.experimental.reprompt.run_qwen25_3b_reprompt import ScriptArgs, _train_args
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from transformers import Qwen2Config


@pytest.fixture
def config(monkeypatch, tmp_path):
    """The launcher's argv, parsed by Miles against a stand-in Qwen2.5 config."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    Qwen2Config(
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        intermediate_size=256,
        tie_word_embeddings=True,
    ).save_pretrained(tmp_path / "Qwen2.5-3B-Instruct")
    (tmp_path / "Qwen2.5-3B-Instruct_torch_dist").mkdir()
    args = ScriptArgs(run_id="test", model_dir=str(tmp_path), reward_urls="http://a:1,http://b:2")
    return parse_megatron_test_config("--ffn-hidden-size", "256", *shlex.split(_train_args(args)))


def test_the_grpo_settings_are_reprompts(config):
    assert (config.rollout_batch_size, config.n_samples_per_prompt, config.global_batch_size) == (32, 4, 128)
    assert config.num_epoch == 3
    assert (config.rollout_temperature, config.rollout_max_response_len) == (0.9, 256)
    assert (config.rollout_top_p, config.rollout_top_k) == (1.0, -1)
    assert config.rollout_skip_special_tokens
    assert config.advantage_estimator == "grpo"
    assert (config.use_kl_loss, config.kl_loss_coef, config.kl_loss_type) == (True, 0.04, "k3")
    assert config.eps_clip == 0.2 and config.eps_clip_high == 0.2
    megatron = config.raw_megatron.base_args
    assert (megatron["lr"], megatron["min_lr"], megatron["lr_decay_style"]) == (2e-6, 0.0, "linear")
    assert (megatron["lr_warmup_iters"], megatron["weight_decay"], megatron["clip_grad"]) == (0, 0.0, 1.0)
    assert (megatron["adam_beta1"], megatron["adam_beta2"], megatron["adam_eps"]) == (0.9, 0.999, 1e-8)
    assert (megatron["hidden_dropout"], megatron["attention_dropout"]) == (0.0, 0.0)


def test_the_reward_and_log_hooks_get_their_settings(config):
    assert config.custom_rm_path.path == "examples.experimental.reprompt.reward.reprompt_rm"
    assert config.custom_rm_path.config == RePromptRewardConfig(reprompt_reward_urls="http://a:1,http://b:2")
    assert not config.group_rm
    log_hook = config.custom_rollout_log_function_path
    assert log_hook.config == RePromptLogConfig(reprompt_sample_log="/root/shared_data/test/samples")


def test_the_job_holds_only_the_trainer_and_the_policy_engine(config):
    (policy,) = config.sglang.models
    assert policy.update_weights
    assert config.rollout_num_gpus == 1 and config.actor_num_gpus_per_node == 1 and not config.colocate


def test_a_split_that_does_not_fill_the_job_is_refused():
    with pytest.raises(AssertionError, match="must add up to"):
        ScriptArgs(train_gpus=2)
