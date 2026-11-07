import shlex

import pytest
from examples.experimental.video_gen_agent.multiturn.run_qwen35_9b_fastwan import ScriptArgs, _agent_env, _train_args
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from transformers import Qwen3_5Config


def test_the_launchers_layout_parses_with_the_generator_and_judge_on_gpus_of_their_own(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    # Megatron probes the GPU architecture for TP > 1; a CPU host has none.
    monkeypatch.setattr("megatron.training.arguments.get_device_arch_version", lambda: 10)
    text = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 1, "intermediate_size": 256}
    text |= {"rms_norm_eps": 1e-5, "tie_word_embeddings": True}
    Qwen3_5Config(text_config=text).save_pretrained(tmp_path / "Qwen3.5-9B")
    args = ScriptArgs(run_id="test", model_dir=str(tmp_path))

    config = parse_megatron_test_config("--ffn-hidden-size", "256", *shlex.split(_train_args(args)))

    policy, generator = config.sglang.models
    assert [group.engine_kind for group in generator.server_groups] == ["diffusion"]
    assert policy.update_weights and not generator.update_weights
    assert config.sglang_scaling.group(model_name="generator", group_index=0).num_gpus == 2
    (judge,) = config.reward_pools
    assert judge.name == "judge" and judge.num_workers == 1 and not judge.colocated
    assert config.reward_scaling.pools["judge"].gpu_offset == 3
    assert not config.colocate and config.actor_num_gpus_per_node == 4
    assert config.use_session_server == "v2" and config.tito_model == "qwen35"
    assert config.multimodal_token_space_processing and not config.group_rm
    assert config.custom_rm_path.config.video_reward_pool == "judge"
    assert [dataset.name for dataset in config.eval_datasets] == ["heldout"]


def test_the_env_is_set_for_clip_observations_with_phase_0s_clips():
    env = _agent_env(ScriptArgs(run_id="test"))

    assert env["AGENT_VIDEO_GEN_CLIP_OBSERVATIONS"] == "1" and env["AGENT_VIDEO_GEN_MAX_GENERATIONS"] == "3"
    assert (env["AGENT_VIDEO_GEN_WIDTH"], env["AGENT_VIDEO_GEN_HEIGHT"], env["AGENT_VIDEO_GEN_FPS"]) == (
        "832",
        "480",
        "24",
    )
    assert env["AGENT_VIDEO_GEN_CLIP_DIR"] == "/root/shared_data/test/clips"


def test_the_feature_cache_is_on_by_default_and_shared_under_the_run():
    argv = _train_args(ScriptArgs(run_id="test"))

    assert "--sglang-enable-token-space-preprocess-cache --sglang-token-space-preprocess-cache-trust-url" in argv
    assert "--sglang-token-space-preprocess-cache-dir /root/shared_data/test/feature_cache " in argv
    assert "preprocess-cache" not in _train_args(ScriptArgs(run_id="test", feature_cache=False))


def test_a_gpu_split_that_does_not_fill_the_node_is_refused():
    with pytest.raises(AssertionError, match="adds up to 7, not 8"):
        ScriptArgs(judge_gpus=0)
