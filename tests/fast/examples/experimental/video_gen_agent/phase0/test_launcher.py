import shlex

import pytest
from examples.experimental.video_gen_agent.phase0.reward import PlannerRewardConfig
from examples.experimental.video_gen_agent.phase0.run_qwen3_8b_fastwan import ScriptArgs, _train_args
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config
from transformers import Qwen3Config


def test_the_launchers_layout_parses_with_the_generator_and_judge_on_gpus_of_their_own(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    Qwen3Config(num_hidden_layers=2).save_pretrained(tmp_path / "Qwen3-8B")

    config = parse_fsdp_test_config(*shlex.split(_train_args(ScriptArgs(run_id="test", model_dir=str(tmp_path)))))

    policy, generator = config.sglang.models
    assert [group.engine_kind for group in generator.server_groups] == ["diffusion"]
    assert policy.update_weights and not generator.update_weights
    assert config.sglang_scaling.group(model_name="generator", group_index=0).num_gpus == 3
    (judge,) = config.reward_pools
    assert judge.name == "judge" and judge.num_workers == 2 and not judge.colocated
    assert config.reward_scaling.pools["judge"].gpu_offset == 4
    assert not config.colocate and config.actor_num_gpus_per_node == 2
    assert config.rollout_skip_special_tokens
    assert not config.group_rm and isinstance(config.custom_rm_path.config, PlannerRewardConfig)
    rm_config = config.custom_rm_path.config
    assert rm_config.video_clip_dir == "/root/shared_data/test/clips"
    assert (rm_config.video_num_inference_steps, rm_config.video_guidance_scale) == (3, 1.0)
    assert [dataset.name for dataset in config.eval_datasets] == ["heldout", "heldout_raw"]


def test_a_gpu_split_that_does_not_fill_the_node_is_refused():
    with pytest.raises(AssertionError, match="adds up to 7, not 8"):
        ScriptArgs(judge_gpus=1)
