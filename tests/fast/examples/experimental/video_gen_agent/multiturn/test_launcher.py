import shlex

import pytest
from examples.experimental.video_gen_agent.multiturn.run_qwen35_9b_fastwan import ScriptArgs, _agent_env, _train_args
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from transformers import Qwen3_5Config

from miles.ray.rollout import metrics
from miles.ray.specs.inference import _compute_spec_inference_engine
from miles.utils.args.custom_view import compute_custom_function_config
from miles.utils.types import Sample
from miles.utils.workers.addr_allocator import PortAllocator


@pytest.fixture
def recipe_config(monkeypatch, tmp_path):
    """The launcher's argv, parsed by Miles against a stand-in Qwen3.5 config."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    # Megatron probes the GPU architecture for TP > 1; a CPU host has none.
    monkeypatch.setattr("megatron.training.arguments.get_device_arch_version", lambda: 10)
    text = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 1, "intermediate_size": 256}
    text |= {"rms_norm_eps": 1e-5, "tie_word_embeddings": True}
    Qwen3_5Config(text_config=text).save_pretrained(tmp_path / "Qwen3.5-9B")
    args = ScriptArgs(run_id="test", model_dir=str(tmp_path))
    return parse_megatron_test_config("--ffn-hidden-size", "256", *shlex.split(_train_args(args)))


def test_the_launchers_layout_parses_with_the_generator_and_eval_engine_on_gpus_of_their_own(recipe_config):
    config = recipe_config

    policy, generator, eval_model = config.sglang.models
    assert [group.engine_kind for group in generator.server_groups] == ["diffusion"]
    assert policy.update_weights and not generator.update_weights
    assert config.sglang_scaling.group(model_name="generator", group_index=0).num_gpus == 2
    (judge,) = config.reward_pools
    assert judge.name == "judge" and judge.num_workers == 1 and judge.colocated and not judge.needs_offload
    assert config.reward_scaling.pools["judge"].gpu_offset == 0
    assert not config.colocate and config.actor_num_gpus_per_node == 4
    assert config.use_session_server == "v2" and config.tito_model == "qwen35"
    # The env acts on parsed tool calls only, so the engines must parse Qwen3.5's native calls.
    assert config.sglang.common_value("tool_call_parser") == "qwen3_coder"
    assert config.sglang.common_value("reasoning_parser") == "qwen3"
    assert config.eval_num_gpus == 1 and config.eval_uses_snapshots and eval_model.name == "eval"
    assert config.eval_hf_dir == "/root/shared_data/test/eval_snapshots"
    assert config.multimodal_token_space_processing and not config.group_rm
    # Group-level submission and no over-sampling: a dropped group is replaced and nothing is aborted at round end.
    assert config.rollout_submission_granularity in (None, "group")
    assert config.over_sampling_batch_size == config.rollout_batch_size
    assert config.custom_rm_path.config.video_reward_pool == "judge"
    assert [dataset.name for dataset in config.eval_datasets] == ["heldout"]
    # Each fold opens a branch with a different request; the default picker keeps it as a sample.
    assert config.session_sample_picker_path == "miles.rollout.session.v2.picker_hub.drop_same_prompt_retries"
    assert config.custom_eval_rollout_log_function_path.path.endswith("multiturn.episode_metrics.log_eval_episodes")
    log_hook = config.custom_rollout_log_function_path
    assert log_hook.path.endswith("multiturn.episode_metrics.log_episodes")
    # The hook prunes the cache through the view of the args Miles hands it.
    hook_args = compute_custom_function_config(config, log_hook)
    assert hook_args.sglang.common_value("token_space_preprocess_cache_dir") == "/root/shared_data/test/feature_cache"


def test_the_env_is_set_for_clip_observations_with_phase_0s_clips():
    env = _agent_env(ScriptArgs(run_id="test"))

    assert env["AGENT_VIDEO_GEN_CLIP_OBSERVATIONS"] == "1" and env["AGENT_VIDEO_GEN_MAX_GENERATIONS"] == "3"
    assert (env["AGENT_VIDEO_GEN_WIDTH"], env["AGENT_VIDEO_GEN_HEIGHT"], env["AGENT_VIDEO_GEN_FPS"]) == (
        "832",
        "480",
        "24",
    )
    assert env["AGENT_VIDEO_GEN_CLIP_DIR"] == "/root/shared_data/test/clips"
    assert (env["AGENT_VIDEO_GEN_KEEP_CLIPS"], env["AGENT_VIDEO_GEN_MAX_EXTENDS"]) == ("0", "0")


def test_multi_shot_folding_reaches_the_env():
    env = _agent_env(
        ScriptArgs(run_id="test", max_extends=5, min_shots=4, max_turns=12, keep_clips=2, fold="keyframe")
    )

    assert env["AGENT_VIDEO_GEN_MAX_EXTENDS"] == "5" and env["AGENT_VIDEO_GEN_MAX_TURNS"] == "12"
    assert env["AGENT_VIDEO_GEN_MIN_SHOTS"] == "4"
    assert env["AGENT_VIDEO_GEN_KEEP_CLIPS"] == "2" and env["AGENT_VIDEO_GEN_FOLD"] == "keyframe"


def test_the_feature_cache_is_on_by_default_and_shared_under_the_run():
    argv = _train_args(ScriptArgs(run_id="test"))

    assert "--sglang-enable-token-space-preprocess-cache --sglang-token-space-preprocess-cache-trust-url" in argv
    assert "--sglang-token-space-preprocess-cache-dir /root/shared_data/test/feature_cache " in argv
    without = _train_args(ScriptArgs(run_id="test", feature_cache=False))
    assert "preprocess-cache" not in without and "multiturn.episode_metrics.log_episodes" in without


def test_a_gpu_split_that_does_not_fill_the_node_is_refused():
    with pytest.raises(AssertionError, match="adds up to 7, not 8"):
        ScriptArgs(eval_gpus=0)


def _eval_sample(session_id: str, reward: float) -> Sample:
    sample = Sample(tokens=[1, 2, 3], response_length=1, loss_mask=[1], reward=reward, status=Sample.Status.COMPLETED)
    sample.metadata = {"session_id": session_id}
    return sample


def test_heldout_reward_counts_each_episode_once_however_many_branches_folding_left(recipe_config, monkeypatch):
    logged = []
    monkeypatch.setattr(metrics.tracking, "log", lambda args, log_dict, step_key: logged.append(log_dict))
    # Episode "a" scored 1 and folded twice (three branches); episode "b" scored 0 and never folded.
    samples = [_eval_sample("a", 1.0)] * 3 + [_eval_sample("b", 0.0)]
    data = {"heldout": {"rewards": [s.reward for s in samples], "truncated": [False] * 4, "samples": samples}}

    metrics.log_eval_rollout_data(0, recipe_config, data)

    (log_dict,) = logged
    assert log_dict["eval/heldout"] == 0.5


def test_heldout_reward_is_also_logged_per_request_category(recipe_config, monkeypatch):
    logged = []
    monkeypatch.setattr(metrics.tracking, "log", lambda args, log_dict, step_key: logged.append(log_dict))
    samples = [_eval_sample(s, r) for s, r in (("a", 1.0), ("a", 1.0), ("b", 0.0), ("c", 0.5), ("d", 1.0))]
    for sample, category in zip(samples, ["spatial", "spatial", "spatial", "numeracy", None], strict=True):
        sample.metadata |= {"category": category} if category else {}
    data = {"compbench": {"rewards": [s.reward for s in samples], "truncated": [False] * 5, "samples": samples}}

    metrics.log_eval_rollout_data(0, recipe_config, data)

    (log_dict,) = logged
    assert log_dict["eval/compbench"] == 0.625
    assert (log_dict["eval/compbench/spatial"], log_dict["eval/compbench/numeracy"]) == (0.5, 0.5)
    assert not any(key.startswith("eval/compbench/None") for key in log_dict)


def test_heldout_reward_is_also_logged_per_value_of_each_request_split(recipe_config, monkeypatch):
    logged = []
    monkeypatch.setattr(metrics.tracking, "log", lambda args, log_dict, step_key: logged.append(log_dict))
    samples = [_eval_sample(s, r) for s, r in (("a", 1.0), ("a", 1.0), ("b", 0.0), ("c", 0.5), ("d", 1.0))]
    splits = [{"heuristic_seconds": "4-7s", "heuristic_setting": "yes"}] * 3
    splits += [{"heuristic_seconds": "12-15s", "heuristic_setting": "yes"}, None]
    for sample, split in zip(samples, splits, strict=True):
        sample.metadata |= {"splits": split} if split else {}
    data = {"vidprom": {"rewards": [s.reward for s in samples], "truncated": [False] * 5, "samples": samples}}

    metrics.log_eval_rollout_data(0, recipe_config, data)

    (log_dict,) = logged
    assert log_dict["eval/vidprom"] == 0.625
    assert log_dict["eval/vidprom/heuristic_seconds=4-7s"] == 0.5
    assert log_dict["eval/vidprom/heuristic_seconds=12-15s"] == 0.5
    assert log_dict["eval/vidprom/heuristic_setting=yes"] == 0.5
    assert not any("None" in key for key in log_dict)


async def test_no_engine_is_given_the_port_the_generators_broker_binds(recipe_config):
    specs = {
        model.name: _compute_spec_inference_engine(
            recipe_config,
            model_idx=model_idx,
            group_index=0,
            model_cfg=model,
            server_group_config=model.server_groups[0],
            group_scaling=recipe_config.sglang_scaling.group(model_name=model.name, group_index=0),
        )
        for model_idx, model in enumerate(recipe_config.sglang.models)
    }
    allocator, ports = PortAllocator(), {}

    # The generator first, as in the failing run: the next engine's primary used to land on its HTTP port + 1.
    for name in ("generator", "policy", "eval"):
        for info in specs[name].port_infos:
            port = await allocator.alloc(_ScanningNode(), node_ip="node", consecutive=info.num_consecutive)
            ports[name, info.name] = port

    broker = ports["generator", "primary"] + 1
    assert broker not in ports.values()


class _ScanningNode:
    """A node with every port free, scanning from the allocator's cursor as ``NodeProbeMixin`` does."""

    class _Probe:
        @staticmethod
        async def remote(*, start_port, count):
            return start_port or 8000

    _get_free_port_block = _Probe()
