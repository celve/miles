import shlex

import examples.experimental.video_gen_agent.keyframe.run_qwen35_9b_keyframe as launcher
import pytest
from examples.experimental.video_gen_agent.keyframe.env import KeyframeEnvOptions
from examples.experimental.video_gen_agent.keyframe.run_qwen35_9b_keyframe import (
    LIGHTX2V,
    ScriptArgs,
    _agent_env,
    _train_args,
)
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from tests.fast.launch_scripts.py_harness import call_entrypoint, freeze_environment, install_command_recorder
from transformers import Qwen3_5Config


def _config(monkeypatch, tmp_path, **script_args):
    """The launcher's argv, parsed by Miles against a stand-in Qwen3.5 config."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    args = ScriptArgs(run_id="test", model_dir=str(tmp_path), **script_args)
    text = {"hidden_size": 128, "num_attention_heads": 4, "num_key_value_heads": 4, "num_hidden_layers": 1}
    text |= {"intermediate_size": 256, "rms_norm_eps": 1e-5, "tie_word_embeddings": True}
    Qwen3_5Config(text_config=text).save_pretrained(args.local(args.recipe.repo))
    return parse_megatron_test_config(
        "--ffn-hidden-size", "256", "--num-attention-heads", "4", *shlex.split(_train_args(args))
    )


def _loras(model):
    (group,) = model.server_groups
    return [(lora.path.rsplit("/", 1)[1], lora.target) for lora in group.loras]


@pytest.mark.parametrize(
    "num_gpus, gpus_by_model, eval_gpus",
    [(8, {"policy": 1, "image": 2, "video": 3}, 1), (4, {"policy": 1, "image": 1, "video": 1}, 0)],
)
def test_each_layout_parses_with_two_frozen_generators_on_gpus_of_their_own(
    monkeypatch, tmp_path, num_gpus, gpus_by_model, eval_gpus
):
    config = _config(monkeypatch, tmp_path, num_gpus_per_node=num_gpus)

    models = {model.name: model for model in config.sglang.models}
    assert set(models) == set(gpus_by_model) | ({"eval"} if eval_gpus else set())
    for name, gpus in gpus_by_model.items():
        assert config.sglang_scaling.group(model_name=name, group_index=0).num_gpus == gpus
    for name in ("image", "video"):
        assert not models[name].update_weights
        assert [(g.engine_kind, g.num_gpus_per_engine) for g in models[name].server_groups] == [("diffusion", 1)]
    assert models["policy"].update_weights
    assert models["image"].model_path == f"{tmp_path}/Z-Image-Turbo"
    assert models["video"].model_path == f"{tmp_path}/Wan2.2-T2V-A14B-Diffusers"
    assert _loras(models["video"]) == [
        ("wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors", "transformer"),
        ("wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors", "transformer_2"),
    ]
    assert not models["image"].server_groups[0].loras
    assert config.eval_num_gpus == eval_gpus and config.eval_uses_snapshots == bool(eval_gpus)
    assert not config.colocate and config.actor_num_gpus_per_node == 1
    hook = config.custom_generate_function_path
    assert hook.path.endswith("keyframe.generate.generate")
    assert (hook.config.video_generator_model, hook.config.image_generator_model) == ("video", "image")
    assert config.custom_agent_function_path.path.endswith("keyframe.env.run_agent")
    assert config.rollout_max_response_len == 2048 and config.max_seq_len == 16384
    assert config.custom_rm_path.path.endswith("gated_reward.reward_func")
    assert config.custom_reward_post_process_path.path.endswith("gated_reward.group_normalized_rewards")
    reward = config.custom_rm_path.config
    assert reward.video_reward_signal_pools == ["hpsv3", "vqascore"]
    assert len(reward.video_reward_signal_ranges) == 4 and reward.video_reward_adherence_pool is None
    assert (reward.video_reward_motion_pool, reward.video_reward_motion_threshold) == ("motion", 0.4)
    assert (reward.video_reward_ocr_pool, reward.video_reward_ocr_factor) == ("ocr", 0.5)
    pools = {pool.name: pool for pool in config.reward_pools}
    # One colocated pool per rollout GPU, from the planner's: the motion judge, HPSv3, VQAScore.
    assert [config.reward_scaling.pools[n].gpu_offset for n in ("motion", "hpsv3", "vqascore")] == [0, 1, 2]
    assert all(pools[n].colocated and pools[n].uses_gpu for n in ("motion", "hpsv3", "vqascore"))
    assert pools["vqascore"].scorer_kwargs == {"model_path": f"{tmp_path}/Qwen3-VL-8B-Instruct"}
    assert pools["motion"].scorer_kwargs["axis_weights"] == {"motion": 1.0}
    assert pools["hpsv3"].scorer_kwargs["top_fraction"] == 0.3
    assert not pools["ocr"].uses_gpu and pools["ocr"].scorer_path.endswith("ocr.VideoOcrScorer")


@pytest.mark.parametrize(
    "reward, adherence_gate, signals, pools",
    [
        ("pickscore", False, ["pickscore"], {"motion", "pickscore", "ocr"}),
        ("hpsv3", True, ["hpsv3"], {"motion", "hpsv3", "adherence", "ocr"}),
        ("hpsv3+unifiedreward", False, ["hpsv3", "unifiedreward"], {"motion", "hpsv3", "unifiedreward", "ocr"}),
    ],
)
def test_the_reward_is_a_flag(monkeypatch, tmp_path, reward, adherence_gate, signals, pools):
    config = _config(monkeypatch, tmp_path, num_gpus_per_node=8, reward=reward, adherence_gate=adherence_gate)

    hook = config.custom_rm_path.config
    assert hook.video_reward_signal_pools == signals and {pool.name for pool in config.reward_pools} == pools
    if adherence_gate:
        assert hook.video_reward_adherence_pool == "adherence" and hook.video_reward_adherence_threshold > 0
        adherence = next(pool for pool in config.reward_pools if pool.name == "adherence")
        assert adherence.scorer_kwargs["axis_weights"] == {"prompt_adherence": 1.0}


def test_the_27b_planner_trains_at_tp4_with_its_own_template_and_eval_on_the_planner_engine(monkeypatch, tmp_path):
    argv = _train_args(ScriptArgs(run_id="test", planner="qwen3.8-27b"))
    assert "--tensor-model-parallel-size 4 " in argv and "--actor-num-gpus-per-node 4 " in argv
    assert "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer " in argv

    # Megatron asks the GPU's architecture to validate TP above 1, so the rest parses at TP1.
    config = _config(monkeypatch, tmp_path, planner="qwen3.8-27b", extra_args="--tensor-model-parallel-size 1")

    assert config.hf_checkpoint == f"{tmp_path}/Qwen3.8-27B" and config.tito_model == "qwen38small"
    assert [config.sglang.common_value(name) for name in ("reasoning_parser", "tool_call_parser")] == [
        "qwen3",
        "qwen3_coder",
    ]
    for name, gpus in {"policy": 1, "image": 1, "video": 2}.items():
        assert config.sglang_scaling.group(model_name=name, group_index=0).num_gpus == gpus
    assert "eval" not in {model.name for model in config.sglang.models} and config.eval_num_gpus == 0
    assert config.raw_megatron.base_args["optimizer_cpu_offload"]


def test_the_9b_planner_keeps_its_template_and_optimizer_on_gpu(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path)

    assert config.tito_model == "qwen35" and config.raw_megatron.base_args["tensor_model_parallel_size"] == 1
    assert not config.raw_megatron.base_args["optimizer_cpu_offload"]


def test_i2v_mode_serves_image_to_video_with_its_own_loras(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path, num_gpus_per_node=4, video_mode="i2v")

    video = next(model for model in config.sglang.models if model.name == "video")
    assert video.model_path == f"{tmp_path}/Wan2.2-I2V-A14B-Diffusers"
    assert [name.split("_")[1] for name, _ in _loras(video)] == ["i2v", "i2v"]
    env = _agent_env(ScriptArgs(run_id="test", video_mode="i2v"))
    assert (env["AGENT_VIDEO_GEN_VIDEO_MODE"], env["AGENT_VIDEO_GEN_VIDEO_MODEL"]) == (
        "i2v",
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    )


@pytest.mark.parametrize("num_gpus, image_gpus, video_gpus", [(4, 1, 1), (8, 1, 4)])
def test_reference_mode_serves_vace_through_the_diffusers_backend_from_the_built_directory(
    monkeypatch, tmp_path, num_gpus, image_gpus, video_gpus
):
    config = _config(monkeypatch, tmp_path, num_gpus_per_node=num_gpus, video_mode="reference")

    video = next(model for model in config.sglang.models if model.name == "video")
    (group,) = video.server_groups
    assert video.model_path == f"{tmp_path}/Wan2.1-VACE-14B-diffusers-reference" and not video.update_weights
    assert group.overrides["backend"] == "diffusers" and not group.loras and group.engine_kind == "diffusion"
    for name, gpus in {"image": image_gpus, "video": video_gpus}.items():
        assert config.sglang_scaling.group(model_name=name, group_index=0).num_gpus == gpus
    env = _agent_env(ScriptArgs(run_id="test", video_mode="reference"))
    assert (env["AGENT_VIDEO_GEN_VIDEO_MODE"], env["AGENT_VIDEO_GEN_REFERENCE_SCALE"]) == ("reference", "0.5")
    assert env["AGENT_VIDEO_GEN_VIDEO_MODEL"] == "Wan-AI/Wan2.1-VACE-14B-diffusers"
    assert "AGENT_VIDEO_GEN_GUIDANCE_SCALE_2" not in env and env["AGENT_VIDEO_GEN_NUM_INFERENCE_STEPS"] == "4"
    scaled = _agent_env(ScriptArgs(run_id="test", video_mode="reference", reference_scale=0.7))
    assert scaled["AGENT_VIDEO_GEN_REFERENCE_SCALE"] == "0.7"


@pytest.mark.parametrize(
    "script_args, timeout",
    [
        # One VACE engine renders a step's 128 episodes and as many eval episodes one at a time, 20 s each.
        (dict(video_mode="reference", num_gpus_per_node=4), 5120.0),
        (dict(video_mode="reference", num_gpus_per_node=8), 1280.0),
        (dict(video_mode="reference", num_gpus_per_node=8, rollout_batch_size=2, n_samples_per_prompt=2), 600.0),
        (dict(video_mode="reference", generation_timeout=900.0), 900.0),
        (dict(generation_timeout=900.0), 900.0),
        (dict(), 600.0),
    ],
)
def test_the_env_waits_for_the_video_queue_the_layout_holds(monkeypatch, script_args, timeout):
    monkeypatch.delenv("AGENT_VIDEO_GEN_TIMEOUT_S", raising=False)
    for name, value in _agent_env(ScriptArgs(run_id="test", **script_args)).items():
        monkeypatch.setenv(name, value)

    assert KeyframeEnvOptions.from_env().timeout_s == timeout


def test_prepare_builds_the_reference_directory_with_the_wan21_lightx2v_lora(monkeypatch, tmp_path):
    commands = _recorded(monkeypatch, tmp_path, "prepare", video_mode="reference")

    models = tmp_path / "models"
    assert any(f"hf download Wan-AI/Wan2.1-VACE-14B-diffusers --local-dir {models}/" in c for c in commands)
    lora = "wan2.1_t2v_14b_lora_rank64_lightx2v_4step.safetensors"
    assert any(f"hf download lightx2v/Wan2.1-Distill-Loras {lora} " in c for c in commands)
    assert not any(LIGHTX2V in command for command in commands)
    (build,) = [command for command in commands if "keyframe.reference_model" in command]
    assert f"--src {models}/Wan2.1-VACE-14B-diffusers --dst {models}/Wan2.1-VACE-14B-diffusers-reference" in build
    assert f"--flow-shift 5.0 --lora {models}/Wan2.1-Distill-Loras/{lora}:transformer" in build


@pytest.mark.parametrize(
    "image_model, steps, guidance",
    [("Tongyi-MAI/Z-Image-Turbo", "9", "0.0"), ("Qwen/Qwen-Image-2.1-Turbo", "8", "1.0")],
)
def test_each_image_model_gets_its_schedule_and_is_recorded(image_model, steps, guidance):
    env = _agent_env(ScriptArgs(run_id="test", image_model=image_model))

    assert (env["AGENT_VIDEO_GEN_IMAGE_NUM_INFERENCE_STEPS"], env["AGENT_VIDEO_GEN_IMAGE_GUIDANCE_SCALE"]) == (
        steps,
        guidance,
    )
    assert env["AGENT_VIDEO_GEN_IMAGE_MODEL"] == image_model


def test_the_split_and_models_are_flags():
    args = ScriptArgs(
        run_id="test",
        image_gpus=1,
        video_gpus=4,
        image_model="org/Image",
        image_steps=4,
        image_guidance_scale=2.0,
        video_model="org/Wan-T2V",
    )

    assert (args.train_gpus, args.policy_gpus, args.image_gpus, args.video_gpus, args.eval_gpus) == (1, 1, 1, 4, 1)
    assert args.video.model == "org/Wan-T2V" and "t2v" in args.video.loras[0][0]
    assert (args.image_steps, args.image_guidance_scale) == (4, 2.0)
    assert "--rollout-num-gpus 6 " in _train_args(args)


def test_the_env_gets_the_recipes_clip_and_both_generators():
    env = _agent_env(ScriptArgs(run_id="test"))

    assert env["AGENT_VIDEO_GEN_GENERATOR"].endswith("multiturn.generator.RouterClipGenerator")
    assert env["AGENT_VIDEO_GEN_IMAGE_GENERATOR"].endswith("keyframe.generator.RouterImageGenerator")
    assert (env["AGENT_VIDEO_GEN_WIDTH"], env["AGENT_VIDEO_GEN_HEIGHT"], env["AGENT_VIDEO_GEN_FPS"]) == (
        "832",
        "480",
        "16",
    )
    assert (env["AGENT_VIDEO_GEN_SECONDS"], env["AGENT_VIDEO_GEN_MAX_IMAGES"], env["AGENT_VIDEO_GEN_MAX_TURNS"]) == (
        "2.0",
        "3",
        "6",
    )
    assert env["AGENT_VIDEO_GEN_MAX_PROMPT_WORDS"] == "0"
    assert _agent_env(ScriptArgs(run_id="test", max_prompt_words=120))["AGENT_VIDEO_GEN_MAX_PROMPT_WORDS"] == "120"
    assert env["AGENT_VIDEO_GEN_NUM_INFERENCE_STEPS"] == "4" and env["AGENT_VIDEO_GEN_GUIDANCE_SCALE_2"] == "1.0"
    assert env["AGENT_VIDEO_GEN_VIDEO_MODE"] == "preview"
    assert env["AGENT_VIDEO_GEN_IMAGE_DIR"] == "/root/shared_data/test/images"


def _recorded(monkeypatch, tmp_path, entrypoint, **script_args) -> list[str]:
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    overrides = dict(run_id="test", model_dir=str(tmp_path / "models"), data_dir=str(tmp_path / "data")) | script_args
    call_entrypoint(launcher, entrypoint, overrides, tmp_path)
    return recording.commands


def test_prepare_downloads_the_defaults_models_and_only_the_modes_lightx2v_loras(monkeypatch, tmp_path):
    commands = _recorded(monkeypatch, tmp_path, "prepare")

    downloads = [command for command in commands if "hf download" in command]
    for repo in (
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3-VL-8B-Instruct",
        "Tongyi-MAI/Z-Image-Turbo",
        "MizzenAI/HPSv3",
        "Qwen/Qwen2-VL-7B-Instruct",
    ):
        assert any(f"hf download {repo} " in command for command in downloads)
    assert not any("PickScore" in command or "UnifiedReward" in command for command in downloads)
    assert any("pip install paddleocr==2.9.1" in command for command in commands)
    assert any("Wan-AI/Wan2.2-T2V-A14B-Diffusers" in command for command in downloads)
    (loras,) = [command for command in downloads if LIGHTX2V in command]
    assert loras.count("t2v_A14b") == 2 and "i2v" not in loras
    assert any("make_requests" in command for command in commands)


def test_prepare_downloads_and_converts_the_27b_planner(monkeypatch, tmp_path):
    commands = _recorded(monkeypatch, tmp_path, "prepare", planner="qwen3.8-27b")

    assert any("hf download Qwen/Qwen3.8-27B " in command for command in commands)
    assert not any("Qwen/Qwen3.5-9B" in command for command in commands)
    (convert,) = [command for command in commands if "convert_hf_to_torch_dist" in command]
    assert "Qwen3.8-27B_torch_dist" in convert and "--num-layers 64 " in convert


def test_execute_submits_the_run_with_the_keyframe_env(monkeypatch, tmp_path):
    commands = _recorded(monkeypatch, tmp_path, "execute", num_gpus_per_node=4)

    (submit,) = [command for command in commands if "ray job submit" in command]
    assert "keyframe.env.run_agent" in submit and "AGENT_VIDEO_GEN_VIDEO_MODE" in submit
    assert "--eval-num-gpus" not in submit and "--num-gpus-per-node 4" in submit


@pytest.mark.parametrize(
    "script_args, message",
    [
        (dict(eval_gpus=0), "adds up to 7, not 8"),
        (dict(num_gpus_per_node=2), "no layout for 2 GPUs"),
        (dict(planner="qwen3.8-27b", num_gpus_per_node=4), "qwen3.8-27b has no layout for 4 GPUs"),
        (dict(planner="qwen3.6-27b"), "--planner is one of"),
        (dict(video_mode="t2v"), "--video-mode is one of"),
        (dict(reference_scale=0.5), "--reference-scale applies to --video-mode reference only"),
        (dict(video_mode="reference", reference_scale=0), "--reference-scale must be positive"),
        (dict(generation_timeout=0), "--generation-timeout must be positive"),
        (dict(image_model="org/Image"), "has no recipe"),
        (dict(reward="judge"), "--reward is one of"),
    ],
)
def test_a_bad_layout_or_mode_is_refused(script_args, message):
    with pytest.raises(AssertionError, match=message):
        ScriptArgs(run_id="test", **script_args)
