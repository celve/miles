import shlex

import examples.experimental.video_gen_agent.keyframe.run_qwen35_9b_keyframe as launcher
import pytest
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
    text = {"hidden_size": 128, "num_attention_heads": 2, "num_hidden_layers": 1, "intermediate_size": 256}
    text |= {"rms_norm_eps": 1e-5, "tie_word_embeddings": True}
    Qwen3_5Config(text_config=text).save_pretrained(tmp_path / "Qwen3.5-9B")
    args = ScriptArgs(run_id="test", model_dir=str(tmp_path), **script_args)
    return parse_megatron_test_config("--ffn-hidden-size", "256", *shlex.split(_train_args(args)))


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
    assert config.custom_rm_path.config.video_reward_pool == "judge"
    (judge,) = config.reward_pools
    assert judge.colocated and config.reward_scaling.pools["judge"].gpu_offset == 0


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
    assert args.video.model == "org/Wan-T2V" and "t2v" in args.video.high_noise_lora
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
    for repo in ("Qwen/Qwen3.5-9B", "Qwen/Qwen3-VL-8B-Instruct", "Tongyi-MAI/Z-Image-Turbo"):
        assert any(f"hf download {repo} " in command for command in downloads)
    assert any("Wan-AI/Wan2.2-T2V-A14B-Diffusers" in command for command in downloads)
    (loras,) = [command for command in downloads if LIGHTX2V in command]
    assert loras.count("t2v_A14b") == 2 and "i2v" not in loras
    assert any("make_requests" in command for command in commands)


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
        (dict(video_mode="t2v"), "--video-mode is one of"),
        (dict(image_model="org/Image"), "has no recipe"),
    ],
)
def test_a_bad_layout_or_mode_is_refused(script_args, message):
    with pytest.raises(AssertionError, match=message):
        ScriptArgs(run_id="test", **script_args)
