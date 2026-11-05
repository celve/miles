import json
import re

import pytest
import typer

from tests.fast.launch_scripts.py_harness import (
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

LAUNCHER = REPO_ROOT / "scripts/run_qwen3_vl_4b_harbor_cua.py"


def _train_command(monkeypatch, tmp_path, **overrides) -> tuple[str, dict[str, str]]:
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)

    # execute() exports MODEL_ARGS_ROTARY_BASE; call_entrypoint restores the environment after it.
    call_entrypoint(import_launch_script(LAUNCHER), "execute", {"harbor_env_type": "e2b", **overrides}, tmp_path)

    (command,) = [cmd for cmd in recording.commands if "ray job submit" in cmd]
    runtime_env = json.loads(re.search(r"--runtime-env-json='(.*?)'", command).group(1))
    return command, runtime_env["env_vars"]


def test_the_session_path_trains_computer_1_through_the_agent_function(monkeypatch, tmp_path):
    command, env = _train_command(monkeypatch, tmp_path, rollout_path="session", max_turns=7)

    assert "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate" in command
    assert "--custom-agent-function-path harbor_agent_function.run" in command
    assert "--use-session-server v2 --use-session-images" in command
    assert "--tito-model qwen3_vl" in command
    assert "--max-seq-len 32768" in command
    assert "--pad-steps-for-dp-alignment" in command
    assert "--harbor-cua-" not in command
    assert str(REPO_ROOT / "examples/experimental/harbor") in env["PYTHONPATH"].split(":")
    assert env["HARBOR_AGENT_MAX_ITERATIONS"] == "7"


def test_the_generate_path_stays_the_default(monkeypatch, tmp_path):
    command, env = _train_command(monkeypatch, tmp_path)

    assert "--custom-generate-function-path examples.experimental.harbor_cua.rollout.generate" in command
    assert "--use-session-server" not in command
    assert "HARBOR_AGENT_MAX_ITERATIONS" not in env


def test_the_session_path_refuses_the_osworld_runner():
    launcher = import_launch_script(LAUNCHER)

    with pytest.raises(typer.BadParameter, match="OSWorld runner"):
        launcher.main(harbor_env_type="e2b", rollout_path="session", osworld_runner_url="http://kvm:8000")


def test_media_refs_hand_the_trainer_files_instead_of_pixels(monkeypatch, tmp_path):
    command, _ = _train_command(monkeypatch, tmp_path, rollout_path="session", media_refs=True)

    assert "--multimodal-token-space-processing --sglang-enable-token-space-processor" in command
    assert re.search(r"--shared-media-dir \S+/media ", command)
    assert "--use-miles-router" in command
    assert "--multimodal-rebuild-per-micro-batch --multimodal-rebuild-keep-on-device" in command


def test_the_session_path_ships_pixels_by_default(monkeypatch, tmp_path):
    command, _ = _train_command(monkeypatch, tmp_path, rollout_path="session")

    assert "--multimodal-token-space-processing" not in command
    assert "--shared-media-dir" not in command
    assert "--use-miles-router" not in command


def test_media_refs_refuse_the_generate_path():
    launcher = import_launch_script(LAUNCHER)

    with pytest.raises(typer.BadParameter, match="--rollout-path session only"):
        launcher.main(harbor_env_type="e2b", media_refs=True)


@pytest.mark.parametrize("rollout_path", ["generate", "session"])
def test_engines_keep_image_features_on_the_gpu(monkeypatch, tmp_path, rollout_path):
    command, env = _train_command(monkeypatch, tmp_path, rollout_path=rollout_path)

    assert "--sglang-mm-feature-transport cuda_ipc" in command
    assert env["SGLANG_MM_FEATURE_CACHE_MB"] == "4096"
    assert env["SGLANG_VLM_CACHE_SIZE_MB"] == "4096"


def _sglang_at(site, *, clears_cache: bool) -> str:
    managers = site / "sglang/srt/managers"
    managers.mkdir(parents=True)
    (site / "sglang/__init__.py").write_text("")
    body = "def clear_mm_embedding_cache() -> None:\n    pass\n" if clears_cache else "embedding_cache = None\n"
    (managers / "mm_schedule.py").write_text(body)
    return str(site)


def _sglang_on_path(monkeypatch, root, *, clears_cache: bool) -> None:
    """As if the launcher was started with this SGLang on PYTHONPATH: in its environment and on its sys.path."""
    site = _sglang_at(root / "site", clears_cache=clears_cache)
    monkeypatch.setenv("PYTHONPATH", site)
    monkeypatch.syspath_prepend(site)


def _launcher_without_harbor_preflight(monkeypatch):
    launcher = import_launch_script(LAUNCHER)
    # main's harbor preflight needs the harbor package, which the CPU test environment lacks.
    monkeypatch.setattr(launcher, "harbor_env_vars", lambda args: {})
    return launcher


def test_the_cache_refuses_an_sglang_that_never_clears_it(monkeypatch, tmp_path):
    """Without the SGLang fix, a screenshot seen again after a weight update is embedded by the old vision weights."""
    freeze_environment(monkeypatch)
    _sglang_on_path(monkeypatch, tmp_path, clears_cache=False)
    recording = install_command_recorder(monkeypatch)
    launcher = _launcher_without_harbor_preflight(monkeypatch)

    with pytest.raises(typer.BadParameter, match="clears its ViT embedding cache"):
        call_entrypoint(launcher, "main", {"harbor_env_type": "e2b", "skip_prepare": True}, tmp_path)
    assert recording.commands == []


@pytest.mark.parametrize("override", ["cache", "pythonpath"])
def test_extra_env_vars_cannot_route_around_the_refusal(monkeypatch, tmp_path, override):
    """--extra-env-vars lands after the recipe's worker env, so the workers' cache size and SGLang are what count."""
    freeze_environment(monkeypatch)
    _sglang_on_path(monkeypatch, tmp_path, clears_cache=override == "pythonpath")
    recording = install_command_recorder(monkeypatch)
    launcher = _launcher_without_harbor_preflight(monkeypatch)
    if override == "cache":
        overrides = {"vlm_cache_size_mb": 0, "extra_env_vars": "SGLANG_VLM_CACHE_SIZE_MB=4096"}
    else:
        old_sglang = _sglang_at(tmp_path / "old", clears_cache=False)
        overrides = {"extra_env_vars": f"PYTHONPATH={old_sglang}"}

    with pytest.raises(typer.BadParameter, match="clears its ViT embedding cache"):
        call_entrypoint(launcher, "main", {"harbor_env_type": "e2b", "skip_prepare": True, **overrides}, tmp_path)
    assert recording.commands == []


@pytest.mark.parametrize(("clears_cache", "cache_mb"), [(True, 4096), (False, 0)])
def test_the_run_starts_with_a_clearing_sglang_or_no_cache(monkeypatch, tmp_path, clears_cache, cache_mb):
    freeze_environment(monkeypatch)
    _sglang_on_path(monkeypatch, tmp_path, clears_cache=clears_cache)
    recording = install_command_recorder(monkeypatch)
    launcher = _launcher_without_harbor_preflight(monkeypatch)

    call_entrypoint(
        launcher,
        "main",
        {"harbor_env_type": "e2b", "skip_prepare": True, "vlm_cache_size_mb": cache_mb},
        tmp_path,
    )

    (command,) = [cmd for cmd in recording.commands if "ray job submit" in cmd]
    assert f'"SGLANG_VLM_CACHE_SIZE_MB": "{cache_mb}"' in command


def test_an_agent_worker_runs_the_session_paths_trials(monkeypatch, tmp_path):
    command, env = _train_command(
        monkeypatch,
        tmp_path,
        rollout_path="session",
        agent_worker_url="http://100.87.162.43:11100",
        session_server_external_host="127.0.0.1",
    )

    assert "--custom-agent-function-mode worker" in command
    assert "--custom-agent-function-worker-url http://100.87.162.43:11100" in command
    assert "--session-server-external-host 127.0.0.1" in command


def test_an_agent_worker_needs_the_session_path(monkeypatch, tmp_path):
    freeze_environment(monkeypatch)
    install_command_recorder(monkeypatch)
    with pytest.raises(typer.BadParameter, match="--rollout-path session"):
        call_entrypoint(
            import_launch_script(LAUNCHER),
            "main",
            {"harbor_env_type": "e2b", "skip_prepare": True, "agent_worker_url": "http://w:11100"},
            tmp_path,
        )


@pytest.mark.parametrize(("worker_url", "warms_here"), [("", True), ("http://w:11100", False)])
def test_templates_are_warmed_where_the_sandboxes_are_created(monkeypatch, tmp_path, worker_url, warms_here):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)

    call_entrypoint(
        import_launch_script(LAUNCHER),
        "prepare",
        {"harbor_env_type": "e2b", "rollout_path": "session", "agent_worker_url": worker_url},
        tmp_path,
    )

    assert any("make_harbor_cua_tasks" in cmd for cmd in recording.commands)
    assert any("warm_templates" in cmd for cmd in recording.commands) == warms_here


@pytest.mark.parametrize(
    ("overrides", "eval_gpus", "save_interval"),
    [({}, 1, 5), ({"eval_interval": 1}, 1, 1), ({"backend": "fsdp"}, 0, 5), ({"eval_num_gpus": 0, "eval_interval": 1}, 0, 5)],
)
def test_snapshot_eval_gets_its_own_gpu_and_a_save_at_every_eval_point(
    monkeypatch, tmp_path, overrides, eval_gpus, save_interval
):
    command, _ = _train_command(monkeypatch, tmp_path, **overrides)

    assert (f"--eval-num-gpus 1 --eval-server-concurrency 32" in command) == bool(eval_gpus)
    assert f"--save-interval {save_interval} " in command


@pytest.mark.parametrize(
    ("overrides", "freezes"),
    [({}, False), ({"num_rollout": 100}, True), ({"num_rollout": 100, "freeze_vision_model": "off"}, False),
     ({"freeze_vision_model": "on"}, True)],
)
def test_the_vision_tower_freezes_from_3000_training_episodes(monkeypatch, tmp_path, overrides, freezes):
    command, _ = _train_command(monkeypatch, tmp_path, **overrides)

    assert ("--freeze-vision-model" in command) == freezes


def test_qwen38_trains_with_thinking_off_through_its_own_tito_family(monkeypatch, tmp_path):
    command, env = _train_command(monkeypatch, tmp_path, model_name="qwen3.8-27b", rollout_path="session")

    assert "--tito-model qwen38small" in command
    assert """--apply-chat-template-kwargs '{"enable_thinking": false}'""" in command
    assert re.search(r"--hf-checkpoint \S+/Qwen3.8-27B ", command)
    assert "--tensor-model-parallel-size 4" in command
    assert "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1" in command
    assert "MODEL_ARGS_ROTARY_BASE" not in env


def test_qwen3_vl_keeps_its_recipe(monkeypatch, tmp_path):
    command, _ = _train_command(monkeypatch, tmp_path, rollout_path="session")

    assert "--tito-model qwen3_vl" in command
    assert "--apply-chat-template-kwargs" not in command
    assert "--tensor-model-parallel-size 2" in command
    assert "--recompute-granularity" not in command


def test_qwen38_refuses_the_generate_path():
    launcher = import_launch_script(LAUNCHER)

    with pytest.raises(typer.BadParameter, match="--rollout-path session only"):
        launcher.main(harbor_env_type="e2b", model_name="qwen3.8-27b")


def test_dead_groups_are_dropped_only_when_asked(monkeypatch, tmp_path):
    dropping, _ = _train_command(
        monkeypatch, tmp_path, model_name="qwen3.8-27b", rollout_path="session", drop_zero_std_groups=True
    )
    keeping, _ = _train_command(monkeypatch, tmp_path, model_name="qwen3.8-27b", rollout_path="session")

    assert "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std" in dropping
    assert "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted" in keeping


def test_tis_replaces_the_rollout_logprob_ratio(monkeypatch, tmp_path):
    tis, _ = _train_command(monkeypatch, tmp_path, rollout_path="session", use_tis=True)
    default, _ = _train_command(monkeypatch, tmp_path, rollout_path="session")

    assert "--use-tis" in tis and "--use-rollout-logprobs" not in tis
    assert "--use-rollout-logprobs" in default and "--use-tis" not in default
