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


def _sglang_on_path(monkeypatch, root, *, clears_cache: bool) -> None:
    managers = root / "site/sglang/srt/managers"
    managers.mkdir(parents=True)
    (root / "site/sglang/__init__.py").write_text("")
    body = "def clear_mm_embedding_cache() -> None:\n    pass\n" if clears_cache else "embedding_cache = None\n"
    (managers / "mm_schedule.py").write_text(body)
    monkeypatch.syspath_prepend(str(root / "site"))


def test_the_cache_refuses_an_sglang_that_never_clears_it(monkeypatch, tmp_path):
    """Without the SGLang fix, a screenshot seen again after a weight update is embedded by the old vision weights."""
    _sglang_on_path(monkeypatch, tmp_path, clears_cache=False)
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)

    with pytest.raises(typer.BadParameter, match="clears its ViT embedding cache"):
        call_entrypoint(
            import_launch_script(LAUNCHER), "main", {"harbor_env_type": "e2b", "skip_prepare": True}, tmp_path
        )
    assert recording.commands == []


@pytest.mark.parametrize(("clears_cache", "cache_mb"), [(True, 4096), (False, 0)])
def test_the_run_starts_with_a_clearing_sglang_or_no_cache(monkeypatch, tmp_path, clears_cache, cache_mb):
    _sglang_on_path(monkeypatch, tmp_path, clears_cache=clears_cache)
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)

    call_entrypoint(
        import_launch_script(LAUNCHER),
        "main",
        {"harbor_env_type": "e2b", "skip_prepare": True, "vlm_cache_size_mb": cache_mb},
        tmp_path,
    )

    (command,) = [cmd for cmd in recording.commands if "ray job submit" in cmd]
    assert f'"SGLANG_VLM_CACHE_SIZE_MB": "{cache_mb}"' in command
