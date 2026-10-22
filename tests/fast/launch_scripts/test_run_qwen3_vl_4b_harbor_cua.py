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
