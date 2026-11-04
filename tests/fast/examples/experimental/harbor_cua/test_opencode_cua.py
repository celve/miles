"""OpenCode as a computer-use harness: only the computer tool, no side requests, the first screen in the first turn."""

import asyncio
import json
import shlex
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor", reason="pip install harbor -- the recipe's one extra dependency")

from examples.experimental.harbor_cua import opencode_cua as oc  # noqa: E402


class FakeEnvironment:
    def __init__(self, *, version: str = oc.OPENCODE_VERSION, mcp_log: str = ""):
        self.commands: list[str] = []
        self.uploads: list[tuple[str, str]] = []
        self._version = version
        self._mcp_log = mcp_log

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None, **_):
        self.commands.append(command)
        if "opencode --version" in command:
            return SimpleNamespace(return_code=0, stdout=f"{self._version}\n", stderr="")
        if command.startswith("cat ") and "computer_mcp.jsonl" in command:
            return SimpleNamespace(return_code=0, stdout=self._mcp_log, stderr="")
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source_path, target_path):
        self.uploads.append((str(source_path), target_path))


def _agent(tmp_path, **kwargs):
    provider = {"openai-compatible": {"models": {"m": {"modalities": {"input": ["text", "image"]}}}}}
    return oc.OpenCodeCUA(
        logs_dir=tmp_path,
        model_name="openai-compatible/m",
        opencode_config={"provider": provider},
        **kwargs,
    )


def _written_config(agent) -> dict:
    command = agent._build_register_config_command()
    return json.loads(shlex.split(command.split("echo ", 1)[1])[0])


def test_the_config_leaves_only_the_computer_tool_and_no_side_requests(tmp_path):
    config = _written_config(_agent(tmp_path, max_turns=30, temperature=1.0))

    assert config["permission"] == {"*": "deny", "computer_*": "allow"}
    assert config["agent"]["title"] == {"disable": True}
    assert config["compaction"] == {"auto": False, "prune": False}
    build = config["agent"]["build"]
    assert build["steps"] == 30 and build["temperature"] == 1.0
    assert build["prompt"].endswith(oc.RESPONSE_FORMAT)
    command = config["mcp"]["computer"]["command"]
    assert command[:2] == ["python3", "/installed-agent/computer/computer_mcp.py"]
    assert command[command.index("--log") + 1] == "/logs/agent/computer_mcp.jsonl"
    assert config["provider"]["openai-compatible"]["models"]["m"]["modalities"]["input"] == ["text", "image"]


def test_a_baked_in_release_skips_the_install(tmp_path):
    env = FakeEnvironment()
    asyncio.run(_agent(tmp_path).setup(env))

    assert not any("npm i -g" in c for c in env.commands)
    assert env.uploads == [(str(oc.MCP_SERVER), "/installed-agent/computer/computer_mcp.py")]


def test_another_release_is_installed_at_the_pin(tmp_path):
    env = FakeEnvironment(version="1.0.0")
    asyncio.run(_agent(tmp_path).setup(env))

    assert any(f"npm i -g opencode-ai@{oc.OPENCODE_VERSION}" in c for c in env.commands)


def test_the_first_screen_rides_the_first_turn(tmp_path):
    env = FakeEnvironment()
    asyncio.run(_agent(tmp_path, display=":0").run("Open the file.", env, SimpleNamespace()))

    capture = next(c for c in env.commands if "--screenshot" in c)
    assert "--display :0" in capture
    run = next(c for c in env.commands if " run --format=json" in c)
    assert "--model=openai-compatible/m" in run
    assert "--file /installed-agent/computer/first.png" in run
    assert run.index("--file") < run.index("-- 'Open the file.'")


def test_a_lost_desktop_is_raised_after_the_run(tmp_path):
    env = FakeEnvironment(mcp_log=json.dumps({"action": {"action": "wait"}, "infra_error": "capture failed"}) + "\n")
    with pytest.raises(oc.DesktopLost):
        asyncio.run(_agent(tmp_path).run("Open the file.", env, SimpleNamespace()))


def test_a_policy_error_is_not_a_lost_desktop(tmp_path):
    env = FakeEnvironment(mcp_log=json.dumps({"action": {"action": "fly"}, "error": "unknown action"}) + "\n")
    asyncio.run(_agent(tmp_path).run("Open the file.", env, SimpleNamespace()))


def test_the_mcp_environment_reaches_the_server_and_the_first_capture(tmp_path):
    agent = _agent(tmp_path, mcp_env={"PATH": "/opt/env/bin:/usr/bin"})
    assert _written_config(agent)["mcp"]["computer"]["environment"] == {"PATH": "/opt/env/bin:/usr/bin"}
