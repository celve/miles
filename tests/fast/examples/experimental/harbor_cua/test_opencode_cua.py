"""OpenCode as a computer-use harness: only the computer tool, no side requests, the first screen in the first turn."""

import asyncio
import json
import shlex
from datetime import UTC, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("harbor", reason="pip install harbor -- the recipe's one extra dependency")

from examples.experimental.harbor_cua import opencode_cua as oc  # noqa: E402
from harbor.agents.installed.base import NonZeroAgentExitCodeError  # noqa: E402


class FakeEnvironment:
    def __init__(
        self, *, version: str = oc.OPENCODE_VERSION, mcp_log: str = "", opencode_exit: int = 0, capture_exit: int = 0
    ):
        self.commands: list[str] = []
        self.envs: list[dict] = []
        self.uploads: list[tuple[str, str]] = []
        self._version = version
        self._mcp_log = mcp_log
        self._opencode_exit = opencode_exit
        self._capture_exit = capture_exit

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None, **_):
        self.commands.append(command)
        self.envs.append(env or {})
        if "opencode --version" in command:
            return SimpleNamespace(return_code=0, stdout=f"{self._version}\n", stderr="")
        if command.startswith("cat ") and "computer_mcp.jsonl" in command:
            return SimpleNamespace(return_code=0, stdout=self._mcp_log, stderr="")
        if "--screenshot" in command:
            return SimpleNamespace(return_code=self._capture_exit, stdout="", stderr="infra_error: no screenshot")
        if " run --format=json" in command:
            return SimpleNamespace(return_code=self._opencode_exit, stdout="", stderr="")
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
    command = agent._register_config_command()
    return json.loads(shlex.split(command.split("echo ", 1)[1])[0])


def test_the_config_leaves_only_the_computer_tool_and_no_side_requests(tmp_path):
    config = _written_config(_agent(tmp_path, max_turns=30, temperature=1.0))

    assert config["permission"] == {"*": "deny", "computer_*": "allow"}
    assert config["agent"]["title"] == {"disable": True}
    assert config["compaction"] == {"auto": False, "prune": False}
    assert config["snapshot"] is False
    build = config["agent"]["build"]
    assert build["steps"] == 31 and build["temperature"] == 1.0
    assert build["prompt"].endswith(oc.RESPONSE_FORMAT)
    command = config["mcp"]["computer"]["command"]
    assert command[:2] == ["python3", "/installed-agent/computer/computer_mcp.py"]
    assert command[command.index("--log") + 1] == "/logs/agent/computer_mcp.jsonl"
    assert "--stop-on-terminate" in command
    assert command[command.index("--max-calls") + 1] == "30"
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


def test_every_opencode_invocation_uses_its_own_config_dir(tmp_path):
    """An image's ~/.config can belong to another user than the agent's (lite.osworld's /home/user)."""
    env = FakeEnvironment()
    agent = _agent(tmp_path)
    asyncio.run(agent.setup(env))
    asyncio.run(agent.run("Open the file.", env, SimpleNamespace()))

    calls = [
        (c, e) for c, e in zip(env.commands, env.envs, strict=True) if "opencode" in c and "--screenshot" not in c
    ]
    assert calls and all(e.get("XDG_CONFIG_HOME") == "/installed-agent/computer/xdg-config" for _, e in calls)
    config = next(c for c, _ in calls if "opencode.json" in c)
    assert '"$XDG_CONFIG_HOME/opencode"/opencode.json' in config and "~/.config" not in config


def _row(action: str, **extra) -> str:
    return json.dumps({"action": {"action": action}, **extra}) + "\n"


def test_opencode_stopped_by_a_terminate_ends_the_episode_cleanly(tmp_path):
    log = _row("left_click", screenshot=True) + _row("terminate") + json.dumps({"stop": "terminate"}) + "\n"
    env = FakeEnvironment(mcp_log=log, opencode_exit=143)
    asyncio.run(_agent(tmp_path).run("Open the file.", env, SimpleNamespace()))


def test_opencode_failing_without_the_server_stopping_it_still_fails(tmp_path):
    env = FakeEnvironment(mcp_log=_row("left_click", screenshot=True), opencode_exit=1)
    with pytest.raises(NonZeroAgentExitCodeError):
        asyncio.run(_agent(tmp_path).run("Open the file.", env, SimpleNamespace()))


def test_a_first_screen_that_cannot_be_captured_is_a_lost_desktop(tmp_path):
    """Before any policy call, so the trial is discarded rather than scored."""
    env = FakeEnvironment(capture_exit=1)
    with pytest.raises(oc.DesktopLost):
        asyncio.run(_agent(tmp_path).run("Open the file.", env, SimpleNamespace()))
    assert not any(" run --format=json" in c for c in env.commands)


def test_opencode_runs_where_it_is_about_noon(tmp_path):
    env = FakeEnvironment()
    asyncio.run(_agent(tmp_path).run("Open the file.", env, SimpleNamespace()))

    (run_env,) = [e for c, e in zip(env.commands, env.envs, strict=True) if " run --format=json" in c]
    assert 11 <= datetime.now(ZoneInfo(run_env["TZ"])).hour <= 13


@pytest.mark.parametrize("hour", range(24))
def test_every_utc_hour_maps_to_a_zone_at_noon(hour):
    now = datetime(2026, 11, 4, hour, 30, tzinfo=UTC)
    assert now.astimezone(ZoneInfo(oc._noon_zone(now))).hour == 12
