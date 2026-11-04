"""Harbor's installed OpenCode as a computer-use harness: Qwen3-VL's ``computer_use`` tool through ``computer_mcp.py``.

OpenCode's coding setup is switched off so a request carries only what the policy acts on: the
``computer`` MCP server is the one tool source, OpenCode's coding prompt is replaced, and the title
call and compaction, which would add requests outside the episode or rewrite its history, are off.
The desktop is the image's own (an X server already on ``display``), and the first screen rides the
first user message, since ``computer_use`` has no screenshot action.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import NonZeroAgentExitCodeError, with_prompt_template
from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from examples.experimental.harbor_cua.osworld_qwen.qwen3vl import _system_prompt

# The release the request shapes were probed on (LIN-2696); OpenCode's 2.x line is a different package.
OPENCODE_VERSION = "1.19.5"
MCP_SERVER = Path(__file__).with_name("computer_mcp.py")
_REMOTE_DIR = "/installed-agent/computer"
_MCP_LOG = "/logs/agent/computer_mcp.jsonl"
_FIRST_SCREEN = f"{_REMOTE_DIR}/first.png"
_PATH = 'export PATH="$HOME/.opencode/bin:$PATH"; [ -s "$HOME/.nvm/nvm.sh" ] && . "$HOME/.nvm/nvm.sh"; '
RESPONSE_FORMAT = "# Response format" + _system_prompt().split("# Response format", 1)[1]
SYSTEM_PROMPT = "You operate a Linux desktop to complete the user's task.\n\n" + RESPONSE_FORMAT


class DesktopLost(RuntimeError):
    """The MCP server could not capture the screen: the sandbox's desktop is gone, not the policy's doing."""


class OpenCodeCUA(OpenCode):
    def __init__(
        self,
        *args: Any,
        display: str = ":1",
        settle_sec: float = 2.0,
        capture_cmd: str = "import -window root {out}",
        mcp_python: str = "python3",
        max_turns: int | None = None,
        temperature: float | None = None,
        opencode_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("version", OPENCODE_VERSION)
        super().__init__(*args, opencode_config=opencode_config, **kwargs)
        self._display = display
        self._settle_sec = settle_sec
        self._capture_cmd = capture_cmd
        self._mcp_python = mcp_python
        self._opencode_config = self._deep_merge(self._cua_config(max_turns, temperature), self._opencode_config)

    def _mcp_command(self, *extra: str) -> list[str]:
        return [
            self._mcp_python,
            f"{_REMOTE_DIR}/computer_mcp.py",
            "--display",
            self._display,
            "--settle-sec",
            str(self._settle_sec),
            "--capture-cmd",
            self._capture_cmd,
            *extra,
        ]

    def _cua_config(self, max_turns: int | None, temperature: float | None) -> dict[str, Any]:
        build: dict[str, Any] = {"prompt": SYSTEM_PROMPT}
        if max_turns is not None:
            build["steps"] = max_turns
        if temperature is not None:
            build["temperature"] = temperature
        return {
            "mcp": {"computer": {"type": "local", "command": self._mcp_command("--log", _MCP_LOG)}},
            "permission": {"*": "deny", "computer_*": "allow"},
            "compaction": {"auto": False, "prune": False},
            "share": "disabled",
            "autoupdate": False,
            "agent": {"title": {"disable": True}, "build": build},
        }

    async def install(self, environment: BaseEnvironment) -> None:
        # An image with this release baked in skips the per-trial apt + nvm + npm install.
        result = await environment.exec(command=_PATH + "opencode --version")
        if result.return_code != 0 or (result.stdout or "").strip() != self._version:
            await super().install(environment)

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        await self.exec_as_root(environment, f"mkdir -p {_REMOTE_DIR} /logs/agent && chmod 777 {_REMOTE_DIR}")
        await environment.upload_file(MCP_SERVER, f"{_REMOTE_DIR}/computer_mcp.py")
        await self.exec_as_agent(
            environment,
            f"for _ in $(seq 60); do DISPLAY={shlex.quote(self._display)} xdotool getdisplaygeometry && exit 0; sleep 1; done; exit 1",
        )

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        self._instruction = instruction
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")
        env = {
            "OPENCODE_FAKE_VCS": "git",
            "XDG_DATA_HOME": "/logs/agent/opencode/xdg-data",
            "XDG_STATE_HOME": "/logs/agent/opencode/xdg-state",
        }
        await self.exec_as_agent(environment, self._build_register_config_command(), env=env)
        first = " ".join(shlex.quote(part) for part in self._mcp_command("--screenshot", _FIRST_SCREEN))
        await self.exec_as_agent(environment, first)
        try:
            await self.exec_as_agent(
                environment,
                _PATH + f"opencode --model={shlex.quote(self.model_name)} run --format=json --thinking "
                f"--file {_FIRST_SCREEN} --dangerously-skip-permissions -- {shlex.quote(instruction)} "
                "2>&1 </dev/null | stdbuf -oL tee /logs/agent/opencode.txt",
                env=env,
            )
            if messages := self._error_messages():
                raise NonZeroAgentExitCodeError("OpenCode emitted error event(s): " + "; ".join(messages[:3]))
        finally:
            await self._raise_if_desktop_lost(environment)

    async def _raise_if_desktop_lost(self, environment: BaseEnvironment) -> None:
        result = await environment.exec(command=f"cat {_MCP_LOG} 2>/dev/null || true")
        for line in (result.stdout or "").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "infra_error" in row:
                raise DesktopLost(row["infra_error"])
