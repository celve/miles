"""``Qwen3VLComputer1`` on a desktop its sandbox image already runs, such as CUA-Lite's GNOME template.

Harbor's computer-1 brings up its own Xvfb, XFCE and Chromium and treats Chromium's death as
the session's; here the image's display is used as it is, and only the X server's socket says
the session is alive.
"""

import asyncio
import shlex
from pathlib import PurePosixPath
from typing import Any

from examples.experimental.harbor_cua.qwen_computer import Qwen3VLComputer1, Qwen3VLJsonProvider
from harbor.agents.computer_1.runtime import (
    _DEFAULT_DISPLAY,
    _RUNTIME_DIR,
    _SCREENSHOT_DIR,
    TERMINAL_ACTION_TYPES,
    Computer1Session,
    ComputerAction,
    RuntimeRequestError,
    _bash_inline,
)
from harbor.environments.base import BaseEnvironment

# zoom crops the next frame, against which Qwen3-VL's grid clicks are then mis-scaled.
UNAVAILABLE_ACTIONS = frozenset({"zoom", "navigate", "go_back", "go_forward"})
_UNCHANGED_SCREEN_ACTIONS = frozenset({"wait", "sleep", *TERMINAL_ACTION_TYPES})

# Each passage is removed from Harbor's computer-1-json.txt, and a missing one fails the build.
_PROMPT_CUTS = (
    " zoom, navigate,",
    '\n    "url": <string, optional, used by navigate>,',
    '\n    "zoom_region": <[x0, y0, x1, y1] in desktop pixels, optional, used by zoom>,',
    '\n- For "zoom": provide "zoom_region" as [x0, y0, x1, y1] in desktop pixels.'
    "\n  The NEXT screenshot is cropped (no resize) to that region, then auto-resets."
    "\n  Use this to inspect a small UI area at native pixel density.",
    '\n- For "navigate": provide the destination URL in "url".',
)


def desktop_prompt_template(template: str) -> str:
    for cut in _PROMPT_CUTS:
        if template.count(cut) != 1:
            raise ValueError(f"computer-1's prompt template no longer holds {cut!r} exactly once")
        template = template.replace(cut, "")
    return template


class DesktopJsonProvider(Qwen3VLJsonProvider):
    screenshot_format = "png"

    @classmethod
    def from_agent(cls, agent: Qwen3VLComputer1) -> "DesktopJsonProvider":
        provider = super().from_agent(agent)
        provider._prompt_template = desktop_prompt_template(provider._prompt_template)
        return provider


class DesktopSession(Computer1Session):
    def __init__(self, *args: Any, settle_sec: float, screenshot_size: tuple[int, int] | None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._settle_sec = settle_sec
        self._screenshot_size = screenshot_size

    async def start(self) -> None:
        if self._started:
            return
        dirs = " ".join(shlex.quote(d) for d in (_RUNTIME_DIR, _SCREENSHOT_DIR, str(self._agent_dir)))
        await self._exec(_bash_inline(f"mkdir -p {dirs}"), timeout_sec=60, label="mkdir runtime dirs", retries=2)
        await self._wait_for_x11()
        self._started = True

    async def is_session_alive(self) -> bool:
        try:
            result = await self._raw_exec("test -S /tmp/.X11-unix/X1 && echo ok || echo down", timeout_sec=5)
        except Exception:
            return False
        return "ok" in (result.stdout or "")

    async def reset(self) -> None:
        raise RuntimeError("a desktop session has no browser to relaunch")

    async def _recover_chromium_if_needed(self, action_type: str, exc: Exception) -> dict[str, Any] | None:
        return None

    async def execute(self, action: ComputerAction) -> dict[str, Any]:
        if action.type in UNAVAILABLE_ACTIONS:
            raise RuntimeRequestError(action.type, 400, f"{action.type!r} is not available on this desktop")
        result = await super().execute(action)
        if action.type not in _UNCHANGED_SCREEN_ACTIONS and self._settle_sec > 0:
            await asyncio.sleep(self._settle_sec)
        return result

    async def fetch_screenshot(self, env_path: PurePosixPath | str) -> str:
        target = str(env_path)
        if PurePosixPath(target).suffix.lower() != ".png":
            raise ValueError(f"a desktop session writes PNG screenshots, not {target!r}")
        env_png = f"{_SCREENSHOT_DIR}/latest.png"
        resize = ""
        if self._screenshot_size is not None:
            width, height = self._screenshot_size
            resize = f" && convert {shlex.quote(env_png)} -resize {width}x{height}! {shlex.quote(env_png)}"
        script = (
            f"set -e; export DISPLAY={_DEFAULT_DISPLAY}; "
            f"mkdir -p {shlex.quote(_SCREENSHOT_DIR)} {shlex.quote(str(PurePosixPath(target).parent))}; "
            f"import -window root {shlex.quote(env_png)}{resize}; "
            f"cp {shlex.quote(env_png)} {shlex.quote(target)}"
        )
        await self._exec(_bash_inline(script), timeout_sec=max(30, self._request_timeout_sec), label="screenshot")
        return target


class DesktopQwen3VLComputer1(Qwen3VLComputer1):
    """``settle_sec`` waits after every screen-changing action; ``keep_screenshots`` caps the images a request carries."""

    def __init__(
        self,
        *args: Any,
        settle_sec: float = 2.0,
        screenshot_width: int | None = None,
        screenshot_height: int | None = None,
        keep_screenshots: int | None = None,
        **kwargs: Any,
    ) -> None:
        if (screenshot_width is None) != (screenshot_height is None):
            raise ValueError("set screenshot_width and screenshot_height together")
        if keep_screenshots is not None and keep_screenshots < 1:
            raise ValueError("keep_screenshots must be at least 1: the request needs the current screen")
        super().__init__(*args, **kwargs)
        self._settle_sec = settle_sec
        self._screenshot_size = (screenshot_width, screenshot_height) if screenshot_width is not None else None
        self._keep_screenshots = keep_screenshots

    def _build_provider(self) -> DesktopJsonProvider:
        return DesktopJsonProvider.from_agent(self)

    async def setup(self, environment: BaseEnvironment) -> None:
        geometry = self._desktop_geometry
        self._session = DesktopSession(
            environment=environment,
            agent_dir=self._env_io_dir,
            desktop_width=geometry.desktop_width,
            desktop_height=geometry.desktop_height,
            window_width=geometry.desktop_width,
            window_height=geometry.desktop_height,
            readiness_timeout_sec=self._runtime_readiness_timeout_sec,
            request_timeout_sec=self._runtime_request_timeout_sec,
            extra_env=self._extra_env,
            user=environment.default_user,
            enable_bash=self._enable_bash,
            settle_sec=self._settle_sec,
            screenshot_size=self._screenshot_size,
        )
        await self._session.start()

    async def _query_litellm(self, chat: Any, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        # The new screen rides ``prompt``, so the history keeps one fewer.
        if self._keep_screenshots is not None:
            self._compactor._trim_old_screenshots(chat, keep_last=self._keep_screenshots - 1)
        return await super()._query_litellm(chat, prompt, *args, **kwargs)
