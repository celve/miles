"""``Qwen3VLComputer1`` on a desktop its sandbox image already runs, such as CUA-Lite's GNOME template.

Harbor's computer-1 brings up its own Xvfb, XFCE and Chromium and treats Chromium's death as
the session's; here the image's display is used as it is, and only the X server's socket says
the session is alive.
"""

import asyncio
import json
import shlex
from dataclasses import asdict
from pathlib import PurePosixPath
from types import SimpleNamespace
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


# CUA-Lite's LoopDetectWrapper looks for a repeated block of up to this many actions.
_LOOP_MAX_PERIOD = 5
_SUMMARY_HEADER = "Earlier steps, oldest first (their screenshots and full replies are no longer shown):"


class LoopDetected(Exception):
    """The policy repeated one block of actions ``loop_limit`` times; the episode ends on the state it reached."""


def _loops(fingerprints: list[str], times: int) -> bool:
    return any(
        len(fingerprints) >= times * period and fingerprints[-times * period :] == fingerprints[-period:] * times
        for period in range(1, _LOOP_MAX_PERIOD + 1)
    )


def _action_line(reply: Any) -> str:
    """The action a computer-1 JSON reply took, on one line."""
    text = reply.get("content") if isinstance(reply, dict) else None
    if isinstance(text, str) and "{" in text:
        try:
            action = json.loads(text[text.index("{") : text.rindex("}") + 1]).get("action")
        except (ValueError, AttributeError):
            action = None
        if isinstance(action, dict):
            return json.dumps(action, separators=(",", ":"))
    return "(a reply with no parsable action)"


def _images_in(prompt: Any) -> int:
    turns = [prompt] if isinstance(prompt, dict) else prompt if isinstance(prompt, list) else []
    return sum(
        isinstance(part, dict) and part.get("type") == "image_url"
        for turn in turns
        for part in (turn.get("content") if isinstance(turn, dict) and isinstance(turn.get("content"), list) else [])
    )


def desktop_prompt_template(template: str) -> str:
    for cut in _PROMPT_CUTS:
        if template.count(cut) != 1:
            raise ValueError(f"computer-1's prompt template no longer holds {cut!r} exactly once")
        template = template.replace(cut, "")
    return template


async def _no_compaction(*args: Any, **kwargs: Any) -> None:
    return None


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

    def _capture_script(self, env_png: str) -> str:
        """Shell that writes the desktop to ``env_png`` as a PNG, at ``screenshot_size`` when one is set."""
        resize = ""
        if self._screenshot_size is not None:
            width, height = self._screenshot_size
            resize = f" && convert {shlex.quote(env_png)} -resize {width}x{height}! {shlex.quote(env_png)}"
        return f"import -window root {shlex.quote(env_png)}{resize}"

    async def fetch_screenshot(self, env_path: PurePosixPath | str) -> str:
        target = str(env_path)
        if PurePosixPath(target).suffix.lower() != ".png":
            raise ValueError(f"a desktop session writes PNG screenshots, not {target!r}")
        env_png = f"{_SCREENSHOT_DIR}/latest.png"
        script = (
            f"set -e; export DISPLAY={_DEFAULT_DISPLAY}; "
            f"mkdir -p {shlex.quote(_SCREENSHOT_DIR)} {shlex.quote(str(PurePosixPath(target).parent))}; "
            f"{self._capture_script(env_png)}; "
            f"cp {shlex.quote(env_png)} {shlex.quote(target)}"
        )
        await self._exec(_bash_inline(script), timeout_sec=max(30, self._request_timeout_sec), label="screenshot")
        return target


class DesktopQwen3VLComputer1(Qwen3VLComputer1):
    """``settle_sec`` waits after every screen-changing action; ``keep_screenshots`` caps the images a request carries.

    ``keep_turns`` keeps that many replies in full and collapses older turns into one line per action; ``loop_limit``
    ends the episode once a block of up to five actions repeats that many times; ``compaction=False`` and
    ``final_answer_fallback=False`` keep Harbor's summarizer and final-answer calls off the policy's session.
    """

    session_class: type[DesktopSession] = DesktopSession

    def __init__(
        self,
        *args: Any,
        settle_sec: float = 2.0,
        screenshot_width: int | None = None,
        screenshot_height: int | None = None,
        keep_screenshots: int | None = None,
        keep_turns: int | None = None,
        loop_limit: int | None = None,
        compaction: bool = True,
        final_answer_fallback: bool = True,
        **kwargs: Any,
    ) -> None:
        if (screenshot_width is None) != (screenshot_height is None):
            raise ValueError("set screenshot_width and screenshot_height together")
        if keep_screenshots is not None and keep_screenshots < 1:
            raise ValueError("keep_screenshots must be at least 1: the request needs the current screen")
        if keep_turns is not None and keep_turns < 1:
            raise ValueError("keep_turns must be at least 1")
        if loop_limit is not None and loop_limit < 2:
            raise ValueError("loop_limit must be at least 2: one action is no repetition")
        super().__init__(*args, **kwargs)
        self._settle_sec = settle_sec
        self._screenshot_size = (screenshot_width, screenshot_height) if screenshot_width is not None else None
        self._keep_screenshots = keep_screenshots
        self._keep_turns = keep_turns
        self._loop_limit = loop_limit
        self._final_answer_fallback = final_answer_fallback
        self._fingerprints: list[str] = []
        self._summary: dict[str, Any] | None = None
        self._summary_lines: list[str] = []
        if not compaction:
            self._compactor.maybe_proactively_compact = _no_compaction
            self._compactor.reactive_compaction = _no_compaction

    def _build_provider(self) -> DesktopJsonProvider:
        return DesktopJsonProvider.from_agent(self)

    async def setup(self, environment: BaseEnvironment) -> None:
        geometry = self._desktop_geometry
        self._session = self.session_class(
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

    def _trim_history(self, messages: Any, prompt_images: int) -> None:
        if self._keep_screenshots is not None:
            keep = max(0, self._keep_screenshots - prompt_images)
            self._compactor._trim_old_screenshots(SimpleNamespace(messages=messages), keep_last=keep)

    def _collapse_turns(self, messages: list[Any]) -> None:
        """Keep the instruction and the last ``keep_turns`` turns; every older turn becomes one summary line."""
        if self._keep_turns is None or len(messages) < 2:
            return
        head, body = messages[0], messages[1:]
        if body and body[0] is self._summary:
            body = body[1:]
        # Each turn is the prompts that led to one reply, then that reply.
        turns, turn = [], []
        for message in body:
            turn.append(message)
            if isinstance(message, dict) and message.get("role") == "assistant":
                turns.append(turn)
                turn = []
        if turn or len(turns) <= self._keep_turns:
            return
        cut = len(turns) - self._keep_turns
        start = len(self._summary_lines) + 1
        self._summary_lines += [f"{start + i}. {_action_line(t[-1])}" for i, t in enumerate(turns[:cut])]
        self._summary = {"role": "user", "content": "\n".join([_SUMMARY_HEADER, *self._summary_lines])}
        messages[:] = [head, self._summary, *(m for t in turns[cut:] for m in t)]

    async def _execute_action(self, action: Any, episode: int) -> Any:
        if action is not None and action.type not in TERMINAL_ACTION_TYPES:
            self._fingerprints.append(json.dumps(asdict(action), sort_keys=True, default=str))
        return await super()._execute_action(action, episode)

    async def _query_litellm(self, chat: Any, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        if self._loop_limit is not None and _loops(self._fingerprints, self._loop_limit):
            self._early_termination_reason = "loop_detected"
            raise LoopDetected(self._fingerprints[-1])
        self._collapse_turns(chat.messages)
        # A parse-error retry carries no screen of its own, so the history keeps the current one.
        self._trim_history(chat.messages, _images_in(prompt))
        return await super()._query_litellm(chat, prompt, *args, **kwargs)

    async def run(self, instruction: str, environment: BaseEnvironment, context: Any) -> None:
        try:
            await super().run(instruction, environment, context)
        except LoopDetected:
            # Harbor has recorded the trial by now; the verifier scores the state the loop left.
            pass

    async def _maybe_write_final_answer_fallback(self, instruction: str) -> None:
        if self._final_answer_fallback:
            await super()._maybe_write_final_answer_fallback(instruction)

    def _fallback_message_history(self) -> list[Any]:
        # The turn-cap fallback sends this history with the latest screen attached whenever images are on.
        history = super()._fallback_message_history()
        self._trim_history(history, int(self._enable_images and self._latest_screenshot_path is not None))
        return history
