"""A stdio MCP server that gives an installed agent CLI Qwen3-VL's ``computer_use`` tool on an X desktop.

Copied into the sandbox and started by the CLI, so it uses the standard library only. Registered as
server ``computer`` with one tool ``use``, which OpenCode names ``computer_use``: the tool name and
schema upstream OSWorld's Qwen3-VL agent prompts with (``osworld_qwen/qwen3vl.py``). Coordinates are
on Qwen3-VL's 0..999 grid; the xdotool calls follow Harbor computer-1's (``runtime.build_xdotool_argv``).
Every action but ``terminate`` and ``answer`` returns a screenshot taken after ``--settle-sec``.

    python3 computer_mcp.py [--display :1] [--settle-sec 2] [--capture-cmd 'import -window root {out}'] [--log f.jsonl]
    python3 computer_mcp.py --screenshot out.png   # one capture with the same settings, then exit

With ``--stop-on-terminate`` a ``terminate`` call, and with ``--max-calls N`` the N-th call, ends the client that
started the server (its parent process), after logging ``{"stop": <reason>}``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time

PROTOCOL_VERSION = "2025-06-18"
GRID = 999

ACTIONS = (
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "scroll",
    "hscroll",
    "wait",
    "terminate",
    "answer",
)
_ACTION_DESCRIPTION = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
"""
TOOL = {
    "name": "use",
    "description": "\n".join(
        [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.",
            "* The screen's resolution is 1000x1000.",
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ]
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {"description": _ACTION_DESCRIPTION, "enum": list(ACTIONS), "type": "string"},
            "keys": {"description": "Required only by `action=key`.", "type": "array"},
            "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"},
            "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
            "pixels": {"description": "The amount of scrolling.", "type": "number"},
            "time": {"description": "The seconds to wait.", "type": "number"},
            "status": {"description": "The status of the task.", "type": "string", "enum": ["success", "failure"]},
        },
        "required": ["action"],
    },
}

# computer-1's runtime.XDOTOOL_KEY_ALIASES.
KEY_ALIASES = {
    "alt": "alt",
    "arrowdown": "Down",
    "arrowleft": "Left",
    "arrowright": "Right",
    "arrowup": "Up",
    "backspace": "BackSpace",
    "cmd": "super",
    "command": "super",
    "control": "ctrl",
    "ctrl": "ctrl",
    "delete": "Delete",
    "down": "Down",
    "end": "End",
    "enter": "Return",
    "esc": "Escape",
    "escape": "Escape",
    "home": "Home",
    "insert": "Insert",
    "left": "Left",
    "meta": "super",
    "option": "alt",
    "pagedown": "Next",
    "pageup": "Prior",
    "return": "Return",
    "right": "Right",
    "shift": "shift",
    "space": "space",
    "spacebar": "space",
    "tab": "Tab",
    "up": "Up",
}
_BUTTONS = {"left_click": "1", "middle_click": "2", "right_click": "3", "double_click": "1", "triple_click": "1"}
_REPEATS = {"double_click": 2, "triple_click": 3}
_NO_SCREEN = frozenset({"terminate", "answer"})


class ActionError(ValueError):
    """An action the policy got wrong; returned to it as a tool error, never raised out of the server."""


class CaptureError(RuntimeError):
    """The desktop could not be captured, which no action causes; logged as ``infra_error`` for the harness."""


def scale(value: object, size: int) -> int:
    """computer-1's ``scale_normalized_coordinate`` for one axis."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionError(f"coordinate values must be numbers, got {value!r}")
    return max(0, min(size - 1, round(value * (size - 1) / GRID)))


def point(args: dict, geometry: tuple[int, int]) -> list[str]:
    coordinate = args.get("coordinate")
    if not isinstance(coordinate, list) or len(coordinate) != 2:
        raise ActionError(f"action={args.get('action')} needs coordinate [x, y], got {coordinate!r}")
    return [str(scale(coordinate[0], geometry[0])), str(scale(coordinate[1], geometry[1]))]


def key_combo(keys: object) -> str:
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, list) or not keys or not all(isinstance(k, str) and k.strip() for k in keys):
        raise ActionError(f"action=key needs keys as a non-empty list of strings, got {keys!r}")
    parts = [part.strip() for key in keys for part in key.split("+") if part.strip()]
    return "+".join(KEY_ALIASES.get(part.lower(), part) for part in parts)


def xdotool_argv(args: dict, geometry: tuple[int, int]) -> list[str] | None:
    """The xdotool arguments for one ``computer_use`` call; None for an action xdotool has no part in."""
    action = args.get("action")
    if action not in ACTIONS:
        raise ActionError(f"unknown action {action!r}; use one of {', '.join(ACTIONS)}")
    if action in _BUTTONS:
        x, y = point(args, geometry)
        repeat = _REPEATS.get(action, 1)
        return ["mousemove", x, y, "click", *(["--repeat", str(repeat)] if repeat > 1 else []), _BUTTONS[action]]
    if action == "mouse_move":
        return ["mousemove", *point(args, geometry)]
    if action == "left_click_drag":
        return ["mousedown", "1", "mousemove", *point(args, geometry), "mouseup", "1"]
    if action == "type":
        text = args.get("text")
        if not isinstance(text, str):
            raise ActionError(f"action=type needs text as a string, got {text!r}")
        return ["type", "--clearmodifiers", "--", text]
    if action == "key":
        return ["key", "--clearmodifiers", key_combo(args.get("keys"))]
    if action in ("scroll", "hscroll"):
        pixels = args.get("pixels", 0)
        if isinstance(pixels, bool) or not isinstance(pixels, (int, float)):
            raise ActionError(f"action={action} needs pixels as a number, got {pixels!r}")
        if not pixels:
            return None
        # pyautogui's sign, as Qwen3-VL's OSWorld agent executes it: positive scrolls up (or left), one click a unit.
        button = ("4" if pixels > 0 else "5") if action == "scroll" else ("6" if pixels > 0 else "7")
        moves = ["mousemove", *point(args, geometry)] if "coordinate" in args else []
        return [*moves, "click", "--repeat", str(max(1, round(abs(pixels)))), button]
    return None


class Desktop:
    def __init__(self, display: str, settle_sec: float, capture_cmd: str, log_path: str | None) -> None:
        self.env = {**os.environ, "DISPLAY": display}
        self.settle_sec = settle_sec
        self.capture_cmd = capture_cmd
        self.log_path = log_path
        self._geometry: tuple[int, int] | None = None

    def geometry(self) -> tuple[int, int]:
        if self._geometry is None:
            try:
                out = self._run(["xdotool", "getdisplaygeometry"]).split()
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                raise CaptureError(f"display geometry unavailable: {exc}") from exc
            self._geometry = (int(out[0]), int(out[1]))
        return self._geometry

    def _run(self, argv: list[str], timeout: float = 60) -> str:
        result = subprocess.run(argv, env=self.env, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"{argv[0]} exited {result.returncode}: {result.stderr.strip()[:500]}")
        return result.stdout

    def screenshot(self, out: str) -> None:
        try:
            self._run(["bash", "-c", self.capture_cmd.replace("{out}", shlex.quote(out))])
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            raise CaptureError(f"screen capture failed: {exc}") from exc

    def screenshot_png(self) -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "screen.png")
            self.screenshot(path)
            try:
                with open(path, "rb") as f:
                    return f.read()
            except OSError as exc:
                raise CaptureError(f"the capture command wrote no screenshot: {exc}") from exc

    def call(self, args: dict) -> dict:
        started = time.monotonic()
        action = args.get("action")
        if action == "terminate":
            content = [
                {"type": "text", "text": f"The task is terminated with status {args.get('status', 'success')}."}
            ]
        elif action == "answer":
            content = [{"type": "text", "text": "The answer is recorded."}]
        else:
            argv = xdotool_argv(args, self.geometry())
            if argv is not None:
                try:
                    self._run(["xdotool", *argv])
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    raise ActionError(str(exc)) from exc
            if action == "wait":
                wait = args.get("time", self.settle_sec)
                if isinstance(wait, bool) or not isinstance(wait, (int, float)) or wait < 0:
                    raise ActionError(f"action=wait needs time as a non-negative number, got {wait!r}")
                time.sleep(min(float(wait), 60.0))
            else:
                time.sleep(self.settle_sec)
            png = self.screenshot_png()
            content = [
                {"type": "text", "text": f"Performed {action}."},
                {"type": "image", "data": base64.b64encode(png).decode(), "mimeType": "image/png"},
            ]
        self._log(
            {"action": args, "screenshot": action not in _NO_SCREEN, "sec": round(time.monotonic() - started, 3)}
        )
        return {"content": content}

    def _log(self, row: dict) -> None:
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row) + "\n")


def handle(desktop: Desktop, message: dict) -> dict | None:
    """The JSON-RPC response to one MCP message; None for a notification."""
    if "id" not in message:
        return None
    method, params = message.get("method"), message.get("params") or {}
    if method == "initialize":
        result = {
            "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "computer", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": [TOOL]}
    elif method == "tools/call":
        if params.get("name") != TOOL["name"]:
            return {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32602, "message": "unknown tool"}}
        args = params.get("arguments") or {}
        try:
            result = desktop.call(args)
        except ActionError as exc:
            desktop._log({"action": args, "error": str(exc)})
            result = {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        except CaptureError as exc:
            desktop._log({"action": args, "infra_error": str(exc)})
            result = {"content": [{"type": "text", "text": "The screen could not be captured."}], "isError": True}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": f"no method {method}"}}
    return {"jsonrpc": "2.0", "id": message["id"], "result": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--display", default=":1")
    parser.add_argument("--settle-sec", type=float, default=2.0)
    parser.add_argument("--capture-cmd", default="import -window root {out}", help="shell writing a PNG to {out}")
    parser.add_argument("--log", default=None, help="append one JSON line per tool call")
    parser.add_argument("--screenshot", default=None, help="write one capture here and exit")
    parser.add_argument(
        "--stop-on-terminate", action="store_true", help="end the client (this server's parent) after a terminate"
    )
    parser.add_argument("--max-calls", type=int, default=None, help="end the client after this many tool calls")
    args = parser.parse_args()
    desktop = Desktop(args.display, args.settle_sec, args.capture_cmd, args.log)
    if args.screenshot:
        desktop.screenshot(args.screenshot)
        return
    calls = 0
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        response = handle(desktop, message)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        calls += message.get("method") == "tools/call"
        if args.stop_on_terminate and _terminated(message, response):
            reason = "terminate"
        elif args.max_calls is not None and calls >= args.max_calls:
            reason = "max_calls"
        else:
            continue
        # An agent CLI's loop ends only on a reply without a tool call, which Qwen3-VL keeps not giving.
        desktop._log({"stop": reason})
        os.kill(os.getppid(), signal.SIGTERM)
        return


def _terminated(message: dict, response: dict | None) -> bool:
    arguments = (message.get("params") or {}).get("arguments") or {}
    succeeded = response is not None and "result" in response and not response["result"].get("isError")
    return message.get("method") == "tools/call" and arguments.get("action") == "terminate" and succeeded


if __name__ == "__main__":
    main()
