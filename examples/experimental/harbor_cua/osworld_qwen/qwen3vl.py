"""Upstream OSWorld's Qwen3-VL computer-use policy, ported for Harbor's OSWorld runner.

Ported from xlang-ai/OSWorld ``mm_agents/qwen3vl_agent.py`` and
``mm_agents/utils/qwen_vl_utils.py`` at b138d348256078fa634fc3b73567a7337c793e6b
(Apache-2.0): the prompt, the history window and the parser are kept as upstream wrote
them, so the policy matches the one LIN-2044 ran through NeMo Gym. Two parser cases that
raised upstream are logged and skipped instead: a malformed coordinate, and naming a
response whose only action is ``DONE`` or ``WAIT``. The model call and the sampling
parameters are left to the caller.
"""

from __future__ import annotations

import base64
import json
import logging
import math
from io import BytesIO

from PIL import Image

logger = logging.getLogger(__name__)

_RESIZE_FACTOR = 32
_MAX_PIXELS = 16 * 16 * 4 * 12800

_ACTION_DESCRIPTION = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
        """


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
    max_long_side: int = 8192,
) -> tuple[int, int]:
    if height < 2 or width < 2:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 100, got {height} / {width}")
    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)
    h_bar = _round_by_factor(height, factor)
    w_bar = _round_by_factor(width, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def process_image(image_bytes: bytes) -> tuple[str, int, int]:
    """The screenshot resized for the model and PNG-encoded, as base64, with its new size."""
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height=height, width=width, factor=_RESIZE_FACTOR, max_pixels=_MAX_PIXELS
    )
    buffer = BytesIO()
    image.resize((resized_width, resized_height)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8"), resized_width, resized_height


def _system_prompt() -> str:
    description = "\n".join(
        [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.",
            "* The screen's resolution is 1000x1000.",
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ]
    )
    tools_def = {
        "type": "function",
        "function": {
            "name_for_human": "computer_use",
            "name": "computer_use",
            "description": description,
            "parameters": {
                "properties": {
                    "action": {
                        "description": _ACTION_DESCRIPTION,
                        "enum": [
                            "key",
                            "type",
                            "mouse_move",
                            "left_click",
                            "left_click_drag",
                            "right_click",
                            "middle_click",
                            "double_click",
                            "scroll",
                            "wait",
                            "terminate",
                        ],
                        "type": "string",
                    },
                    "keys": {"description": "Required only by `action=key`.", "type": "array"},
                    "text": {"description": "Required only by `action=type`.", "type": "string"},
                    "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                    "pixels": {"description": "The amount of scrolling.", "type": "number"},
                    "time": {"description": "The seconds to wait.", "type": "number"},
                    "status": {
                        "description": "The status of the task.",
                        "type": "string",
                        "enum": ["success", "failure"],
                    },
                },
                "required": ["action"],
                "type": "object",
            },
            "args_format": "Format the arguments as a JSON object.",
        },
    }
    return (
        """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
"""
        + json.dumps(tools_def)
        + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call."""
    )


def _image_part(b64_png: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_png}"}}


class Qwen3VLPolicy:
    """Upstream's message construction and response parsing, holding one episode's history."""

    def __init__(self, history_n: int = 4) -> None:
        self.history_n = history_n
        self.actions: list[str] = []
        self.responses: list[str] = []
        self.screenshots: list[str] = []

    def build_messages(self, instruction: str, screenshot: bytes) -> tuple[list[dict], tuple[int, int]]:
        """Messages for this step, and the screenshot's original (width, height) for the parser."""
        width, height = Image.open(BytesIO(screenshot)).size
        processed, _, _ = process_image(screenshot)
        self.screenshots.append(processed)

        history_start = max(0, len(self.actions) - self.history_n)
        previous = [f"Step {i + 1}: {self.actions[i]}" for i in range(history_start)]
        instruction_prompt = f"""
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {instruction}

Previous actions:
{chr(10).join(previous) if previous else "None"}"""

        messages: list[dict] = [{"role": "system", "content": [{"type": "text", "text": _system_prompt()}]}]
        history_len = min(self.history_n, len(self.responses))
        if history_len:
            history_screenshots = self.screenshots[-history_len - 1 : -1]
            for idx, response in enumerate(self.responses[-history_len:]):
                content = [_image_part(history_screenshots[idx])]
                if idx == 0:
                    content.append({"type": "text", "text": instruction_prompt})
                messages.append({"role": "user", "content": content})
                messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
            messages.append({"role": "user", "content": [_image_part(processed)]})
        else:
            messages.append(
                {"role": "user", "content": [_image_part(processed), {"type": "text", "text": instruction_prompt}]}
            )
        return messages, (width, height)

    def record(self, response: str, screen: tuple[int, int]) -> list[str]:
        """Append the model's response to the history and return its pyautogui actions."""
        self.responses.append(response)
        low_level_instruction, code = parse_response(response, *screen)
        self.actions.append(low_level_instruction)
        return code


def parse_response(response: str, original_width: int, original_height: int) -> tuple[str, list[str]]:
    """The response's one-line instruction and its pyautogui code, coordinates rescaled from 0..999."""
    low_level_instruction = ""
    code: list[str] = []
    if response is None or not response.strip():
        return low_level_instruction, code

    def adjust(x: float, y: float) -> tuple[int, int]:
        return int(x * original_width / 999), int(y * original_height / 999)

    def process_tool_call(json_str: str) -> None:
        try:
            tool_call = json.loads(json_str)
            if tool_call.get("name") != "computer_use":
                return
            args = tool_call["arguments"]
            action = args["action"]
            if action in ("left_click", "right_click", "middle_click", "double_click"):
                function = {
                    "left_click": "click",
                    "right_click": "rightClick",
                    "middle_click": "middleClick",
                    "double_click": "doubleClick",
                }[action]
                if "coordinate" in args:
                    x, y = adjust(*args["coordinate"])
                    code.append(f"pyautogui.{function}({x}, {y})")
                else:
                    code.append(f"pyautogui.{function}()")
            elif action == "type":
                lines = args.get("text", "").split("\n")
                for idx, line in enumerate(lines):
                    if line:
                        code.append(f"pyautogui.typewrite({repr(line)}, interval=0.03)")
                    if idx < len(lines) - 1:
                        code.append("pyautogui.press('enter')")
            elif action == "key":
                keys = args.get("keys", [])
                if isinstance(keys, list):
                    keys = [_clean_key(key) if isinstance(key, str) else key for key in keys]
                keys_str = ", ".join(f"'{key}'" for key in keys)
                code.append(f"pyautogui.hotkey({keys_str})" if len(keys) > 1 else f"pyautogui.press({keys_str})")
            elif action == "scroll":
                code.append(f"pyautogui.scroll({args.get('pixels', 0)})")
            elif action == "wait":
                code.append("WAIT")
            elif action == "terminate":
                code.append("DONE")
            elif action == "mouse_move":
                if "coordinate" in args:
                    x, y = adjust(*args["coordinate"])
                    code.append(f"pyautogui.moveTo({x}, {y})")
                else:
                    code.append("pyautogui.moveTo(0, 0)")
            elif action == "left_click_drag":
                if "coordinate" in args:
                    x, y = adjust(*args["coordinate"])
                    code.append(f"pyautogui.dragTo({x}, {y}, duration={args.get('duration', 0.5)})")
                else:
                    code.append("pyautogui.dragTo(0, 0)")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.error("Failed to parse tool call: %s", exc)

    inside_tool_call = False
    current: list[str] = []
    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("action:"):
            if not low_level_instruction:
                low_level_instruction = line.split("Action:")[-1].strip()
            continue
        if line.startswith("<tool_call>"):
            inside_tool_call = True
            continue
        if line.startswith("</tool_call>"):
            if current:
                process_tool_call("\n".join(current))
                current = []
            inside_tool_call = False
            continue
        if inside_tool_call:
            current.append(line)
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "name" in parsed and "arguments" in parsed:
                process_tool_call(line)
    if current:
        process_tool_call("\n".join(current))
    if not low_level_instruction and code:
        first = code[0]
        low_level_instruction = (
            f"Performing {first.split('.', 1)[1].split('(', 1)[0]} action"
            if "." in first
            else f"Performing {first} action"
        )
    return low_level_instruction, code


def _clean_key(key: str) -> str:
    if key.startswith("keys=["):
        key = key[6:]
    if key.endswith("]"):
        key = key[:-1]
    if key.startswith("['") or key.startswith('["'):
        key = key[2:] if len(key) > 2 else key
    if key.endswith("']") or key.endswith('"]'):
        key = key[:-2] if len(key) > 2 else key
    return key.strip()
