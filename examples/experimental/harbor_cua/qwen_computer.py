"""Harbor's ``computer-1``, reading Qwen3-VL's clicks on the 0..999 grid it emits whatever the prompt asks for."""

import json
from dataclasses import asdict, replace
from typing import Any

from harbor.agents.computer_1.computer_1 import Computer1
from harbor.agents.computer_1.providers.base import ModelStep
from harbor.agents.computer_1.providers.generic import GenericJsonProvider, _extract_json_object
from harbor.agents.computer_1.runtime import CoordinateSpace
from harbor.llms.base import LLMResponse

from miles.utils.eval_record import EpisodeWriter, decode_data_url

# The screen upstream OSWorld's Qwen3-VL agent names (osworld_qwen/qwen3vl.py); the runtime scales it to the desktop.
QWEN3_VL_SCREEN = 1000
# Harbor reads a point it cannot parse as none, and executes such an action at (0, 0).
_NEEDS_POINT = frozenset(
    {
        "click",
        "double_click",
        "right_click",
        "triple_click",
        "mouse_move",
        "mouse_down",
        "mouse_up",
        "drag",
        "type_text_at",
    }
)


def _scalar_point(action: dict) -> dict:
    """The action with its point as scalar ``x`` and ``y`` when the reply wrote it as a pair, as Qwen3.8 often does."""
    x, y = action.get("x"), action.get("y")
    if isinstance(x, list) and len(x) == 2 and (y is None or y == x):
        return {**action, "x": x[0], "y": x[1]}
    if isinstance(x, list) and len(x) == 1 and isinstance(y, list) and len(y) == 1:
        return {**action, "x": x[0], "y": y[0]}
    coordinate = action.get("coordinate")
    if x is None and isinstance(coordinate, list) and len(coordinate) == 2:
        return {**action, "x": coordinate[0], "y": coordinate[1]}
    return action


def _with_scalar_point(response: LLMResponse) -> LLMResponse:
    json_str, _ = _extract_json_object(response.content)
    try:
        data = json.loads(json_str) if json_str else None
    except json.JSONDecodeError:
        return response
    if not isinstance(data, dict) or not isinstance(data.get("action"), dict):
        return response
    action = _scalar_point(data["action"])
    if action is data["action"]:
        return response
    return replace(response, content=response.content.replace(json_str, json.dumps({**data, "action": action})))


class Qwen3VLJsonProvider(GenericJsonProvider):
    # With a writer, each parsed reply becomes a step paired with the screenshot its prompt carried.
    record: EpisodeWriter | None = None
    _screenshot: tuple[bytes, str] | None = None

    @classmethod
    def from_agent(cls, agent: Computer1) -> "Qwen3VLJsonProvider":
        provider = super().from_agent(agent)
        provider.desktop_width = provider.desktop_height = QWEN3_VL_SCREEN
        if (record_dir := getattr(agent, "record_dir", None)) is not None:
            provider.record = EpisodeWriter(record_dir)
        return provider

    def initial_messages(self, instruction: str, screenshot_ref: str) -> list:
        if self.record is not None:
            self.record.set_instruction(instruction)
            self._screenshot = decode_data_url(screenshot_ref)
        return super().initial_messages(instruction, screenshot_ref)

    def follow_up_messages(self, step: ModelStep, observation: str, screenshot_ref: str) -> list:
        if self.record is not None:
            self._screenshot = decode_data_url(screenshot_ref)
        return super().follow_up_messages(step, observation, screenshot_ref)

    def parse(self, llm_response: LLMResponse) -> ModelStep:
        parsed = super().parse(_with_scalar_point(llm_response))
        # The history keeps the reply the engine generated; the session server matches its tokens against it.
        step = replace(parsed, llm_response=llm_response, message=llm_response.content if parsed.message else "")
        if step.action is not None and step.action.type in _NEEDS_POINT and None in (step.action.x, step.action.y):
            step = ModelStep(
                needs_retry=True,
                feedback=f'ERROR: a {step.action.type!r} action needs integer "x" and "y", e.g. "x": 500, "y": 300',
                llm_response=llm_response,
            )
        if step.action is not None:
            step.action.source = CoordinateSpace.NORMALIZED_0_999.value
        if self.record is not None:
            screenshot, media_type = self._screenshot or (None, "image/png")
            self.record.add_step(
                output=llm_response.content,
                screenshot=screenshot,
                media_type=media_type,
                action=asdict(step.action) if step.action is not None else None,
                error=step.feedback if step.needs_retry else None,
            )
        return step


class Qwen3VLComputer1(Computer1):
    def __init__(self, *args: Any, record_dir: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.record_dir = record_dir

    def _build_provider(self) -> Qwen3VLJsonProvider:
        return Qwen3VLJsonProvider.from_agent(self)
