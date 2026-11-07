"""Harbor's ``computer-1``, reading Qwen3-VL's clicks on the 0..999 grid it emits whatever the prompt asks for."""

from dataclasses import asdict
from typing import Any

from harbor.agents.computer_1.computer_1 import Computer1
from harbor.agents.computer_1.providers.base import ModelStep
from harbor.agents.computer_1.providers.generic import GenericJsonProvider
from harbor.agents.computer_1.runtime import CoordinateSpace
from harbor.llms.base import LLMResponse

from miles.utils.eval_record import EpisodeWriter, decode_data_url

# The screen upstream OSWorld's Qwen3-VL agent names (osworld_qwen/qwen3vl.py); the runtime scales it to the desktop.
QWEN3_VL_SCREEN = 1000


class Qwen3VLJsonProvider(GenericJsonProvider):
    # With a writer, each parsed reply becomes a step paired with the screenshot its prompt carried.
    record: EpisodeWriter | None = None
    _screenshot: tuple[bytes, str] | None = None

    @classmethod
    def from_agent(cls, agent: Computer1) -> "Qwen3VLJsonProvider":
        provider = super().from_agent(agent)
        provider.desktop_width = provider.desktop_height = QWEN3_VL_SCREEN
        if (
            record_dir := getattr(agent, "record_dir", None)
        ) is not None:  # config-access-exempt: only recording agents carry record_dir
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
        step = super().parse(llm_response)
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
