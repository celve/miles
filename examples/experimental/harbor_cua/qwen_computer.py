"""Harbor's ``computer-1``, reading Qwen3-VL's clicks on the 0..999 grid it emits whatever the prompt asks for."""

from harbor.agents.computer_1.computer_1 import Computer1
from harbor.agents.computer_1.providers.base import ModelStep
from harbor.agents.computer_1.providers.generic import GenericJsonProvider
from harbor.agents.computer_1.runtime import CoordinateSpace
from harbor.llms.base import LLMResponse

# The screen upstream OSWorld's Qwen3-VL agent names (osworld_qwen/qwen3vl.py); the runtime scales it to the desktop.
QWEN3_VL_SCREEN = 1000


class Qwen3VLJsonProvider(GenericJsonProvider):
    @classmethod
    def from_agent(cls, agent: Computer1) -> "Qwen3VLJsonProvider":
        provider = super().from_agent(agent)
        provider.desktop_width = provider.desktop_height = QWEN3_VL_SCREEN
        return provider

    def parse(self, llm_response: LLMResponse) -> ModelStep:
        step = super().parse(llm_response)
        if step.action is not None:
            step.action.source = CoordinateSpace.NORMALIZED_0_999.value
        return step


class Qwen3VLComputer1(Computer1):
    def _build_provider(self) -> Qwen3VLJsonProvider:
        return Qwen3VLJsonProvider.from_agent(self)
