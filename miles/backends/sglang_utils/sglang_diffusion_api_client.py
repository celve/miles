import dataclasses

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient


@dataclasses.dataclass(frozen=True)
class DiffusionApiClient(SGLangApiClient):
    """Control client of a frozen ``sglang.multimodal_gen`` server, which serves none of the text engine's
    generation-control routes; the controller sends these to every engine, so here they do nothing."""

    async def flush_cache(self):
        return None

    async def pause_generation(self, mode: str = "retract"):
        return None

    async def continue_generation(self):
        return None

    async def abort_all_requests(self, timeout: float | None = None):
        return None
