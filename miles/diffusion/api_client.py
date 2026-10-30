import dataclasses

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient


@dataclasses.dataclass(frozen=True)
class DiffusionApiClient(SGLangApiClient):
    """Client of a sglang.multimodal_gen server, which has no KV cache and sleeps or wakes as a whole."""

    async def flush_cache(self):
        return None

    async def release_memory_occupation(self, tags: list[str] = None):
        return await self._make_request("release_memory_occupation")

    async def resume_memory_occupation(self, tags: list[str] = None):
        return await self._make_request("resume_memory_occupation")
