import dataclasses

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient


@dataclasses.dataclass(frozen=True)
class DiffusionApiClient(SGLangApiClient):
    """Client of a sglang.multimodal_gen server, which has no KV cache, sleeps or wakes as a whole, serves no
    weight-update session and reports no weight version."""

    async def flush_cache(self):
        return None

    async def release_memory_occupation(self, tags: list[str] = None):
        return await self._make_request("release_memory_occupation")

    async def resume_memory_occupation(self, tags: list[str] = None):
        return await self._make_request("resume_memory_occupation")

    async def get_weight_version(self):
        raise NotImplementedError("sglang.multimodal_gen does not report a weight version")

    async def update_lora_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: list[str],
        payload_gpu_uuids: list[str],
        target_module: str,
        lora_rank: int,
        lora_alpha: int,
    ):
        """Merge one bucket of LoRA A/B pairs, handed over as CUDA-IPC handles, into ``target_module``."""
        return await self._make_request(
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": serialized_named_tensors,
                "payload_gpu_uuids": payload_gpu_uuids,
                "load_format": "flattened_bucket",
                "target_modules": [target_module],
                "weight_update_mode": "lora_merge",
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
            },
        )
