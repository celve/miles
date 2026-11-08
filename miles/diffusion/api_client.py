import asyncio
import dataclasses
from typing import Any

import torch

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.diffusion.clip import save_clip
from miles.diffusion.rollout.response import parse_rollout_response
from miles.utils.http_utils import GeneralHttpClientProvider


@dataclasses.dataclass(frozen=True)
class DiffusionApiClient(SGLangApiClient):
    """Client of a sglang.multimodal_gen server, which has no KV cache, sleeps or wakes as a whole, serves no
    weight-update session and reports no weight version."""

    async def flush_cache(self):
        return None

    # multimodal_gen cannot pause or abort a generation; a frozen generator's weights never change under it, and a
    # trained diffusion run's loop never pauses or aborts.
    async def pause_generation(self, mode: str = "retract"):
        return None

    async def continue_generation(self):
        return None

    async def abort_all_requests(self, timeout: float | None = None):
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

    async def generate_clip(
        self,
        *,
        prompt: str,
        seed: int,
        num_frames: int,
        fps: int,
        width: int,
        height: int,
        output_dir: str,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        negative_prompt: str | None = None,
        image_path: str | None = None,
        extra_sampling_params: dict[str, Any] | None = None,
    ) -> str:
        """Generate one clip and return the path of its mp4, ``<output_dir>/<sha256>.mp4``.

        ``POST /rollout/generate`` answers in the same request, so ``server_url`` may be a router that sends each
        request to a different engine.
        """
        payload = {
            "prompt": prompt,
            "seed": seed,
            "num_frames": num_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
            "image_path": [image_path] if image_path else None,
            "extra_sampling_params": extra_sampling_params,
            "num_outputs_per_prompt": 1,
            "rollout": False,
            "rollout_video_dtype": "uint8",
            "suppress_logs": True,
        }
        response = await GeneralHttpClientProvider.client().post(
            f"{self.server_url}/rollout/generate",
            json={key: value for key, value in payload.items() if value is not None},
            headers=self._headers,
        )
        response.raise_for_status()
        (rollout,) = await asyncio.to_thread(parse_rollout_response, response.content)
        video = rollout.generated_output
        if image_path and video.shape[1] == num_frames:
            # Wan2.2 TI2V picks its own size from the image (SGLang's input_validation, 704x1280 area), whatever we ask.
            video = _fit(video, height=height, width=width)
        if tuple(video.shape[1:]) != (num_frames, height, width):
            raise ValueError(
                f"asked for {num_frames} frames of {width}x{height}, the engine generated {video.shape[1]} frames of "
                f"{video.shape[3]}x{video.shape[2]}; pick a frame count and size the model takes as they are"
            )
        return await asyncio.to_thread(save_clip, video, fps=fps, output_dir=output_dir)


def _fit(video: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """Center-crop a uint8 ``[C, F, H, W]`` clip to ``width/height``'s aspect, then resize it to that size."""
    _, _, source_height, source_width = video.shape
    if (source_height, source_width) == (height, width):
        return video
    crop_height = min(source_height, round(source_width * height / width))
    crop_width = min(source_width, round(source_height * width / height))
    top, left = (source_height - crop_height) // 2, (source_width - crop_width) // 2
    frames = video[:, :, top : top + crop_height, left : left + crop_width].permute(1, 0, 2, 3).float()
    resized = torch.nn.functional.interpolate(frames, size=(height, width), mode="bilinear", antialias=True)
    return resized.round().clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3).contiguous()
