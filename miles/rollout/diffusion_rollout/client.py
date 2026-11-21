"""Client for sglang.multimodal_gen's msgpack rollout endpoint."""

import asyncio
import logging
from argparse import Namespace
from typing import Any

import httpx
import msgpack
import torch

from miles.rollout.diffusion_rollout.dispatch import Dispatcher, MicrogroupDispatcher
from miles.utils.diffusion.types import CondKwargs, DenoisingEnv, DiffusionRollout, DiTTrajectory, decode_tensor
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_MAX_RETRIES = 60
_IMAGE_CHANNEL_COUNTS = (1, 3, 4)


class ImageClient:
    def __init__(self, args: Namespace, reward_kwargs: dict[str, Any]) -> None:
        self.url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/rollout/generate"
        self._http = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=0), timeout=httpx.Timeout(None))
        self.dispatcher: Dispatcher = MicrogroupDispatcher(args, self._request, reward_kwargs)

    async def generate(self, sample: Sample, sampling_params: dict[str, Any]) -> tuple[DiffusionRollout, Any]:
        return await self.dispatcher.generate(sample, sampling_params)

    async def generate_eval(self, sample: Sample, sampling_params: dict[str, Any]) -> DiffusionRollout:
        (rollout,) = await self._request({**sampling_params, "prompt": sample.prompt, "num_outputs_per_prompt": 1})
        rollout.seed = sampling_params["seed"]
        return rollout

    async def _request(self, body: dict[str, Any]) -> list[DiffusionRollout]:
        raw = await self._post(body)
        return await asyncio.to_thread(parse_rollout_response, raw)

    async def _post(self, payload: dict[str, Any]) -> bytes:
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._http.post(self.url, json=payload)
                response.raise_for_status()
                return response.content
            except Exception:
                if attempt + 1 >= _MAX_RETRIES:
                    raise
                logger.warning(f"/rollout/generate attempt {attempt} failed, retrying", exc_info=True)
                await asyncio.sleep(1)
        raise AssertionError("unreachable")

    async def dispose(self) -> None:
        await self._http.aclose()


def parse_rollout_response(raw: bytes) -> list[DiffusionRollout]:
    return [_parse_output(body) for body in msgpack.unpackb(raw, raw=False)]


def _parse_output(body: dict[str, Any]) -> DiffusionRollout:
    return DiffusionRollout(
        seed=int(body["seed"]) if body.get("seed") is not None else None,
        request_id=body.get("request_id"),
        generated_output=_normalize_generated_output(_tensor(body.get("generated_output"))),
        rollout_log_probs=_tensor(body.get("rollout_log_probs")),
        denoising_env=_parse_denoising_env(body.get("denoising_env")),
        dit_trajectory=_parse_dit_trajectory(body.get("dit_trajectory")),
        inference_time_s=float(body["inference_time_s"]) if body.get("inference_time_s") is not None else None,
    )


def _tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, dict) and value.get("__tensor__"):
        return decode_tensor(value["data"]).detach().cpu()
    raise TypeError(f"Cannot deserialize {type(value)}")


def _tensor_list(value: Any) -> list[torch.Tensor] | None:
    """SD3 sends a single serialized tensor; other families send a list of them."""
    if value is None:
        return None
    if isinstance(value, dict):
        return [_tensor(value)]
    return [_tensor(item) for item in value]


def _parse_cond_kwargs(data: dict[str, Any] | None) -> CondKwargs | None:
    if not data:
        return None
    return CondKwargs(
        encoder_hidden_states=_tensor_list(data.get("encoder_hidden_states")),
        pooled_projections=_tensor_list(data.get("pooled_projections")),
    )


def _parse_denoising_env(data: dict[str, Any] | None) -> DenoisingEnv | None:
    if not data:
        return None
    return DenoisingEnv(
        pos_cond_kwargs=_parse_cond_kwargs(data.get("pos_cond_kwargs")),
        neg_cond_kwargs=_parse_cond_kwargs(data.get("neg_cond_kwargs")),
    )


def _parse_dit_trajectory(data: dict[str, Any] | None) -> DiTTrajectory | None:
    if not data:
        return None
    return DiTTrajectory(
        latents=_tensor(data.get("latents")),
        timesteps=_tensor(data.get("timesteps")),
        sigmas=_tensor(data.get("sigmas")),
        latent_step_indices=_tensor(data.get("latent_step_indices")),
    )


def _normalize_generated_output(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Normalize a per-sample image or video to ``[C, F, H, W]``."""
    if tensor is None:
        return None
    if tensor.ndim not in (3, 4):
        raise ValueError(f"generated_output must be CHW/HWC or CFHW/FHWC, got shape {tuple(tensor.shape)}")
    first_is_channel = tensor.shape[0] in _IMAGE_CHANNEL_COUNTS
    last_is_channel = tensor.shape[-1] in _IMAGE_CHANNEL_COUNTS
    if first_is_channel == last_is_channel:
        raise ValueError(f"generated_output has no unambiguous channel dimension: {tuple(tensor.shape)}")
    if first_is_channel:
        canonical = tensor.unsqueeze(1) if tensor.ndim == 3 else tensor
    else:
        fhwc = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
        canonical = fhwc.permute(3, 0, 1, 2)
    return canonical.contiguous()
