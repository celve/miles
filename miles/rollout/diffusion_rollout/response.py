"""Parse sglang.multimodal_gen's msgpack ``POST /rollout/generate`` response into `DiffusionRollout`s."""

from typing import Any

import msgpack
import torch

from miles.diffusion.types import CondKwargs, DenoisingEnv, DiffusionRollout, DiTTrajectory, decode_tensor

_IMAGE_CHANNEL_COUNTS = (1, 3, 4)


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
