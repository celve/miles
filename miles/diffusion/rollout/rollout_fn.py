"""Flow-GRPO rollout: SDE-sampled images from sglang.multimodal_gen through the Miles router, scored by OCR."""

import asyncio
import logging
from argparse import Namespace
from typing import Any

import httpx

from miles.diffusion.rollout.ocr import OcrRewardPool, image_to_rgb_uint8
from miles.diffusion.rollout.response import parse_rollout_response
from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnInput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.inference_rollout.inference_rollout_train import get_worker_urls
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_MAX_RETRIES = 60


class DiffusionRolloutFn(BaseRolloutFn):
    def __init__(self, input: RolloutFnConstructorInput) -> None:
        super().__init__(input)
        args = input.args
        self.args = args
        self.data_source = input.data_source
        self.url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/rollout/generate"
        self.step_strategy = load_function(args.diffusion_step_strategy_path)
        self.sampling_params = build_sampling_params(args)
        self.semaphore: asyncio.Semaphore | None = None
        self.reward_pool = OcrRewardPool(num_workers=args.diffusion_ocr_num_workers)
        self.client = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=0), timeout=httpx.Timeout(None))

    async def __call__(self, input: RolloutFnInput) -> RolloutFnTrainOutput:
        assert isinstance(input, RolloutFnTrainInput), "diffusion runs have no eval rollout yet"
        if self.semaphore is None:
            num_engines = len(await get_worker_urls(self.args))
            self.semaphore = asyncio.Semaphore(self.args.sglang_server_concurrency * num_engines)
        groups = self.data_source.get_samples(self.args.rollout_batch_size)
        groups = await asyncio.gather(*(self._generate_group(group) for group in groups))
        logger.info(f"Rollout {input.rollout_id}: first prompt {groups[0][0].prompt!r}, reward {groups[0][0].reward}")
        return RolloutFnTrainOutput(samples=sorted(groups, key=lambda group: group[0].index))

    async def _generate_group(self, group: list[Sample]) -> list[Sample]:
        # N-spaced seeds keep each request's [seed, seed + n) disjoint across prompt groups.
        seed_base = (self.args.rollout_seed + group[0].group_index * self.args.n_samples_per_prompt) % (2**31)
        size = self.args.diffusion_microgroup_size
        microgroups = await asyncio.gather(
            *(
                self._generate_microgroup(group[start : start + size], seed=seed_base + start)
                for start in range(0, len(group), size)
            )
        )
        return [sample for microgroup in microgroups for sample in microgroup]

    async def _generate_microgroup(self, microgroup: list[Sample], *, seed: int) -> list[Sample]:
        num_steps = self.sampling_params["num_inference_steps"]
        sde_steps, _returned_latents = self.step_strategy(self.args, microgroup[0], num_steps, seed)
        payload = {
            **self.sampling_params,
            "seed": seed,
            # The trainer slices the full trajectory itself, so the engine returns every latent.
            "rollout_sde_step_indices": sde_steps,
            "rollout_return_step_indices": None,
            "prompt": microgroup[0].prompt,
            "num_outputs_per_prompt": len(microgroup),
        }
        async with self.semaphore:
            raw = await self._post(payload)
        rollouts = await asyncio.to_thread(parse_rollout_response, raw)
        for index, (sample, rollout) in enumerate(zip(microgroup, rollouts, strict=True)):
            # The engine expands one seed to [seed, seed + n); record each image's own.
            rollout.seed = seed + index
            rollout.sde_step_indices = list(sde_steps)
            sample.diffusion = rollout
            sample.status = Sample.Status.COMPLETED

        rewards = await self.reward_pool.score(
            [image_to_rgb_uint8(sample.diffusion.generated_output) for sample in microgroup],
            [sample.prompt for sample in microgroup],
        )
        for sample, reward in zip(microgroup, rewards, strict=True):
            sample.reward = reward
        return microgroup

    async def _post(self, payload: dict[str, Any]) -> bytes:
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self.client.post(self.url, json=payload)
                response.raise_for_status()
                return response.content
            except Exception:
                if attempt + 1 >= _MAX_RETRIES:
                    raise
                logger.warning(f"/rollout/generate attempt {attempt} failed, retrying", exc_info=True)
                await asyncio.sleep(1)
        raise AssertionError("unreachable")

    async def dispose(self) -> None:
        await self.client.aclose()
        self.reward_pool.shutdown()


def build_sampling_params(args: Namespace) -> dict[str, Any]:
    """Static fields of the ``POST /rollout/generate`` body for training rollouts."""
    params: dict[str, Any] = {
        "generator_device": "cuda",
        "width": args.diffusion_width,
        "height": args.diffusion_height,
        "num_inference_steps": args.diffusion_num_steps,
        "guidance_scale": args.diffusion_guidance_scale,
        "true_cfg_scale": None,
        "rollout": True,
        "rollout_sde_type": args.diffusion_sde_type,
        "rollout_noise_level": args.diffusion_noise_level,
        "rollout_log_prob_no_const": False,
        "rollout_debug_mode": False,
        "rollout_video_dtype": "keep",
        "rollout_return_denoising_env": True,
        "rollout_return_dit_trajectory": True,
        "num_frames": 1,
        "extra_sampling_params": {"quality": args.diffusion_quality},
    }
    if args.diffusion_negative_prompt is not None:
        params["negative_prompt"] = args.diffusion_negative_prompt
    return params
