"""SD3.5 Flow-GRPO training on the shared per-sample inference rollout."""

from argparse import Namespace
from contextlib import nullcontext
from typing import Any

from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.diffusion_rollout.client import ImageClient
from miles.rollout.diffusion_rollout.generate import use_image_client
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn, RolloutGenerationHooks
from miles.rollout.rm_hub.ocr import OcrRewardPool, use_ocr_pool


class DiffusionRolloutFn(InferenceRolloutFn):
    def __init__(self, input: RolloutFnConstructorInput) -> None:
        super().__init__(input)
        args = input.args
        # rm_hub runs --custom-rm-path ahead of --rm-type.
        uses_ocr = args.rm_type == "ocr" and args.custom_rm_path is None
        self.reward_pool = OcrRewardPool(num_workers=args.diffusion_ocr_num_workers) if uses_ocr else None
        reward_kwargs = {"ocr_pool": self.reward_pool} if self.reward_pool is not None else {}
        self.client = ImageClient(args, reward_kwargs)

    def generation_hooks(self) -> RolloutGenerationHooks:
        return RolloutGenerationHooks(
            sampling_params=build_sampling_params,
            load_tokenizer=False,
            engine_abort=False,
            retry_group_reward_errors=False,
        )

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        self.client.dispatcher.reset(index_offset=self.data_source.sample_index_offset)
        pool = self.reward_pool
        reward_scope = use_ocr_pool(pool) if self.state.args.group_rm and pool is not None else nullcontext()
        with use_image_client(self.client), reward_scope:
            output = await super()._call_train(input)
        for group in output.samples:
            for sample in group:
                sample.diffusion.rollout_id = input.rollout_id
        return output

    async def _call_eval(self, input: RolloutFnEvalInput):
        raise AssertionError("diffusion runs have no eval rollout yet")

    async def dispose(self) -> None:
        await self.client.dispose()
        if self.reward_pool is not None:
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
