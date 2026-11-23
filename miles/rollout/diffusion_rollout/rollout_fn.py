"""SD3.5 Flow-GRPO training and evaluation on the shared per-sample inference rollout."""

from argparse import Namespace
from contextlib import nullcontext
from typing import Any

from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.diffusion_rollout.client import ImageClient
from miles.rollout.diffusion_rollout.data_source import read_eval_prompts
from miles.rollout.diffusion_rollout.generate import use_image_client
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn, RolloutGenerationHooks
from miles.rollout.rm_hub.ocr import OcrRewardPool, use_ocr_pool
from miles.utils.eval_config import EvalDatasetConfig


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
            eval_prompts=read_eval_prompts,
            eval_sampling_params=build_eval_sampling_params,
            eval_seed_key="seed",
            retry_group_reward_errors=False,
        )

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        pool = self.reward_pool
        reward_scope = use_ocr_pool(pool) if pool is not None else nullcontext()
        with use_image_client(self.client), reward_scope:
            return await super().__call__(input)

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        self.client.dispatcher.reset(index_offset=self.data_source.sample_index_offset)
        output = await super()._call_train(input)
        for group in output.samples:
            for sample in group:
                sample.diffusion.rollout_id = input.rollout_id
        return output

    async def dispose(self) -> None:
        await self.client.dispose()
        if self.reward_pool is not None:
            self.reward_pool.shutdown()


def build_sampling_params(args: Namespace) -> dict[str, Any]:
    """Static fields of the ``POST /rollout/generate`` body for training rollouts."""
    return _image_params(args, args.diffusion_num_steps, rollout=True)


def build_eval_sampling_params(args: Namespace, dataset_cfg: EvalDatasetConfig) -> dict[str, Any]:
    return _image_params(args, args.diffusion_eval_num_steps or args.diffusion_num_steps, rollout=False)


def _image_params(args: Namespace, num_steps: int, *, rollout: bool) -> dict[str, Any]:
    params: dict[str, Any] = {
        "generator_device": "cuda",
        "width": args.diffusion_width,
        "height": args.diffusion_height,
        "num_inference_steps": num_steps,
        "guidance_scale": args.diffusion_guidance_scale,
        "true_cfg_scale": None,
        "rollout": rollout,
    }
    if rollout:
        params.update(
            rollout_sde_type=args.diffusion_sde_type,
            rollout_noise_level=args.diffusion_noise_level,
            rollout_log_prob_no_const=False,
            rollout_debug_mode=False,
        )
    params.update(
        rollout_video_dtype="keep",
        rollout_return_denoising_env=rollout,
        rollout_return_dit_trajectory=rollout,
        num_frames=1,
        extra_sampling_params={"quality": args.diffusion_quality},
    )
    if args.diffusion_negative_prompt is not None:
        params["negative_prompt"] = args.diffusion_negative_prompt
    return params
