"""Diffusion training on the shared per-sample inference rollout."""

import asyncio
from argparse import Namespace
from typing import Any

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.diffusion_rollout.client import ImageClient
from miles.rollout.diffusion_rollout.generate import use_image_client
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn, RolloutGenerationHooks
from miles.rollout.rm_hub.pools import use_reward_pools
from miles.utils.diffusion.rewards import BUILT_IN_REWARD_POOLS


class DiffusionRolloutFn(InferenceRolloutFn):
    def __init__(self, input: RolloutFnConstructorInput) -> None:
        self.reward_pools = dict(input.reward_pools)
        # A global custom reward replaces every built-in reward, including prompt overrides.
        if input.args.custom_rm_path is None:
            _check_selected_rewards_have_pools(
                _selected_rm_types(input.args, input.data_source), declared=self.reward_pools
            )
        super().__init__(input)
        self.client = ImageClient(input.args, {f"{name}_pool": pool for name, pool in self.reward_pools.items()})

    def generation_hooks(self) -> RolloutGenerationHooks:
        return RolloutGenerationHooks(sampling_params=build_sampling_params, load_tokenizer=False, engine_abort=False)

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        self.client.dispatcher.reset(index_offset=self.data_source.sample_index_offset)
        await asyncio.gather(*(pool.ensure_healthy() for pool in self.reward_pools.values()))
        with use_image_client(self.client), use_reward_pools(self.reward_pools):
            output = await super()._call_train(input)
        for group in output.samples:
            for sample in group:
                sample.diffusion.rollout_id = input.rollout_id
        return output

    async def _call_eval(self, input):
        raise AssertionError("diffusion runs have no eval rollout yet")

    async def dispose(self) -> None:
        await self.client.dispose()


def _selected_rm_types(args: Namespace, data_source) -> set[str]:
    """The launch reward and every reward a prompt's metadata `rm_type` overrides it with."""
    overrides = {
        str(metadata.get("rm_type")).strip() for _, metadata, _ in data_source.dataset if metadata.get("rm_type")
    }
    return {args.rm_type} | overrides


def _check_selected_rewards_have_pools(rm_types: set[str], *, declared: dict) -> None:
    """Reward pools start at launch, before the prompts are read, so every built-in reward a prompt selects must have
    been declared."""
    if missing := sorted((rm_types & set(BUILT_IN_REWARD_POOLS)) - set(declared)):
        raise ValueError(
            f"prompts select the {', '.join(missing)} reward, whose pool this run does not start; launch with "
            f"--diffusion-reward-pools {' '.join(missing)}"
        )


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
        # The engine refuses full log-probs of ODE steps, which carry no noise to score.
        "rollout_log_prob_no_const": args.diffusion_sde_type == "ode",
        "rollout_debug_mode": False,
        "rollout_video_dtype": "keep",
        "rollout_return_denoising_env": True,
        "rollout_return_dit_trajectory": True,
        "num_frames": args.diffusion_output_num_frames,
        "extra_sampling_params": {"quality": args.diffusion_quality},
    }
    if args.diffusion_guidance_scale_2 is not None:
        params["extra_sampling_params"]["guidance_scale_2"] = args.diffusion_guidance_scale_2
    if args.diffusion_negative_prompt is not None:
        params["negative_prompt"] = args.diffusion_negative_prompt
    return params
