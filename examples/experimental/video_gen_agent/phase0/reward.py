"""Phase 0's reward step: `--custom-rm-path examples.experimental.video_gen_agent.phase0.reward.planner_rm`, with or
without `--group-rm`.

The planner's answer is one prompt for the frozen generator. The reward step makes the clip through the generator's
router and scores it against the user's request on a reward pool, with `video_rm`. A generator that fails over HTTP,
or a clip the scorer cannot read, scores None, which drops the sample's group; an empty answer is the planner's own
failure and scores 0.
"""

import asyncio
import hashlib
import logging
import time

import httpx
import numpy as np

from miles.diffusion.api_client import DiffusionApiClient
from miles.rollout.rm_hub.video.reward import CLIP_PATH_KEY, REQUEST_KEY, VideoRewardConfig, video_rm
from miles.rollout.sglang_rollout import get_model_url
from miles.utils.args.schema import A, Arg
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.tracking_utils import tracking
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# Set on a held-out row to send the request itself to the generator: the raw-request baseline.
SEND_REQUEST_KEY = "send_request_to_generator"
GENERATOR_PROMPT_KEY = "generator_prompt"
TIMING_KEY = "phase0_seconds"


class PlannerRewardConfig(VideoRewardConfig):
    video_generator_model: A[str, Arg(help="The --sglang-config model that serves the frozen generator.")] = (
        "generator"
    )
    video_clip_dir: A[str, Arg(help="Directory the generated clips are written to, on the rollout host.")]
    video_num_frames: A[int, Arg(help="Frames per clip; the generator takes 4k+1.")] = 49
    video_fps: A[int, Arg(help="Frames per second of each clip.")] = 24
    video_width: A[int, Arg(help="Clip width in pixels.")] = 832
    video_height: A[int, Arg(help="Clip height in pixels.")] = 480
    video_num_inference_steps: A[int | None, Arg(help="Denoising steps; the generator's default when unset.")] = None
    video_guidance_scale: A[float | None, Arg(help="CFG scale; the generator's default when unset.")] = None


async def planner_rm(args, samples: Sample | list[Sample], **pools) -> float | None | list[float | None]:
    batch = samples if isinstance(samples, list) else [samples]
    client = DiffusionApiClient(server_url=get_model_url(args, args.video_generator_model, ""))
    started = time.monotonic()
    await asyncio.gather(*(_generate_clip(args, client, sample) for sample in batch))
    generated = time.monotonic()
    rewards = await video_rm(args, batch, **pools)
    scored = time.monotonic()
    for sample in batch:
        sample.metadata[TIMING_KEY] = {"generate": generated - started, "score": scored - generated}
    rewards = [
        0.0 if not sample.metadata[GENERATOR_PROMPT_KEY] else r for sample, r in zip(batch, rewards, strict=True)
    ]
    return rewards if isinstance(samples, list) else rewards[0]


planner_rm.config_class = PlannerRewardConfig


def planner_prompt(response: str) -> str:
    """The generator prompt in a planner's answer: the text after any reasoning block."""
    return response.rpartition("</think>")[2].strip()


def request_seed(request: str) -> int:
    """One seed per request, so a group's clips and both baselines differ only in their prompt."""
    return int(hashlib.sha256(request.encode()).hexdigest()[:8], 16)


async def _generate_clip(args, client: DiffusionApiClient, sample: Sample) -> None:
    request = sample.metadata[REQUEST_KEY]
    prompt = request if sample.metadata.get(SEND_REQUEST_KEY) else planner_prompt(sample.response)
    sample.metadata[GENERATOR_PROMPT_KEY] = prompt
    if not prompt:
        return
    try:
        sample.metadata[CLIP_PATH_KEY] = await client.generate_clip(
            prompt=prompt,
            seed=request_seed(request),
            num_frames=args.video_num_frames,
            fps=args.video_fps,
            width=args.video_width,
            height=args.video_height,
            output_dir=args.video_clip_dir,
            num_inference_steps=args.video_num_inference_steps,
            guidance_scale=args.video_guidance_scale,
        )
    except httpx.HTTPError as error:
        logger.warning(f"generator failed for request {request!r}: {error!r}")


def log_rollout_data(rollout_id, args, samples: list[Sample], rollout_extra_metrics, rollout_time) -> bool:
    """`--custom-rollout-log-function-path`: Phase 0's share of the rollout time, logged beside the default metrics.

    The seconds are means over reward calls (one per sample, or per group under --group-rm), which overlap one another
    and the planner's generation. Only kept groups reach here: Miles counts the groups a failure dropped as
    `rollout/dynamic_filter/drop_group_has_missing_reward`.
    """
    stepped = [sample for sample in samples if TIMING_KEY in sample.metadata]
    if stepped:
        metrics = {
            "rollout/phase0/generate_seconds": np.mean([s.metadata[TIMING_KEY]["generate"] for s in stepped]),
            "rollout/phase0/score_seconds": np.mean([s.metadata[TIMING_KEY]["score"] for s in stepped]),
            "rollout/phase0/empty_answer_ratio": np.mean([not s.metadata[GENERATOR_PROMPT_KEY] for s in stepped]),
            "rollout/phase0/generator_prompt_words": np.mean(
                [len(s.metadata[GENERATOR_PROMPT_KEY].split()) for s in stepped]
            ),
            "rollout/step": compute_rollout_step(args, rollout_id),
        }
        logger.info(f"phase0 {rollout_id}: {metrics}")
        tracking.log(args, {key: float(value) for key, value in metrics.items()}, step_key="rollout/step")
    return False
