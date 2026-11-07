"""`--custom-rm-path miles.rollout.rm_hub.video.reward.video_rm`: a sample's generated clip, scored against the user's
request on a reward pool.

Whatever produced the clip (the reward step of a single-turn recipe, or an agent's env) records it in the sample's
metadata; a sample with no clip, because generation failed, scores None, so the missing-reward filter drops its group.
"""

from miles.utils.args.schema import A, Arg, BaseConfig
from miles.utils.types import Sample

CLIP_PATH_KEY = "video_clip_path"
# The user's request, never the planner's generator prompt: a judge shown the planner's words could score those.
REQUEST_KEY = "video_request"


class VideoRewardConfig(BaseConfig):
    video_reward_pool: A[
        str, Arg(help="The reward pool, from --reward-pool-config, that scores each sample's generated clip.")
    ]


async def video_rm(args, samples: Sample | list[Sample], **pools) -> float | None | list[float | None]:
    """One reward per sample, with or without --group-rm; a group's clips go to the pool in one call."""
    batch = samples if isinstance(samples, list) else [samples]
    pool_kwarg = f"{args.video_reward_pool}_pool"
    assert (
        pool_kwarg in pools
    ), f"--video-reward-pool {args.video_reward_pool} is not a declared pool, got {sorted(pools)}"
    with_clip = [index for index, sample in enumerate(batch) if sample.metadata.get(CLIP_PATH_KEY) is not None]
    rewards: list[float | None] = [None] * len(batch)
    if with_clip:
        scores = await pools[pool_kwarg].score(
            [batch[index].metadata[CLIP_PATH_KEY] for index in with_clip],
            [batch[index].metadata[REQUEST_KEY] for index in with_clip],
        )
        for index, score in zip(with_clip, scores, strict=True):
            rewards[index] = score
    return rewards if isinstance(samples, list) else rewards[0]


video_rm.config_class = VideoRewardConfig
