"""The reward pools a `--model-kind diffusion` run declares, from its reward flags."""

from miles.utils.args.configs.reward_pool import RewardPoolConfig

PICKSCORE_SCORER = "miles.diffusion.rollout.pickscore.PickScoreBatchScorer"


def declare_reward_pools(args) -> list[RewardPoolConfig]:
    pools = []
    if args.rm_type == "pickscore":
        pools.append(_pickscore_pool(args))
    return pools


def _pickscore_pool(args) -> RewardPoolConfig:
    return RewardPoolConfig(
        name="pickscore",
        scorer_path=PICKSCORE_SCORER,
        scorer_kwargs=dict(
            processor_path=args.diffusion_pickscore_processor_path,
            model_path=args.diffusion_pickscore_model_path,
            frames_per_forward=args.diffusion_pickscore_batch_size,
        ),
        num_workers=args.diffusion_pickscore_num_workers,
        uses_gpu=args.diffusion_pickscore_num_gpus_per_worker > 0,
        batch_size=args.diffusion_pickscore_batch_size,
        num_gpus_per_node=args.num_gpus_per_node,
    )
