"""The reward pools a `--model-kind diffusion` run declares, from its reward flags."""

from miles.utils.args.configs.reward_pool import RewardPoolConfig

OCR_SCORER = "miles.rollout.rm_hub.ocr.OcrBatchScorer"
PICKSCORE_SCORER = "miles.rollout.rm_hub.pickscore.PickScoreBatchScorer"
BUILT_IN_REWARD_POOLS = ("ocr", "pickscore")


def declared_reward_pool_names(args) -> list[str]:
    """--rm-type's pool, when it has one, then every --diffusion-reward-pools entry."""
    names = [args.rm_type] if args.rm_type in BUILT_IN_REWARD_POOLS else []
    return list(dict.fromkeys(names + list(args.diffusion_reward_pools)))


def declare_reward_pools(args) -> list[RewardPoolConfig]:
    # A global --custom-rm-path scores every sample itself, so it starts no built-in pool.
    if args.custom_rm_path is not None:
        return []
    declare = {"ocr": _ocr_pool, "pickscore": _pickscore_pool}
    return [declare[name](args) for name in declared_reward_pool_names(args)]


def _ocr_pool(args) -> RewardPoolConfig:
    # One image per call, as the process pool it replaces scored them.
    return RewardPoolConfig(
        name="ocr",
        scorer_path=OCR_SCORER,
        num_workers=args.diffusion_ocr_num_workers,
        uses_gpu=False,
        batch_size=1,
        num_gpus_per_node=args.num_gpus_per_node,
    )


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
        colocated=args.diffusion_pickscore_placement == "colocated",
    )
