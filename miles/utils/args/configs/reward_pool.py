from typing import Any

import pydantic

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class RewardPoolConfig(FrozenStrictBaseModel):
    """A pool of reward workers: `num_workers` cells of one worker each, every worker running one `scorer_path`.

    The rollout reaches a pool through `rm_hub` kwargs as `<name>_pool`; a GPU pool's workers each take one bundle of
    the run's placement group, after the rollout and eval GPUs.
    """

    name: str
    scorer_path: str
    scorer_kwargs: dict[str, Any] = {}
    num_workers: int = pydantic.Field(gt=0)
    uses_gpu: bool
    batch_size: int = pydantic.Field(gt=0)
    num_gpus_per_node: int = pydantic.Field(gt=0)


class RewardPoolScalingConfig(FrozenStrictBaseModel):
    num_gpus: int = pydantic.Field(ge=0)
    gpu_offset: int = pydantic.Field(ge=0)


class RewardScalingConfig(FrozenStrictBaseModel):
    pools: dict[str, RewardPoolScalingConfig] = {}

    @property
    def num_gpus(self) -> int:
        return sum(pool.num_gpus for pool in self.pools.values())

    @classmethod
    def resolve(cls, pools: list[RewardPoolConfig], *, gpu_offset: int) -> "RewardScalingConfig":
        """Each GPU pool's slots follow the ones before it, as server groups' do."""
        resolved = {}
        for pool in pools:
            num_gpus = pool.num_workers if pool.uses_gpu else 0
            resolved[pool.name] = RewardPoolScalingConfig(num_gpus=num_gpus, gpu_offset=gpu_offset)
            gpu_offset += num_gpus
        return cls(pools=resolved)


def compute_reward_pools(args) -> tuple[list[RewardPoolConfig], RewardScalingConfig]:
    """The run's reward pools and their slots, which start after the rollout and eval GPUs."""
    pools: list[RewardPoolConfig] = []
    return pools, RewardScalingConfig.resolve(pools, gpu_offset=(args.rollout_num_gpus or 0) + args.eval_num_gpus)
