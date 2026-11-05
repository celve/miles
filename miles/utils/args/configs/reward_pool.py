from typing import Any

import pydantic
from typing_extensions import Self

from miles.backends.sglang_utils.sglang_config import rollout_slot_needs_offload
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
    # A colocated GPU pool shares the first rollout GPUs instead of having GPUs of its own.
    colocated: bool = False
    # Set at parse from the pool's slots by the engines' rule: the pool sleeps while training holds its GPUs.
    needs_offload: bool = False

    @pydantic.model_validator(mode="after")
    def _colocate_only_a_gpu_pool(self) -> Self:
        assert (
            self.uses_gpu or not self.colocated
        ), f"the {self.name} reward pool scores on the CPU, so has no GPU to share"
        return self


class RewardPoolScalingConfig(FrozenStrictBaseModel):
    num_gpus: int = pydantic.Field(ge=0)
    gpu_offset: int = pydantic.Field(ge=0)


class RewardScalingConfig(FrozenStrictBaseModel):
    pools: dict[str, RewardPoolScalingConfig] = {}

    @property
    def num_gpus(self) -> int:
        return sum(pool.num_gpus for pool in self.pools.values())

    @classmethod
    def resolve(
        cls, pools: list[RewardPoolConfig], *, gpu_offset: int, num_rollout_gpus: int | None = None
    ) -> "RewardScalingConfig":
        """A dedicated GPU pool's slots follow the ones before it, as server groups' do, and count as GPUs of their
        own; a colocated pool's slots run from the first rollout slot and add none."""
        resolved, colocated_offset = {}, 0
        for pool in pools:
            if pool.colocated:
                assert num_rollout_gpus is not None and colocated_offset + pool.num_workers <= num_rollout_gpus, (
                    f"the colocated {pool.name} reward pool needs {pool.num_workers} rollout slots, but "
                    f"{num_rollout_gpus} rollout GPUs hold only {num_rollout_gpus - colocated_offset} more"
                )
                resolved[pool.name] = RewardPoolScalingConfig(num_gpus=0, gpu_offset=colocated_offset)
                colocated_offset += pool.num_workers
                continue
            num_gpus = pool.num_workers if pool.uses_gpu else 0
            resolved[pool.name] = RewardPoolScalingConfig(num_gpus=num_gpus, gpu_offset=gpu_offset)
            gpu_offset += num_gpus
        return cls(pools=resolved)


def compute_reward_pools(args) -> tuple[list[RewardPoolConfig], RewardScalingConfig]:
    """The run's reward pools and their slots: dedicated ones after the rollout and eval GPUs, colocated ones on the
    first rollout GPUs."""
    declared = _declare_reward_pools(args)
    scaling = RewardScalingConfig.resolve(
        declared,
        gpu_offset=(args.rollout_num_gpus or 0) + args.eval_num_gpus,
        num_rollout_gpus=args.rollout_num_gpus or 0,
    )
    pools = [
        pool.model_copy(
            update=dict(
                needs_offload=pool.colocated
                and rollout_slot_needs_offload(args, gpu_offset=scaling.pools[pool.name].gpu_offset)
            )
        )
        for pool in declared
    ]
    return pools, scaling


def _declare_reward_pools(args) -> list[RewardPoolConfig]:
    if args.model_kind != "diffusion":
        return []
    # Local import: the diffusion package pulls in optional dependencies text runs never install.
    from miles.diffusion.rewards import declare_reward_pools

    return declare_reward_pools(args)
