import asyncio
from argparse import Namespace
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from miles.rollout.diffusion_rollout.plan import ImageRequestPlan, plan_for, prompt_sample_at, window_group_for
from miles.rollout.rm_hub import batched_async_rm
from miles.utils.diffusion.algorithms import get_algorithm
from miles.utils.diffusion.types import DiffusionRollout
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

Reward = float | dict[str, Any] | None
Request = Callable[[dict[str, Any]], Awaitable[list[DiffusionRollout]]]


class Dispatcher(Protocol):
    async def generate(self, sample: Sample, sampling_params: dict[str, Any]) -> tuple[DiffusionRollout, Reward]: ...

    def reset(self, *, index_offset: int = 0) -> None: ...


@dataclass
class _SharedRequest:
    task: asyncio.Task[tuple[list[DiffusionRollout], list[Reward]]]
    remaining: int


class MicrogroupDispatcher:
    def __init__(self, args: Namespace, request: Request, reward_kwargs: dict[str, Any]) -> None:
        self._args = args
        self._request = request
        self._reward_kwargs = reward_kwargs
        self._algorithm = get_algorithm(args.loss_type)
        self._step_strategy = load_function(args.diffusion_step_strategy_path)
        self._index_offset = 0
        self._inflight: dict[tuple[int, int], _SharedRequest] = {}

    async def generate(self, sample: Sample, sampling_params: dict[str, Any]) -> tuple[DiffusionRollout, Reward]:
        group_key, position = window_group_for(self._args, sample, self._index_offset)
        shared = self._inflight.get(group_key)
        if shared is None:
            plan = plan_for(
                self._args, sample, sampling_params["num_inference_steps"], self._step_strategy, self._index_offset
            )
            task = asyncio.create_task(self._generate_batch(plan, sample, sampling_params))
            shared = self._inflight[group_key] = _SharedRequest(task, plan.size)

        rollouts, rewards = await shared.task
        shared.remaining -= 1
        if shared.remaining == 0:
            del self._inflight[group_key]
        return rollouts[position], rewards[position]

    async def _generate_batch(
        self, plan: ImageRequestPlan, sample: Sample, sampling_params: dict[str, Any]
    ) -> tuple[list[DiffusionRollout], list[Reward]]:
        # The shared group loop adds a text sampling seed for deterministic inference.
        params = {key: value for key, value in sampling_params.items() if key != "sampling_seed"}
        body = {
            **params,
            "seed": plan.seed,
            **self._algorithm.rollout_request(plan.sde_steps, plan.returned_latents),
            "prompt": sample.prompt,
            "num_outputs_per_prompt": plan.size,
        }
        rollouts = await self._request(body)
        if len(rollouts) != plan.size:
            raise ValueError(f"expected {plan.size} diffusion outputs, got {len(rollouts)}")
        for position, rollout in enumerate(rollouts):
            rollout.seed = plan.seed + position
            rollout.sde_step_indices = list(plan.sde_steps or [])
        if self._args.group_rm:
            return rollouts, [None] * plan.size

        first_index = sample.index - plan.position
        batch = []
        for position, rollout in enumerate(rollouts):
            shell = prompt_sample_at(sample, first_index + position)
            shell.diffusion = rollout
            shell.status = Sample.Status.COMPLETED
            batch.append(shell)
        await batched_async_rm(self._args, batch, inplace_set_reward_field=True, **self._reward_kwargs)
        return rollouts, [shell.reward for shell in batch]

    def reset(self, *, index_offset: int = 0) -> None:
        self._inflight.clear()
        self._index_offset = index_offset
