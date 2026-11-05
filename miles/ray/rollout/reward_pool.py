"""The rollout's handle on one reward pool: batches go round robin to its workers, and a worker found dead is
restarted and sent the same batch again.

Nothing else restarts a reward cell: the worker manager's liveness scan only tears a dead cell down, and the fault
tolerance controller that resumes cells is not part of the runs that declare reward pools.
"""

import asyncio
import itertools
import logging

from miles.ray.specs.reward import reward_cell_id, reward_pool_id, reward_worker_name
from miles.utils.args.configs.reward_pool import RewardPoolConfig
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError

logger = logging.getLogger(__name__)

_MAX_RESTARTS_PER_CALL = 3
# A restarted worker loads its scorer before it answers, which for a CLIP-H checkpoint takes minutes on a cold cache.
_READY_TIMEOUT_SECONDS = 1800.0


class RewardPoolClient:
    def __init__(
        self,
        *,
        config: RewardPoolConfig,
        capability: BackendCapability,
        max_restarts_per_call: int = _MAX_RESTARTS_PER_CALL,
        ready_timeout: float = _READY_TIMEOUT_SECONDS,
    ) -> None:
        self._config = config
        self._capability = capability
        self._max_restarts_per_call = max_restarts_per_call
        self._ready_timeout = ready_timeout
        self._handles: dict[int, BaseWorkerHandle] = {}
        self._restart_locks = [asyncio.Lock() for _ in range(config.num_workers)]
        # A call that saw a worker fail restarts it only if no other call restarted it since that call began.
        self._num_restarts = [0] * config.num_workers
        self._next_worker = 0

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def needs_offload(self) -> bool:
        return self._config.needs_offload

    async def score(self, outputs: list, prompts: list[str]) -> list[float]:
        """Scores in input order; each chunk of `batch_size` samples is one worker call."""
        size = self._config.batch_size
        chunks = [
            (outputs[start : start + size], prompts[start : start + size]) for start in range(0, len(outputs), size)
        ]
        workers = [self._take_next_worker() for _ in chunks]
        results = await asyncio.gather(
            *(self._call(worker, "score_batch", *chunk) for worker, chunk in zip(workers, chunks, strict=True))
        )
        return [float(score) for chunk in results for score in chunk]

    async def sleep(self) -> None:
        await asyncio.gather(*(self._call(index, "sleep") for index in range(self._config.num_workers)))

    async def wake(self) -> None:
        await asyncio.gather(*(self._call(index, "wake") for index in range(self._config.num_workers)))

    async def ensure_healthy(self) -> None:
        """Restart every worker whose cell is down, so a rollout starts on a whole pool."""
        cell_infos = await self._capability.cell_operations().cell_infos(
            pool_ids=[reward_pool_id(self.name)], category=None
        )
        for index in range(self._config.num_workers):
            info = cell_infos.get(reward_cell_id(self.name, cell_index=index))
            if info is None or not info.alive:
                logger.warning(
                    f"Reward worker {reward_worker_name(self.name, cell_index=index)} is down; restarting it"
                )
                await self._restart(index, seen_restarts=self._num_restarts[index])

    def _take_next_worker(self) -> int:
        index = self._next_worker % self._config.num_workers
        self._next_worker += 1
        return index

    async def _call(self, index: int, method: str, *args):
        for attempt in itertools.count():
            seen_restarts = self._num_restarts[index]
            try:
                handle = await self._handle(index)
                return await getattr(handle, method)(*args)  # config-access-exempt: worker method chosen by caller
            except WorkerUnreachableError:
                if attempt >= self._max_restarts_per_call:
                    raise
                logger.warning(
                    f"Reward worker {reward_worker_name(self.name, cell_index=index)} is unreachable in {method}; "
                    f"restarting it and calling it again (restart {attempt + 1}/{self._max_restarts_per_call})",
                    exc_info=True,
                )
                await self._restart(index, seen_restarts=seen_restarts)
        raise AssertionError("unreachable")

    async def _handle(self, index: int) -> BaseWorkerHandle:
        if (handle := self._handles.get(index)) is None:
            handle = await self._lookup_handle(index)
            await handle.wait_ready(timeout=self._ready_timeout)
            self._handles[index] = handle
        return handle

    async def _lookup_handle(self, index: int) -> BaseWorkerHandle:
        provider = self._capability.static_worker_provider(pool_id=reward_pool_id(self.name))
        name = reward_worker_name(self.name, cell_index=index)
        # The Ray provider's lookup blocks on ray.get.
        (infos,) = await asyncio.to_thread(
            provider.get_worker_infos, cell_ids=[reward_cell_id(self.name, cell_index=index)]
        )
        if (handle := provider.get_handles_of_worker_infos(infos).get(name)) is None:
            # The liveness scan tore the cell down, so the manager lists no worker for it.
            raise WorkerUnreachableError(f"Reward worker {name} has no live actor")
        return handle

    async def _restart(self, index: int, *, seen_restarts: int) -> None:
        async with self._restart_locks[index]:
            if self._num_restarts[index] != seen_restarts:
                return
            self._handles.pop(index, None)
            cell_id = reward_cell_id(self.name, cell_index=index)
            operations = self._capability.cell_operations()
            # A dead cell the liveness scan has not reached yet still holds its actors, and resume skips such a cell.
            await operations.suspend(cell_id=cell_id)
            await operations.resume(cell_id=cell_id)
            self._num_restarts[index] += 1
