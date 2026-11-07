from __future__ import annotations

import asyncio

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability

from miles.ray.rollout.reward_pool import RewardPoolClient
from miles.utils.args.configs.reward_pool import RewardPoolConfig
from miles.utils.workers.naming import parse_worker_name
from miles.utils.workers.worker_handle import WorkerUnreachableError
from miles.utils.workers.worker_provider.base import CellInfo


def _config(*, num_workers: int = 2, batch_size: int = 2) -> RewardPoolConfig:
    return RewardPoolConfig(
        name="pickscore",
        scorer_path="unused.Scorer",
        num_workers=num_workers,
        uses_gpu=True,
        batch_size=batch_size,
        num_gpus_per_node=8,
    )


class _Handle:
    def __init__(self, cluster: _Cluster, cell_index: int, generation: int) -> None:
        self.cluster, self.cell_index, self.generation = cluster, cell_index, generation

    async def score_batch(self, outputs, prompts):
        await asyncio.sleep(0)
        if self.generation != self.cluster.generation[self.cell_index]:
            raise WorkerUnreachableError("stale actor")
        if self.cell_index in self.cluster.dead:
            raise WorkerUnreachableError("actor died")
        self.cluster.calls.append((self.cell_index, list(prompts)))
        return [None if prompt == "unscorable" else len(prompt) for prompt in prompts]

    async def sleep(self):
        self._lifecycle("sleep")

    async def wake(self):
        self._lifecycle("wake")

    def _lifecycle(self, method):
        if self.generation != self.cluster.generation[self.cell_index] or self.cell_index in self.cluster.dead:
            raise WorkerUnreachableError("actor died")
        self.cluster.calls.append((self.cell_index, method))

    async def wait_ready(self, *, timeout):
        pass


class _Cluster:
    """Stands in for the worker manager: handles per cell generation, suspend/resume bump the generation."""

    def __init__(self, num_workers: int) -> None:
        self.generation = [0] * num_workers
        self.dead: set[int] = set()
        self.torn_down: set[int] = set()
        self.calls: list[tuple[int, list[str]]] = []
        self.operations: list[tuple[str, str]] = []

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[str]]:
        """A cell the manager's liveness scan tore down lists no worker, as RayWorkerManager.get_worker_infos does."""
        return [
            [] if int(cell_id.rsplit("-", 1)[1]) in self.torn_down else [f"{cell_id}-00000"] for cell_id in cell_ids
        ]

    def get_handles_of_worker_infos(self, infos: list[str]) -> dict[str, _Handle]:
        handles = {}
        for name in infos:
            _, cell_index, _ = parse_worker_name(name)
            handles[name] = _Handle(self, cell_index, self.generation[cell_index])
        return handles

    async def suspend(self, *, cell_id: str) -> None:
        self.operations.append(("suspend", cell_id))

    async def resume(self, *, cell_id: str) -> None:
        self.operations.append(("resume", cell_id))
        cell_index = int(cell_id.rsplit("-", 1)[1])
        self.generation[cell_index] += 1
        self.dead.discard(cell_index)
        self.torn_down.discard(cell_index)

    async def cell_infos(self, *, pool_ids, category):
        return {
            f"reward-pickscore-{index:05d}": CellInfo(
                cell_id=f"reward-pickscore-{index:05d}",
                pool_id="reward-pickscore",
                alive=index not in self.dead and index not in self.torn_down,
                worker_names=[],
                workers_hash="h",
                meta={},
            )
            for index in range(len(self.generation))
        }


def _client(cluster: _Cluster, config: RewardPoolConfig, **kwargs) -> RewardPoolClient:
    capability = FakeBackendCapability(static_provider=cluster, cell_operations=cluster)
    return RewardPoolClient(config=config, capability=capability, **kwargs)


async def test_batches_go_round_robin_across_calls_and_scores_keep_input_order():
    """Same dealing as the actor pool it replaces: chunks of batch_size, one counter that runs across calls."""
    cluster = _Cluster(num_workers=2)
    client = _client(cluster, _config())

    first = await client.score([None] * 3, ["a", "bb", "ccc"])
    second = await client.score([None] * 2, ["dddd", "e"])

    assert first == [1.0, 2.0, 3.0] and second == [4.0, 1.0]
    assert sorted(cluster.calls) == [(0, ["a", "bb"]), (0, ["dddd", "e"]), (1, ["ccc"])]


async def test_a_scorer_none_stays_none_so_the_missing_reward_filter_can_drop_it():
    client = _client(_Cluster(num_workers=1), _config(num_workers=1))

    assert await client.score([None] * 3, ["a", "unscorable", "ccc"]) == [1.0, None, 3.0]


async def test_a_worker_that_died_is_restarted_and_scores_the_same_batch():
    cluster = _Cluster(num_workers=2)
    client = _client(cluster, _config())
    await client.score([None] * 2, ["a", "bb"])
    cluster.dead.add(1)

    scores = await client.score([None] * 2, ["xyz", "q"])

    assert scores == [3.0, 1.0]
    assert cluster.operations == [("suspend", "reward-pickscore-00001"), ("resume", "reward-pickscore-00001")]
    assert cluster.calls[-1] == (1, ["xyz", "q"])


async def test_two_batches_that_find_the_same_worker_dead_restart_it_once():
    cluster = _Cluster(num_workers=1)
    client = _client(cluster, _config(num_workers=1, batch_size=1))
    await client.score([None], ["a"])
    cluster.dead.add(0)

    scores = await client.score([None] * 2, ["bb", "ccc"])

    assert scores == [2.0, 3.0]
    assert [op for op, _ in cluster.operations] == ["suspend", "resume"]


async def test_a_worker_that_keeps_dying_fails_the_call_after_the_restart_budget():
    cluster = _Cluster(num_workers=1)
    client = _client(cluster, _config(num_workers=1), max_restarts_per_call=2)

    async def always_dead(*, cell_id):
        cluster.operations.append(("resume", cell_id))

    cluster.resume = always_dead
    cluster.dead.add(0)

    with pytest.raises(WorkerUnreachableError):
        await client.score([None], ["a"])
    assert [op for op, _ in cluster.operations].count("resume") == 2


async def test_ensure_healthy_restarts_every_cell_that_is_down_before_the_rollout():
    cluster = _Cluster(num_workers=3)
    client = _client(cluster, _config(num_workers=3))
    cluster.dead.update({0, 2})

    await client.ensure_healthy()

    assert [cell for op, cell in cluster.operations if op == "resume"] == [
        "reward-pickscore-00000",
        "reward-pickscore-00002",
    ]
    assert not cluster.dead


async def test_a_worker_torn_down_before_its_first_call_is_restarted_by_that_call():
    """ensure_healthy saw it alive, then it died and the liveness scan dropped its cell before the first lookup."""
    cluster = _Cluster(num_workers=2)
    client = _client(cluster, _config())
    await client.ensure_healthy()
    cluster.torn_down.add(0)

    scores = await client.score([None] * 2, ["ab", "c"])

    assert scores == [2.0, 1.0]
    assert cluster.operations == [("suspend", "reward-pickscore-00000"), ("resume", "reward-pickscore-00000")]


async def test_sleep_and_wake_reach_every_worker_and_restart_one_that_died():
    cluster = _Cluster(num_workers=2)
    client = _client(cluster, _config())
    cluster.dead.add(1)

    await client.sleep()
    await client.wake()

    assert sorted(cluster.calls) == [(0, "sleep"), (0, "wake"), (1, "sleep"), (1, "wake")]
    assert [op for op, _ in cluster.operations] == ["suspend", "resume"]
