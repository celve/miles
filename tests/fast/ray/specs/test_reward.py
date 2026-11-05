from __future__ import annotations

from argparse import Namespace

import pytest
import torch
from tests.fast.ray.rollout.conftest import make_rollout_config

from miles.backends.sglang_utils.sglang_scaling_config import SglangScalingConfig
from miles.ray.rollout.reward_worker import RewardWorker
from miles.ray.specs.reward import REWARD_WORKER_CLASS, RewardPoolSpec, reward_cell_id, reward_worker_name
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils.args.configs import reward_pool as reward_pool_module
from miles.utils.args.configs.reward_pool import RewardPoolConfig, RewardScalingConfig, compute_reward_pools
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.function_registry import load_function
from miles.utils.misc import NodeProbeMixin
from miles.utils.workers.ray_worker_manager import bootstrapped_worker_class
from miles.utils.workers.worker_spec import WorkerLaunchContext


def _pool(name: str, *, uses_gpu: bool, num_workers: int = 1) -> RewardPoolConfig:
    return RewardPoolConfig(
        name=name,
        scorer_path=f"{__name__}.RecordingScorer",
        num_workers=num_workers,
        uses_gpu=uses_gpu,
        batch_size=8,
        num_gpus_per_node=8,
    )


def _spec(pool: RewardPoolConfig) -> RewardPoolSpec:
    (spec,) = RewardPoolSpec.create(make_rollout_config().model_copy(update={"reward_pools": [pool]}))
    return spec


def _scaling(pools: list[RewardPoolConfig], *, gpu_offset: int) -> ScalingConfig:
    return ScalingConfig(
        sglang_scaling=SglangScalingConfig(groups={}),
        reward_scaling=RewardScalingConfig.resolve(pools, gpu_offset=gpu_offset),
    )


def _launch_context(spec: RewardPoolSpec) -> WorkerLaunchContext:
    return WorkerLaunchContext(
        args=spec.args, cell_index=0, worker_in_cell_index=0, num_workers_per_cell=1, gpu_ids=[]
    )


class RecordingScorer:
    def __init__(self, *, device: torch.device, scale: float = 1.0) -> None:
        self.device = device
        self.scale = scale

    def score_batch(self, outputs, prompts):
        return [self.scale * len(prompt) for prompt in prompts]


class TestRewardPoolSpec:
    def test_gpu_pools_take_rollout_slots_after_the_rollout_and_eval_gpus_in_declaration_order(self):
        """Reward GPUs are the tail of the rollout view, so each pool starts where the one before it ends."""
        pools = [
            _pool("pickscore", uses_gpu=True, num_workers=2),
            _pool("ocr", uses_gpu=False),
            _pool("hps", uses_gpu=True),
        ]
        scaling = _scaling(pools, gpu_offset=4 + 1)

        schedulings = {pool.name: _spec(pool).scheduling(scaling) for pool in pools}

        assert scaling.reward_scaling.num_gpus == 3
        assert {name: (s.pg_name, s.pg_slot_offset) for name, s in schedulings.items() if s.pg_name} == {
            "pickscore": ("rollout", 5),
            "hps": ("rollout", 7),
        }
        assert (schedulings["pickscore"].num_cells, schedulings["pickscore"].num_workers_per_cell) == (2, 1)
        assert schedulings["pickscore"].num_gpu_slots_per_worker == 1

    def test_a_gpu_worker_fits_beside_a_trainer_and_an_engine_on_one_bundle(self):
        """0.4 trainer + 0.2 engine + the reward share must fit one bundle of 1 GPU and 1 CPU."""
        pool = _pool("pickscore", uses_gpu=True)
        scheduling = _spec(pool).scheduling(_scaling([pool], gpu_offset=0))

        assert 0.4 + 0.2 + scheduling.num_gpus_per_worker <= 1.0
        assert 0.4 + 0.2 + scheduling.num_cpus_per_worker <= 1.0

    def test_a_cpu_pool_claims_no_gpu_and_no_placement_group(self):
        pool = _pool("ocr", uses_gpu=False, num_workers=4)
        spec = _spec(pool)
        scheduling = spec.scheduling(_scaling([pool], gpu_offset=2))

        assert (scheduling.num_cells, scheduling.num_gpus_per_worker, scheduling.pg_name) == (4, 0, None)
        assert spec.env_var(_launch_context(spec)) == {}

    def test_a_gpu_worker_keeps_every_device_visible_to_pick_its_own(self):
        """Ray would narrow CUDA_VISIBLE_DEVICES to a fractional share; the worker reads its GPU from gpu_ids."""
        spec = _spec(_pool("pickscore", uses_gpu=True))

        assert spec.env_var(_launch_context(spec)) == {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST}

    def test_pool_and_worker_names_follow_the_pool_cell_worker_scheme(self):
        assert _spec(_pool("pickscore", uses_gpu=True)).name == "reward-pickscore"
        assert reward_cell_id("pickscore", cell_index=1) == "reward-pickscore-00001"
        assert reward_worker_name("pickscore", cell_index=1) == "reward-pickscore-00001-00000"

    def test_the_worker_class_is_the_reward_worker_and_answers_the_node_probe(self):
        assert load_function(REWARD_WORKER_CLASS) is RewardWorker
        assert issubclass(bootstrapped_worker_class(REWARD_WORKER_CLASS), NodeProbeMixin)


class TestRewardWorker:
    def test_a_cpu_worker_builds_its_scorer_on_the_cpu_with_the_pools_kwargs(self):
        pool = _pool("ocr", uses_gpu=False).model_copy(update=dict(scorer_kwargs={"scale": 2.0}))
        worker = RewardWorker(args=make_rollout_config(), pool=pool, gpu_ids=[])

        assert worker._scorer.device == torch.device("cpu")
        assert worker.score_batch([None, None], ["ab", "abc"]) == [4.0, 6.0]

    @pytest.mark.parametrize("mask,physical,local", [("4,5,6,7,8", 8, 4), ("3,0,1,2", 0, 1)])
    def test_a_gpu_worker_maps_its_physical_gpu_through_an_inherited_device_mask(
        self, monkeypatch, mask, physical, local
    ):
        """The bundle names a physical GPU; a process that sees a masked or reordered set numbers it differently."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
        bound = []
        monkeypatch.setattr(torch.accelerator, "set_device_index", bound.append)
        worker = RewardWorker(args=make_rollout_config(), pool=_pool("pickscore", uses_gpu=True), gpu_ids=[physical])

        assert bound == [local] and worker._scorer.device.index == local

    def test_a_gpu_worker_binds_the_device_its_bundle_names(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
        bound = []
        monkeypatch.setattr(torch.accelerator, "set_device_index", bound.append)
        worker = RewardWorker(args=make_rollout_config(), pool=_pool("pickscore", uses_gpu=True), gpu_ids=[3])

        assert bound == [3]
        assert worker._scorer.device.index == 3


def test_a_run_that_declares_no_pool_has_no_reward_spec():
    assert RewardPoolSpec.create(make_rollout_config()) == []


class TestColocatedRewardPools:
    @staticmethod
    def _args(**overrides) -> Namespace:
        values = dict(
            rollout_num_gpus=4,
            eval_num_gpus=0,
            offload_rollout=True,
            colocate=True,
            debug_rollout_only=False,
            starts_inference_engines=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=4,
        )
        return Namespace(**(values | overrides))

    def test_a_colocated_pool_takes_the_first_rollout_slots_adds_no_gpu_and_sleeps_with_the_engines(self, monkeypatch):
        colocated = _pool("pickscore", uses_gpu=True).model_copy(update=dict(colocated=True))
        monkeypatch.setattr(
            reward_pool_module, "_declare_reward_pools", lambda args: [colocated, _pool("hps", uses_gpu=True)]
        )

        pools, scaling = compute_reward_pools(self._args())

        assert {name: (pool.gpu_offset, pool.num_gpus) for name, pool in scaling.pools.items()} == {
            "pickscore": (0, 0),
            "hps": (4, 1),
        }
        assert [(pool.name, pool.needs_offload) for pool in pools] == [("pickscore", True), ("hps", False)]
        assert _spec(pools[0]).scheduling(_scaling_of(scaling)).pg_slot_offset == 0

    def test_a_colocated_pool_never_sleeps_when_rollout_is_not_offloaded(self, monkeypatch):
        colocated = _pool("pickscore", uses_gpu=True).model_copy(update=dict(colocated=True))
        monkeypatch.setattr(reward_pool_module, "_declare_reward_pools", lambda args: [colocated])

        (pool,), _ = compute_reward_pools(self._args(offload_rollout=False))

        assert not pool.needs_offload

    def test_a_colocated_pool_larger_than_the_rollout_gpus_is_rejected(self):
        colocated = _pool("pickscore", uses_gpu=True, num_workers=3).model_copy(update=dict(colocated=True))

        with pytest.raises(AssertionError, match="rollout GPUs hold only 2 more"):
            RewardScalingConfig.resolve([colocated], gpu_offset=2, num_rollout_gpus=2)


def _scaling_of(reward_scaling: RewardScalingConfig) -> ScalingConfig:
    return ScalingConfig(sglang_scaling=SglangScalingConfig(groups={}), reward_scaling=reward_scaling)
