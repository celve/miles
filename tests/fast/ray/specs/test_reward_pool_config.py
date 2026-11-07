"""A text run's reward pools, declared with --reward-pool-config."""

import pytest
import yaml
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

from miles.ray.placement_group import _get_placement_group_layout
from miles.ray.specs.entrypoint import compute_specs
from miles.ray.specs.reward import RewardPoolSpec
from miles.utils.args.configs.scaling import ScalingConfig

_JUDGE = dict(
    name="judge",
    scorer_path="miles.rollout.rm_hub.video.vlm_judge.VlmJudgeScorer",
    scorer_kwargs=dict(model_path="/models/Qwen3-VL-8B-Instruct"),
    num_workers=1,
    uses_gpu=True,
    batch_size=4,
)


def _parse(tmp_path, pools: list[dict], *argv: str):
    path = tmp_path / "pools.yaml"
    path.write_text(yaml.safe_dump(pools))
    return parse_fsdp_test_config("--reward-pool-config", str(path), *argv)


def test_a_text_run_declares_its_pool_on_a_gpu_after_the_rollout_gpus(tmp_path):
    gpus = ("--rollout-num-gpus", "1", "--num-gpus-per-node", "2")
    args = _parse(tmp_path, [_JUDGE], *gpus)

    (pool,) = args.reward_pools
    (spec,) = [spec for spec in compute_specs(args) if isinstance(spec, RewardPoolSpec)]
    scheduling = spec.scheduling(ScalingConfig.slice_from(args))

    assert pool.name == "judge" and pool.scorer_kwargs == _JUDGE["scorer_kwargs"] and pool.num_gpus_per_node == 2
    assert (scheduling.pg_name, scheduling.pg_slot_offset) == ("rollout", 1)
    assert _get_placement_group_layout(args)[0] == _get_placement_group_layout(parse_fsdp_test_config(*gpus))[0] + 1


def test_a_colocated_text_pool_shares_the_first_rollout_gpu_and_sleeps_while_training_holds_it(tmp_path):
    args = _parse(tmp_path, [_JUDGE | dict(colocated=True)], "--colocate")

    (pool,) = args.reward_pools

    assert pool.colocated and pool.needs_offload
    assert args.reward_scaling.pools["judge"].num_gpus == 0
    assert _get_placement_group_layout(args) == (1, 0)


def test_a_run_without_the_flag_declares_no_pool():
    assert parse_fsdp_test_config().reward_pools == []


@pytest.mark.parametrize("derived", ["num_gpus_per_node", "needs_offload"])
def test_a_pool_may_not_set_what_miles_derives_from_the_run(tmp_path, derived):
    with pytest.raises(AssertionError, match=derived):
        _parse(tmp_path, [_JUDGE | {derived: 1}])


def test_two_pools_of_one_name_are_rejected(tmp_path):
    with pytest.raises(AssertionError, match="unique"):
        _parse(tmp_path, [_JUDGE, _JUDGE])


@pytest.mark.parametrize(
    "argv",
    [("--cluster-backend", "kubernetes"), ("--worker-comm-backend", "rpc")],
)
def test_a_declared_pool_runs_only_in_one_ray_deployment(tmp_path, argv):
    with pytest.raises(ValueError, match="one Ray deployment"):
        _parse(tmp_path, [_JUDGE], *argv)
