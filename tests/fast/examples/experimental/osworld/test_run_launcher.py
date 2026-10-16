"""Offline tests: no GPU, no network, no checkpoint.

The launcher's one arithmetic claim: an episode's fixed sample count and the
global batch size cannot be set apart, and a geometry that would not divide
across the ranks is refused at construction rather than at the optimizer.
"""

from __future__ import annotations

import pytest

from scripts.run_qwen3_vl_4b_osworld import ScriptArgs, _refuse_geometry_overrides


def test_global_batch_size_is_derived_from_the_episode_sample_count():
    args = ScriptArgs(rollout_batch_size=2, n_samples_per_prompt=4, samples_per_episode=15, num_gpus_per_node=4)
    assert args.episodes_per_step == 8
    assert args.global_batch_size == 120


def test_a_geometry_that_does_not_divide_across_ranks_is_refused():
    """1 x 2 x 15 = 30 samples over 4 ranks gives shards of 8, 8, 7, 7. The
    train side then floor-divides its own shard, so rank 0 holds 8 rewards
    against 7 forward rows -- an IndexError in get_grpo_returns rather than a
    refusal anyone can read."""
    with pytest.raises(ValueError, match="does not divide across 4 ranks"):
        ScriptArgs(rollout_batch_size=1, n_samples_per_prompt=2, samples_per_episode=15, num_gpus_per_node=4)


def test_a_micro_batch_that_does_not_divide_the_rank_share_is_refused():
    with pytest.raises(ValueError, match="must divide"):
        ScriptArgs(
            rollout_batch_size=2,
            n_samples_per_prompt=4,
            samples_per_episode=15,
            num_gpus_per_node=4,
            micro_batch_size=7,
        )


def test_smoke_mode_is_exempt_because_it_runs_no_optimizer_step():
    """Smoke passes --debug-rollout-only, so nothing consumes the batch and a
    geometry that would not divide is not a fault there."""
    args = ScriptArgs(mode="smoke", samples_per_episode=15, num_gpus_per_node=4)
    assert args.episodes_per_step == 2
    assert args.global_batch_size == 30


def test_an_extra_arg_overriding_the_geometry_is_refused():
    """__post_init__ sees the fields; --extra-args is appended to the argv
    afterwards and wins last, so the geometry the constructor accepted can
    still reach the trainer changed."""
    with pytest.raises(ValueError, match="--micro-batch-size is set more than once"):
        _refuse_geometry_overrides("--micro-batch-size 1 --global-batch-size 120 --micro-batch-size 7", "")


def test_an_extra_env_var_overriding_the_sample_count_is_refused():
    """--extra-env-vars is merged over the launcher's own environment, so it
    can change the count --global-batch-size was derived from."""
    with pytest.raises(ValueError, match="MILES_SCRIPT_OSWORLD_SAMPLES_PER_EPISODE"):
        _refuse_geometry_overrides("--micro-batch-size 1", "MILES_SCRIPT_OSWORLD_SAMPLES_PER_EPISODE=16")


def test_an_unrelated_extra_arg_is_left_alone():
    assert _refuse_geometry_overrides("--micro-batch-size 1 --log-passrate --seed 7", "WANDB_MODE=offline") is None
