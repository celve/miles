"""Offline tests: no GPU, no network, no checkpoint.

The launcher's one arithmetic claim: an episode's fixed sample count and the
global batch size cannot be set apart, and a geometry that would not divide
across the ranks is refused at construction rather than at the optimizer.
"""

from __future__ import annotations

import pytest

from scripts.run_qwen3_vl_4b_osworld import ScriptArgs


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
