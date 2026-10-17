"""Offline tests: no GPU, no network, no checkpoint.

The launcher's one arithmetic claim: an episode's fixed sample count and the
global batch size cannot be set apart, and a geometry that would not divide
across the ranks is refused at construction rather than at the optimizer.
"""

from __future__ import annotations

import shlex

import pytest

from miles.utils.external_utils.model_args_utils import load_model_args
from scripts import run_qwen3_vl_4b_osworld as launcher
from scripts.run_qwen3_vl_4b_osworld import ScriptArgs, _refuse_geometry_overrides, execute


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


def test_megatron_divides_the_batch_over_data_parallel_ranks_not_gpus():
    """Tensor parallelism shards each sample, not the batch, so 30 samples that
    four FSDP ranks cannot split are one data-parallel rank's share at TP 4."""
    args = ScriptArgs(
        train_backend="megatron",
        tensor_parallel=4,
        rollout_batch_size=1,
        n_samples_per_prompt=2,
        samples_per_episode=15,
        num_gpus_per_node=4,
    )
    assert args.data_parallel_size == 1
    with pytest.raises(ValueError, match="does not divide 4 GPUs"):
        ScriptArgs(train_backend="megatron", tensor_parallel=3, samples_per_episode=15, num_gpus_per_node=4)


def test_smoke_mode_is_exempt_because_it_runs_no_optimizer_step():
    """Smoke passes --debug-rollout-only, so nothing consumes the batch and a
    geometry that would not divide is not a fault there."""
    args = ScriptArgs(mode="smoke", samples_per_episode=15, num_gpus_per_node=4)
    assert args.episodes_per_step == 2
    assert args.global_batch_size == 30


LAUNCHER_ENV = {"MILES_SCRIPT_OSWORLD_SAMPLES_PER_EPISODE": "15", "MILES_SCRIPT_GYM_URL": "http://x:11001"}


def _args(extra_env_vars=""):
    return ScriptArgs(samples_per_episode=15, num_gpus_per_node=4, extra_env_vars=extra_env_vars)


@pytest.mark.parametrize(
    "argv",
    [
        # Space-separated, the obvious spelling.
        "--micro-batch-size 1 --global-batch-size 120 --micro-batch-size 7",
        # `=`, which argparse accepts identically.
        "--micro-batch-size 1 --micro-batch-size=7",
        # Quoted forms: the backend tokenizes with shlex, so str.split kept
        # these as single tokens and the guard never saw the flag.
        "--micro-batch-size 1 \'--micro-batch-size=7\'",
        '--micro-batch-size 1 "--micro-batch-size=7"',
        "--micro-batch-size 1 --micro-batch-size\'=7\'",
    ],
)
def test_an_extra_arg_redeclaring_the_geometry_is_refused(argv):
    """__post_init__ sees the fields; --extra-args is appended to the argv
    afterwards and wins last, so the geometry the constructor accepted can
    still reach the trainer changed."""
    with pytest.raises(ValueError, match="--micro-batch-size is declared more than once"):
        _refuse_geometry_overrides(argv, _args(), LAUNCHER_ENV)


@pytest.mark.parametrize(
    "argv",
    [
        "--micro-batch-size 1 --micro-batch=7",
        "--micro-batch-size 1 --global-batch 8",
        # Quoted, which str.split would have kept as one opaque token.
        "--micro-batch-size 1 \'--micro-batch=7\'",
    ],
)
def test_an_abbreviated_geometry_flag_is_refused(argv):
    """Miles's parser leaves argparse's abbreviation matching on, so an
    unambiguous prefix reaches the same destination under another spelling."""
    with pytest.raises(ValueError, match="abbreviates"):
        _refuse_geometry_overrides(argv, _args(), LAUNCHER_ENV)


@pytest.mark.parametrize(
    "extra",
    [
        "MILES_SCRIPT_OSWORLD_SAMPLES_PER_EPISODE=16",
        '{"MILES_SCRIPT_OSWORLD_SAMPLES_PER_EPISODE":"16"}',
    ],
)
def test_an_extra_env_var_overriding_the_sample_count_is_refused(extra):
    """--extra-env-vars is merged over the launcher's own environment in both
    its JSON and NAME=value forms, so either can change the count
    --global-batch-size was derived from."""
    with pytest.raises(ValueError, match="MILES_SCRIPT_OSWORLD_SAMPLES_PER_EPISODE"):
        _refuse_geometry_overrides("--micro-batch-size 1", _args(extra), LAUNCHER_ENV)


def test_unrelated_extras_are_left_alone():
    argv = "--micro-batch-size 1 --log-passrate --seed 7 --num-rollout=3"
    assert _refuse_geometry_overrides(argv, _args("WANDB_MODE=offline"), LAUNCHER_ENV) is None


GEOMETRY_FLAGS = (
    "--rollout-batch-size",
    "--n-samples-per-prompt",
    "--global-batch-size",
    "--micro-batch-size",
    "--actor-num-nodes",
    "--actor-num-gpus-per-node",
    "--num-gpus-per-node",
)
MEGATRON_GEOMETRY_FLAGS = (
    *GEOMETRY_FLAGS,
    "--tensor-model-parallel-size",
    "--pipeline-model-parallel-size",
    "--context-parallel-size",
)


def _emitted(**kwargs):
    """What execute() hands the backend, captured without running anything."""
    captured = {}

    class _Backend:
        def exec_command_cpu(self, cmd, **kw):
            pass

        def execute_train(self, train_args, **kw):
            captured["argv"] = shlex.split(train_args)
            captured["kwargs"] = kw

    args = ScriptArgs(samples_per_episode=15, num_gpus_per_node=4, **kwargs)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(ScriptArgs, "create_backend", lambda self: _Backend())
    try:
        execute(args)
    finally:
        monkey.undo()
    return captured


def _emitted_argv(**kwargs):
    return _emitted(**kwargs)["argv"]


@pytest.mark.parametrize(
    "backend, flags",
    [("fsdp", GEOMETRY_FLAGS), ("megatron", MEGATRON_GEOMETRY_FLAGS)],
)
def test_the_geometry_is_declared_once_and_last(backend, flags):
    """This is what makes the guard belt-and-braces rather than load-bearing:
    argparse's store action overwrites, so the launcher's declaration being
    last is what defeats an override of any spelling."""
    argv = _emitted_argv(train_backend=backend)
    for flag in flags:
        assert argv.count(flag) == 1, f"{flag} is declared {argv.count(flag)} times"
    first_geometry = min(argv.index(flag) for flag in flags)
    non_geometry_after = [token for token in argv[first_geometry:] if token.startswith("--") and token not in flags]
    assert non_geometry_after == [], f"these flags follow the geometry block: {non_geometry_after}"


def test_an_unrelated_extra_arg_still_takes_effect():
    """Only the geometry block moved, so --extra-args keeps its purpose: a
    later declaration of a non-geometry flag still wins."""
    argv = _emitted_argv(extra_args="--lr 2e-6")
    assert [index for index, token in enumerate(argv) if token == "--lr"][-1] > argv.index("--optimizer")


def test_megatron_runs_the_vl_model_definition_through_the_bridge():
    emitted = _emitted(train_backend="megatron")
    argv = emitted["argv"]
    assert emitted["kwargs"]["megatron_model_type"] == "qwen3-vl-4B"
    assert argv[argv.index("--train-backend") + 1] == "megatron"
    assert argv[argv.index("--megatron-to-hf-mode") + 1] == "bridge"
    assert "--gradient-checkpointing" not in argv and "--attn-implementation" not in argv
    # LIN-2045 measured all 18 DeepStack tensors bit-identical to the base after six steps with recompute on.
    assert "--recompute-granularity" not in argv
    assert _emitted(train_backend="fsdp")["kwargs"]["megatron_model_type"] is None


def test_the_vl_model_definition_fixes_qwen3_vls_rotary_base(monkeypatch):
    """Qwen3-VL-4B's language tower resolves to Qwen3-4B's definition, whose
    rotary base defaults to the text model's 1,000,000; the VL checkpoint's
    rope_theta is 5,000,000, and nothing raises on the wrong one."""
    monkeypatch.setenv("MODEL_ARGS_ROTARY_BASE", "1000000")
    vl, text = load_model_args("qwen3-vl-4B").split(), load_model_args("qwen3-4B").split()
    assert vl[vl.index("--rotary-base") + 1] == "5000000"
    assert [token for token in vl if token != "5000000"] == [token for token in text if token != "1000000"]


@pytest.mark.parametrize("flag", ["--context-parallel-size", "--pipeline-model-parallel-size"])
def test_a_megatron_parallel_size_override_is_refused(flag):
    """Both enter the data-parallel size the batch was divided by, and CP > 1
    mis-places Qwen3-VL's vision embeddings while the plugin only warns."""
    with pytest.raises(ValueError, match=f"{flag} is declared more than once"):
        _emitted_argv(train_backend="megatron", extra_args=f"{flag} 2")


def test_a_launcher_driving_another_trees_miles_is_refused(monkeypatch):
    """On an rx devbox the image's PYTHONPATH put /root/miles ahead of the
    checkout, so the job ran the image's train.py while its Ray workers ran the
    checkout. That split was harmless only because the two trees agreed."""
    monkeypatch.setattr(launcher.command_utils, "repo_base_dir", "/root/miles")
    with pytest.raises(RuntimeError, match="imported miles from /root/miles"):
        _emitted_argv()
