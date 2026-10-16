"""The conversion handoff the launcher's flags have to satisfy.

Every sample this recipe returns carries a ``rollout_id``, which makes the batch
compact. ``postprocess_rollout_data`` then skips its whole non-compact block and
returns no ``dynamic_global_batch_size``, and ``convert_samples_to_train_data``
asserts that against ``--use-dynamic-global-batch-size``. So the launcher must
pass dynamic micro-batching without the dynamic *global* batch size, and this
test is what keeps the two in step.
"""

from argparse import Namespace
from pathlib import Path

import pytest
import torch

from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.utils.types import Sample

LAUNCHER = "scripts/run_qwen3_vl_4b_harbor_cua.py"


def compact_sample(index: int, rollout_id: int, reward: float) -> Sample:
    return Sample(
        index=index,
        rollout_id=rollout_id,
        group_index=0,
        tokens=[1, 2, 3, 4, 5],
        response_length=2,
        loss_mask=[1, 1],
        rollout_log_probs=[-0.1, -0.2],
        reward=reward,
        status=Sample.Status.COMPLETED,
        multimodal_train_inputs={"pixel_values": torch.zeros(4, 8), "image_grid_thw": torch.tensor([[1, 2, 2]])},
    )


def conversion_args(**overrides):
    """Only the fields the conversion path reads, so the test runs without an engine."""
    defaults = {
        "global_batch_size": 2,
        "use_dynamic_global_batch_size": False,
        "disable_rollout_trim_samples": False,
        "advantage_estimator": "grpo",
        "rewards_normalization": True,
        "grpo_std_normalization": True,
        "reward_key": None,
        "n_samples_per_prompt": 2,
        "rollout_batch_size": 1,
        "use_rollout_entropy": False,
        "enable_sample_ownership_checker": False,
        "save_debug_rollout_data": None,
        "use_critic": False,
        "use_sampling_support_replay": False,
        "rollout_top_logprobs_num": 0,
        "use_opd": False,
        "loss_mask_type": None,
    }
    return Namespace(**{**defaults, **overrides})


def convert(args, data):
    data, metadata = postprocess_rollout_data(args, data, {"dp_size": 1})
    return convert_samples_to_train_data(args, data, metadata, None, None)


def test_compact_siblings_convert_without_the_dynamic_global_batch_flag():
    args = conversion_args()
    data = [[compact_sample(0, 0, 1.0), compact_sample(1, 0, 1.0)], [compact_sample(2, 1, 0.0)]]

    train_data = convert(args, [sample for group in data for sample in group])

    assert train_data["rollout_ids"] == [0, 0, 1]
    # Every sibling carries the whole rollout's mask total, so splitting adds no weight.
    assert train_data["rollout_mask_sums"] == [4, 4, 2]
    assert "dynamic_global_batch_size" not in train_data


def test_the_dynamic_global_batch_flag_is_what_breaks_a_compact_batch():
    """The failure the launcher's comment names, pinned so a future flag change sees it."""
    args = conversion_args(use_dynamic_global_batch_size=True)
    data = [compact_sample(0, 0, 1.0), compact_sample(1, 0, 1.0)]

    with pytest.raises(AssertionError):
        convert(args, data)


def test_siblings_disagreeing_on_reward_are_refused():
    args = conversion_args()
    data = [compact_sample(0, 0, 1.0), compact_sample(1, 0, 0.0)]

    with pytest.raises(ValueError, match="must share one reward"):
        convert(args, data)


def test_the_launcher_passes_dynamic_microbatching_and_not_the_global_flag():
    source = (Path(__file__).parents[5] / LAUNCHER).read_text()

    assert "--use-dynamic-batch-size" in source
    assert "--max-tokens-per-gpu" in source
    assert "--use-dynamic-global-batch-size " not in source.replace("NOT --use-dynamic-global-batch-size", "")


def _launcher_source() -> str:
    return (Path(__file__).parents[5] / LAUNCHER).read_text()


def test_the_default_backend_is_the_one_with_working_multimodal_rope():
    """FSDP trains Qwen3-VL with linear positions, so megatron is the default (LIN-2046)."""
    source = _launcher_source()

    assert 'backend: Literal["megatron", "fsdp"] = "megatron"' in source
    assert "--megatron-to-hf-mode bridge" in source


def test_the_importance_ratio_uses_the_recorded_rollout_logprobs():
    """Without this the ratio is taken against a trainer recomputation, not the engine."""
    assert "--use-rollout-logprobs" in _launcher_source()


def test_exported_values_are_shell_quoted():
    assert "shlex.quote" in _launcher_source()


def test_the_rotary_base_is_set_where_the_model_args_are_expanded():
    """scripts/models/qwen3-4B.py reads it in the launcher process, not on the workers."""
    source = _launcher_source()

    assert 'os.environ["MODEL_ARGS_ROTARY_BASE"]' in source
    assert "ROTARY_BASE = 5_000_000" in source
    assert "MODEL_ARGS_ROTARY_BASE" not in source.split("def _worker_env")[1]
