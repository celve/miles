"""The reward pools a diffusion run declares, and where Miles places them."""

import pytest
from tests.fast.diffusion.test_validation import _parse_diffusion_run

from miles.ray.placement_group import _get_placement_group_layout
from miles.ray.specs.entrypoint import compute_specs
from miles.ray.specs.reward import RewardPoolSpec
from miles.utils.args.configs.scaling import ScalingConfig

_PICKSCORE = (
    *("--rm-type", "pickscore", "--diffusion-pickscore-model-path", "/models/PickScore_v1"),
    *("--diffusion-pickscore-processor-path", "/models/CLIP-ViT-H-14", "--global-batch-size", "64"),
    *("--actor-num-gpus-per-node", "4", "--num-gpus-per-node", "5"),
)


def _reward_specs(args) -> dict[str, RewardPoolSpec]:
    return {spec.name: spec for spec in compute_specs(args) if isinstance(spec, RewardPoolSpec)}


def test_the_wan_layout_appends_the_pickscore_gpu_as_slot_4_of_the_one_placement_group():
    """4 colocated train/rollout GPUs plus PickScore: the group holds 5 bundles, and PickScore takes the last."""
    args = _parse_diffusion_run(*_PICKSCORE)

    (spec,) = _reward_specs(args).values()
    scheduling = spec.scheduling(ScalingConfig.slice_from(args))

    assert spec.name == "reward-pickscore"
    assert (scheduling.pg_name, scheduling.pg_slot_offset, scheduling.num_cells) == ("rollout", 4, 1)
    assert _get_placement_group_layout(args) == (5, 0)
    assert spec.pool.batch_size == 8 and spec.pool.scorer_kwargs["frames_per_forward"] == 8


def test_a_cpu_pickscore_pool_leaves_the_group_at_the_train_gpus():
    args = _parse_diffusion_run(*_PICKSCORE, "--diffusion-pickscore-num-gpus-per-worker", "0")

    (spec,) = _reward_specs(args).values()

    assert spec.scheduling(ScalingConfig.slice_from(args)).pg_name is None
    assert _get_placement_group_layout(args) == (4, 0)


@pytest.mark.parametrize("gpus", ["0.5", "2"])
def test_a_pickscore_worker_on_a_fraction_of_a_gpu_or_several_is_rejected(gpus):
    with pytest.raises(ValueError, match="one whole GPU or on the CPU"):
        _parse_diffusion_run(*_PICKSCORE, "--diffusion-pickscore-num-gpus-per-worker", gpus)


def test_an_ocr_run_declares_no_gpu_and_keeps_its_two_bundles():
    args = _parse_diffusion_run("--global-batch-size", "64")

    assert all(not spec.pool.uses_gpu for spec in _reward_specs(args).values())
    assert _get_placement_group_layout(args) == (2, 0)
