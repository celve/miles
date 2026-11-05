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


def test_an_ocr_run_starts_one_cpu_pool_scoring_one_image_per_call():
    args = _parse_diffusion_run("--global-batch-size", "64", "--diffusion-ocr-num-workers", "3")

    (spec,) = _reward_specs(args).values()

    assert spec.name == "reward-ocr"
    assert (spec.pool.num_workers, spec.pool.batch_size, spec.pool.uses_gpu) == (3, 1, False)


def test_reward_pools_flag_starts_a_pool_for_a_reward_only_some_prompts_select():
    args = _parse_diffusion_run(*_PICKSCORE, "--diffusion-reward-pools", "ocr", "pickscore")

    assert [pool.name for pool in args.reward_pools] == ["pickscore", "ocr"]
    assert _get_placement_group_layout(args) == (5, 0)


def test_the_reward_pools_flag_reads_the_pickscore_flags_under_an_ocr_launch():
    args = _parse_diffusion_run(
        *("--global-batch-size", "64", "--diffusion-reward-pools", "pickscore"),
        *("--diffusion-pickscore-model-path", "/m", "--diffusion-pickscore-processor-path", "/p"),
    )

    assert [pool.name for pool in args.reward_pools] == ["ocr", "pickscore"]
    with pytest.raises(ValueError, match="diffusion-pickscore-model-path"):
        _parse_diffusion_run("--global-batch-size", "64", "--diffusion-reward-pools", "pickscore")


def test_an_ocr_pool_in_a_video_run_is_rejected_at_startup():
    with pytest.raises(ValueError, match="only the first frame"):
        _parse_diffusion_run(*_PICKSCORE, "--diffusion-output-num-frames", "5", "--diffusion-reward-pools", "ocr")


def test_an_ocr_worker_scores_each_image_of_a_call_on_its_own(monkeypatch):
    from miles.rollout.rm_hub import ocr

    monkeypatch.setattr(ocr, "OcrScorer", lambda: lambda image, prompt: float(image + len(prompt)))
    scorer = ocr.OcrBatchScorer(device=None)

    assert scorer.score_batch([1, 2], ['"a"', '"bb"']) == [4.0, 6.0]


@pytest.mark.parametrize(
    "override",
    [dict(cluster_backend="kubernetes"), dict(worker_comm_backend="rpc"), dict(deploy_component="trainer")],
)
def test_reward_pools_run_only_in_one_ray_deployment(override):
    from tests.fast.diffusion.test_validation import _supported

    from miles.utils.diffusion.arguments import validate_diffusion_args

    with pytest.raises(ValueError, match="a reward pool outside one Ray deployment"):
        validate_diffusion_args(_supported(**override))


def test_a_global_custom_reward_starts_no_pool_so_any_deployment_may_run_it():
    from tests.fast.diffusion.test_validation import _supported

    from miles.utils.diffusion.arguments import validate_diffusion_args

    validate_diffusion_args(_supported(worker_comm_backend="rpc", custom_rm_path="my.rewards.score"))


async def constant_reward(args, samples):
    return [0.5] * len(samples)


def test_a_global_custom_reward_declares_no_pickscore_pool():
    args = _parse_diffusion_run(*_PICKSCORE, "--custom-rm-path", f"{__name__}.constant_reward")

    assert _reward_specs(args) == {}
    assert _get_placement_group_layout(args) == (4, 0)


def test_reward_pools_flag_with_a_global_custom_reward_is_rejected_at_startup():
    with pytest.raises(ValueError, match="diffusion-reward-pools with a global --custom-rm-path"):
        _parse_diffusion_run(
            *_PICKSCORE, "--custom-rm-path", f"{__name__}.constant_reward", "--diffusion-reward-pools", "ocr"
        )
