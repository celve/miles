from argparse import Namespace
from types import SimpleNamespace

import pytest

from tests.fast.fixtures.args_fixtures import _RESOLVED_AFTER_PARSING, parser_defaults

from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.diffusion.arguments import _HOOKS, validate_diffusion_args, validate_diffusion_sglang_config


def _supported(**overrides):
    # The CLI defaults the validator really sees: fields derived after parsing (e.g. multi_lora) are absent.
    defaults = {k: v for k, v in parser_defaults().items() if k not in _RESOLVED_AFTER_PARSING}
    recipe = dict(
        train_backend="fsdp",
        colocate=True,
        use_miles_router=True,
        rollout_batch_size=8,
        n_samples_per_prompt=16,
        global_batch_size=64,
        hf_checkpoint="/models/stable-diffusion-3.5-medium",
        **_HOOKS,
    )
    return Namespace(**(defaults | recipe | overrides))


def test_the_recipe_layout_passes_and_resolves_the_family_defaults():
    args = _supported()

    validate_diffusion_args(args)

    assert args.diffusion_model_family == "sd3"
    assert "attn.to_q" in args.diffusion_lora_target_modules


@pytest.mark.parametrize(
    "override,message",
    [
        (dict(train_backend="megatron"), "fsdp"),
        (dict(colocate=False), "colocate"),
        (dict(rollout_num_gpus_per_engine=2), "several GPUs"),
        (dict(use_miles_router=False), "router"),
        (dict(advantage_estimator="ppo"), "critic"),
        (dict(ref_load="/ref"), "reference"),
        (dict(use_kl_loss=True), "reference"),
        (dict(use_opd=True), "distillation"),
        (dict(eval_interval=5), "evaluation"),
        (dict(fully_async=True), "async"),
        (dict(partial_rollout=True), "async"),
        (dict(dynamic_sampling_filter_path="f"), "filters"),
        (dict(rm_type="math"), "reward"),
        (dict(use_output_store=True), "output store"),
        (dict(object_store_backend="mooncake"), "Mooncake"),
        (dict(use_fault_tolerance=True), "fault tolerance"),
        (dict(delay_split_train_data_by_dp=True), "delay-split"),
        (dict(indep_dp=True), "independent"),
        (dict(lora_rank=8), "text LoRA"),
        (dict(multi_lora_n_adapters=2), "text LoRA"),
        (dict(true_on_policy_mode=True), "on-policy"),
        (dict(debug_rollout_only=True), "debug"),
        (dict(ci_test=True), "ci-test"),
    ],
)
def test_each_unsupported_option_is_rejected_at_startup(override, message):
    with pytest.raises(ValueError, match=message):
        validate_diffusion_args(_supported(**override))


def test_a_foreign_hook_is_rejected():
    with pytest.raises(ValueError, match="hooks"):
        validate_diffusion_args(_supported(rollout_function_path="miles.rollout.other.Fn"))


def _group(worker_type=WorkerType.REGULAR, num_gpus_per_engine=1):
    return SimpleNamespace(worker_type=worker_type, num_gpus_per_engine=num_gpus_per_engine)


def test_the_engine_layout_is_one_model_of_regular_single_gpu_groups():
    validate_diffusion_sglang_config(SimpleNamespace(models=[SimpleNamespace(name="m", server_groups=[_group()])]))
    with pytest.raises(ValueError, match="exactly one model"):
        validate_diffusion_sglang_config(
            SimpleNamespace(models=[SimpleNamespace(name=n, server_groups=[_group()]) for n in "ab"])
        )
    for group in (_group(WorkerType.PREFILL), _group(num_gpus_per_engine=2)):
        with pytest.raises(ValueError, match="regular single-GPU"):
            validate_diffusion_sglang_config(
                SimpleNamespace(models=[SimpleNamespace(name="m", server_groups=[group])])
            )
