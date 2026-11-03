from argparse import Namespace
from types import SimpleNamespace

import pytest
from tests.fast.fixtures.args_fixtures import _RESOLVED_AFTER_PARSING, parser_defaults

from miles.backends.fsdp_utils.arguments import FSDPArgs
from miles.backends.fsdp_utils.diffusion import pipeline_config
from miles.backends.fsdp_utils.diffusion.pipeline_config import SD3TrainPipelineConfig
from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.utils.args.configs.diffusion import DiffusionConfig
from miles.utils.diffusion.arguments import (
    _HOOKS,
    validate_diffusion_args,
    validate_diffusion_batch_size,
    validate_diffusion_sglang_config,
)


def _supported(**overrides):
    # The CLI defaults the validator really sees: fields derived after parsing (e.g. multi_lora) are absent.
    defaults = {k: v for k, v in parser_defaults().items() if k not in _RESOLVED_AFTER_PARSING}
    recipe = dict(
        train_backend="fsdp",
        colocate=True,
        router="miles",
        rollout_batch_size=8,
        n_samples_per_prompt=16,
        global_batch_size=64,
        hf_checkpoint="/models/stable-diffusion-3.5-medium",
        # An FSDP backend flag, absent from the shared parser's defaults.
        optimizer_cpu_offload=False,
        micro_batch_size=1,
        rm_type="ocr",
        eps_clip=1e-4,
    )
    return Namespace(**(defaults | vars(FSDPArgs()) | recipe | overrides))


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
        (dict(router="sglang"), "router"),
        (dict(advantage_estimator="ppo"), "critic"),
        (dict(ref_load="/ref"), "reference"),
        (dict(use_kl_loss=True), "reference"),
        (dict(use_opd=True), "distillation"),
        (dict(eval_interval=5), "evaluation"),
        (dict(fully_async=True), "async"),
        (dict(partial_rollout=True), "async"),
        (dict(dynamic_sampling_filter_path="f"), "filters"),
        (dict(rm_type="math"), "reward"),
        (dict(rm_type=None), "reward"),
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
        (dict(eps_clip=0.2), "explicitly"),
        (dict(loss_type="custom_loss"), "--loss-type other than the diffusion algorithms"),
        (dict(loss_type="sft_loss"), "--loss-type other than the diffusion algorithms"),
        (dict(use_dynamic_batch_size=True), "token-based batching"),
        (dict(use_dynamic_global_batch_size=True), "token-based batching"),
        (dict(micro_batch_size=8), "--micro-batch-size"),
        (dict(eps_clip_high=0.3), "asymmetric"),
        (dict(eps_clip_c=3.0), "dual-clip"),
        (dict(save_debug_rollout_data="/dump/{rollout_id}.pt"), "debug dumps"),
        (dict(save_debug_train_data="/dump/{rollout_id}_{rank}.pt"), "debug dumps"),
        (dict(dump_details="/dump"), "debug dumps"),
    ],
)
def test_each_unsupported_option_is_rejected_at_startup(override, message):
    with pytest.raises(ValueError, match=message):
        validate_diffusion_args(_supported(**override))


def test_a_custom_reward_replaces_the_ocr_reward():
    validate_diffusion_args(_supported(rm_type=None, custom_rm_path="my.rewards.score"))


def test_a_text_run_rejects_the_image_reward():
    from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

    with pytest.raises(ValueError, match="--rm-type ocr"):
        parse_fsdp_test_config("--rm-type", "ocr")


def test_optimizer_cpu_offload_is_supported():
    validate_diffusion_args(_supported(optimizer_cpu_offload=True))


def test_model_kind_sets_its_hooks_and_leaves_rollout_logging_to_miles():
    args = _supported()

    validate_diffusion_args(args)

    assert {name: vars(args)[name] for name in _HOOKS} == _HOOKS
    assert args.custom_rollout_log_function_path is None


def test_the_diffusion_flags_hold_no_copy_of_the_ppo_clip_flags():
    assert not {"diffusion_clip_range", "diffusion_clip_grad"} & DiffusionConfig.model_fields.keys()


def test_a_parsed_diffusion_run_needs_an_explicit_eps_clip():
    from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

    with pytest.raises(ValueError, match="--eps-clip 0.2"):
        parse_fsdp_test_config("--model-kind", "diffusion")


def test_a_foreign_hook_is_rejected():
    with pytest.raises(ValueError, match="hooks"):
        validate_diffusion_args(_supported(rollout_function_path="miles.rollout.other.Fn"))


def _group(worker_type=WorkerType.REGULAR, num_gpus_per_engine=1, engine_kind="diffusion"):
    return SimpleNamespace(worker_type=worker_type, num_gpus_per_engine=num_gpus_per_engine, engine_kind=engine_kind)


def _model(name, groups, update_weights=True):
    return SimpleNamespace(name=name, server_groups=groups, update_weights=update_weights)


def test_a_frozen_sole_model_is_rejected():
    with pytest.raises(ValueError, match="update_weights"):
        validate_diffusion_sglang_config(SimpleNamespace(models=[_model("m", [_group()], update_weights=False)]))


def test_the_engine_layout_is_one_model_of_regular_single_gpu_groups():
    validate_diffusion_sglang_config(SimpleNamespace(models=[_model("m", [_group()])]))
    with pytest.raises(ValueError, match="exactly one model"):
        validate_diffusion_sglang_config(SimpleNamespace(models=[_model(n, [_group()]) for n in "ab"]))
    for group in (_group(WorkerType.PREFILL), _group(num_gpus_per_engine=2)):
        with pytest.raises(ValueError, match="regular single-GPU"):
            validate_diffusion_sglang_config(SimpleNamespace(models=[_model("m", [group])]))


def test_a_model_path_that_differs_from_the_checkpoint_freezes_the_model_and_is_rejected(tmp_path):
    from tests.fast.ray.rollout.conftest import make_sglang_config_yaml

    from miles.backends.sglang_utils.sglang_config import SglangConfig

    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(
        make_sglang_config_yaml(
            model_path="/models/another-copy",
            server_groups=[{"worker_type": "regular", "num_gpus": 2, "num_gpus_per_engine": 1}],
        )
    )
    supported = _supported(sglang_config=str(config_path), rollout_num_gpus=2, actor_num_gpus_per_node=2)
    sglang, _ = SglangConfig.parse_args(Namespace(**(parser_defaults() | vars(supported))))

    assert sglang.models[0].update_weights is False
    with pytest.raises(ValueError, match="update_weights"):
        validate_diffusion_sglang_config(sglang)


def test_a_text_engine_group_is_rejected_in_a_diffusion_run():
    with pytest.raises(ValueError, match="engine_kind text"):
        validate_diffusion_sglang_config(SimpleNamespace(models=[_model("m", [_group(), _group(engine_kind="text")])]))


def _parse_diffusion_run(*argv):
    from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

    return parse_fsdp_test_config(
        *("--model-kind", "diffusion", "--colocate", "--use-miles-router", "--rm-type", "ocr", "--eps-clip", "1e-4"),
        *("--hf-checkpoint", "/models/stable-diffusion-3.5-medium", "--actor-num-gpus-per-node", "2"),
        *("--rollout-batch-size", "8", "--n-samples-per-prompt", "16"),
        *argv,
    )


def test_a_parsed_diffusion_run_derives_its_batch_size_from_steps_per_rollout():
    assert _parse_diffusion_run("--num-steps-per-rollout", "2").global_batch_size == 64


@pytest.mark.parametrize(
    "override,flag",
    [
        (dict(custom_loss_function_path="my.loss"), "--custom-loss-function-path"),
        (dict(entropy_coef=0.01), "--entropy-coef"),
        (dict(use_tis=True), "--use-tis"),
        (dict(rollout_temperature=0.7), "--rollout-temperature"),
        (dict(rollout_max_response_len=64), "--rollout-max-response-len"),
        (dict(custom_generate_function_path="my.generate"), "--custom-generate-function-path"),
        (dict(fp16=True), "--fp16.*--diffusion-forward-dtype"),
        (dict(attn_implementation="sdpa"), "--attn-implementation"),
        (dict(apply_chat_template=True), "--apply-chat-template"),
        (dict(multimodal_keys={"image": "img"}), "--multimodal-keys"),
        (dict(rollout_data_yaml_config="mix.yaml"), "--rollout-data-yaml-config"),
        (dict(multimodal_token_space_processing=True), "--multimodal-token-space-processing"),
        (dict(length_penalty_coeff=0.1), "--length-penalty-coeff"),
    ],
)
def test_a_text_flag_the_diffusion_path_never_reads_is_rejected(override, flag):
    with pytest.raises(ValueError, match=f"never reads: .*{flag}"):
        validate_diffusion_args(_supported(**override))


def test_a_parsed_fp16_run_is_pointed_at_the_diffusion_dtype():
    with pytest.raises(ValueError, match="--fp16.*--diffusion-forward-dtype"):
        _parse_diffusion_run("--global-batch-size", "64", "--fp16")


def test_a_parsed_length_penalty_run_is_rejected():
    with pytest.raises(ValueError, match="--length-penalty-coeff"):
        _parse_diffusion_run("--global-batch-size", "64", "--length-penalty-coeff", "0.1")


def test_the_shared_loop_save_sentinel_stays_accepted():
    config = _parse_diffusion_run(
        "--global-batch-size", "64", "--save", "/ckpts", "--save-trigger-sentinel", "/tmp/checkpoint-now"
    )

    assert config.save_trigger_sentinel == "/tmp/checkpoint-now"


def test_the_parsed_recipe_flags_pass_the_unread_flag_check():
    assert _parse_diffusion_run("--global-batch-size", "64").global_batch_size == 64


def test_a_step_count_that_does_not_divide_the_images_is_rejected():
    with pytest.raises(ValueError, match="num-steps-per-rollout must divide"):
        _parse_diffusion_run("--num-steps-per-rollout", "48")


@pytest.mark.parametrize("global_batch_size,message", [(None, "num-steps-per-rollout"), (48, "must divide")])
def test_the_batch_size_must_be_set_and_divide_the_rollout(global_batch_size, message):
    with pytest.raises(ValueError, match=message):
        validate_diffusion_batch_size(_supported(global_batch_size=global_batch_size))


def _nft(**overrides):
    recipe = dict(
        loss_type="nft",
        eps_clip=0.2,
        diffusion_sde_type="ode",
        diffusion_noise_level=0.0,
        diffusion_guidance_scale=1.0,
        diffusion_step_strategy_path="miles.rollout.diffusion_rollout.step_strategy.ode_and_return_last",
        diffusion_nft_num_dropped_timesteps=1,
    )
    return _supported(**(recipe | overrides))


def test_the_nft_recipe_layout_passes():
    validate_diffusion_args(_nft())


@pytest.mark.parametrize(
    "override,message",
    [
        (dict(diffusion_sde_type="sde"), "SDE rollouts"),
        (dict(diffusion_noise_level=0.7), "noise-level"),
        (dict(diffusion_guidance_scale=4.5), "classifier-free guidance"),
        (dict(diffusion_kl_beta=0.04), "KL"),
        (dict(diffusion_nft_beta=0.0), "nft-beta"),
        (dict(diffusion_nft_num_dropped_timesteps=10), "dropped-timesteps"),
        (dict(diffusion_ema_decay_max=1.5), "EMA decay"),
        (dict(diffusion_ema_decay_ramp=-0.1), "EMA decay"),
    ],
)
def test_each_unsupported_nft_option_is_rejected_at_startup(override, message):
    with pytest.raises(ValueError, match=message):
        validate_diffusion_args(_nft(**override))


@pytest.mark.parametrize(
    "override,flag",
    [
        (dict(eps_clip=1e-4), "--eps-clip"),
        (dict(diffusion_num_sde_steps=4), "--diffusion-num-sde-steps"),
        (dict(diffusion_sde_window_range="0,5"), "--diffusion-sde-window-range"),
        (dict(diffusion_negative_prompt=" "), "--diffusion-negative-prompt"),
    ],
)
def test_a_flow_grpo_flag_nft_never_reads_is_rejected(override, flag):
    with pytest.raises(ValueError, match=f"--loss-type nft never reads: .*{flag}"):
        validate_diffusion_args(_nft(**override))


@pytest.mark.parametrize(
    "override,flag",
    [
        (dict(diffusion_nft_beta=0.5), "--diffusion-nft-beta"),
        (dict(diffusion_ema_decay_max=0.9), "--diffusion-ema-decay-max"),
        (dict(diffusion_nft_adaptive_weight=False), "--diffusion-nft-adaptive-weight"),
    ],
)
def test_an_nft_flag_flow_grpo_never_reads_is_rejected(override, flag):
    with pytest.raises(ValueError, match=f"--loss-type flow_grpo never reads: .*{flag}"):
        validate_diffusion_args(_supported(**override))


class _TwoExperts(SD3TrainPipelineConfig):
    model_family = "two_experts"
    name_patterns = ("two-experts",)
    components = ("transformer", "transformer_2")
    guidance_scale_2_component = "transformer_2"


@pytest.fixture
def two_experts(monkeypatch):
    monkeypatch.setitem(pipeline_config._REGISTRY, "two_experts", _TwoExperts)
    return dict(diffusion_model_family="two_experts", diffusion_guidance_scale_2=3.0)


def test_an_ema_algorithm_is_rejected_for_a_family_of_several_components(two_experts):
    validate_diffusion_args(_supported(**two_experts))
    with pytest.raises(ValueError, match="EMA of one adapter"):
        validate_diffusion_args(_nft(**two_experts))


def test_the_second_guidance_scale_is_required_exactly_by_a_family_that_guides_with_it(two_experts):
    with pytest.raises(ValueError, match="so pass it"):
        validate_diffusion_args(_supported(**(two_experts | dict(diffusion_guidance_scale_2=None))))
    with pytest.raises(ValueError, match="no second guidance scale"):
        validate_diffusion_args(_supported(diffusion_guidance_scale_2=3.0))


_PICKSCORE = dict(
    rm_type="pickscore",
    diffusion_pickscore_model_path="/models/PickScore_v1",
    diffusion_pickscore_processor_path="/models/CLIP-ViT-H-14",
)


def test_the_pickscore_reward_needs_its_model_and_processor():
    validate_diffusion_args(_supported(**_PICKSCORE))
    with pytest.raises(ValueError, match="pickscore-model-path"):
        validate_diffusion_args(_supported(**(_PICKSCORE | dict(diffusion_pickscore_model_path=None))))


def test_a_pickscore_flag_without_the_pickscore_reward_is_rejected():
    with pytest.raises(ValueError, match="--diffusion-pickscore-num-frames"):
        validate_diffusion_args(_supported(diffusion_pickscore_num_frames=3))


def test_a_text_run_rejects_the_pickscore_reward():
    from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

    with pytest.raises(ValueError, match="--rm-type pickscore"):
        parse_fsdp_test_config("--rm-type", "pickscore")


def _wan(**overrides):
    recipe = dict(
        hf_checkpoint="/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        diffusion_guidance_scale_2=3.0,
        diffusion_output_num_frames=5,
        diffusion_flow_shift=3.0,
        diffusion_step_strategy_path="miles.rollout.diffusion_rollout.step_strategy.epoch_global_random_choice",
        diffusion_sde_candidate_steps="1,2,3",
        diffusion_num_sde_steps=1,
    )
    return _supported(**(_PICKSCORE | recipe | overrides))


def test_the_wan22_recipe_passes_and_trains_both_experts():
    args = _wan()

    validate_diffusion_args(args)

    assert args.diffusion_model_family == "wan2_2" and "attn2.to_q" in args.diffusion_lora_target_modules
    with pytest.raises(ValueError, match="guides transformer_2"):
        validate_diffusion_args(_wan(diffusion_guidance_scale_2=None))


@pytest.mark.parametrize(
    "override,flag",
    [
        (dict(diffusion_sde_candidate_steps="1,2,3"), "--diffusion-sde-candidate-steps"),
        (
            dict(
                diffusion_step_strategy_path="miles.rollout.diffusion_rollout.step_strategy.epoch_global_random_choice",
                diffusion_sde_candidate_steps="1,2,3",
                diffusion_sde_window_range="7,9",
            ),
            "--diffusion-sde-window-range",
        ),
    ],
)
def test_a_step_flag_the_built_in_strategy_never_reads_is_rejected(override, flag):
    with pytest.raises(ValueError, match=f"never reads: {flag}"):
        validate_diffusion_args(_supported(**override))


def test_a_custom_step_strategy_may_read_any_step_flag():
    validate_diffusion_args(
        _supported(diffusion_step_strategy_path="my_pkg.strategy", diffusion_sde_candidate_steps="1,2,3")
    )


def test_nft_never_reads_the_candidate_steps():
    with pytest.raises(ValueError, match="--diffusion-sde-candidate-steps"):
        validate_diffusion_args(_nft(diffusion_sde_candidate_steps="1,2,3"))


def test_the_ocr_reward_is_rejected_on_video():
    with pytest.raises(ValueError, match="only the first frame"):
        validate_diffusion_args(_supported(diffusion_output_num_frames=5))


def test_flow_grpo_rejects_ode_rollouts():
    with pytest.raises(ValueError, match="ODE rollouts under Flow-GRPO"):
        validate_diffusion_args(_supported(diffusion_sde_type="ode"))


@pytest.mark.parametrize("loss_type", ["flow_grpo", "nft"])
def test_a_text_run_rejects_the_diffusion_losses(loss_type):
    from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

    with pytest.raises(ValueError, match=f"--loss-type {loss_type} is a diffusion algorithm"):
        parse_fsdp_test_config("--loss-type", loss_type)


@pytest.mark.parametrize("argv", [(), ("--loss-type", "policy_loss"), ("--loss-type", "flow_grpo")])
def test_a_diffusion_run_trains_flow_grpo_by_default_or_by_name(argv):
    assert _parse_diffusion_run("--global-batch-size", "64", *argv).loss_type == "flow_grpo"


def test_a_parsed_nft_run_passes_and_carries_the_seed_the_converter_reads():
    config = _parse_diffusion_run(
        *("--loss-type", "nft", "--eps-clip", "0.2", "--num-steps-per-rollout", "1"),
        *("--diffusion-sde-type", "ode", "--diffusion-noise-level", "0", "--diffusion-guidance-scale", "1"),
    )

    assert config.loss_type == "nft" and config.seed == 1234
