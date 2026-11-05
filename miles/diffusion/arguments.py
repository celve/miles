"""Startup validation and derived defaults of `--model-kind diffusion` runs.

The diffusion path supports one layout: a single SD3-family model, Flow-GRPO or DiffusionNFT (`--loss-type`) with
the OCR reward, a LoRA adapter on FSDP, colocated single-GPU engines and the synchronous loop. Everything else is
rejected here, before any worker starts, rather than failing inside one.
"""

from miles.backends.fsdp_utils.arguments import FSDPArgs
from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.diffusion.algorithms import ALGORITHMS
from miles.diffusion.rollout import step_strategy
from miles.diffusion.train.pipeline_config import get_train_pipeline_config, resolve_model_family
from miles.utils.args.configs.algo import AlgoConfig
from miles.utils.args.configs.backend_fields import TrainerBackendTraitConfig
from miles.utils.args.configs.data import DataConfig
from miles.utils.args.configs.diffusion import DiffusionConfig
from miles.utils.args.configs.rollout import RolloutRelatedConfig, RolloutRelatedRolloutOnlyConfig

_HOOKS = {
    "rollout_function_path": "miles.diffusion.rollout.rollout_fn.DiffusionRolloutFn",
    "data_source_path": "miles.diffusion.rollout.data_source.DiffusionPromptDataSource",
    "custom_convert_samples_to_train_data_path": "miles.diffusion.conversion.convert_samples_to_train_data",
}

# (what is unsupported, whether this run asks for it)
_UNSUPPORTED = [
    ("a training backend other than --train-backend fsdp", lambda a: a.train_backend != "fsdp"),
    ("separate rollout GPUs: weights sync over CUDA IPC, so --colocate is required", lambda a: not a.colocate),
    (
        "engines spanning several GPUs (--rollout-num-gpus-per-engine > 1)",
        lambda a: a.rollout_num_gpus_per_engine != 1,
    ),
    ("a router other than --use-miles-router", lambda a: not a.use_miles_router),
    ("an advantage estimator other than grpo, including a critic", lambda a: a.advantage_estimator != "grpo"),
    (
        "a separate reference model or token KL (--ref-load, --use-kl-loss, --kl-coef)",
        lambda a: a.ref_load is not None or a.use_kl_loss or a.kl_coef != 0,
    ),
    ("on-policy distillation (--use-opd)", lambda a: a.use_opd),
    (
        "evaluation (--eval-interval, --eval-prompt-data)",
        lambda a: a.eval_interval is not None or bool(a.eval_prompt_data),
    ),
    ("fully async or partial rollout", lambda a: a.fully_async or a.partial_rollout),
    ("dynamic sampling filters", lambda a: a.dynamic_sampling_filter_path is not None),
    (
        "a reward other than --rm-type ocr, --rm-type pickscore or a --custom-rm-path",
        lambda a: a.custom_rm_path is None and a.rm_type not in ("ocr", "pickscore"),
    ),
    (
        "--rm-type pickscore without --diffusion-pickscore-model-path and --diffusion-pickscore-processor-path",
        lambda a: a.rm_type == "pickscore"
        and not (a.diffusion_pickscore_model_path and a.diffusion_pickscore_processor_path),
    ),
    (
        "a PickScore worker on a fraction of a GPU or several (--diffusion-pickscore-num-gpus-per-worker other than 0 "
        "or 1); each worker scores on one whole GPU or on the CPU",
        lambda a: a.rm_type == "pickscore" and a.diffusion_pickscore_num_gpus_per_worker not in (0, 1),
    ),
    (
        "the OCR reward on video (--diffusion-output-num-frames > 1); it reads only the first frame",
        lambda a: a.rm_type == "ocr" and a.diffusion_output_num_frames != 1,
    ),
    (
        "the output store or the Mooncake object store",
        lambda a: a.use_output_store or str(a.object_store_backend).lower().endswith("mooncake"),
    ),
    ("fault tolerance (--use-fault-tolerance)", lambda a: a.use_fault_tolerance),
    ("--delay-split-train-data-by-dp", lambda a: a.delay_split_train_data_by_dp),
    ("independent data parallelism (--indep-dp)", lambda a: a.indep_dp),
    (
        "the text LoRA stack (--lora-rank, --multi-lora); use --diffusion-lora-rank",
        lambda a: a.lora_rank > 0 or a.multi_lora_n_adapters > 0,
    ),
    ("routing replay or true on-policy mode", lambda a: a.use_rollout_routing_replay or a.true_on_policy_mode),
    ("--debug-train-only or --debug-rollout-only", lambda a: a.debug_train_only or a.debug_rollout_only),
    ("--ci-test: its version readback and sample-ownership audit read token fields", lambda a: a.ci_test),
    (
        f"a --loss-type other than the diffusion algorithms {sorted(ALGORITHMS)}",
        lambda a: a.loss_type not in ALGORITHMS,
    ),
    (
        "token-based batching (--use-dynamic-batch-size, --use-dynamic-global-batch-size); images are tiled by "
        "--diffusion-micro-batch-size-sample/-tstep into --global-batch-size steps",
        lambda a: a.use_dynamic_batch_size or a.use_dynamic_global_batch_size,
    ),
    (
        "--micro-batch-size: use --diffusion-micro-batch-size-sample and --diffusion-micro-batch-size-tstep",
        lambda a: a.micro_batch_size != TrainerBackendTraitConfig.model_fields["micro_batch_size"].default,
    ),
    (
        "asymmetric or dual-clip PPO (--eps-clip-high, --eps-clip-c); --eps-clip sets the symmetric ratio clip",
        lambda a: a.eps_clip_high is not None or a.eps_clip_c is not None,
    ),
    (
        "debug dumps (--save-debug-rollout-data, --save-debug-train-data, --dump-details)",
        lambda a: a.save_debug_rollout_data is not None
        or a.save_debug_train_data is not None
        or a.dump_details is not None,
    ),
]


# Text-training and text-sampling flags the diffusion path never reads (the ones above have their own messages);
# each must keep its default, or the run would silently differ from what was asked.
_UNREAD = {
    AlgoConfig: sorted(
        AlgoConfig.model_fields.keys()
        - {"eps_clip", "advantage_estimator", "ref_load", "kl_coef", "use_kl_loss", "eps_clip_high", "eps_clip_c"}
        # Selects the diffusion algorithm, which validates the flags it reads and those it does not.
        - {"loss_type"}
        - {"use_rollout_routing_replay"}
        # Derived from --advantage-estimator after parsing, which the grpo check above already pins.
        - {"use_critic"}
        # Read by the shared train loop (train.py), which saves through the diffusion actor's checkpointing.
        - {"save_trigger_sentinel"}
    ),
    RolloutRelatedConfig: [
        "rollout_temperature",
        "rollout_top_p",
        "rollout_top_k",
        "rollout_max_response_len",
        "rollout_top_logprobs_num",
        "custom_generate_function_path",
        "custom_agent_function_path",
        "rollout_data_postprocess_path",
    ],
    DataConfig: [
        # --apply-chat-template-kwargs only acts with --apply-chat-template, and its parser default is not the model's.
        *("apply_chat_template", "chat_template_path"),
        *("multimodal_keys", "tool_key"),
    ],
    RolloutRelatedRolloutOnlyConfig: [
        "rollout_max_context_len",
        "rollout_max_prompt_len",
        "rollout_stop",
        "rollout_stop_token_ids",
        "buffer_filter_path",
    ],
}
_UNREAD_DEFAULTS = {
    **{
        name: config.model_fields[name].get_default(call_default_factory=True)
        for config, names in _UNREAD.items()
        for name in names
    },
    # The text FSDP trainer's precision, attention and kernel options; the DiT takes its own.
    **{
        name: vars(FSDPArgs())[name]
        for name in (
            *("fp16", "bf16", "keep_fp32_master", "attn_implementation"),
            *("kernel_backend", "kernel_mapping_path", "kernel_strict"),
        )
    },
}
_PRECISION_FLAGS = {"--fp16", "--bf16", "--keep-fp32-master"}
_PICKSCORE_FLAGS = sorted(name for name in DiffusionConfig.model_fields if name.startswith("diffusion_pickscore_"))


def _default(name: str):
    (config,) = (config for config in (AlgoConfig, DiffusionConfig) if name in config.model_fields)
    return config.model_fields[name].get_default(call_default_factory=True)


def validate_diffusion_args(args) -> None:
    # Miles' default text loss names no diffusion algorithm; Flow-GRPO is the diffusion default.
    if args.loss_type == AlgoConfig.model_fields["loss_type"].default:
        args.loss_type = "flow_grpo"
    algorithm = ALGORITHMS.get(args.loss_type)
    checks = _UNSUPPORTED + (algorithm.unsupported if algorithm else [])
    unsupported = [what for what, requested in checks if requested(args)]
    if unsupported:
        raise ValueError("--model-kind diffusion does not support: " + "; ".join(unsupported))
    unread_defaults = _UNREAD_DEFAULTS | {name: _default(name) for name in algorithm.unread_flags}
    unread = [
        f"--{name.replace('_', '-')}" for name, default in unread_defaults.items() if vars(args)[name] != default
    ]
    if unread:
        hint = "; set precision with --diffusion-forward-dtype" if _PRECISION_FLAGS & set(unread) else ""
        raise ValueError(
            f"--model-kind diffusion with --loss-type {args.loss_type} never reads: " + ", ".join(unread) + hint
        )
    _derive_hooks(args)
    _check_step_strategy_flags(args)
    if args.rm_type != "pickscore":
        if unread_pickscore := [flag for flag in _PICKSCORE_FLAGS if vars(args)[flag] != _default(flag)]:
            raise ValueError(
                "without --rm-type pickscore no PickScore pool is built, so these are never read: "
                + ", ".join(f"--{flag.replace('_', '-')}" for flag in unread_pickscore)
            )
    if args.diffusion_lora_rank <= 0 or args.diffusion_num_sde_steps <= 0:
        raise ValueError("--diffusion-lora-rank and --diffusion-num-sde-steps must be positive")

    if args.diffusion_model_family is None:
        args.diffusion_model_family = resolve_model_family(args.hf_checkpoint)
    config = get_train_pipeline_config(args.diffusion_model_family)
    if args.diffusion_lora_target_modules is None:
        args.diffusion_lora_target_modules = config.lora_target_modules
    if (args.diffusion_guidance_scale_2 is None) != (config.guidance_scale_2_component is None):
        raise ValueError(
            f"family {config.model_family} guides {config.guidance_scale_2_component} with "
            "--diffusion-guidance-scale-2, which the engine otherwise replaces by its own default, so pass it"
            if config.guidance_scale_2_component
            else f"family {config.model_family} has no second guidance scale; drop --diffusion-guidance-scale-2"
        )
    if len(config.components) > 1 and args.diffusion_step_strategy_path == f"{step_strategy.__name__}.sde_window":
        raise ValueError(
            f"family {config.model_family} routes timesteps to the components {list(config.components)}, which every "
            "DP rank must forward in lockstep; sde_window draws each request's steps apart, so use "
            "epoch_global_random_choice"
        )
    if algorithm.uses_ema and len(config.components) > 1:
        raise ValueError(
            f"--loss-type {args.loss_type} keeps an EMA of one adapter, but family {config.model_family} trains "
            f"the components {list(config.components)}"
        )
    if (
        args.diffusion_guidance_scale_2 is not None
        and args.diffusion_guidance_scale <= 1.0 < args.diffusion_guidance_scale_2
    ):
        raise ValueError(
            "the engine runs classifier-free guidance only when --diffusion-guidance-scale is above 1, so "
            f"--diffusion-guidance-scale-2 {args.diffusion_guidance_scale_2} would never be read"
        )


def validate_diffusion_batch_size(args) -> None:
    """Run once `--num-steps-per-rollout` has been turned into `--global-batch-size`."""
    num_images = args.rollout_batch_size * args.n_samples_per_prompt
    if args.global_batch_size is None:
        raise ValueError("--model-kind diffusion needs --global-batch-size or --num-steps-per-rollout")
    # The derivation floor-divides, so a non-dividing step count would silently train a different number of steps.
    if args.num_steps_per_rollout is not None and num_images % args.num_steps_per_rollout:
        raise ValueError("--num-steps-per-rollout must divide --rollout-batch-size x --n-samples-per-prompt")
    if num_images % args.global_batch_size:
        raise ValueError("--global-batch-size must divide --rollout-batch-size x --n-samples-per-prompt")


def _check_step_strategy_flags(args) -> None:
    """A built-in step strategy reads only some of the step flags; a custom one may read any."""
    module, _, name = args.diffusion_step_strategy_path.rpartition(".")
    if module != step_strategy.__name__ or name not in step_strategy.FLAGS_READ:
        return
    flags = [flag for flag in step_strategy.STEP_FLAGS if flag not in step_strategy.FLAGS_READ[name]]
    if unread := [f"--{flag.replace('_', '-')}" for flag in flags if vars(args)[flag] != _default(flag)]:
        raise ValueError(f"the {name} step strategy never reads: " + ", ".join(unread))


def _derive_hooks(args) -> None:
    """`--model-kind diffusion` sets its four hooks; an explicit different one is an error."""
    unset = {None, DataConfig.model_fields["data_source_path"].default.path}
    wrong = {}
    for name, hook in _HOOKS.items():
        if vars(args)[name] in unset:
            vars(args)[name] = hook
        elif vars(args)[name] != hook:
            wrong[name] = vars(args)[name]
    if wrong:
        raise ValueError(f"--model-kind diffusion sets its own hooks {_HOOKS}; got {wrong}")


def validate_diffusion_sglang_config(sglang) -> None:
    """One model served by regular single-GPU engine groups: the trainer syncs one DiT into one engine kind."""
    if len(sglang.models) != 1:
        raise ValueError(f"--model-kind diffusion serves exactly one model, got {[m.name for m in sglang.models]}")
    if not sglang.models[0].update_weights:
        # A frozen model's engines get no LoRA sync, yet every weight version would still be published.
        raise ValueError("--model-kind diffusion trains its one model, so that model needs update_weights: true")
    for group in sglang.models[0].server_groups:
        if group.engine_kind != "diffusion":
            raise ValueError("--model-kind diffusion serves its model from diffusion engines, not engine_kind text")
        if group.worker_type != WorkerType.REGULAR or group.num_gpus_per_engine != 1:
            raise ValueError(
                f"--model-kind diffusion serves regular single-GPU engine groups, got {group.worker_type.value} "
                f"with {group.num_gpus_per_engine} GPUs per engine"
            )
