"""Startup validation and derived defaults of `--model-kind diffusion` runs.

The diffusion path supports one layout: a single SD3-family model, Flow-GRPO with the OCR reward, a LoRA adapter on
FSDP, colocated single-GPU engines and the synchronous loop. Everything else is rejected here, before any worker
starts, rather than failing inside one.
"""

from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.diffusion.train.pipeline_config import get_train_pipeline_config, resolve_model_family

_HOOKS = {
    "rollout_function_path": "miles.diffusion.rollout.rollout_fn.DiffusionRolloutFn",
    "data_source_path": "miles.diffusion.rollout.data_source.DiffusionPromptDataSource",
    "custom_convert_samples_to_train_data_path": "miles.diffusion.conversion.convert_samples_to_train_data",
    "custom_rollout_log_function_path": "miles.diffusion.conversion.log_rollout_data",
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
        "text reward models (--rm-type, --custom-rm-path); the OCR reward is built in",
        lambda a: a.rm_type is not None or a.custom_rm_path is not None,
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
]


def validate_diffusion_args(args) -> None:
    unsupported = [what for what, requested in _UNSUPPORTED if requested(args)]
    if unsupported:
        raise ValueError("--model-kind diffusion does not support: " + "; ".join(unsupported))
    hooks = {name: vars(args)[name] for name in _HOOKS}
    wrong_hooks = {name: value for name, value in hooks.items() if value != _HOOKS[name]}
    if wrong_hooks:
        raise ValueError(f"--model-kind diffusion needs its own hooks {_HOOKS}; got {wrong_hooks}")
    if (args.rollout_batch_size * args.n_samples_per_prompt) % args.global_batch_size:
        raise ValueError("--global-batch-size must divide --rollout-batch-size x --n-samples-per-prompt")
    if args.diffusion_lora_rank <= 0 or args.diffusion_num_sde_steps <= 0:
        raise ValueError("--diffusion-lora-rank and --diffusion-num-sde-steps must be positive")

    if args.diffusion_model_family is None:
        args.diffusion_model_family = resolve_model_family(args.hf_checkpoint)
    if args.diffusion_lora_target_modules is None:
        args.diffusion_lora_target_modules = get_train_pipeline_config(args.diffusion_model_family).lora_target_modules


def validate_diffusion_sglang_config(sglang) -> None:
    """One model served by regular single-GPU engine groups: the trainer syncs one DiT into one engine kind."""
    if len(sglang.models) != 1:
        raise ValueError(f"--model-kind diffusion serves exactly one model, got {[m.name for m in sglang.models]}")
    if not sglang.models[0].update_weights:
        # A frozen model's engines get no LoRA sync, yet every weight version would still be published.
        raise ValueError("--model-kind diffusion trains its one model, so that model needs update_weights: true")
    for group in sglang.models[0].server_groups:
        if group.worker_type != WorkerType.REGULAR or group.num_gpus_per_engine != 1:
            raise ValueError(
                f"--model-kind diffusion serves regular single-GPU engine groups, got {group.worker_type.value} "
                f"with {group.num_gpus_per_engine} GPUs per engine"
            )
