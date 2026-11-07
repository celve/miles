"""Launch command and environment of a `--model-kind diffusion` inference engine (sglang.multimodal_gen)."""

import json
import shlex


def compute_diffusion_engine_launch_cmd(
    args,
    *,
    interpreter_prefix: list[str],
    base_gpu_id: int,
    num_gpus_per_engine: int,
    sglang_overrides: dict,
    update_weights: bool,
    nccl_port: int,
    host: str,
    port: int,
) -> str:
    server_args = compute_diffusion_server_args(
        args,
        num_gpus_per_engine=num_gpus_per_engine,
        sglang_overrides=sglang_overrides,
        update_weights=update_weights,
        nccl_port=nccl_port,
        host=host.strip("[]"),
        port=port,
    )
    return shlex.join(
        [
            *interpreter_prefix,
            "-m",
            "miles.diffusion.engine_main",
            "--base-gpu-id",
            str(base_gpu_id),
            "--server-args",
            json.dumps(server_args),
        ]
    )


def compute_diffusion_server_args(
    args,
    *,
    num_gpus_per_engine: int,
    sglang_overrides: dict,
    update_weights: bool,
    nccl_port: int,
    host: str,
    port: int,
) -> dict:
    """Keyword arguments of sglang.multimodal_gen's `ServerArgs.from_kwargs`; the server group's overrides,
    which carry the model's own `model_path` and any `tp_size`, `sp_degree`, `ulysses_degree`, `ring_degree` or
    `enable_cfg_parallel`, win over the run-wide defaults, as they do for text engines."""
    server_args = {
        "model_path": args.hf_checkpoint,
        "trust_remote_code": True,
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        # Distinct per engine so concurrent port probes of co-located engines do not race on the default.
        "master_port": nccl_port + 10000,
        # sglang splits these GPUs into tensor, sequence and CFG parallelism itself unless the group's overrides do.
        "num_gpus": num_gpus_per_engine,
        "warmup_mode": "off",
        # Miles places rollout memory itself; sglang's automatic policy would offload by startup free memory.
        "performance_mode": "manual",
    }
    if update_weights:
        # The engine holds no adapter: each weight sync merges the trainer's LoRA delta into its base weights.
        server_args["lora_merge_mode"] = "merge"
        server_args["lora_target_modules"] = args.diffusion_lora_target_modules
    if args.diffusion_dit_precision is not None:
        server_args["dit_precision"] = args.diffusion_dit_precision
    if args.diffusion_flow_shift is not None:
        server_args["flow_shift"] = args.diffusion_flow_shift
    return {**server_args, **sglang_overrides}


def compute_diffusion_engine_env_vars(args, *, update_weights: bool) -> dict[str, str]:
    if not update_weights:
        return {}
    # The engine merges LoRA deltas in this dtype; matching the trainer's forward dtype keeps log-probs aligned.
    return {"SGLANG_DIFFUSION_LORA_MERGE_FP32": "1" if args.diffusion_forward_dtype == "fp32" else "0"}
