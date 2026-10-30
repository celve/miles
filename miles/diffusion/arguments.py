"""Startup validation and derived defaults of `--model-kind diffusion` runs."""

from miles.diffusion.train.pipeline_config import get_train_pipeline_config, resolve_model_family


def validate_diffusion_args(args) -> None:
    assert args.train_backend == "fsdp", "--model-kind diffusion trains with --train-backend fsdp only"
    assert args.colocate, "--model-kind diffusion syncs weights over CUDA IPC, so it needs --colocate"
    assert args.diffusion_lora_rank > 0, "--model-kind diffusion trains a LoRA adapter"
    assert args.use_miles_router, "--model-kind diffusion routes /rollout/generate through --use-miles-router"
    assert not args.fully_async, "--model-kind diffusion runs the synchronous loop only"
    assert args.rollout_num_gpus_per_engine == 1, "--model-kind diffusion serves one GPU per engine"
    assert (args.rollout_batch_size * args.n_samples_per_prompt) % args.global_batch_size == 0
    assert args.diffusion_sde_type == "sde" and args.diffusion_num_sde_steps > 0

    if args.diffusion_model_family is None:
        args.diffusion_model_family = resolve_model_family(args.hf_checkpoint)
    if args.diffusion_lora_target_modules is None:
        args.diffusion_lora_target_modules = get_train_pipeline_config(args.diffusion_model_family).lora_target_modules
