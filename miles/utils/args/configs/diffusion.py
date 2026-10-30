from miles.utils.args.schema import A, Arg, BaseConfig


class DiffusionConfig(BaseConfig):
    """Flags of `--model-kind diffusion` runs: Flow-GRPO on a diffusers DiT served by sglang.multimodal_gen."""

    model_kind: A[
        str,
        Arg(
            choices=["text", "diffusion"],
            help="What the run trains. `diffusion` selects the DiT trainer, the sglang.multimodal_gen engine "
            "and the denoising-trajectory data path; `--hf-checkpoint` is then a diffusers pipeline directory.",
        ),
    ] = "text"

    # rollout request
    diffusion_num_steps: A[int, Arg(help="Denoising steps per rollout image.")] = 10
    diffusion_guidance_scale: A[float, Arg(help="Classifier-free guidance scale, used in rollout and training.")] = 4.5
    diffusion_negative_prompt: A[str | None, Arg(help="Negative prompt for classifier-free guidance.")] = None
    diffusion_height: A[int, Arg(help="Rollout image height.")] = 512
    diffusion_width: A[int, Arg(help="Rollout image width.")] = 512
    diffusion_sde_type: A[str, Arg(choices=["sde"], help="Stochastic sampler the engine runs on the SDE steps.")] = (
        "sde"
    )
    diffusion_noise_level: A[float, Arg(help="SDE noise level, used in rollout and in the trainer's log-prob.")] = 0.7
    diffusion_quality: A[
        str, Arg(help="sglang.multimodal_gen `quality` sampling knob; `exact` keeps the engine batch-invariant.")
    ] = "exact"
    diffusion_step_strategy_path: A[
        str,
        Arg(help="Function `(args, sample, num_steps, seed) -> (sde_steps, returned_latents)` choosing SDE steps."),
    ] = "miles.diffusion.rollout.step_strategy.sde_window"
    diffusion_num_sde_steps: A[int, Arg(help="SDE steps per image; each becomes one train pair.")] = 10
    diffusion_sde_window_range: A[
        str | None, Arg(help="'lo,hi' range the SDE window is drawn from; the whole schedule when unset.")
    ] = None
    diffusion_microgroup_size: A[
        int, Arg(help="Images per /rollout/generate request; a prompt's group is split into such requests.")
    ] = 8
    diffusion_dit_precision: A[str | None, Arg(help="Rollout DiT precision passed to the engine (fp16/bf16).")] = None
    diffusion_ocr_num_workers: A[int, Arg(help="CPU PaddleOCR reward actors.")] = 4

    # training
    diffusion_model_family: A[
        str | None, Arg(help="Train pipeline family (e.g. sd3); resolved from --hf-checkpoint when unset.")
    ] = None
    diffusion_forward_dtype: A[
        str, Arg(choices=["fp16", "bf16", "fp32"], help="DiT forward dtype; fp32 master weights are kept regardless.")
    ] = "bf16"
    diffusion_clip_range: A[float, Arg(help="PPO ratio clip range.")] = 1e-4
    diffusion_kl_beta: A[
        float, Arg(help="KL coefficient against the LoRA base model (adapter disabled); 0 disables it.")
    ] = 0.0
    diffusion_adv_clip_max: A[float, Arg(help="Advantages are clamped to +-this value.")] = 5.0
    diffusion_clip_grad: A[float, Arg(help="Gradient-norm clip.")] = 1.0
    diffusion_globalize_reward_std: A[
        bool, Arg(help="GRPO: divide by the rollout-wide reward std instead of each prompt group's.")
    ] = False
    diffusion_micro_batch_size_sample: A[int, Arg(help="Images per micro-batch tile.")] = 16
    diffusion_micro_batch_size_tstep: A[int, Arg(help="SDE steps per micro-batch tile.")] = 5
    diffusion_lora_rank: A[int, Arg(help="Rank of the trained LoRA adapter.")] = 32
    diffusion_lora_alpha: A[int, Arg(help="LoRA alpha; the adapter is scaled by alpha / rank.")] = 64
    diffusion_lora_init: A[
        str, Arg(choices=["gaussian", "kaiming-uniform"], help="PEFT LoRA initialization of the trained adapter.")
    ] = "gaussian"
    diffusion_lora_target_modules: A[
        list[str] | None, Arg(type_parser=str, nargs="+", help="LoRA target modules; the family default when unset.")
    ] = None
