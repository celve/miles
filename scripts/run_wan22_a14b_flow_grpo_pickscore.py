"""Flow-GRPO on Wan2.2-T2V-A14B with the PickScore reward: LoRA r64 on both experts, 5-frame 480x480 videos, 4 GPUs
colocated (FSDP DP=4 and four sglang.multimodal_gen engines time-sharing them) plus a fifth GPU for PickScore.

Ported from radixark/miles_diffusion at d608b4ec (scripts/run_diffusion_grpo_wan22_pickscore_5gpu.py), which the fork
has not run to a full curve. Each rollout draws one SDE step from steps 1-3 (epoch_global_random_choice), so it trains
one expert: steps 1-2 sit above the t = 875 boundary and train `transformer`, step 3 trains `transformer_2`. Evaluation
is dropped (diffusion runs have none yet). Gradient checkpointing stays off: under FSDP2 mixed precision it trips on
Wan's fp32 RoPE buffers.

Args:
    --num-rollout: rollouts to train; each is --rollout-batch-size prompts x --n-samples-per-prompt videos.
    --model-path: a local Wan2.2-T2V-A14B-Diffusers directory; downloaded to --model-dir when empty.
    --save-interval: rollouts between checkpoints; 0 saves none.
    --pickscore-gpus: GPUs of the PickScore actor; 0 scores on CPU and leaves the node at the 4 train GPUs.
    --extra-args: appended to the train argv.

    python scripts/run_wan22_a14b_flow_grpo_pickscore.py --num-rollout 300
"""

import math
from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils

MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
PICKSCORE_MODEL = "yuvalkirstain/PickScore_v1"
PICKSCORE_PROCESSOR = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
DATASET = "rockdu/miles-diffusion-datasets"
DATASET_SUBSET = "flowgrpo_pickscore"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    num_rollout: int = 300
    rollout_batch_size: int = 48
    n_samples_per_prompt: int = 16
    pickscore_gpus: float = 1.0
    model_path: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    save_interval: int = 0
    extra_args: str = ""

    @property
    def resolved_model_path(self) -> str:
        return self.model_path or f"{self.model_dir}/{MODEL.split('/')[-1]}"

    @property
    def dataset_dir(self) -> str:
        return f"{self.data_dir}/{DATASET.split('/')[-1]}"


def _local(model_dir: str, repo: str) -> str:
    return f"{model_dir}/{repo.split('/')[-1]}"


def prepare(args: ScriptArgs) -> None:
    U = args.create_backend()
    U.exec_command_cpu(
        f"hf download --repo-type dataset {DATASET} --include '{DATASET_SUBSET}/**' --local-dir {args.dataset_dir}"
    )
    U.exec_command_cpu(f"hf download {PICKSCORE_MODEL} --local-dir {_local(args.model_dir, PICKSCORE_MODEL)}")
    U.exec_command_cpu(
        f"hf download {PICKSCORE_PROCESSOR} --exclude '*.bin' --exclude 'open_clip*' "
        f"--local-dir {_local(args.model_dir, PICKSCORE_PROCESSOR)}"
    )
    if not args.model_path:
        U.exec_command_cpu(f"hf download {MODEL} --local-dir {args.resolved_model_path}")


def execute(args: ScriptArgs) -> None:
    U = args.create_backend()
    save_dir = f"{args.output_dir}/{args.run_id}/checkpoints"

    ckpt_args = f"--hf-checkpoint {args.resolved_model_path} --model-kind diffusion "
    if args.save_interval:
        ckpt_args += f"--save {save_dir} --save-interval {args.save_interval} "

    rollout_args = (
        f"--prompt-data {args.dataset_dir}/{DATASET_SUBSET}/train.jsonl "
        "--input-key input "
        # File order, as miles_diffusion reads its prompts; Miles' own default shuffles them.
        "--no-rollout-shuffle "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--num-rollout {args.num_rollout} "
        "--num-steps-per-rollout 2 "
        "--diffusion-microgroup-size 8 "
        "--diffusion-num-steps 10 "
        "--diffusion-output-num-frames 5 "
        "--diffusion-guidance-scale 4.0 "
        "--diffusion-guidance-scale-2 3.0 "
        "--diffusion-noise-level 0.9 "
        "--diffusion-height 480 "
        "--diffusion-width 480 "
        # Overrides the engine's serving default of 12 for this schedule.
        "--diffusion-flow-shift 3.0 "
        "--diffusion-step-strategy-path miles.diffusion.rollout.step_strategy.epoch_global_random_choice "
        "--diffusion-num-sde-steps 1 "
        "--diffusion-sde-candidate-steps 1,2,3 "
    )

    grpo_args = "--advantage-estimator grpo --eps-clip 1e-4 "

    reward_args = (
        "--rm-type pickscore "
        "--diffusion-pickscore-num-workers 1 "
        f"--diffusion-pickscore-num-gpus-per-worker {args.pickscore_gpus} "
        "--diffusion-pickscore-batch-size 8 "
        f"--diffusion-pickscore-model-path {_local(args.model_dir, PICKSCORE_MODEL)} "
        f"--diffusion-pickscore-processor-path {_local(args.model_dir, PICKSCORE_PROCESSOR)} "
    )

    optimizer_args = "--lr 1e-4 --adam-beta2 0.999 --weight-decay 1e-4 "

    lora_args = "--diffusion-lora-rank 64 --diffusion-lora-alpha 128 --diffusion-lora-init gaussian "

    sglang_args = "--use-miles-router --sglang-server-concurrency 8 --update-weight-buffer-size 2147483648 "

    train_args = (
        "--train-backend fsdp "
        "--diffusion-forward-dtype bf16 "
        # Two images per micro-batch, the most an H200 holds for both experts.
        "--diffusion-micro-batch-size-sample 2 "
        "--diffusion-micro-batch-size-tstep 1 "
        "--deterministic-mode "
    )

    num_gpus = 4 + math.ceil(args.pickscore_gpus)
    misc_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 4 "
        "--rollout-num-gpus-per-engine 1 "
        f"--num-gpus-per-node {num_gpus} "
        "--colocate "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {grpo_args} {reward_args} {optimizer_args} {lora_args} {sglang_args} "
            f"{train_args} {misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=num_gpus,
        megatron_model_type=None,
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"},
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs) -> None:
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
