"""Flow-GRPO on SD3.5-medium with the OCR reward: LoRA r32, KL to the LoRA base, 2 GPUs colocated (FSDP DP=2 and
two sglang.multimodal_gen engines time-sharing the same GPUs).

Ported from radixark/miles_diffusion at 221f965f (scripts/run_diffusion_grpo_sd3_ocr_sglang.py). SD3.5 is gated on
the Hugging Face Hub, so export HF_TOKEN or pass --model-path to a local copy. `prepare` installs the OCR reward's
PaddleOCR (CPU) when the image lacks it.

Args:
    --num-rollout: rollouts to train; each is 8 prompts x 16 images and two optimizer steps.
    --model-path: a local SD3.5-medium diffusers directory; downloaded to --model-dir when empty.
    --save-interval: rollouts between checkpoints; 0 saves none.
    --extra-args: appended to the train argv.

    python scripts/run_sd35_medium_flow_grpo_ocr.py --num-rollout 600
"""

from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils

MODEL = "stabilityai/stable-diffusion-3.5-medium"
DATASET = "rockdu/miles-diffusion-datasets"
DATASET_SUBSET = "flowgrpo_ocr"

# paddleocr's resolver downgrades numpy below 2 and adds GUI OpenCV wheels; the image's numpy and headless OpenCV
# are restored afterwards, which PaddleOCR 2.9 runs on (miles.diffusion.rollout.ocr patches the one numpy 2 gap).
_OCR_DEPS_INSTALL = (
    'python -c \'import importlib.metadata as m; [m.version(p) for p in ("paddleocr", "paddlepaddle", '
    '"python-Levenshtein")]\' 2>/dev/null || ('
    "set -e; "
    "NUMPY=$(python -c 'import numpy; print(numpy.__version__)'); "
    "OPENCV=$(pip show opencv-python-headless | sed -n 's/^Version: //p'); "
    "pip install paddleocr==2.9.1 paddlepaddle==2.6.2 python-Levenshtein==0.27.3; "
    "pip uninstall -y opencv-python opencv-contrib-python; "
    'pip install "numpy==$NUMPY"; '
    'pip install --force-reinstall --no-deps "opencv-python-headless==$OPENCV")'
)


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    num_rollout: int = 600
    model_path: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    save_interval: int = 0
    extra_args: str = ""

    @property
    def resolved_model_path(self) -> str:
        return self.model_path or f"{self.model_dir}/stable-diffusion-3.5-medium"

    @property
    def dataset_dir(self) -> str:
        return f"{self.data_dir}/{DATASET.split('/')[-1]}"


def prepare(args: ScriptArgs) -> None:
    U = args.create_backend()
    U.exec_command_cpu(_OCR_DEPS_INSTALL)
    U.exec_command_cpu(
        f"hf download --repo-type dataset {DATASET} --include '{DATASET_SUBSET}/**' --local-dir {args.dataset_dir}"
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
        "--rollout-function-path miles.diffusion.rollout.rollout_fn.DiffusionRolloutFn "
        "--data-source-path miles.diffusion.rollout.data_source.DiffusionPromptDataSource "
        "--custom-convert-samples-to-train-data-path miles.diffusion.conversion.convert_samples_to_train_data "
        "--custom-rollout-log-function-path miles.diffusion.conversion.log_rollout_data "
        f"--prompt-data {args.dataset_dir}/{DATASET_SUBSET}/train.jsonl "
        "--input-key input "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 16 "
        f"--num-rollout {args.num_rollout} "
        "--global-batch-size 64 "
        "--diffusion-microgroup-size 8 "
        "--diffusion-num-steps 10 "
        "--diffusion-guidance-scale 4.5 "
        "--diffusion-negative-prompt ' ' "
        "--diffusion-noise-level 0.7 "
        "--diffusion-height 512 "
        "--diffusion-width 512 "
        "--diffusion-num-sde-steps 10 "
        "--diffusion-sde-window-range 0,10 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--diffusion-globalize-reward-std "
        "--diffusion-clip-range 1e-4 "
        "--diffusion-kl-beta 0.04 "
    )

    optimizer_args = "--lr 3e-4 --adam-beta2 0.999 --weight-decay 1e-4 "

    lora_args = "--diffusion-lora-rank 32 --diffusion-lora-alpha 64 --diffusion-lora-init gaussian "

    sglang_args = (
        "--use-miles-router "
        "--sglang-server-concurrency 8 "
        "--diffusion-dit-precision fp16 "
        "--update-weight-buffer-size 2147483648 "
    )

    train_args = (
        "--train-backend fsdp "
        "--diffusion-forward-dtype fp16 "
        "--gradient-checkpointing "
        "--diffusion-micro-batch-size-sample 16 "
        "--diffusion-micro-batch-size-tstep 5 "
        "--deterministic-mode "
    )

    misc_args = (
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 2 "
        "--rollout-num-gpus-per-engine 1 "
        "--num-gpus-per-node 2 "
        "--colocate "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {grpo_args} {optimizer_args} {lora_args} {sglang_args} "
            f"{train_args} {misc_args} {args.extra_args}"
        ),
        num_gpus_per_node=2,
        megatron_model_type=None,
        extra_env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs) -> None:
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
