"""Phase 0: GRPO on a Qwen3-8B planner that rewrites each video request into one prompt for a frozen
FastWan2.2-TI2V-5B generator, rewarded by a Qwen3-VL judge of the clip against the request.

One node. The trainer has GPUs of its own (a frozen generator cannot share GPUs with training, so the run is not
colocated); then the planner's engines, the generator's engines, and the judge's reward pool. Every eval, including
the one before training, scores two held-out sets on the same requests and seeds: `heldout` (the planner; at step 0
the untrained one) and `heldout_raw` (the request sent to the generator as it is).

`prepare` downloads the three models and VBench's prompts, and writes the planner data with `make_planner_data.py`.

Args:
    --train-gpus, --policy-gpus, --generator-gpus, --judge-gpus: the node's split, summing to --num-gpus-per-node.
    --num-rollout, --rollout-batch-size, --n-samples-per-prompt: GRPO steps, requests per step, clips per request.
    --model-dir, --data-dir: where models and the planner data live; checkpoints and clips go under --output-dir.

Example:
    python examples/experimental/video_gen_agent/phase0/run_qwen3_8b_fastwan.py
    python examples/experimental/video_gen_agent/phase0/run_qwen3_8b_fastwan.py --num-rollout 2 --rollout-batch-size 4
"""

from dataclasses import dataclass

import typer
import yaml

from miles.utils.external_utils import command_utils

PLANNER = "Qwen/Qwen3-8B"
GENERATOR = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
JUDGE = "Qwen/Qwen3-VL-8B-Instruct"
VBENCH = "https://github.com/Vchitect/VBench.git"
PHASE0 = "examples.experimental.video_gen_agent.phase0"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    num_gpus_per_node: int = 8
    train_gpus: int = 2
    policy_gpus: int = 1
    generator_gpus: int = 3
    judge_gpus: int = 2
    num_rollout: int = 200
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    eval_interval: int = 10
    heldout_percent: int = 10
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    extra_args: str = ""

    def __post_init__(self):
        split = self.train_gpus + self.policy_gpus + self.generator_gpus + self.judge_gpus
        assert self.num_nodes == 1, "the recipe places its trainer, engines and judge on one node"
        assert split == self.num_gpus_per_node, f"the GPU split adds up to {split}, not {self.num_gpus_per_node}"

    @property
    def phase0_data_dir(self) -> str:
        return f"{self.data_dir}/video_gen_phase0"

    @property
    def run_dir(self) -> str:
        return f"{self.output_dir}/{self.run_id}"


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    for repo in (PLANNER, GENERATOR, JUDGE):
        U.exec_command_cpu(f"hf download {repo} --local-dir {args.model_dir}/{repo.split('/')[1]}")
    U.exec_command_cpu(f"[ -d {args.data_dir}/VBench ] || git clone --depth 1 {VBENCH} {args.data_dir}/VBench")
    U.exec_command_cpu(
        f"cd {command_utils.repo_base_dir} && python -m {PHASE0}.make_planner_data "
        f"{args.data_dir}/VBench/prompts/all_dimension.txt {args.phase0_data_dir} "
        f"--heldout-percent {args.heldout_percent}"
    )


def execute(args: ScriptArgs):
    U = args.create_backend()
    U.execute_train(train_args=_train_args(args), num_gpus_per_node=args.num_gpus_per_node, megatron_model_type=None)


def _train_args(args: ScriptArgs) -> str:
    planner_path = f"{args.model_dir}/{PLANNER.split('/')[1]}"

    ckpt_args = f"--hf-checkpoint {planner_path} --save {args.run_dir}/checkpoints --save-interval 50 "

    rollout_args = (
        f"--prompt-data {args.phase0_data_dir}/train.jsonl "
        "--input-key messages "
        "--apply-chat-template "
        """--apply-chat-template-kwargs '{"enable_thinking":false}' """
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-max-response-len 320 "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--balance-data "
    )

    # No --group-rm: eval refuses it, and the judge scores clip by clip, so a group's call would batch nothing.
    reward_args = (
        f"--custom-rm-path {PHASE0}.reward.planner_rm "
        f"--reward-pool-config {command_utils.encode_pseudo_file(yaml.safe_dump(_reward_pools(args)))} "
        "--video-reward-pool judge "
        "--video-generator-model generator "
        f"--video-clip-dir {args.run_dir}/clips "
        f"--custom-rollout-log-function-path {PHASE0}.reward.log_rollout_data "
    )

    eval_args = (
        f"--eval-interval {args.eval_interval} "
        f"--eval-config {command_utils.encode_pseudo_file(yaml.safe_dump(_eval_config(args)))} "
    )

    grpo_args = "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "

    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )

    sglang_args = (
        f"--sglang-config {command_utils.encode_pseudo_file(yaml.safe_dump(_sglang_config(args)))} "
        f"--rollout-num-gpus {args.policy_gpus + args.generator_gpus} "
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 "
        "--use-miles-router "
    )

    train_backend_args = (
        "--train-backend fsdp "
        "--gradient-checkpointing "
        f"--update-weight-buffer-size {512 * 1024 * 1024} "
        """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
        "--use-dynamic-batch-size --max-tokens-per-gpu 16384 "
    )

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.train_gpus} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--dump-details {args.run_dir}/dump_details "
    )

    return (
        f"{ckpt_args} {rollout_args} {reward_args} {eval_args} {grpo_args} {optimizer_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{sglang_args} {train_backend_args} {misc_args} {args.extra_args} "
    )


def _sglang_config(args: ScriptArgs) -> dict:
    # Clip settings are item 1's benchmark ones (LIN-2735): 832x480, 49 frames at 24 fps, 3 steps without CFG.
    generator_group = {
        "worker_type": "regular",
        "num_gpus": args.generator_gpus,
        "num_gpus_per_engine": 1,
        "engine_kind": "diffusion",
    }
    return {
        "sglang": [
            {"name": "policy", "server_groups": [{"worker_type": "regular", "num_gpus": args.policy_gpus}]},
            {
                "name": "generator",
                "model_path": f"{args.model_dir}/{GENERATOR.split('/')[1]}",
                "update_weights": False,
                "server_groups": [generator_group],
            },
        ]
    }


def _reward_pools(args: ScriptArgs) -> list[dict]:
    return [
        {
            "name": "judge",
            "scorer_path": "miles.rollout.rm_hub.video.vlm_judge.VlmJudgeScorer",
            "scorer_kwargs": {"model_path": f"{args.model_dir}/{JUDGE.split('/')[1]}"},
            "num_workers": args.judge_gpus,
            "uses_gpu": True,
            "batch_size": 4,
        }
    ]


def _eval_config(args: ScriptArgs) -> dict:
    return {
        "eval": {
            "defaults": {"n_samples_per_eval_prompt": 1, "top_k": 1, "max_response_len": 320},
            "datasets": [
                {"name": "heldout", "path": f"{args.phase0_data_dir}/heldout.jsonl"},
                {"name": "heldout_raw", "path": f"{args.phase0_data_dir}/heldout_raw.jsonl"},
            ],
        }
    }


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
