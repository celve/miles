"""Multi-turn planner RL: a Qwen3.5-9B planner generates clips with a frozen FastWan2.2-TI2V-5B, watches them, retries
up to three times and submits one, rewarded by a Qwen3-VL judge of the submitted clip against the request.

One node, not colocated (a frozen generator cannot share GPUs with training): the trainer (Qwen3.5 plugin model under
raw Megatron, its vision tower frozen), the planner's engines, the generator's engines and the judge's reward pool.
The env is ``examples.experimental.video_gen_agent.env`` under session server v2; each clip comes back to the planner
as a ``video_url`` part in token space, and the engines and the trainer share SGLang's disk item feature cache.

Every eval, including the one before training, runs the planner greedily on the held-out requests (``eval/heldout``;
at step 0, the untrained planner). The raw-request baseline is Phase 0's ``eval/heldout_raw``: the same requests,
seeds, generator settings and judge.

Args:
    --train-gpus, --policy-gpus, --generator-gpus, --judge-gpus: the node's split, summing to --num-gpus-per-node.
    --num-rollout, --rollout-batch-size, --n-samples-per-prompt: GRPO steps, requests per step, episodes per request.
    --feature-cache / --no-feature-cache: share the engines' disk item feature cache with the trainer.
    --model-dir, --data-dir: where models and the requests live; checkpoints and clips go under --output-dir.

Example:
    python examples/experimental/video_gen_agent/multiturn/run_qwen35_9b_fastwan.py
    python examples/experimental/video_gen_agent/multiturn/run_qwen35_9b_fastwan.py --num-rollout 2 --rollout-batch-size 4
"""

import json
from dataclasses import dataclass

import typer
import yaml

from miles.utils.external_utils import command_utils

PLANNER = "Qwen/Qwen3.5-9B"
GENERATOR = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
JUDGE = "Qwen/Qwen3-VL-8B-Instruct"
VBENCH = "https://github.com/Vchitect/VBench.git"
AGENT = "examples.experimental.video_gen_agent"

# Phase 0's clip settings (LIN-2737), so both recipes share the raw-request baseline: 832x480, 2 s at 24 fps
# (49 frames), FastWan's 3-step schedule without CFG.
_CLIP_ENV = {
    "AGENT_VIDEO_GEN_WIDTH": "832",
    "AGENT_VIDEO_GEN_HEIGHT": "480",
    "AGENT_VIDEO_GEN_FPS": "24",
    "AGENT_VIDEO_GEN_MAX_SECONDS": "2",
    "AGENT_VIDEO_GEN_NUM_INFERENCE_STEPS": "3",
    "AGENT_VIDEO_GEN_GUIDANCE_SCALE": "1.0",
}

# The planner watches each clip at 4 fps: 8 frames of a 2 s clip.
_MM_PROCESS_CONFIG = {"video": {"fps": 4.0, "max_frames": 16}, "video_preprocessing_device": "cpu"}


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    num_gpus_per_node: int = 8
    train_gpus: int = 4
    policy_gpus: int = 1
    generator_gpus: int = 2
    judge_gpus: int = 1
    num_rollout: int = 200
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    max_generations: int = 3
    eval_interval: int = 10
    heldout_percent: int = 10
    feature_cache: bool = True
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""

    def __post_init__(self):
        split = self.train_gpus + self.policy_gpus + self.generator_gpus + self.judge_gpus
        assert self.num_nodes == 1, "the recipe places its trainer, engines and judge on one node"
        assert split == self.num_gpus_per_node, f"the GPU split adds up to {split}, not {self.num_gpus_per_node}"

    @property
    def requests_dir(self) -> str:
        return f"{self.data_dir}/video_gen_requests"

    @property
    def run_dir(self) -> str:
        return f"{self.output_dir}/{self.run_id}"


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    for repo in (PLANNER, GENERATOR, JUDGE):
        U.exec_command_cpu(f"hf download {repo} --local-dir {args.model_dir}/{repo.split('/')[1]}")
    U.convert_checkpoint(
        model_name=PLANNER.split("/")[1],
        megatron_model_type="qwen3.5-9B",
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{PLANNER.split('/')[1]}",
        megatron_path=args.megatron_path,
    )
    U.exec_command_cpu(f"[ -d {args.data_dir}/VBench ] || git clone --depth 1 {VBENCH} {args.data_dir}/VBench")
    U.exec_command_cpu(
        f"cd {command_utils.repo_base_dir} && python -m {AGENT}.make_requests "
        f"{args.data_dir}/VBench/prompts/all_dimension.txt {args.requests_dir} "
        f"--heldout-percent {args.heldout_percent}"
    )


def execute(args: ScriptArgs):
    U = args.create_backend()
    U.execute_train(
        train_args=_train_args(args),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3.5-9B",
        megatron_path=args.megatron_path,
        extra_env_vars={"PYTHONPATH": args.megatron_path, **_agent_env(args)},
    )


def _agent_env(args: ScriptArgs) -> dict[str, str]:
    return _CLIP_ENV | {
        "AGENT_VIDEO_GEN_GENERATOR": f"{AGENT}.multiturn.generator.RouterClipGenerator",
        "AGENT_VIDEO_GEN_CLIP_DIR": f"{args.run_dir}/clips",
        "AGENT_VIDEO_GEN_SUBMITTED_DIR": f"{args.run_dir}/submitted",
        "AGENT_VIDEO_GEN_MAX_GENERATIONS": str(args.max_generations),
        "AGENT_VIDEO_GEN_CLIP_OBSERVATIONS": "1",
    }


def _train_args(args: ScriptArgs) -> str:
    planner_path = f"{args.model_dir}/{PLANNER.split('/')[1]}"

    # A fresh --load directory starts the run from the --ref-load weights at rollout 0.
    ckpt_args = (
        f"--hf-checkpoint {planner_path} "
        f"--ref-load {planner_path}_torch_dist "
        f"--load {args.run_dir}/checkpoints --save {args.run_dir}/checkpoints --save-interval 50 "
        "--megatron-to-hf-mode raw "
        "--custom-model-provider-path miles_plugins.models.qwen3_5.qwen3_5_mm_model_provider "
    )

    rollout_args = (
        f"--prompt-data {args.requests_dir}/train.jsonl "
        "--input-key messages "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--rollout-max-response-len 1024 "
        "--max-seq-len 16384 "
        "--rollout-temperature 1 "
    )

    # Video turns through the session server: Qwen3.5's fixed template, clips in token space behind the Miles router.
    agent_args = (
        f"--custom-generate-function-path {AGENT}.multiturn.generate.generate "
        f"--custom-agent-function-path {AGENT}.env.run_agent "
        "--use-session-server v2 "
        "--tito-model qwen35 "
        "--router miles "
        # Generation dominates an episode, so every episode of a round is in flight at once (LIN-2634).
        f"--sglang-server-concurrency {args.rollout_batch_size * args.n_samples_per_prompt} "
    )

    multimodal_args = (
        "--multimodal-token-space-processing "
        "--sglang-enable-token-space-processor "
        f"--shared-media-dir {args.run_dir}/media "
        f"--sglang-mm-process-config '{json.dumps(_MM_PROCESS_CONFIG)}' "
        "--trainer-mm-processor-worker-num 8 --trainer-mm-io-worker-num 8 "
    )
    if args.feature_cache:
        # Clips are written by content hash and never rewritten, so a cached path never goes stale.
        multimodal_args += (
            "--sglang-enable-token-space-preprocess-cache "
            "--sglang-token-space-preprocess-cache-trust-url "
            f"--sglang-token-space-preprocess-cache-dir {args.run_dir}/feature_cache "
        )

    # No --group-rm: the env reward scores episodes sample by sample, and eval refuses it.
    reward_args = (
        f"--custom-rm-path {AGENT}.reward.reward_func "
        f"--reward-pool-config {command_utils.encode_pseudo_file(yaml.safe_dump(_reward_pools(args)))} "
        "--video-reward-pool judge "
    )

    eval_args = (
        f"--eval-interval {args.eval_interval} "
        f"--eval-config {command_utils.encode_pseudo_file(yaml.safe_dump(_eval_config(args)))} "
    )

    # TimeLens GRPO on Qwen3.5 (#4633): no KL, constant lr 1e-6, TIS against the rollout log-probs.
    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--use-tis "
    )

    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.999 "
    )

    sglang_args = (
        f"--sglang-config {command_utils.encode_pseudo_file(yaml.safe_dump(_sglang_config(args)))} "
        f"--rollout-num-gpus {args.policy_gpus + args.generator_gpus} "
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {args.train_gpus} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 20480 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.train_gpus} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--dump-details {args.run_dir}/dump_details "
    )

    return (
        f"{ckpt_args} {rollout_args} {agent_args} {multimodal_args} {reward_args} {eval_args} {grpo_args} "
        f"{optimizer_args} {command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{sglang_args} {perf_args} {misc_args} {args.extra_args} "
    )


def _sglang_config(args: ScriptArgs) -> dict:
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
            "defaults": {"n_samples_per_eval_prompt": 1, "top_k": 1, "max_response_len": 1024},
            "datasets": [{"name": "heldout", "path": f"{args.requests_dir}/heldout.jsonl"}],
        }
    }


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
