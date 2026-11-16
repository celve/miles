"""RePrompt's RL stage (https://arxiv.org/abs/2505.17540) in Miles: GRPO on Qwen2.5-3B-Instruct rewriting short
text-to-image prompts, rewarded by ImageReward on the rendered image plus RePrompt's format and length terms.

The settings are RePrompt's `train_reprompt.py` under TRL's GRPOTrainer, so a run here can be set against that script
on the same prompts: 1108 prompts for 3 epochs of 34 steps, 32 prompts x 4 completions per step, lr 2e-6 decaying
linearly to 0, AdamW without weight decay, gradient clip 1, sampling temperature 0.9 and 256 new tokens, and a k3 KL
loss to the initial policy at 0.04. Its generator is Z-Image-Turbo at 512x512, not the paper's FLUX.1-dev.

The image half of the reward runs in `reward_server.py` processes on GPUs outside this job; start them first and pass
their URLs. This job's Ray cluster holds only the trainer and the policy's engine.

Args:
    --train-gpus, --policy-gpus: the job's GPUs; --num-gpus-per-node is their sum.
    --reward-urls: comma-separated reward_server.py URLs.
    --num-epoch, --rollout-batch-size, --n-samples-per-prompt: as RePrompt's 3 epochs, 32 prompts, 4 completions.
    --model-dir, --data-dir: where the policy and the prompt data live; checkpoints go under --output-dir.

Example:
    python examples/experimental/reprompt/run_qwen25_3b_reprompt.py --reward-urls http://127.0.0.1:18002
"""

from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils

POLICY = "Qwen/Qwen2.5-3B-Instruct"
MEGATRON_MODEL_TYPE = "qwen2.5-3B"
REPROMPT_REPO = "https://github.com/microsoft/DKI_LLM.git"
REPROMPT_COMMIT = "c090b7d57e3996c8900e86fecd16085b27e07f29"
RECIPE = "examples.experimental.reprompt"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    num_gpus_per_node: int = 2
    train_gpus: int = 1
    policy_gpus: int = 1
    reward_urls: str = "http://127.0.0.1:18002"
    num_epoch: int = 3
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 4
    save_interval: int = 50
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    extra_args: str = ""

    def __post_init__(self):
        assert self.num_nodes == 1, "the recipe places its trainer and engine on one node"
        assert self.train_gpus + self.policy_gpus == self.num_gpus_per_node, (
            f"--train-gpus {self.train_gpus} and --policy-gpus {self.policy_gpus} must add up to "
            f"--num-gpus-per-node {self.num_gpus_per_node}"
        )

    @property
    def policy_path(self) -> str:
        return f"{self.model_dir}/{POLICY.split('/')[1]}"

    @property
    def reprompt_dir(self) -> str:
        return f"{self.data_dir}/reprompt"

    @property
    def run_dir(self) -> str:
        return f"{self.output_dir}/{self.run_id}"


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.reprompt_dir}")
    U.exec_command_cpu(f"hf download {POLICY} --local-dir {args.policy_path}")
    U.exec_command_cpu(
        f"[ -d {args.reprompt_dir}/DKI_LLM ] || git clone {REPROMPT_REPO} {args.reprompt_dir}/DKI_LLM; "
        f"git -C {args.reprompt_dir}/DKI_LLM checkout {REPROMPT_COMMIT}"
    )
    U.exec_command_cpu(
        f"cd {command_utils.repo_base_dir} && python -m {RECIPE}.make_data "
        f"{args.reprompt_dir}/DKI_LLM/RePrompt/data/generation_prompts_1k_filterd.txt {args.policy_path} "
        f"{args.reprompt_dir}/train.jsonl"
    )
    U.convert_checkpoint(
        model_name=POLICY.split("/")[1],
        megatron_model_type=MEGATRON_MODEL_TYPE,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=args.policy_path,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    U = args.create_backend()
    U.execute_train(
        train_args=_train_args(args),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=MEGATRON_MODEL_TYPE,
    )


def _train_args(args: ScriptArgs) -> str:
    ckpt_args = (
        f"--hf-checkpoint {args.policy_path} "
        f"--ref-load {args.policy_path}_torch_dist "
        f"--load {args.run_dir}/checkpoints --save {args.run_dir}/checkpoints --save-interval {args.save_interval} "
        f"--save-hf {args.run_dir}/hf/iter_{{rollout_id}} "
        "--megatron-to-hf-mode raw "
    )

    rollout_args = (
        f"--prompt-data {args.reprompt_dir}/train.jsonl "
        "--input-key text "
        "--rollout-shuffle "
        f"--num-epoch {args.num_epoch} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--rollout-max-response-len 256 "
        "--rollout-temperature 0.9 "
        "--rollout-skip-special-tokens "
    )

    reward_args = (
        f"--custom-rm-path {RECIPE}.reward.reprompt_rm "
        f"--reprompt-reward-urls {args.reward_urls} "
        f"--custom-rollout-log-function-path {RECIPE}.reward.log_rollout_data "
        f"--reprompt-sample-log {args.run_dir}/samples "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss --kl-loss-coef 0.04 --kl-loss-type k3 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--use-tis "
    )

    optimizer_args = (
        "--optimizer adam --lr 2e-6 --lr-decay-style linear --min-lr 0 --lr-warmup-iters 0 "
        "--weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --clip-grad 1.0 "
    )

    sglang_args = (
        f"--rollout-num-gpus {args.policy_gpus} "
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
        # Qwen2.5 trains without dropout; Megatron's default is 0.1.
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
    )

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.train_gpus} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )

    return (
        f"{ckpt_args} {rollout_args} {reward_args} {grpo_args} {optimizer_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{sglang_args} {perf_args} {misc_args} {args.extra_args} "
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
