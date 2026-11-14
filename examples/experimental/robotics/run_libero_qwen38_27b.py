"""GRPO for Qwen3.8-27B on LIBERO episodes in PEAR's protocol, trainer and engines colocated on one 8-GPU node.

Each episode runs ``libero_agent_function.run`` through the session server: a libero-sim sandbox per episode on the
sandbox service, PEAR's prompts and action checks, thinking on, and one context budget shared by every turn
(``--context-tokens``, PEAR's 160k at most). The trainer loads the HF checkpoint through the Megatron bridge, so no
conversion runs. Needs the libero-sim template built (``libero_sandbox.py build``) and an API key for the service.

Args:
    --suites: LIBERO suites the training rows come from (default libero_goal); held-out trials 40-49 never train.
    --context-tokens: an episode's whole context, prompt and every turn (default 65536).
    --num-rollout / --rollout-batch-size / --n-samples-per-prompt: rounds, rows a round, episodes a row.
    --eval-interval: rounds between held-out evaluations on trials 40-49; 0 turns evaluation off (default).
    --e2b-api-url / --e2b-sandbox-url / --e2b-api-key-file: the sandbox service.
    --libero-e2b-proxy: a proxy to the service for a node off its network, such as socks5h://127.0.0.1:1080.

Example:
    python examples/experimental/robotics/run_libero_qwen38_27b.py \\
        --e2b-api-url https://sandbox-service-control-plane.tail134ba0.ts.net \\
        --e2b-sandbox-url http://sandbox-service-control-plane --e2b-api-key-file ~/.config/e2b/api_key
"""

from dataclasses import dataclass
from pathlib import Path

import typer

from miles.utils.external_utils import command_utils

MODEL_NAME = "Qwen3.8-27B"
MEGATRON_MODEL_TYPE = "qwen3.8-27B"
HERE = Path(__file__).resolve().parent


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    num_gpus_per_node: int = 8
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    suites: str = "libero_goal"
    context_tokens: int = 65536
    num_rollout: int = 1
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    eval_interval: int = 0
    lr: float = 1e-6
    e2b_api_url: str = ""
    e2b_sandbox_url: str = ""
    e2b_api_key_file: str = ""
    libero_e2b_proxy: str = ""
    libero_template: str = ""
    extra_args: str = ""

    @property
    def rows_dir(self) -> str:
        return f"{self.data_dir}/libero"

    @property
    def run_dir(self) -> str:
        return f"{self.output_dir}/{self.run_id}"


def prepare(args: ScriptArgs):
    U = args.create_backend()
    if not Path(f"{args.model_dir}/{MODEL_NAME}").exists():
        U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir {args.model_dir}/{MODEL_NAME}")
    U.exec_command_cpu(
        f"python3 {HERE}/make_libero_data.py --out-dir {args.rows_dir} --suites {args.suites.replace(',', ' ')}"
    )


def _env_vars(args: ScriptArgs) -> dict[str, str]:
    env = {
        "PYTHONPATH": f"{args.megatron_path}:{HERE}",
        "LIBERO_CONTEXT_TOKENS": str(args.context_tokens),
        # Miles flushes no image-embedding cache on a weight update, so a resent frame would keep old embeddings.
        "SGLANG_VLM_CACHE_SIZE_MB": "0",
    }
    optional = {
        "E2B_API_URL": args.e2b_api_url,
        "E2B_SANDBOX_URL": args.e2b_sandbox_url,
        "E2B_API_KEY_FILE": args.e2b_api_key_file,
        "LIBERO_E2B_PROXY": args.libero_e2b_proxy,
        "LIBERO_TEMPLATE": args.libero_template,
    }
    return {**env, **{name: value for name, value in optional.items() if value}}


def execute(args: ScriptArgs):
    U = args.create_backend()
    episodes = args.rollout_batch_size * args.n_samples_per_prompt

    # The bridge reads the HF checkpoint directly; a fresh --load directory starts from it at rollout 0.
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{MODEL_NAME} "
        f"--ref-load {args.model_dir}/{MODEL_NAME} "
        "--megatron-to-hf-mode bridge "
        f"--load {args.run_dir}/checkpoints "
        f"--save {args.run_dir}/checkpoints "
        "--save-interval 5 "
    )

    # One context budget for the whole episode, as PEAR's response budget; a turn may use all that is left.
    rollout_args = (
        f"--prompt-data {args.rows_dir}/libero.train.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--metadata-key metadata "
        "--apply-chat-template "
        """--apply-chat-template-kwargs '{"enable_thinking": true}' """
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--num-steps-per-rollout 1 "
        f"--rollout-max-response-len {args.context_tokens} "
        f"--max-seq-len {args.context_tokens} "
        "--rollout-temperature 1.0 "
        "--rollout-top-p 1.0 "
        "--rollout-top-k -1 "
    )

    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path libero_agent_function.run "
        "--custom-rm-path libero_agent_function.reward_func "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        "--use-session-server v2 "
        "--tito-model qwen38small "
        "--session-server-workers 4 "
        "--use-miles-router "
    )

    # SGLang expands each frame's placeholders; the trainer rebuilds the same tensors from the shared files.
    multimodal_args = (
        "--multimodal-token-space-processing "
        "--sglang-enable-token-space-processor "
        f"--shared-media-dir {args.run_dir}/media "
    )

    eval_args = "--skip-eval-before-train "
    if args.eval_interval:
        eval_args = (
            f"--eval-interval {args.eval_interval} "
            f"--eval-prompt-data heldout {args.rows_dir}/libero.heldout.jsonl "
            "--n-samples-per-eval-prompt 8 "
            "--eval-temperature 1.0 "
            "--eval-top-p 1.0 "
            "--eval-top-k -1 "
        )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--eps-clip-c 3.0 "
        "--use-tis "
    )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-reasoning-parser qwen3 "
        f"--sglang-server-concurrency {episodes} "
        "--rollout-health-check-timeout 30 "
    )

    # Context parallelism halves each rank's share of a long episode; Qwen3.8's 4 KV heads cap TP at 4.
    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 2 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--distributed-timeout-minutes 60 "
        f"--dump-details {args.run_dir}/dump "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{agent_args} "
        f"{multimodal_args} "
        f"{eval_args} "
        f"{grpo_args} "
        f"{optimizer_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        megatron_path=args.megatron_path,
        extra_env_vars=_env_vars(args),
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
