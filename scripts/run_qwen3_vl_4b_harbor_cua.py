"""GRPO on Qwen3-VL-4B through Harbor's ``computer-1`` CUA agent, one node.

Each episode runs one Harbor trial in the rollout worker, pointed at a recording
proxy in front of Miles's sglang router; the recorded calls become one training
sample per screenshot segment, sharing the episode's rollout id and reward. The
task set is the five browser families under
``examples/experimental/harbor_cua/tasks``, generated here with a disclosed
train/evaluation split.

Needs, before ``execute``:

* a sandbox backend the rollout worker can reach -- ``--harbor-env-type e2b``
  for E2B Cloud or an E2B-compatible endpoint, with its key file at
  ``--e2b-api-key-file``; the local ``docker`` backend is refused, since
  ``Trial.run()`` would need a daemon beside it;
* a Megatron-LM checkout at ``--megatron-path``, because the FSDP log-prob path
  imports ``megatron.core.fusions.fused_cross_entropy``;
* ``harbor`` installed in the trainer's environment from the branch Miles pins
  (``uv pip install "harbor[e2b] @ git+https://github.com/harbor-framework/harbor@harbor-miles-v0.20.0"``).

Args:
    --harbor-env-type: Harbor ``EnvironmentType`` for the trials (required).
    --harbor-tasks-dir: where ``prepare`` writes the generated task directories.
    --harbor-trials-dir: where Harbor writes trial directories on the worker.
    --backend: ``megatron`` is the default because LIN-2046 measured FSDP training
        Qwen3-VL without its multimodal RoPE; ``fsdp`` is for throughput and plumbing
        work, not for a result.
    --skip-prepare: leave the checkpoint, the task directories and the warmed
        templates as they are, for a rerun or a hand-curated data directory.
    --mode: ``rollout`` converts rollouts and stops before any optimizer step,
        ``eval`` scores the evaluation split only, ``train`` does the GRPO run --
        which evaluates once before the first update and every
        ``--eval-interval`` rollouts after it, so a before/after pair comes out
        of one run.

Usage:
    python3 scripts/run_qwen3_vl_4b_harbor_cua.py --harbor-env-type e2b
    python3 scripts/run_qwen3_vl_4b_harbor_cua.py --harbor-env-type e2b --mode rollout
    python3 scripts/run_qwen3_vl_4b_harbor_cua.py --harbor-env-type e2b --mode eval
"""

import os
import shlex
from dataclasses import dataclass
from typing import Literal

import typer

from examples.experimental.harbor.launch_common import harbor_env_vars
from miles.utils.external_utils import command_utils

HF_REPO = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_NAME = "Qwen3-VL-4B-Instruct"
WANDB_GROUP = "qwen3-vl-4b-harbor-cua"
GENERATE_FN = "examples.experimental.harbor_cua.rollout.generate"
# text_config.rope_theta of Qwen/Qwen3-VL-4B-Instruct; the shared qwen3-4B model args
# default to 1000000, which is the text model's, not this checkpoint's.
ROTARY_BASE = 5_000_000


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    num_gpus_per_node: int = 2
    num_rollout: int = 30
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 8
    model_dir: str = "/root/models"
    # Even the FSDP path imports megatron's fused cross-entropy for log probs, so the
    # checkout has to be on the workers' PYTHONPATH.
    megatron_path: str = "/root/Megatron-LM"
    data_dir: str = "/root/datasets/harbor_cua"
    harbor_tasks_dir: str = "/root/harbor_cua_tasks"
    harbor_trials_dir: str = "/tmp/harbor_cua_trials"
    harbor_env_type: str = ""
    harbor_env_kwargs: str = ""
    e2b_api_key_file: str = ""
    # Empty means "whatever the engines serve", which is the checkpoint path Miles passes
    # as --hf-checkpoint; a litellm model name that is not the served one fails every call.
    agent_model_name: str = ""
    agent_timeout: int = 600
    train_per_family: int = 4
    eval_per_family: int = 2
    max_turns: int = 10
    # flash_attention_3 is the house default, but the trainer resolves it by downloading
    # kernels-community/vllm-flash-attn3, which needs a writable HF cache and an image
    # that can reach it; flash_attention_2 keeps the recipe runnable where it cannot.
    attn_implementation: str = "flash_attention_3"
    max_sandboxes: int = 8
    # ``auto`` asks the router for one of its engines, because the router itself answers
    # without the token-evidence fields the recorder needs; ``router`` or an explicit URL
    # overrides that.
    recorder_upstream: str = "auto"
    desktop_width: int = 1024
    desktop_height: int = 900
    mode: Literal["train", "rollout", "eval"] = "train"
    # megatron by default because LIN-2046 measured FSDP training Qwen3-VL with linear
    # 1-D positions, so its multimodal RoPE never runs (max |logit diff| 32.72 on a
    # three-image sample) while megatron-bridge reconstructs the 3-D positions itself.
    # fsdp stays selectable for throughput and plumbing work, never for a result.
    backend: Literal["megatron", "fsdp"] = "megatron"
    tensor_model_parallel_size: int = 2
    # Megatron's own default is transformer_engine, which Miles installs from prebuilt
    # wheels in its image; on a plain sglang image those wheels break torch's NCCL
    # symbols, so `local` -- megatron's pure-PyTorch layer spec -- is what runs there.
    # transformer_engine is faster and is the right choice on the Miles image.
    transformer_impl: Literal["local", "transformer_engine"] = "local"
    skip_prepare: bool = False
    eval_interval: int = 5
    wandb_project: str = "miles-harbor-cua"
    extra_args: str = ""

    @property
    def model_path(self) -> str:
        return f"{self.model_dir}/{MODEL_NAME}"

    @property
    def global_batch_size(self) -> int:
        return self.rollout_batch_size * self.n_samples_per_prompt

    @property
    def served_model_name(self) -> str:
        return self.agent_model_name or self.model_path

    @property
    def rollouts(self) -> int:
        """``--num-rollout 0`` with an eval interval is Miles's eval-only mode."""
        return {"train": self.num_rollout, "rollout": 1, "eval": 0}[self.mode]


def prepare(args: ScriptArgs, harbor_env: dict[str, str] | None = None):
    """``harbor_env`` comes from ``main``; warming needs the same backend as the run."""
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir}")
    U.exec_command_cpu(f"hf download {HF_REPO} --local-dir {args.model_path}")
    U.exec_command_cpu(
        f"cd {command_utils.repo_base_dir} && python -m examples.experimental.harbor_cua.make_harbor_cua_tasks "
        f"--tasks-dir {args.harbor_tasks_dir} --data-dir {args.data_dir} "
        f"--train-per-family {args.train_per_family} --eval-per-family {args.eval_per_family}"
    )
    # Serial, and before any rollout: Harbor has no retry around a template build, so
    # concurrent trials on an unbuilt template lose a whole GRPO group to a 409.
    # Quoted: HARBOR_ENV_KWARGS is JSON and a provider key path can contain spaces, so
    # an unquoted export either fails or silently strips the quotes out of the JSON.
    exports = "".join(
        f"export {name}={shlex.quote(str(value))} && " for name, value in sorted((harbor_env or {}).items())
    )
    U.exec_command_cpu(
        f"{exports}cd {command_utils.repo_base_dir} && "
        "python -m examples.experimental.harbor_cua.warm_templates "
        f"--tasks-dir {args.harbor_tasks_dir} --trials-dir {args.harbor_trials_dir} "
        f"--data {args.data_dir}/train.jsonl {args.data_dir}/eval.jsonl"
    )


def execute(args: ScriptArgs, harbor_env: dict[str, str] | None = None):
    """``harbor_env`` comes from ``main``, which preflights it before any GPU is held."""
    U = args.create_backend()

    ckpt_args = f"--hf-checkpoint {args.model_path} --ref-load {args.model_path} "

    rollout_args = (
        f"--prompt-data {args.data_dir}/train.jsonl "
        "--input-key prompt "
        "--label-key label "
        # carries instance_id, family, variant and split onto Sample.metadata
        "--metadata-key metadata "
        "--apply-chat-template "
        "--rollout-shuffle "
        f"--num-rollout {args.rollouts} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-max-response-len 4096 "
        "--rollout-max-context-len 32768 "
        "--rollout-temperature 1.0 "
        f"--global-batch-size {args.global_batch_size} "
    )

    cua_args = (
        f"--custom-generate-function-path {GENERATE_FN} "
        f"--harbor-cua-max-turns {args.max_turns} "
        f"--harbor-cua-max-sandboxes {args.max_sandboxes} "
        "--harbor-cua-write-off-mode abort_group "
        + f"--harbor-cua-upstream {args.recorder_upstream} "
        + ("--debug-rollout-only " if args.mode == "rollout" else "")
    )

    eval_args = (
        f"--eval-prompt-data harbor_cua {args.data_dir}/eval.jsonl "
        f"--eval-interval {args.eval_interval} "
        "--n-samples-per-eval-prompt 4 "
        "--eval-max-response-len 4096 "
        "--eval-top-p 1 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        # The recorder's whole purpose: the importance ratio is taken against the
        # logprobs the engine reported, not against a trainer recomputation.
        "--use-rollout-logprobs "
    )

    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
    )

    # Dynamic micro-batching, because an episode's segment count is data-dependent.
    # Deliberately NOT --use-dynamic-global-batch-size: every sample here carries a
    # rollout_id, so postprocess_rollout_data skips its whole non-compact block and
    # returns no dynamic_global_batch_size, which convert_samples_to_train_data then
    # asserts against the flag. The compact schedule drops whole trailing rollouts
    # instead, which is what makes a variable sibling count safe without it.
    perf_args = "--use-dynamic-batch-size --max-tokens-per-gpu 16384 "

    fsdp_args = (
        "--train-backend fsdp "
        "--gradient-checkpointing "
        "--update-weight-buffer-size 536870912 "
        f"--attn-implementation {args.attn_implementation} "
        # A rollout batch of screenshot episodes is gigabytes of pixel tensors, and
        # dispatching it keeps rank 0 past the default NCCL watchdog while the other
        # ranks sit in their first collective (miles#2406).
        "--distributed-timeout-minutes 60 "
        """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
    )

    megatron_args = (
        "--train-backend megatron "
        f"--load {args.model_path} "
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--transformer-impl {args.transformer_impl} "
        "--megatron-to-hf-mode bridge "
        "--distributed-timeout-minutes 60 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        "--sglang-attention-backend fa3 "
    )

    wandb_args = (
        f"--use-wandb --wandb-project {args.wandb_project} --wandb-group {WANDB_GROUP} "
        if os.environ.get("WANDB_API_KEY")
        else ""
    )

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        f"--save {args.output_dir}/ckpt "
        "--save-interval 5 "
        f"--dump-details {args.output_dir}/dump "
    )

    backend_args = fsdp_args
    if args.backend == "megatron":
        backend_args = megatron_args
        # Set here, not in the worker environment: scripts/models/qwen3-4B.py reads this
        # while the model args are expanded in *this* process, and defaults to 1000000.
        # Qwen3-VL-4B-Instruct's text_config.rope_theta is 5000000, read off the checkpoint.
        os.environ["MODEL_ARGS_ROTARY_BASE"] = str(ROTARY_BASE)
    U.execute_train(
        train_args=(
            f"{ckpt_args} {rollout_args} {cua_args} {eval_args} {grpo_args} "
            f"{optimizer_args} {perf_args} {backend_args} {sglang_args} "
            f"{wandb_args} {misc_args} {args.extra_args} "
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=_megatron_model_type(args),
        megatron_path=args.megatron_path,
        extra_env_vars=_worker_env(args, harbor_env or {}),
    )


def _megatron_model_type(args: ScriptArgs) -> str | None:
    """The text model Megatron loads for this checkpoint, as geo3k's VLM recipe maps it."""
    if args.backend != "megatron":
        return None
    return "qwen3-4B"


def _worker_env(args: ScriptArgs, harbor_env: dict[str, str]) -> dict:
    env = {
        "PYTHONPATH": str(command_utils.repo_base_dir),
        # The trainer actors resolve the attention kernel through the hub cache, which is
        # read-only on some images unless this is forwarded.
        **({"HF_HOME": os.environ["HF_HOME"]} if os.environ.get("HF_HOME") else {}),
        "HARBOR_CUA_DESKTOP_WIDTH": str(args.desktop_width),
        "HARBOR_CUA_DESKTOP_HEIGHT": str(args.desktop_height),
        "AGENT_MAX_INPUT_TOKENS": os.environ.get("AGENT_MAX_INPUT_TOKENS", "32768"),
        "AGENT_MAX_OUTPUT_TOKENS": os.environ.get("AGENT_MAX_OUTPUT_TOKENS", "2048"),
    }
    env.update(harbor_env)
    return env


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    if not args.harbor_env_type:
        raise typer.BadParameter(
            "--harbor-env-type is required: in-process trials need a sandbox backend the "
            "rollout worker can reach, such as e2b"
        )
    args.agent_model_name = args.served_model_name
    harbor_env = harbor_env_vars(args)
    if not args.skip_prepare:
        prepare(args, harbor_env=harbor_env)
    execute(args, harbor_env=harbor_env)


if __name__ == "__main__":
    typer.run(main)
