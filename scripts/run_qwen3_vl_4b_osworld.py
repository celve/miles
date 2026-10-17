"""Computer-use GRPO on OSWorld through NeMo Gym: Qwen3-VL-4B-Instruct, FSDP or Megatron, single node.

NeMo Gym runs each episode in an OSWorld VM, and ``examples/experimental/osworld/rollout.py``
turns its exact trace into a fixed number of training samples per episode.

What must already exist:

- A NeMo Gym checkout with LIN-1996's enablement patches, its agent server reachable at ``gym_url``.
- ``gym_sglang_proxy.py`` in front of this run's SGLang router, named by Gym's ``POLICY_BASE_URL``.
- The prompt rows and held-out split, built by ``scripts/tools/make_osworld_split.py``.

Args:
    model_name, model_dir: where the HF checkpoint lives; ``prepare`` downloads it.
    prompt_data, eval_prompt_data: the training and held-out splits.
    gym_url: NeMo Gym's agent server, reachable from the train actors.
    router_port: the pinned port of this run's SGLang router, which Gym's proxy forwards to.
    samples_per_episode: OSWorld's step cap; fixes each episode's sample count.
    train_backend: ``fsdp``, or ``megatron`` through megatron-bridge, which loads the HF checkpoint directly.
    tensor_parallel: Megatron's tensor-parallel size; data-parallel ranks are the GPUs divided by it.
    max_output_tokens: Gym's per-call generation cap; a task row's own value wins.
    rollout_batch_size, n_samples_per_prompt: the GRPO group geometry.
    micro_batch_size: must divide each rank's share of the global batch.
    num_rollout: optimizer steps.
    mode: ``normal`` trains; ``smoke`` is two episodes, rollout only.

Usage:
    python3 scripts/run_qwen3_vl_4b_osworld.py --gym-url http://10.0.0.1:11001
    python3 scripts/run_qwen3_vl_4b_osworld.py --mode smoke --gym-url http://10.0.0.1:11001
    python3 scripts/run_qwen3_vl_4b_osworld.py --train-backend megatron --gym-url http://10.0.0.1:11001
"""

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.common import ArgvManipulator

HF_REPO = "Qwen/Qwen3-VL-4B-Instruct"
HF_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # Taken at import, so the snapshot harness's frozen create_run_id applies.
    run_id: str = command_utils.create_run_id()
    num_gpus_per_node: int = 4
    model_name: str = "Qwen3-VL-4B-Instruct"
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    prompt_data: str = ""
    eval_prompt_data: str = ""
    gym_url: str = "http://127.0.0.1:11001"
    # Gym is told this port before the job starts; presetting the IP asserts in resolve_router_addrs.
    router_port: int = 31000
    samples_per_episode: int = 15
    micro_batch_size: int = 1
    max_sandboxes: int = 8
    episode_timeout_s: int = 2400
    max_output_tokens: int = 1500
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 4
    num_rollout: int = 20
    train_backend: Literal["fsdp", "megatron"] = "fsdp"
    tensor_parallel: int = 4
    mode: Literal["normal", "smoke"] = "normal"
    extra_args: str = ""

    def __post_init__(self):
        # The legacy sample-strided split floor-divides an uneven shard, dropping rows' log-probs.
        super_post = getattr(super(), "__post_init__", None)
        if super_post is not None:
            super_post()
        if self.mode == "smoke":
            # Smoke runs no optimizer step.
            return
        total = self.global_batch_size
        dp_size = self.data_parallel_size
        if total % dp_size:
            raise ValueError(
                f"{self.episodes_per_step} episodes x {self.samples_per_episode} samples = {total} samples "
                f"does not divide across {dp_size} ranks. Adjust --rollout-batch-size, "
                "--n-samples-per-prompt or --samples-per-episode until it does."
            )
        per_rank = total // dp_size
        if per_rank < 1 or per_rank % self.micro_batch_size:
            raise ValueError(
                f"each rank takes {per_rank} samples, which --micro-batch-size {self.micro_batch_size} "
                "must divide and which must be at least 1."
            )

    @property
    def data_parallel_size(self) -> int:
        gpus = self.num_nodes * self.num_gpus_per_node
        if self.train_backend == "fsdp":
            return gpus
        if gpus % self.tensor_parallel:
            raise ValueError(f"--tensor-parallel {self.tensor_parallel} does not divide {gpus} GPUs.")
        return gpus // self.tensor_parallel

    @property
    def model_path(self) -> str:
        return f"{self.model_dir}/{self.model_name}"

    @property
    def episodes_per_step(self) -> int:
        """Episodes in one rollout batch, i.e. the GRPO group geometry."""
        if self.mode == "smoke":
            return 1 * 2
        return self.rollout_batch_size * self.n_samples_per_prompt

    @property
    def global_batch_size(self) -> int:
        """Samples in one optimizer step, derived so the batch divides exactly."""
        return self.episodes_per_step * self.samples_per_episode

    @property
    def train_prompt_data(self) -> str:
        return self.prompt_data or f"{self.data_dir}/osworld/train.jsonl"

    @property
    def held_out_prompt_data(self) -> str:
        return self.eval_prompt_data or f"{self.data_dir}/osworld/held_out.jsonl"


# Flags that fix the batch geometry, which --extra-args may not override.
_PROTECTED_FLAGS = (
    "--global-batch-size",
    "--micro-batch-size",
    "--rollout-batch-size",
    "--n-samples-per-prompt",
    "--actor-num-nodes",
    "--actor-num-gpus-per-node",
    "--num-gpus-per-node",
    "--tensor-model-parallel-size",
    "--pipeline-model-parallel-size",
    "--context-parallel-size",
)
_PROTECTED_ENV = ("MILES_SCRIPT_OSWORLD_SAMPLES_PER_EPISODE",)


def _refuse_geometry_overrides(train_args: str, args: ScriptArgs, launcher_env: dict[str, str]) -> None:
    """Check the final effective argv and environment with their consumers' own semantics."""
    argv = shlex.split(train_args)
    for flag in _PROTECTED_FLAGS:
        if len(ArgvManipulator.get(argv, flag)) > 1:
            raise ValueError(
                f"{flag} is declared more than once in the final command line. It fixes this recipe's "
                "batch geometry, which --extra-args may not override; change the matching field instead."
            )
    for token in argv:
        name = token.split("=", maxsplit=1)[0]
        if not name.startswith("--") or name in _PROTECTED_FLAGS:
            continue
        for flag in _PROTECTED_FLAGS:
            if flag.startswith(name):
                raise ValueError(
                    f"{token} abbreviates {flag}, which argparse resolves to the same destination. "
                    f"{flag} fixes this recipe's batch geometry; change the matching field instead."
                )

    effective = command_utils.resolve_extra_env_vars(launcher_env, args)
    for name in _PROTECTED_ENV:
        if effective.get(name) != launcher_env.get(name):
            raise ValueError(
                f"{name} resolves to {effective.get(name)!r} rather than the launcher's "
                f"{launcher_env.get(name)!r}, because --extra-env-vars is merged over the launcher's own "
                "environment. It fixes each episode's sample count, from which --global-batch-size is "
                "derived; change --samples-per-episode instead."
            )


def _refuse_a_shadowed_checkout() -> None:
    """Fail when ``miles`` resolves to a different tree than the one this launcher lives in."""
    checkout = Path(__file__).resolve().parents[1]
    if Path(command_utils.repo_base_dir).resolve() != checkout:
        raise RuntimeError(
            f"this launcher is in {checkout} but imported miles from {command_utils.repo_base_dir}, so the job "
            f"would run that tree's train.py and model definitions. Put the checkout first: "
            f"PYTHONPATH={checkout}:$PYTHONPATH"
        )


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir}")
    U.exec_command_cpu(f"hf download {HF_REPO} --revision {HF_REVISION} --local-dir {args.model_path}")


def execute(args: ScriptArgs):
    _refuse_a_shadowed_checkout()
    U = args.create_backend()
    smoke = args.mode == "smoke"

    # Emitted last, so argparse's last-wins makes these effective whatever --extra-args spells.
    geometry_args = (
        f"--rollout-batch-size {1 if smoke else args.rollout_batch_size} "
        f"--n-samples-per-prompt {2 if smoke else args.n_samples_per_prompt} "
        f"--global-batch-size {args.global_batch_size} "
        f"--micro-batch-size {args.micro_batch_size} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        # CP stays 1: the Qwen3-VL plugin cannot place vision embeddings across CP ranks.
        + (
            f"--tensor-model-parallel-size {args.tensor_parallel} "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            if args.train_backend == "megatron"
            else ""
        )
    )

    ckpt_args = f"--hf-checkpoint {args.model_path} --ref-load {args.model_path} "

    rollout_args = (
        f"--prompt-data {args.train_prompt_data} "
        "--input-key prompt "
        "--label-key label "
        # carries the Gym task row onto Sample.metadata['osworld_task']
        "--metadata-key metadata "
        "--apply-chat-template "
        "--rollout-shuffle "
        f"--num-rollout {1 if smoke else args.num_rollout} "
        # Bounds a whole stitched segment, about 11k tokens for a 15-turn episode.
        "--rollout-max-response-len 12288 "
        "--rollout-max-context-len 16384 "
        "--rollout-temperature 1.0 "
    )

    eval_args = (
        ""
        if smoke
        else (
            "--eval-interval 10 "
            f"--eval-prompt-data osworld_held_out {args.held_out_prompt_data} "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 12288 "
        )
    )

    custom_args = (
        "--custom-generate-function-path examples.experimental.osworld.rollout.generate "
        "--custom-rm-path examples.experimental.osworld.rollout.reward_func "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        # Explicit: the padded global batch makes the derived train_iters 0 below a cap's worth of rollouts.
        f"--lr-decay-iters {args.num_rollout} "
    )

    # Dispatching GBs of pixel tensors keeps rank 0 past NCCL's default 10-minute timeout.
    if args.train_backend == "megatron":
        # examples/geo3k_vlm's Megatron block, its parallel sizes in the geometry block above.
        train_backend_args = (
            "--train-backend megatron "
            f"--load {args.model_path} "
            "--megatron-to-hf-mode bridge "
            "--sequence-parallel "
            "--expert-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
            # No recompute: Megatron-Bridge copies it into the vision tower, where it stops DeepStack training.
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-softmax-in-fp32 "
            "--attention-backend flash "
            "--update-weight-buffer-size 536870912 "
            "--distributed-timeout-minutes 60 "
        )
    else:
        train_backend_args = (
            "--train-backend fsdp "
            "--gradient-checkpointing "
            "--update-weight-buffer-size 536870912 "
            "--distributed-timeout-minutes 60 "
            "--attn-implementation flash_attention_3 "
        )

    # No tool-call parser: OSWorld's agent parses Qwen's <tool_call> text itself.
    sglang_args = (
        f"--sglang-router-port {args.router_port} "
        "--pin-rollout-manager-to-head "
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        "--sglang-attention-backend fa3 "
    )

    telemetry_args = "--use-miles-dashboard --dashboard-gpu-sample-interval 5 --log-multi-turn --log-passrate "

    misc_args = (
        "--colocate "
        f"--save {args.output_dir}/ckpt "
        "--save-interval 5 "
        f"--dump-details {args.output_dir}/dump "
        # Miles's default eval logger scores a null reward as zero.
        "--custom-eval-rollout-log-function-path examples.experimental.osworld.rollout.log_eval_rollout_data "
        + ("--debug-rollout-only " if smoke else "")
    )

    train_args = (
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{eval_args} "
            f"{custom_args} "
            f"{grpo_args} "
            f"{optimizer_args} "
            f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
            f"{train_backend_args} "
            f"{sglang_args} "
            f"{telemetry_args} "
            f"{misc_args} "
            f"{args.extra_args} "
            f"{geometry_args} "
    )
    # Miles's training entry rejects unknown flags, so the converter's knobs travel as env vars.
    launcher_env = {
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "MILES_SCRIPT_GYM_URL": args.gym_url,
        "MILES_SCRIPT_OSWORLD_SAMPLES_PER_EPISODE": str(args.samples_per_episode),
        "MILES_SCRIPT_OSWORLD_MAX_SANDBOXES": str(args.max_sandboxes),
        "MILES_SCRIPT_OSWORLD_EPISODE_TIMEOUT_S": str(args.episode_timeout_s),
        "MILES_SCRIPT_OSWORLD_MAX_OUTPUT_TOKENS": str(args.max_output_tokens),
    }
    if not smoke:
        _refuse_geometry_overrides(train_args, args, launcher_env)

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3-vl-4B" if args.train_backend == "megatron" else None,
        extra_env_vars=launcher_env,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
