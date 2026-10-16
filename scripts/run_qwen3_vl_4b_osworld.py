"""Computer-use GRPO on OSWorld through NeMo Gym: Qwen3-VL-4B-Instruct, FSDP, single node.

NeMo Gym owns the episode -- it boots an OSWorld VM, drives ``DesktopEnv``
against our policy and grades with OSWorld's own evaluator -- and
``examples/experimental/osworld/rollout.py`` converts its exact trace into
training samples. Gym therefore runs as its own process group, reached over
HTTP, and need not live on this node; only ``gym_url`` has to resolve from the
train actors, and the OSWorld VMs run wherever that Gym is.

What must already exist:

- A NeMo Gym checkout with LIN-1996's three enablement patches applied, prepared
  for OSWorld, its agent server reachable at ``gym_url``. Without them the
  policy runner accepts no ``history_policy`` and discards every token id, so
  the trace carries nothing trainable.
- ``gym_sglang_proxy.py`` between Gym and this run's SGLang router: Gym reads
  the exact-trace fields off the message while SGLang puts them on the choice,
  and asks for none of them. Point Gym's ``POLICY_BASE_URL`` at the proxy.
- The prompt rows, built by ``scripts/tools/make_osworld_split.py``, which also
  writes the held-out side and the provenance this run's report cites.

Why ``samples_per_episode`` exists, and why ``--global-batch-size`` is derived
from it: an episode yields one sample per exact-trace segment, and a
data-dependent count cannot be trained on an image batch today. The schedule
that counts ``--global-batch-size`` in rollouts rather than samples is switched
off for any batch carrying ``multimodal_train_inputs``
(``train_data_conversion.py:381``) and needs a schedule config the FSDP backend
never publishes, so the legacy sample-strided split runs instead and
floor-divides the shard into steps, leaving its tail rows without log-probs
while they keep their rewards. Fixing the count makes that division exact. Set
it to OSWorld's step cap and it is an upper bound by construction, since an
episode cannot produce more segments than model calls.

Args:
    model_name, model_dir: where the HF checkpoint lives; ``prepare`` downloads it.
    prompt_data, eval_prompt_data: the training and held-out splits.
    gym_url: NeMo Gym's agent server, reachable from the train actors.
    samples_per_episode: OSWorld's step cap; fixes each episode's sample count.
    rollout_batch_size, n_samples_per_prompt: the GRPO group geometry.
    history_policy: recorded for the run report; Gym reads it from its own config.
    num_rollout: optimizer steps.
    mode: ``normal`` trains; ``smoke`` is two episodes, rollout only.

Usage:
    python3 scripts/run_qwen3_vl_4b_osworld.py --gym-url http://10.0.0.1:11001
    python3 scripts/run_qwen3_vl_4b_osworld.py --mode smoke --gym-url http://10.0.0.1:11001
"""

import os
from dataclasses import dataclass, field
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

HF_REPO = "Qwen/Qwen3-VL-4B-Instruct"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    num_gpus_per_node: int = 4
    model_name: str = "Qwen3-VL-4B-Instruct"
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    prompt_data: str = ""
    eval_prompt_data: str = ""
    gym_url: str = "http://127.0.0.1:11001"
    samples_per_episode: int = 15
    max_sandboxes: int = 8
    episode_timeout_s: int = 2400
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 4
    num_rollout: int = 20
    # Recorded so the run report and the checkpoint agree on which screenshot
    # window produced them; Gym itself reads this from its own env.yaml.
    history_policy: str = "hysteresis:1:5"
    mode: Literal["normal", "smoke"] = "normal"
    wandb_project: str = "miles-cua-osworld"
    extra_args: str = ""
    # Read at call time, never at import: a value captured at import leaks one
    # developer's export into every snapshot.
    wandb_enabled: bool = field(default_factory=lambda: bool(os.environ.get("WANDB_API_KEY")))

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
        """Samples in one optimizer step, derived so the batch divides exactly.

        One step per rollout batch is the point: at any other value the legacy
        sample-strided split floor-divides the shard and drops its tail rows
        from the forward pass while they keep their rewards.
        """
        return self.episodes_per_step * self.samples_per_episode

    @property
    def train_prompt_data(self) -> str:
        return self.prompt_data or f"{self.data_dir}/osworld/train.jsonl"

    @property
    def held_out_prompt_data(self) -> str:
        return self.eval_prompt_data or f"{self.data_dir}/osworld/held_out.jsonl"


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir}")
    U.exec_command_cpu(f"hf download {HF_REPO} --local-dir {args.model_path}")


def execute(args: ScriptArgs):
    U = args.create_backend()
    smoke = args.mode == "smoke"

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
        f"--rollout-batch-size {1 if smoke else args.rollout_batch_size} "
        f"--n-samples-per-prompt {2 if smoke else args.n_samples_per_prompt} "
        # A 15-turn episode at ~2,040 tokens of screenshot per turn reaches
        # ~11k in its longest segment, and the converter drops anything over
        # this bound rather than letting it reach the forward pass.
        "--rollout-max-response-len 12288 "
        "--rollout-max-context-len 16384 "
        "--rollout-temperature 1.0 "
        f"--global-batch-size {args.global_batch_size} "
    )

    # Held-out only: the training split never appears here, so a reported
    # before/after is not measured on what the policy trained on.
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
    )

    # A rollout batch of long multi-image episodes is GBs of pixel tensors;
    # dispatching it keeps rank 0 past the default 10-minute NCCL timeout while
    # the other ranks wait in their first collective, and the watchdog then
    # kills the job at the end of step one.
    train_backend_args = (
        "--train-backend fsdp "
        "--gradient-checkpointing "
        "--update-weight-buffer-size 536870912 "
        "--distributed-timeout-minutes 60 "
        "--attn-implementation flash_attention_3 "
    )

    # No tool-call parser on purpose: the policy emits Qwen's <tool_call> blocks
    # as text and OSWorld's own agent parses them, so a parser would strip what
    # that agent is waiting for.
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        "--sglang-attention-backend fa3 "
    )

    # No --wandb-key on purpose: the backend prints the full command line, so
    # the trainer picks WANDB_API_KEY out of its inherited environment.
    wandb_args = (
        f"--use-wandb --wandb-project {args.wandb_project} --wandb-group {args.model_name}-osworld "
        if args.wandb_enabled
        else ""
    )

    telemetry_args = "--use-miles-dashboard --dashboard-gpu-sample-interval 5 --log-multi-turn --log-passrate "

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        f"--save {args.output_dir}/ckpt "
        "--save-interval 5 "
        f"--dump-details {args.output_dir}/dump "
        + ("--debug-rollout-only " if smoke else "")
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{eval_args} "
            f"{custom_args} "
            f"{grpo_args} "
            f"{optimizer_args} "
            f"{wandb_args} "
            f"{train_backend_args} "
            f"{sglang_args} "
            f"{telemetry_args} "
            f"{misc_args} "
            f"{args.extra_args} "
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=None,
        extra_env_vars={
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            # The converter's knobs reach the train actors here rather than as
            # flags, because Miles's training entry parses argv strictly and
            # each of them is a launch constraint rather than a per-request one.
            "MILES_SCRIPT_GYM_URL": args.gym_url,
            "MILES_SCRIPT_OSWORLD_SAMPLES_PER_EPISODE": str(args.samples_per_episode),
            "MILES_SCRIPT_OSWORLD_MAX_SANDBOXES": str(args.max_sandboxes),
            "MILES_SCRIPT_OSWORLD_EPISODE_TIMEOUT_S": str(args.episode_timeout_s),
            "MILES_SCRIPT_OSWORLD_HISTORY_POLICY": args.history_policy,
        },
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
