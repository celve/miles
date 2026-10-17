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
    router_port: the port this run's SGLang router listens on, pinned so Gym can
        be configured before the job starts. The router is pinned to the Ray
        head rather than given an address, so the proxy forwards to that host.
    samples_per_episode: OSWorld's step cap; fixes each episode's sample count.
    max_output_tokens: Gym's per-call generation cap, the value the variance
        probe ran at. A task row's own value wins.
    rollout_batch_size, n_samples_per_prompt: the GRPO group geometry.
    micro_batch_size: must divide each rank's share of the global batch.
    num_rollout: optimizer steps.
    mode: ``normal`` trains; ``smoke`` is two episodes, rollout only.

Usage:
    python3 scripts/run_qwen3_vl_4b_osworld.py --gym-url http://10.0.0.1:11001
    python3 scripts/run_qwen3_vl_4b_osworld.py --mode smoke --gym-url http://10.0.0.1:11001
"""

import shlex
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.common import ArgvManipulator

HF_REPO = "Qwen/Qwen3-VL-4B-Instruct"
# Pinned, not floating: a recipe that documents one revision while its own
# download command fetches whatever HEAD is cannot be reproduced.
HF_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # Import-time, not the base's call-time factory: the snapshot harness
    # freezes `command_utils.create_run_id`, so only a value taken through that
    # name records deterministically.
    run_id: str = command_utils.create_run_id()
    num_gpus_per_node: int = 4
    model_name: str = "Qwen3-VL-4B-Instruct"
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    prompt_data: str = ""
    eval_prompt_data: str = ""
    gym_url: str = "http://127.0.0.1:11001"
    # The port is pinned and the router is pinned to the Ray head, because Gym
    # is a separate process group that has to reach the policy and is
    # configured before this job exists. The *address* is deliberately not
    # preset: `router_manager.resolve_router_addrs` treats a non-None
    # `sglang_router_ip` as an already-resolved address and asserts that the
    # per-model map exists, so presetting it fails the run before it contacts
    # Gym. Pinning placement gives a stable host without writing into Miles's
    # resolved state; the port is safe to pin because an occupied static port
    # is rejected loudly rather than silently reassigned.
    router_port: int = 31000
    samples_per_episode: int = 15
    micro_batch_size: int = 1
    max_sandboxes: int = 8
    episode_timeout_s: int = 2400
    max_output_tokens: int = 1500
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 4
    num_rollout: int = 20
    mode: Literal["normal", "smoke"] = "normal"
    extra_args: str = ""

    def __post_init__(self):
        # A fixed sample count is necessary but not sufficient: the legacy
        # sample-strided split gives ranks unequal shards when the total does
        # not divide, and the train side then floor-divides its own shard --
        # leaving rows without log-probs while they keep their rewards, which
        # is an IndexError in get_grpo_returns rather than a clear refusal.
        super_post = getattr(super(), "__post_init__", None)
        if super_post is not None:
            super_post()
        if self.mode == "smoke":
            # Smoke runs rollout-only, so no optimizer step consumes the batch
            # and the division below is not a constraint on it.
            return
        total = self.global_batch_size
        dp_size = self.num_nodes * self.num_gpus_per_node
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


# Flags whose value the whole run has to agree on: each fixes the batch
# geometry or the episode's sample count, and the two cannot be set apart
# without leaving rows unscored. launch-and-request-args.md calls these launch
# constraints, which a request may not change.
_PROTECTED_FLAGS = (
    "--global-batch-size",
    "--micro-batch-size",
    "--rollout-batch-size",
    "--n-samples-per-prompt",
    "--actor-num-nodes",
    "--actor-num-gpus-per-node",
    "--num-gpus-per-node",
)
_PROTECTED_ENV = ("MILES_SCRIPT_OSWORLD_SAMPLES_PER_EPISODE",)


def _refuse_geometry_overrides(train_args: str, args: ScriptArgs, launcher_env: dict[str, str]) -> None:
    """Check the final effective argv and environment with their consumers' own semantics.

    ``__post_init__`` sees the dataclass before ``--extra-args`` is appended to
    the argv and before ``--extra-env-vars`` is merged over the launcher's own
    environment, and both of those win last. So a geometry the constructor
    accepted can still reach the trainer changed, and the first sign of it is
    an unscored tail in every shard.

    **Correctness does not rest on this function.** The protected flags are
    emitted once, after ``--extra-args``, so the launcher's declaration already
    wins by position whatever a user writes. This refuses the attempt anyway,
    because a flag that is silently ignored is worse than one that errors.

    Tokenization is ``shlex.split``, matching what the backend itself does
    (``base_backend.py:177``), so a quoted spelling is seen the way argparse
    will see it. ``resolve_extra_env_vars`` performs the real environment merge
    so the *effective* value is what gets compared, covering the JSON form as
    well as ``NAME=value``.
    """
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


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir}")
    U.exec_command_cpu(f"hf download {HF_REPO} --revision {HF_REVISION} --local-dir {args.model_path}")


def execute(args: ScriptArgs):
    U = args.create_backend()
    smoke = args.mode == "smoke"

    # Emitted last, after --extra-args, and nowhere else. argparse's store
    # action overwrites, so the launcher's declaration is the effective one
    # whatever spelling a user writes -- an `=` form, a quoted token or an
    # abbreviation, which resolves to the same destination. Policing the
    # override was tried twice and defeated twice by spellings nobody had
    # enumerated; winning by position needs no enumeration at all.
    # execute_train's num_gpus_per_node argument only sets ray's resource
    # count, so train.py's own flag belongs here too.
    geometry_args = (
        f"--rollout-batch-size {1 if smoke else args.rollout_batch_size} "
        f"--n-samples-per-prompt {2 if smoke else args.n_samples_per_prompt} "
        f"--global-batch-size {args.global_batch_size} "
        f"--micro-batch-size {args.micro_batch_size} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
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
        # A 15-turn episode at ~2,040 tokens of screenshot per turn reaches
        # ~11k in its longest segment, and the converter drops anything over
        # this bound rather than letting it reach the forward pass.
        "--rollout-max-response-len 12288 "
        "--rollout-max-context-len 16384 "
        "--rollout-temperature 1.0 "
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
        # Explicit, because the backend derives train_iters as
        # num_rollout * rollout_batch_size * n_samples_per_prompt //
        # global_batch_size, and this recipe's padded global batch makes that
        # num_rollout // samples_per_episode -- which is 1 at the defaults and
        # 0 below a cap's worth of rollouts, where the scheduler asserts on a
        # zero decay horizon before any update runs. The recipe's own invariant
        # is one update per rollout batch, so that is the horizon.
        f"--lr-decay-iters {args.num_rollout} "
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
        # Infrastructure exclusions must not land in the task-score denominator:
        # the default logger turns a null reward into a zero, so one aborted
        # sandbox would halve a two-task held-out score.
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
    # The converter's knobs reach the train actors here rather than as flags,
    # because Miles's training entry parses argv strictly and each of them is a
    # launch constraint rather than a per-request one.
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
        megatron_model_type=None,
        extra_env_vars=launcher_env,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
