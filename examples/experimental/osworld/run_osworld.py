"""OSWorld computer-use GRPO launcher (Qwen3-VL-4B-Instruct, FSDP, single node).

Modelled on the HUD launcher next door, which is itself the CI-verified
qwen3_vl FSDP recipe; the differences are all consequences of NeMo Gym owning
the episode instead of HUD:

- **No tool-call parser.** The policy emits Qwen's ``<tool_call>`` blocks as
  text and OSWorld's own agent parses them, so ``--sglang-tool-call-parser``
  would strip what the agent is waiting for.
- **The generate function returns a list.** One sample per exact-trace segment,
  siblings sharing a ``rollout_id``, so the group's reward is normalized per
  episode rather than per sample.
- **Prompts are task rows**, not real prompts: the instruction reaches the
  policy through Gym, inside the environment.

Gym runs as its own process group and is reached over HTTP, so it does not have
to live on this node -- only ``MILES_SCRIPT_GYM_URL`` has to resolve from the
train actors. The OSWorld VMs run wherever that Gym is.

Env knobs:
  MILES_SCRIPT_MODEL_NAME   default Qwen3-VL-4B-Instruct
  MILES_SCRIPT_MODEL_DIR    where the HF checkpoint lives
  MILES_SCRIPT_NUM_GPUS     default 4
  MILES_SCRIPT_NUM_ROLLOUT  default 20
  MILES_SCRIPT_GYM_URL      Gym agent server, default http://127.0.0.1:11001
  MILES_SCRIPT_MODE         normal | debug_rollout_only | smoke (2 episodes)
  MILES_SCRIPT_OUTPUT_DIR   checkpoints and dumps; use storage outliving the node

MODE=smoke is the rung for checking the plumbing: two episodes find the
per-episode bugs in minutes, where a full batch takes half an hour. Reward
spread within a group and weight sync need real training, so go to normal next
rather than widening smoke.
"""

import os

from miles.utils.external_utils import command_utils

MODEL_NAME = os.environ.get("MILES_SCRIPT_MODEL_NAME", "Qwen3-VL-4B-Instruct")
MODEL_DIR = os.environ.get("MILES_SCRIPT_MODEL_DIR", "/personal/models")
NUM_GPUS = int(os.environ.get("MILES_SCRIPT_NUM_GPUS", "4"))
NUM_ROLLOUT = int(os.environ.get("MILES_SCRIPT_NUM_ROLLOUT", "20"))
GYM_URL = os.environ.get("MILES_SCRIPT_GYM_URL", "http://127.0.0.1:11001")
MODE = os.environ.get("MILES_SCRIPT_MODE", "normal")
OUTPUT_DIR = os.environ.get("MILES_SCRIPT_OUTPUT_DIR", "/personal/lin1996/osworld-rl")
PROMPT_DATA = os.environ.get("MILES_SCRIPT_PROMPT_DATA", "/personal/lin1996/osworld_train.jsonl")
MAX_TOKENS_PER_GPU = int(os.environ.get("MILES_SCRIPT_MAX_TOKENS_PER_GPU", "24576"))

SMOKE = MODE == "smoke"
# 2 prompts x 4 samples: the group size is what gives GRPO its within-group
# reward spread, so it is a training parameter, not a throughput one. At the
# measured ~35% task success a group of 4 carries spread most of the time.
BATCH_SIZE, SAMPLES_PER_PROMPT = (1, 2) if SMOKE else (2, 4)


def preflight() -> dict[str, str]:
    """Check Gym answers before any GPU is claimed.

    An unreachable Gym would otherwise first appear inside an episode, where
    the sample is written off and the rollout loop refills -- silent churn
    burning GPUs rather than an error.
    """
    import urllib.error  # noqa: PLC0415 - launch-time only
    import urllib.request  # noqa: PLC0415

    try:
        with urllib.request.urlopen(f"{GYM_URL.rstrip('/')}/health", timeout=15) as response:
            if response.status != 200:
                raise SystemExit(f"Gym at {GYM_URL} answered /health with {response.status}")
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(
            f"Gym is not reachable at {GYM_URL} ({exc}). Start its control stack and, when it runs on "
            "another host, relay the agent server onto an address the train actors can reach."
        ) from exc
    print(f"osworld: Gym agent server ready at {GYM_URL}")
    return {"MILES_SCRIPT_GYM_URL": GYM_URL}


def execute() -> None:
    U = command_utils.default_config().create_backend()

    ckpt_args = (
        f"--hf-checkpoint {MODEL_DIR}/{MODEL_NAME} "
        f"--ref-load {MODEL_DIR}/{MODEL_NAME} "
    )

    rollout_args = (
        f"--prompt-data {PROMPT_DATA} "
        "--input-key prompt "
        "--label-key label "
        # carries the Gym task row onto Sample.metadata['osworld_task']
        "--metadata-key metadata "
        "--apply-chat-template "
        "--rollout-shuffle "
        f"--num-rollout {1 if SMOKE else NUM_ROLLOUT} "
        f"--rollout-batch-size {BATCH_SIZE} "
        f"--n-samples-per-prompt {SAMPLES_PER_PROMPT} "
        # A 15-turn episode at ~2k tokens of screenshot per turn reaches ~11k
        # in its longest segment; the context bound has to clear that.
        "--rollout-max-response-len 12288 "
        "--rollout-max-context-len 16384 "
        "--rollout-temperature 1.0 "
        f"--global-batch-size {BATCH_SIZE * SAMPLES_PER_PROMPT} "
    )

    custom_args = (
        "--custom-generate-function-path examples.experimental.osworld.rollout.generate "
        "--custom-rm-path examples.experimental.osworld.rollout.reward_func "
    )

    # A rollout batch of long multi-image episodes is GBs of pixel tensors;
    # dispatching it keeps rank 0 past the default 10-minute NCCL timeout while
    # the other ranks wait in their first collective, and the watchdog then
    # kills the job at the end of step one.
    fsdp_args = (
        "--train-backend fsdp --gradient-checkpointing --update-weight-buffer-size 536870912 "
        "--distributed-timeout-minutes 60 "
    )

    # Deliberately NOT --use-dynamic-global-batch-size: `postprocess_rollout_data`
    # only produces `dynamic_global_batch_size` for a non-compact batch, and
    # setting any `rollout_id` makes this batch compact, so the flag trips
    # `assert args.use_dynamic_global_batch_size == (x is not None)` instead of
    # reconciling anything. The rollout-side schedule that would count the
    # global batch in rollouts is unreachable here on two counts:
    # `can_schedule_on_rollout_side` returns False for any batch carrying
    # `multimodal_train_inputs`, and FSDP publishes only `dp_size` where
    # `has_full_schedule_config` wants four keys. So the converter emits one
    # sample per episode and the legacy sample-strided split divides evenly.
    dynamic_batch_args = ""

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

    telemetry_args = "--use-miles-dashboard --dashboard-gpu-sample-interval 5 --log-multi-turn --log-passrate "

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        "--sglang-attention-backend fa3 "
        "--attn-implementation flash_attention_3 "
    )

    debug_args = "--debug-rollout-only " if MODE in ("debug_rollout_only", "smoke") else ""

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        f"--save {OUTPUT_DIR}/ckpt "
        "--save-interval 5 "
        f"--dump-details {OUTPUT_DIR}/dump "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{custom_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{fsdp_args} "
        f"{dynamic_batch_args} "
        f"{sglang_args} "
        f"{telemetry_args} "
        f"{debug_args} "
        f"{misc_args} "
    )

    extra_env_vars = {"CUDA_DEVICE_MAX_CONNECTIONS": "1", **preflight()}
    if os.environ.get("WANDB_API_KEY"):
        extra_env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
        extra_env_vars=extra_env_vars,
    )


if __name__ == "__main__":
    execute()
