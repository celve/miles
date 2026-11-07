from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.schema import A, Arg, BaseConfig
from miles.utils.eval_config import EvalDatasetConfig


class EvalConfig(BaseConfig):
    eval_uses_snapshots: bool

    eval_prompt_data: A[
        list[str] | None,
        Arg(
            type_parser=str,
            nargs="+",
            help=(
                "Path to the evaluation prompt data, "
                "should first input the name of the eval dataset and then the path, e.g. "
                "aime /path/to/aime.jsonl"
            ),
        ),
    ] = None
    eval_config: A[
        str | None,
        Arg(
            help=(
                "Path to an OmegaConf YAML/JSON file describing evaluation datasets, or an "
                "inline `base64:<payload>` carrying the same document. "
                "When provided, this overrides --eval-prompt-data."
            )
        ),
    ] = None
    skip_eval_before_train: A[bool, Arg(help="Whether to skip evaluation before training.")] = False

    # The following keys are used to override the rollout version during eval.
    eval_input_key: A[str | None, Arg(help="JSON dataset key")] = None
    eval_label_key: A[str | None, Arg(help="JSON dataset key")] = None
    eval_tool_key: A[str | None, Arg(help="JSON dataset key")] = None
    n_samples_per_eval_prompt: A[int, Arg(help="number of responses for each prompt in generation")] = 1
    eval_temperature: A[float | None, Arg()] = None
    eval_top_p: A[float | None, Arg()] = None
    eval_top_k: A[int | None, Arg()] = None
    eval_max_response_len: A[int | None, Arg()] = None
    eval_min_new_tokens: A[int | None, Arg()] = None
    eval_server_concurrency: A[
        int | None,
        Arg(
            help=(
                "Requests in flight per dedicated eval engine (--eval-num-gpus), in place of "
                "--sglang-server-concurrency, which it defaults to. Sizes the eval rollout's "
                "concurrency and the eval router's connections, so a small eval pool can keep as "
                "many episodes going as the training engines do."
            )
        ),
    ] = None
    eval_hf_dir: A[
        str | None,
        Arg(
            help=(
                "Staging directory for per-eval HF snapshots (written to "
                "`{eval_hf_dir}/step_{rollout_id}`). Point at tmpfs (e.g. /dev/shm/...) to "
                "avoid disk. When unset and --save-hf is set, eval reuses the --save-hf "
                "checkpoints instead of exporting its own snapshots."
            )
        ),
    ] = None
    eval_max_in_flight: A[int, Arg(help="Maximum number of concurrently pending async evals.")] = 2
    eval_overflow_policy: A[
        str,
        Arg(
            choices=["backpressure", "skip"],
            help=(
                "What to do when an eval is due but --eval-max-in-flight evals are pending: "
                "'backpressure' awaits the oldest pending eval (deterministic curve, bounded "
                "stall); 'skip' drops the new eval point and logs eval/skipped_busy at that "
                "step (training cadence is never stalled)."
            ),
        ),
    ] = "backpressure"
    eval_record_dir: A[
        str | None,
        Arg(
            help=(
                "Record every evaluated episode under `{eval_record_dir}/eval_{rollout_id}/`: its instruction, "
                "each step's screenshot, model output and parsed action, and its reward (miles.utils.eval_record). "
                "Miles's eval loop hands each eval sample its directory in metadata['eval_record_dir']; the "
                "Harbor computer-1 binding and examples/experimental/cua_lite/record.py write the episodes, and "
                "scripts/tools/render_eval_records.py renders two recordings side by side."
            )
        ),
    ] = None
    eval_keep_snapshots: A[
        int,
        Arg(
            help=(
                "How many snapshot dirs to keep under --eval-hf-dir (consumed snapshots "
                "beyond this are deleted). --save-hf checkpoints are never deleted."
            )
        ),
    ] = 2


class EvalRolloutOnlyConfig(BaseConfig):
    eval_datasets: list[EvalDatasetConfig]
    eval_function_path: A[
        CustomFunctionConfig | None,
        Arg(
            help=(
                "Path to the eval fn. Two kinds fit here. A rollout fn generates against the "
                "engines the framework hands it: the training engines, or the dedicated fleet "
                "when --eval-num-gpus is set. A CheckpointEvalFn subclass gets the snapshot "
                "path instead and owns the rest itself — weight delivery, endpoint, generation. "
                "If not set, defaults to --rollout-function-path."
            ),
        ),
    ] = None

    eval_max_prompt_len: A[int | None, Arg()] = None
