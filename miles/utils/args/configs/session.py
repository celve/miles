from typing import ClassVar

from miles.rollout.session.types import SessionServerInstance
from miles.utils.args.schema import A, Arg, BaseConfig
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType


class SessionConfig(BaseConfig):
    _mutable_fields: ClassVar[frozenset[str]] = frozenset({"session_server_instances"})

    session_server_instances: list[SessionServerInstance] | None = None
    use_session_server: A[
        str | bool,
        Arg(
            type_parser=None,
            nargs="?",
            const=True,
            help="Start a standalone session server for TITO/session support. "
            "Requires --hf-checkpoint. A named --tito-model resolves its registered template; "
            "--tito-model=default uses the checkpoint-native or explicit --chat-template-path template. "
            "Bare flag (or 'v1') selects the append-only linear v1 server; "
            "'--use-session-server v2' selects the tree-serving v2 "
            "(multi-lineage trajectories, always-branch).",
        ),
    ] = False
    session_server_workers: A[int, Arg(help="Number of session server instances.")] = 32
    session_server_ip: A[
        str | None,
        Arg(
            help=(
                "Address the session servers bind to, e.g. 0.0.0.0 to accept traffic from outside "
                "the cluster. Peers still reach them on the address their worker was placed on. "
                "Defaults to that placed address."
            )
        ),
    ] = None
    session_server_external_host: A[
        str | None,
        Arg(
            help=(
                "Host that peers outside the cluster, such as agents in a sandbox, reach every session "
                "server on. Setting it keeps all session servers on the head node, so it must reach the head. "
                "Leave it unset when each node sets MILES_NODE_EXTERNAL_IP to its own reachable address, or "
                "when the placed addresses already route from outside."
            )
        ),
    ] = None
    session_server_port: A[
        int | None,
        Arg(
            help=(
                "Base port for the session servers, so a network policy can whitelist a known range. "
                "Instance i listens on this port plus i. Defaults to a dynamically allocated port."
            )
        ),
    ] = None
    tito_model: A[
        str,
        Arg(
            choices=[t.value for t in TITOTokenizerType],
            help="TITO tokenizer type for pretokenized prefix reuse. "
            "Controls how token IDs are computed for messages appended after "
            "the pretokenized prefix in multi-turn agentic sessions.",
        ),
    ] = "default"
    use_session_images: A[
        bool,
        Arg(
            help="v2 only. Carry images through the session server so a screenshot harness can "
            "train through the agent-function interface. Off by default, which keeps the text-only "
            "behavior and rejects image content. On, the session server loads the checkpoint's processor, "
            "sends image-bearing turns through the router's chat route like text turns, and assembles "
            "pixel_values / image_grid_thw per retained sample; it is refused with "
            "--multimodal-token-space-processing, which carries session media its own way. Requires "
            "--use-session-server v2, "
            "--tito-model qwen3_vl (the model family), an SGLang whose pretokenized chat branch keeps "
            "message images, and a checkpoint whose processor produces image grids; incompatible with the "
            "routing and indexer replay side channels, whose per-token rows are sized on the expanded "
            "sequence. Checked at startup against the args Miles launches the policy engines with: every "
            "engine group must serve --hf-checkpoint unpinned, loading its processor from it, with no "
            "mm_process_config and image_processor_backend 'auto'; external engines and "
            "prefill/decode disaggregation are refused.",
        ),
    ] = False
    session_image_text_only_samples: A[
        str,
        Arg(
            choices=["drop", "keep"],
            help="v2 only, with --use-session-images. What to do with a training sample whose "
            "retained tokens carry no image, such as one built from a text-only summary call. "
            "'drop' (default) removes it: under FSDP the vision tower's weight all-gathers are "
            "collectives, so a text-only sample in an image-bearing batch stalls the ranks that "
            "issued them. 'keep' emits it, for a trainer that tolerates mixed modality. A "
            "placeholder image is never injected into a real sampled context either way.",
        ),
    ] = "drop"
    session_message_matcher: A[
        str,
        Arg(
            help=(
                "Process-wide session history matcher: strict (default), "
                "loose_tool_call, role_content_only, or a trusted dotted import "
                "path. role_content_only is a high-risk opt-in that can collapse "
                "different tool-call lineages and does not reconcile call IDs."
            )
        ),
    ] = "strict"
    session_sample_picker_path: A[
        str,
        Arg(
            help=(
                "v2 only. Import path of the sample-pick hook for the "
                "session samples op: fn(leaf_samples, session_metadata) -> "
                "list[Sample], a pure selection over the per-leaf raw samples. "
                "Runs synchronously inside the session server process; long CPU "
                "work stalls every session on the instance. Default: drop_same_prompt_retries, "
                "which trims identical re-sends, including a re-sent first turn; "
                "drop_rolled_back_leaves also trims a leaf whose later sibling sent a "
                "different request."
            )
        ),
    ] = "miles.rollout.session.v2.picker_hub.drop_same_prompt_retries"
    session_sample_postprocessor_path: A[
        str,
        Arg(
            help=(
                "v2 only. Import path of the post-process hook for the "
                "session samples op: fn(leaf_samples, session_metadata) -> "
                "list[Sample], finalizing loss masks / rewards over the picked "
                "samples. Runs synchronously inside the session server process. "
                "Default: exactly-once completion masking + rewards keyed by "
                "response id."
            )
        ),
    ] = "miles.rollout.session.v2.postprocessor_hub.default_postprocess"
