from typing import Any

from miles.utils.pydantic_utils import FrozenStrictBaseModel

SESSION_IMAGE_SUPPORT_OFF = "off"
SESSION_IMAGE_SUPPORT_QWEN3_VL = "qwen3_vl"


def validate_session_image_args(args) -> None:
    """Refuse at startup the combinations the session image path cannot serve."""
    support = getattr(args, "session_image_support", SESSION_IMAGE_SUPPORT_OFF)
    if support == SESSION_IMAGE_SUPPORT_OFF:
        return
    if args.use_session_server != "v2":
        raise ValueError(
            f"--session-image-support={support} requires --use-session-server v2; "
            "image trajectories need the tree server's branch-per-rewrite assembly"
        )
    if not args.hf_checkpoint:
        raise ValueError(f"--session-image-support={support} requires --hf-checkpoint to load the processor")
    replays = [
        flag
        for enabled, flag in (
            (args.use_rollout_routing_replay, "--use-rollout-routing-replay"),
            (args.use_rollout_indexer_replay, "--use-rollout-indexer-replay"),
        )
        if enabled
    ]
    if replays:
        raise ValueError(
            f"--session-image-support={support} does not support {', '.join(replays)}: those per-token "
            "rows are sized on the sequence the engine expanded, while the session server's offsets are "
            "computed on the unexpanded render, so the two silently misalign"
        )


class SessionServerConfig(FrozenStrictBaseModel):
    host: str
    port: int
    instance_id: str | None
    backend_url: str
    timeout: float | None
    hf_checkpoint: str | None
    chat_template_path: str | None
    tito_model: str
    apply_chat_template_kwargs: dict[str, Any] | None
    use_rollout_routing_replay: bool
    use_rollout_indexer_replay: bool
    use_sampling_support_replay: bool
    sglang_speculative_algorithm: str | None
    num_layers: int | None
    moe_router_topk: int | None
    save_debug_trajectory_data: str | None
    lora_rank: int
    lora_adapter_path: str | None
    lora_train_only: bool
    use_session_server: bool | str | None
    session_image_support: str
    session_image_engine_url: str | None
    session_image_text_only_leaf: str
    session_message_matcher: str
    pause_generation_mode: str | None
    session_sample_picker_path: str | None
    session_sample_postprocessor_path: str | None
    rollout_top_logprobs_num: int
    rollout_sampling_logprobs_mode: str
    rollout_temperature: float


def compute_session_server_config(
    args, *, host: str, port: int, instance_id: str | None, backend_url: str
) -> SessionServerConfig:
    return SessionServerConfig(
        host=host,
        port=port,
        instance_id=instance_id,
        backend_url=backend_url,
        timeout=args.miles_router_timeout,
        hf_checkpoint=args.hf_checkpoint,
        chat_template_path=args.chat_template_path,
        tito_model=args.tito_model,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        use_rollout_routing_replay=args.use_rollout_routing_replay,
        use_rollout_indexer_replay=args.use_rollout_indexer_replay,
        use_sampling_support_replay=args.use_sampling_support_replay,
        sglang_speculative_algorithm=args.sglang_speculative_algorithm,
        num_layers=getattr(args, "num_layers", None),
        moe_router_topk=getattr(args, "moe_router_topk", None),
        save_debug_trajectory_data=args.save_debug_trajectory_data,
        lora_rank=args.lora_rank,
        lora_adapter_path=args.lora_adapter_path,
        lora_train_only=getattr(args, "lora_train_only", False),
        use_session_server=getattr(args, "use_session_server", None),
        session_image_support=getattr(args, "session_image_support", SESSION_IMAGE_SUPPORT_OFF),
        session_image_engine_url=getattr(args, "session_image_engine_url", None),
        session_image_text_only_leaf=getattr(args, "session_image_text_only_leaf", "drop"),
        session_message_matcher=getattr(args, "session_message_matcher", "strict"),
        pause_generation_mode=getattr(args, "pause_generation_mode", None),
        session_sample_picker_path=getattr(args, "session_sample_picker_path", None),
        session_sample_postprocessor_path=getattr(args, "session_sample_postprocessor_path", None),
        rollout_top_logprobs_num=args.rollout_top_logprobs_num,
        rollout_sampling_logprobs_mode=args.rollout_sampling_logprobs_mode,
        rollout_temperature=getattr(args, "rollout_temperature", 1.0),
    )
