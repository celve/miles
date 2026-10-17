from typing import Any

from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.utils.pydantic_utils import FrozenStrictBaseModel

SESSION_IMAGE_SUPPORT_OFF = "off"
SESSION_IMAGE_SUPPORT_QWEN3_VL = "qwen3_vl"

# The engine launch settings that decide what an image becomes and how a reply is parsed.
ENGINE_LAUNCH_SETTING_FIELDS = (
    "revision",
    "tokenizer_path",
    "mm_process_config",
    "image_processor_backend",
    "disable_fast_image_processor",
    "sampling_defaults",
    "tool_call_parser",
    "reasoning_parser",
)


def validate_session_image_args(args) -> None:
    """Refuse at startup the combinations the session image path cannot serve.

    Needs the derived engine topology (``rollout_external``, ``starts_inference_engines``),
    so it runs at the end of ``miles_validate_args``.
    """
    support = getattr(args, "session_image_support", SESSION_IMAGE_SUPPORT_OFF)
    args.session_image_engine_settings = None
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
    args.session_image_engine_settings = resolve_engine_launch_settings(args)


def resolve_engine_launch_settings(args) -> dict[str, Any] | None:
    """The image settings every policy engine is launched with, read from this run's own args.

    Miles launches those engines itself from these args, so their settings are known before
    any engine exists and no request has to ask one. The router may hand an image turn to
    any of them, so every one must carry the settings this path checks.
    """
    support = args.session_image_support
    if args.rollout_external:
        raise ValueError(
            f"--session-image-support={support} needs engines Miles launches itself: an external engine's "
            "image preprocessing is not described by this run's args, so whether its pixels match the "
            "trainer's cannot be checked"
        )
    models = resolve_sglang_config(args).models
    if not models:
        return None
    policy = models[0]
    if policy.has_pd_disaggregation:
        raise ValueError(
            f"--session-image-support={support} does not support prefill/decode disaggregation; image turns "
            "have only been verified on regular engines"
        )
    settings = [
        _group_launch_settings(args, group.overrides)
        for group in policy.server_groups
        if group.worker_type != WorkerType.PLACEHOLDER
    ]
    differing = sorted({key for other in settings[1:] for key in other if other[key] != settings[0][key]})
    if differing:
        raise ValueError(
            f"--session-image-support={support} needs every policy engine group to launch with the same "
            f"image settings, because the router picks any of them per turn; they differ in {differing}"
        )
    check_engine_launch_settings(settings[0], checkpoint=args.hf_checkpoint)
    return settings[0]


def _group_launch_settings(args, overrides: dict[str, Any]) -> dict[str, Any]:
    # The order `_compute_server_args` launches with: a group override beats --sglang-<field>.
    settings = {
        "model_path": overrides["model_path"],
        **{name: overrides.get(name, getattr(args, f"sglang_{name}", None)) for name in ENGINE_LAUNCH_SETTING_FIELDS},
    }
    # SGLang loads the image processor from tokenizer_path, which defaults to model_path.
    settings["tokenizer_path"] = settings["tokenizer_path"] or settings["model_path"]
    return settings


def check_engine_launch_settings(settings: dict[str, Any], *, checkpoint: str) -> None:
    """Require the one engine image configuration whose pixels and parsing this path has verified."""
    missing = [key for key in ("model_path", *ENGINE_LAUNCH_SETTING_FIELDS) if key not in settings]
    if missing:
        raise ValueError(f"the session image engine settings do not state {missing}")
    if settings["model_path"] != checkpoint:
        raise ValueError(
            f"the policy engines serve {settings['model_path']!r} but --hf-checkpoint is {checkpoint!r}; "
            "the two processors would disagree about every image"
        )
    if settings["tokenizer_path"] != checkpoint:
        raise ValueError(
            f"the policy engines load their tokenizer and image processor from {settings['tokenizer_path']!r} "
            f"but --hf-checkpoint is {checkpoint!r}; that processor could normalize differently with every "
            "token id still matching"
        )
    if settings["revision"] is not None:
        raise ValueError(
            f"the policy engines pin revision {settings['revision']!r} while the session server loads the "
            "processor unpinned, so the two could resize or normalize differently with every token id "
            "still matching"
        )
    if settings["mm_process_config"]:
        raise ValueError(
            f"the policy engines launch with mm_process_config {settings['mm_process_config']!r}, which "
            "changes how they decode or normalize images; their pixels would then differ from the "
            "trainer's while every token id stayed identical"
        )
    if settings["disable_fast_image_processor"]:
        raise ValueError(
            "the policy engines launch with disable_fast_image_processor, which forces SGLang's 'pil' image "
            "processor whatever image_processor_backend says; only 'auto' has been verified to match the "
            "trainer's pixels"
        )
    if settings["image_processor_backend"] != "auto":
        raise ValueError(
            f"the policy engines launch with image_processor_backend {settings['image_processor_backend']!r}; "
            "only the default 'auto' has been verified to match the trainer's pixels"
        )
    if settings["sampling_defaults"] not in ("model", "openai"):
        raise ValueError(f"unknown sampling_defaults {settings['sampling_defaults']!r}")
    auto = [name for name in ("tool_call_parser", "reasoning_parser") if settings[name] == "auto"]
    if auto:
        raise ValueError(
            f"the policy engines launch with {auto} set to 'auto', which each engine resolves for itself at "
            "launch, so the session server cannot know which parser an image turn's reply needs; name the "
            "parser explicitly"
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
    session_image_engine_settings: dict[str, Any] | None
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
        session_image_engine_settings=getattr(args, "session_image_engine_settings", None),
        session_image_text_only_leaf=getattr(args, "session_image_text_only_leaf", "drop"),
        session_message_matcher=getattr(args, "session_message_matcher", "strict"),
        pause_generation_mode=getattr(args, "pause_generation_mode", None),
        session_sample_picker_path=getattr(args, "session_sample_picker_path", None),
        session_sample_postprocessor_path=getattr(args, "session_sample_postprocessor_path", None),
        rollout_top_logprobs_num=args.rollout_top_logprobs_num,
        rollout_sampling_logprobs_mode=args.rollout_sampling_logprobs_mode,
        rollout_temperature=getattr(args, "rollout_temperature", 1.0),
    )
