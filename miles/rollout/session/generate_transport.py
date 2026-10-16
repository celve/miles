"""Serve an image-bearing turn through the engine's native ``/generate`` route.

The pretokenized branch of ``/v1/chat/completions`` builds its
``MessageProcessingResult`` with ``image_data=None``, so a chat request that
carries ``input_ids`` can never carry pixels. ``/generate`` takes both, plus
``mm_content_hashes`` and ``return_prompt_token_ids``, so image turns go there
and text turns keep the chat route byte for byte.

What comes back is a bare generation, so this module rebuilds the chat envelope
the session server and every downstream sample consumer expect. Reasoning and
tool-call text is split by SGLang's own parsers, named by the request's TITO
family, rather than by a second implementation of those formats here.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from miles.rollout.session.errors import MessageValidationError, UpstreamResponseError
from miles.rollout.session.media import RequestMedia

logger = logging.getLogger(__name__)

# Chat fields that become sampling_params on the native route, under the same name.
_SAMPLING_PASSTHROUGH = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "stop",
    "stop_token_ids",
    "skip_special_tokens",
    "no_stop_trim",
    "json_schema",
    "regex",
    "ebnf",
    "structural_tag",
)
_SAMPLING_RENAMES = {"max_tokens": "max_new_tokens", "max_completion_tokens": "max_new_tokens", "seed": "seed"}
# Request-level fields the native route reads directly rather than through sampling_params.
_REQUEST_PASSTHROUGH = ("return_sampling_mask", "sampling_logprobs_mode", "lora_path", "extra_key")

# Chat features whose semantics live in the chat serving layer; refusing beats
# serving them differently on this route.
_UNSUPPORTED_WITH_IMAGES = {
    "n": lambda value: value not in (None, 1),
    "best_of": lambda value: value not in (None, 1),
    "logit_bias": lambda value: bool(value),
    "response_format": lambda value: value is not None,
}


@dataclass(frozen=True)
class CompletionParsers:
    """The reasoning and tool-call parsers this model family uses."""

    reasoning_parser: str | None
    tool_call_parser: str | None
    tokenizer: Any


def validate_image_request_features(request_body: dict[str, Any]) -> None:
    """Refuse the chat features the native route would otherwise serve differently."""
    for field, is_set in _UNSUPPORTED_WITH_IMAGES.items():
        if is_set(request_body.get(field)):
            raise MessageValidationError(
                f"{field}={request_body.get(field)!r} is not supported for an image-bearing request: "
                "those turns are served through the engine's native generate route, which does not "
                "implement the chat layer's handling of this field"
            )
    tool_choice = request_body.get("tool_choice")
    if tool_choice not in (None, "none", "auto"):
        raise MessageValidationError(
            f"tool_choice={tool_choice!r} is not supported for an image-bearing request: a forced or "
            "named tool call needs the grammar constraint the chat serving layer builds"
        )


def build_generate_body(
    request_body: dict[str, Any],
    *,
    image_data: list[str],
    content_hashes: list[str],
) -> dict[str, Any]:
    """Translate a resolved chat request into a native generate request."""
    sampling_params: dict[str, Any] = {}
    for field in _SAMPLING_PASSTHROUGH:
        if (value := request_body.get(field)) is not None:
            sampling_params[field] = value
    for field, target in _SAMPLING_RENAMES.items():
        if (value := request_body.get(field)) is not None:
            sampling_params.setdefault(target, value)

    body: dict[str, Any] = {
        "input_ids": request_body["input_ids"],
        "sampling_params": sampling_params,
        "return_logprob": True,
        # The effective ids: what the engine expanded our placeholders into.
        "return_prompt_token_ids": True,
    }
    if (top_logprobs := request_body.get("top_logprobs")) is not None:
        body["top_logprobs_num"] = top_logprobs
    for field in _REQUEST_PASSTHROUGH:
        if (value := request_body.get(field)) is not None:
            body[field] = value
    if image_data:
        body["image_data"] = image_data
        body["mm_content_hashes"] = content_hashes
    return body


def assert_effective_prompt_ids(
    response: dict[str, Any], expected_ids: list[int], media: RequestMedia, counts: list[int]
) -> None:
    """Compare the ids inference consumed against the ids we meant it to consume.

    An image-token count can match while pixels, order, or resize settings do not;
    comparing the sequences settles all of them at once.
    """
    effective = response.get("prompt_token_ids")
    if effective is None:
        raise UpstreamResponseError(
            "generate response carries no prompt_token_ids; the session image path needs "
            "return_prompt_token_ids to prove the ids inference consumed"
        )
    if list(effective) == expected_ids:
        return
    first_diff = next(
        (i for i, (a, b) in enumerate(zip(effective, expected_ids)) if a != b),
        min(len(effective), len(expected_ids)),
    )
    raise UpstreamResponseError(
        f"effective prompt ids differ from the rendered expansion: engine has {len(effective)} tokens, "
        f"the session server expected {len(expected_ids)} (first difference at index {first_diff}); "
        f"media={list(media.asset_ids)} counts={counts} — the engine's processor settings or image "
        "decoding disagree with this run's pinned processor"
    )


def adapt_generate_response(
    result: dict[str, Any],
    response: dict[str, Any],
    *,
    model: str,
    tools: list[dict[str, Any]] | None,
    parsers: CompletionParsers,
    separate_reasoning: bool,
) -> dict[str, Any]:
    """Rebuild the chat-completion envelope the session server records and replies with."""
    meta_info = response.get("meta_info")
    if not isinstance(meta_info, dict):
        raise UpstreamResponseError("generate response carries no meta_info object")
    text = response.get("text")
    if text is None:
        raise UpstreamResponseError("generate response carries no text")

    finish_reason, matched = _finish_reason(meta_info)
    reasoning_content = None
    if separate_reasoning and parsers.reasoning_parser:
        reasoning_content, text = _split_reasoning(text, parsers)
    tool_calls = None
    if tools and parsers.tool_call_parser:
        text, tool_calls = _split_tool_calls(text, tools, parsers)
        if tool_calls:
            finish_reason = "tool_calls"

    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = tool_calls
    choice: dict[str, Any] = {
        "index": 0,
        "message": message,
        "finish_reason": finish_reason,
        "meta_info": meta_info,
    }
    if matched is not None:
        choice["matched_stop"] = matched
    prompt_tokens = int(meta_info.get("prompt_tokens") or 0)
    completion_tokens = len(meta_info.get("output_token_logprobs") or [])
    return {
        "id": meta_info.get("id") or "",
        "object": "chat.completion",
        "created": int(result.get("created") or time.time()),
        "model": model,
        "choices": [choice],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _finish_reason(meta_info: dict[str, Any]) -> tuple[str, Any]:
    """Map the native route's finish-reason object onto the chat route's string."""
    raw = meta_info.get("finish_reason")
    if raw is None:
        return "", None
    if isinstance(raw, str):
        return raw, None
    reason = raw.get("type") or ""
    return reason, raw.get("matched")


def _split_reasoning(text: str, parsers: CompletionParsers) -> tuple[str | None, str]:
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    parser = ReasoningParser(
        model_type=parsers.reasoning_parser,
        stream_reasoning=False,
        tokenizer=parsers.tokenizer,
        tool_call_parser_active=bool(parsers.tool_call_parser),
    )
    return parser.parse_non_stream(text)


def _split_tool_calls(
    text: str, tools: list[dict[str, Any]], parsers: CompletionParsers
) -> tuple[str, list[dict[str, Any]] | None]:
    from sglang.srt.entrypoints.openai.protocol import Tool
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    parser = FunctionCallParser(
        [Tool.model_validate(tool) for tool in tools], parsers.tool_call_parser, tokenizer=parsers.tokenizer
    )
    if not parser.has_tool_call(text):
        return text, None
    remaining, call_info_list = parser.parse_non_stream(text)
    if not call_info_list:
        logger.warning("Tool call marker present but no complete call parsed by %s", parsers.tool_call_parser)
        return remaining, None
    return remaining, [
        {
            "id": f"call_{index}_{info.name}",
            "index": index,
            "type": "function",
            "function": {"name": info.name, "arguments": info.parameters},
        }
        for index, info in enumerate(call_info_list)
    ]
