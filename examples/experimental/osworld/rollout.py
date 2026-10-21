"""Multi-turn computer-use rollout on OSWorld through NeMo Gym."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PIL import Image as PILImage

from examples.experimental.osworld.media_binding import (
    MediaBindingError,
    expanded_token_counts,
    media_digests,
    processor_identity,
    verify_image_token_layout,
)
from miles.utils.types import Sample

if TYPE_CHECKING:
    from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput

logger = logging.getLogger(__name__)

_REJECTIONS = (
    "no_exact_evidence",
    "stitch_violation",
    "image_mismatch",
    "text_only_call",
    "over_context",
)


@dataclass(frozen=True)
class EpisodeConfig:
    """Run-wide knobs this converter needs, resolved once per call site."""

    gym_url: str
    samples_per_episode: int
    max_sandboxes: int
    episode_timeout_s: float
    max_output_tokens: int

    @classmethod
    def read(cls, args: Any) -> EpisodeConfig:
        def knob(name: str, env: str, default: str) -> str:
            value = getattr(args, f"osworld_{name}", None)
            return str(value) if value is not None else os.environ.get(env, default)

        samples_per_episode = int(knob("samples_per_episode", "MILES_SCRIPT_OSWORLD_SAMPLES_PER_EPISODE", "0"))
        if samples_per_episode < 1:
            raise ValueError(
                "MILES_SCRIPT_OSWORLD_SAMPLES_PER_EPISODE must be set to at least 1: an episode's sample count has "
                "to be fixed for the training batch to divide. This module's docstring says why."
            )
        gym_url = knob("gym_url", "MILES_SCRIPT_GYM_URL", "http://127.0.0.1:11001").rstrip("/")
        return cls(
            gym_url=gym_url,
            samples_per_episode=samples_per_episode,
            max_sandboxes=int(knob("max_sandboxes", "MILES_SCRIPT_OSWORLD_MAX_SANDBOXES", "8")),
            episode_timeout_s=float(knob("episode_timeout_s", "MILES_SCRIPT_OSWORLD_EPISODE_TIMEOUT_S", "2400")),
            max_output_tokens=int(knob("max_output_tokens", "MILES_SCRIPT_OSWORLD_MAX_OUTPUT_TOKENS", "1500")),
        )

_gate: asyncio.Semaphore | None = None


def _sandbox_gate(config: EpisodeConfig) -> asyncio.Semaphore:
    """Cap concurrent OSWorld VMs process-wide, independent of batch geometry."""
    global _gate
    if _gate is None:
        _gate = asyncio.Semaphore(config.max_sandboxes)
    return _gate


def _request_body(config: EpisodeConfig, task: dict, sampling_params: dict) -> dict:
    """The /run body, where a task row's own request fields win over this run's defaults."""
    create_params = dict(task.get("responses_create_params") or {})
    # Per model call: Miles's max_new_tokens is this recipe's bound on a whole stitched segment.
    create_params.setdefault("max_output_tokens", config.max_output_tokens)
    create_params.setdefault("temperature", float(sampling_params.get("temperature", 1.0)))
    if sampling_params.get("top_p") is not None:
        create_params.setdefault("top_p", float(sampling_params["top_p"]))
    create_params["metadata"] = {**(create_params.get("metadata") or {}), "nemo_rl_rollout_purpose": "training"}
    return {
        "responses_create_params": create_params,
        "verifier_metadata": task.get("verifier_metadata") or {},
        # Gym emits the exact trace only for a training-purpose run.
        "rollout_purpose": "training",
    }


async def _run_gym_task(config: EpisodeConfig, task: dict, sampling_params: dict) -> dict:
    """POST one task to Gym and return its response envelope."""
    import aiohttp  # noqa: PLC0415

    body = _request_body(config, task, sampling_params)
    timeout = aiohttp.ClientTimeout(total=config.episode_timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{config.gym_url}/run", json=body) as response:
            response.raise_for_status()
            return await response.json()


def _calls(response: dict) -> list[dict]:
    """One record per model call, joining the two arrays transport v3 splits it across."""
    outputs = response.get("output") or []
    metadata = response.get("model_call_metadata") or []
    if len(outputs) != len(metadata):
        raise ValueError(f"exact trace is inconsistent: {len(outputs)} outputs, {len(metadata)} metadata records")
    return [
        {
            "prompt_token_ids": item.get("prompt_token_ids") or [],
            "sampled_token_ids": item.get("generation_token_ids") or [],
            "sampled_logprobs": item.get("generation_log_probs") or [],
            "segment_id": meta.get("segment_id"),
            "segment_index": meta.get("segment_index") or 0,
            "media_ids": meta.get("media_ids") or [],
            "eligible": meta.get("eligible"),
        }
        for item, meta in zip(outputs, metadata, strict=True)
    ]


def _segments(calls: list[dict]) -> list[list[dict]]:
    """Group model calls into prefix-compatible runs, preserving order."""
    grouped: dict[str, list[dict]] = {}
    for call in calls:
        if call.get("eligible") is False:
            continue
        grouped.setdefault(str(call.get("segment_id")), []).append(call)
    return [calls for _, calls in sorted(grouped.items(), key=lambda kv: kv[1][0]["segment_index"])]


def _stitch(segment: list[dict]) -> tuple[list[int], list[int], list[float], int] | None:
    """One token sequence for a segment, with loss on the sampled tokens only."""
    tokens: list[int] = list(segment[0].get("prompt_token_ids") or [])
    if not tokens:
        return None
    response_start = len(tokens)
    loss_mask: list[int] = []
    logprobs: list[float] = []

    context = list(tokens)
    for call in segment:
        prompt = list(call.get("prompt_token_ids") or [])
        sampled = list(call.get("sampled_token_ids") or [])
        sampled_logprobs = list(call.get("sampled_logprobs") or [])
        if not sampled or len(sampled) != len(sampled_logprobs):
            return None
        if context != prompt[: len(context)]:
            return None
        delta = prompt[len(context) :]
        tokens += delta
        loss_mask += [0] * len(delta)
        logprobs += [0.0] * len(delta)
        tokens += sampled
        loss_mask += [1] * len(sampled)
        logprobs += sampled_logprobs
        context = [*prompt, *sampled]

    if not any(loss_mask):
        return None
    return tokens, loss_mask, logprobs, response_start


def _images(segment: list[dict], media_assets: dict[str, dict]) -> tuple[list[PILImage.Image], list[bytes], list[str]]:
    """Decode this segment's screenshots in prompt order, with their bytes and ids."""
    media_ids = _media_occurrences(segment)
    images: list[PILImage.Image] = []
    payloads: list[bytes] = []
    for media_id in media_ids:
        asset = media_assets.get(media_id) or {}
        source = asset.get("source_part") or {}
        # OpenAI's image_url is either the bare data URL or {"url": ...}.
        part = source.get("image_url")
        url = part if isinstance(part, str) else str((part or {}).get("url") or source.get("url") or "")
        if "base64," not in url:
            raise MediaBindingError(f"media {media_id} carries no base64 payload, so its tokens cannot be bound")
        payload = base64.b64decode(url.split("base64,", 1)[1])
        _check_against_gyms_digest(media_id, source, asset.get("content_digest"))
        images.append(PILImage.open(io.BytesIO(payload)).convert("RGB"))
        payloads.append(payload)
    return images, payloads, media_ids


def _media_occurrences(segment: list[dict]) -> list[str]:
    """This segment's screenshots in prompt order, one entry per occurrence."""
    per_call = [[str(media_id) for media_id in call.get("media_ids") or []] for call in segment]
    for index, (earlier, later) in enumerate(zip(per_call, per_call[1:], strict=False)):
        if later[: len(earlier)] != earlier:
            raise MediaBindingError(
                f"call {index} claims media {earlier} which is not an ordered prefix of call {index + 1}'s "
                f"{later}: a screenshot was added, dropped or rewritten inside a segment Gym declared "
                "append-compatible"
            )
    return per_call[-1]


def _check_against_gyms_digest(media_id: str, source: dict, claimed: Any) -> None:
    """Fail unless this asset is the one Gym committed to when it recorded it."""
    if not isinstance(claimed, str) or not claimed:
        return
    canonical = json.dumps(source, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest != claimed:
        raise MediaBindingError(
            f"media {media_id} canonicalizes to {digest} but Gym recorded {claimed}: the screenshot that "
            "reached training is not the one the policy was shown"
        )
    if media_id != f"media-{digest[:24]}":
        raise MediaBindingError(f"media {media_id} does not match its own digest {digest[:24]}")


def _vision_inputs(state: Any, images: list[PILImage.Image], tokens: list[int]) -> dict | None:
    """pixel_values/image_grid_thw for the sequence, verified image by image."""
    if not images:
        return None

    image_token_id = state.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    verify_image_token_layout(tokens, image_token_id, expanded_token_counts(state.processor, images))
    processed = state.processor.image_processor(images=images, return_tensors="pt")
    return {"pixel_values": processed["pixel_values"], "image_grid_thw": processed["image_grid_thw"]}


def _verify_each_call(state: Any, segment: list[dict], media_ids: list[str], images: list[PILImage.Image]) -> None:
    """Check every call's own prompt against its own claimed screenshots."""
    image_token_id = state.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    counts = dict(zip(media_ids, expanded_token_counts(state.processor, images), strict=True))
    for index, call in enumerate(segment):
        claimed = [str(media_id) for media_id in call.get("media_ids") or []]
        try:
            verify_image_token_layout(call.get("prompt_token_ids") or [], image_token_id, [counts[m] for m in claimed])
        except MediaBindingError as e:
            raise MediaBindingError(f"call {index}: {e}") from e


def _inert(sample: Sample, state: Any) -> Sample:
    """Shape a sample so it travels with the batch and trains on nothing."""
    import torch  # noqa: PLC0415

    tokenizer = state.processor.tokenizer
    processed = state.processor.image_processor(images=[PILImage.new("RGB", (64, 64))], return_tensors="pt")
    merge = getattr(state.processor.image_processor, "merge_size", 2)
    n_image_tokens = int(torch.prod(processed["image_grid_thw"], dim=-1).sum().item()) // (merge * merge)

    sample.remove_sample = True
    sample.tokens = (
        [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
        + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * n_image_tokens
        + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
    )
    sample.multimodal_train_inputs = {
        "pixel_values": processed["pixel_values"],
        "image_grid_thw": processed["image_grid_thw"],
    }
    sample.response_length = 0
    sample.response = ""
    sample.loss_mask = []
    sample.rollout_log_probs = []
    return sample


def _aborted(sample: Sample, reason: str, state: Any, counters: dict) -> list[Sample]:
    """Give up on this episode for an infrastructure reason, costing its group."""
    logger.warning("osworld episode aborted: %s", reason)
    sample = _inert(sample, state)
    sample.status = Sample.Status.ABORTED
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["reward"] = 0.0
    sample.metadata["osworld_abort_reason"] = reason
    sample.metadata["osworld_rejections"] = dict(counters)
    return [sample]


def _admitted_reward(envelope: dict) -> float | None:
    """The grade, only if Gym positively says it admitted and scored the episode."""
    if envelope.get("runtime_eligible") is not True or envelope.get("mask_sample") is not False:
        return None
    reward = envelope.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        return None
    return float(reward)


def _graded_only(sample: Sample, reason: str, state: Any, counters: dict, reward: float) -> list[Sample]:
    """An episode Gym graded but this converter cannot train on."""
    logger.info("osworld episode graded but not trainable: %s", reason)
    graded = _inert(sample, state)
    graded.status = Sample.Status.COMPLETED
    graded.metadata = dict(graded.metadata or {})
    graded.metadata["reward"] = reward
    graded.metadata["osworld_untrainable_reason"] = reason
    graded.metadata["osworld_rejections"] = dict(counters)
    return [graded]


def _padding(template: Sample, reward: float, rollout_id: int, index: int, state: Any) -> Sample:
    """Fill a fixed-size episode out to its sample count with an inert sibling."""
    pad = _inert(copy.deepcopy(template), state)
    pad.status = Sample.Status.COMPLETED
    pad.rollout_id = rollout_id
    pad.metadata = dict(pad.metadata or {})
    pad.metadata["reward"] = reward
    pad.metadata["osworld_segment_index"] = index
    pad.metadata["osworld_padding"] = True
    return pad


def _boundary_provenance(response: dict) -> list[dict]:
    """Gym's rewrite boundaries, kept so a segment count can be explained later."""
    return [
        {
            "segment_id": event.get("segment_id"),
            "reason": event.get("reason"),
            "retained_media": event.get("retained_media_count"),
            "removed_media": event.get("removed_media_count"),
            # Gym's own events record the policy it applied.
            "policy": event.get("policy_name"),
            "policy_version": event.get("policy_version"),
            "config_digest": event.get("config_digest"),
        }
        for event in response.get("boundary_events") or []
    ]


async def generate(input: "GenerateFnInput") -> "GenerateFnOutput":
    """One graded episode -> a fixed number of training samples, or one eval row."""
    # Imported here: base_types pulls in sglang_router, which CPU-only boxes lack.
    from miles.rollout.base_types import GenerateFnOutput  # noqa: PLC0415

    args, sample, state = input.args, input.sample, input.state
    sampling_params = input.sampling_params
    assert not args.partial_rollout, "Partial rollout is not supported for OSWorld rollouts."

    task = (sample.metadata or {}).get("osworld_task")
    if not task:
        raise ValueError("no OSWorld task row: expected sample.metadata['osworld_task']")
    config = EpisodeConfig.read(args)
    per_episode = 1 if input.evaluation else config.samples_per_episode
    counters = dict.fromkeys(_REJECTIONS, 0)

    async with _sandbox_gate(config):
        try:
            envelope = await _run_gym_task(config, task, dict(sampling_params))
        except Exception as e:  # noqa: BLE001
            # A provider or timeout failure is routine at this scale and must not stop the rollout.
            logger.warning("osworld episode failed to run: %s", e, exc_info=True)
            return GenerateFnOutput(samples=_aborted(sample, f"transport: {type(e).__name__}: {e}", state, counters))

    response = envelope.get("response") or {}
    graded = _admitted_reward(envelope)
    if graded is None:
        reason = envelope.get("runtime_admission_reason") or "no valid admission or reward"
        return GenerateFnOutput(samples=_aborted(sample, f"not runtime-eligible: {reason}", state, counters))
    reward = graded

    def unusable(reason: str) -> GenerateFnOutput:
        if input.evaluation:
            return GenerateFnOutput(samples=_graded_only(sample, reason, state, counters, reward))
        return GenerateFnOutput(samples=_aborted(sample, reason, state, counters))

    media_assets = response.get("media_assets") or {}
    try:
        calls = _calls(response)
    except ValueError as e:
        return unusable(f"inconsistent exact trace: {e}")
    segments = _segments(calls)
    if not segments:
        counters["no_exact_evidence"] += 1
        return unusable("carried no exact model-call evidence")

    context_limit = int(getattr(args, "rollout_max_context_len", 0) or 0)
    processor = processor_identity(state.processor.image_processor)
    boundaries = _boundary_provenance(response)
    rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
    siblings: list[Sample] = []

    for index, segment in enumerate(segments):
        stitched = _stitch(segment)
        if stitched is None:
            counters["stitch_violation"] += 1
            logger.warning("osworld segment %d did not stitch; dropped", index)
            continue
        tokens, loss_mask, logprobs, response_start = stitched
        # Miles checks the context bound only in its own request builder, which this function bypasses.
        if context_limit and len(tokens) > context_limit:
            counters["over_context"] += 1
            logger.warning("osworld segment %d is %d tokens over %d; dropped", index, len(tokens), context_limit)
            continue
        try:
            images, payloads, media_ids = _images(segment, media_assets)
            vision_inputs = _vision_inputs(state, images, tokens)
            if vision_inputs is not None:
                _verify_each_call(state, segment, media_ids, images)
        except MediaBindingError as e:
            counters["image_mismatch"] += 1
            logger.warning("osworld segment %d dropped: %s", index, e)
            continue
        if vision_inputs is None:
            # FSDP's vision-tower collectives make a text-only sample unsafe.
            counters["text_only_call"] += 1
            logger.info("osworld segment %d is text-only; counted and dropped", index)
            continue

        child = sample if index == 0 else copy.deepcopy(sample)
        child.rollout_id = rollout_id
        child.tokens = tokens
        child.response_length = len(tokens) - response_start
        child.loss_mask = loss_mask
        child.rollout_log_probs = logprobs
        child.multimodal_train_inputs = vision_inputs
        child.response = state.tokenizer.decode(tokens[response_start:], skip_special_tokens=False)
        child.metadata = dict(child.metadata or {})
        child.metadata["reward"] = reward
        child.metadata["osworld_segment_index"] = index
        child.metadata["osworld_segment_id"] = segment[0].get("segment_id")
        child.metadata["osworld_media_ids"] = media_ids
        child.metadata["osworld_media_digests"] = media_digests(payloads)
        child.metadata["osworld_processor"] = processor
        child.metadata["osworld_boundary_events"] = boundaries
        child.status = Sample.Status.COMPLETED
        siblings.append(child)

    if not siblings:
        return unusable(f"kept no segment of {len(segments)}")
    if len(siblings) > per_episode:
        if input.evaluation:
            siblings = siblings[:1]
        else:
            return GenerateFnOutput(
                samples=_aborted(
                    sample,
                    f"produced {len(siblings)} segments over the configured cap of {per_episode}",
                    state,
                    counters,
                )
            )

    kept = len(siblings)
    for index in range(kept, per_episode):
        siblings.append(_padding(sample, reward, rollout_id, index, state))
    siblings[0].metadata["osworld_rejections"] = dict(counters)
    siblings[0].metadata["osworld_segment_stats"] = {
        "calls": len(calls),
        "segments": len(segments),
        "kept": kept,
        "padded": per_episode - kept,
        "boundaries": len(boundaries),
    }

    logger.info(
        "[osworld %s] reward=%.3f calls=%d segments=%d kept=%d padded=%d tokens=%d rejections=%s",
        (task.get("verifier_metadata") or {}).get("task_id", "unknown"),
        reward,
        len(calls),
        len(segments),
        kept,
        per_episode - kept,
        sum(len(s.tokens) for s in siblings),
        {name: count for name, count in counters.items() if count},
    )
    return GenerateFnOutput(samples=siblings)


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """Pass through the reward OSWorld's own evaluator computed inside the VM."""
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict | None = None) -> bool:
    """Score the held-out set over the episodes the environment actually graded."""
    from miles.ray.rollout.metrics import compute_rollout_step  # noqa: PLC0415
    from miles.utils.tracking_utils import tracking  # noqa: PLC0415

    log_dict = dict(extra_metrics or {})
    for key, rows in data.items():
        rewards = rows.get("rewards") or []
        graded = [reward for reward in rewards if reward is not None]
        excluded = len(rewards) - len(graded)
        log_dict[f"eval/{key}-attempted"] = len(rewards)
        log_dict[f"eval/{key}-graded"] = len(graded)
        log_dict[f"eval/{key}-infrastructure_excluded"] = excluded
        if graded:
            log_dict[f"eval/{key}"] = sum(graded) / len(graded)
        else:
            # No graded attempt is no score, not a score of zero.
            logger.warning("eval/%s: all %d attempts were excluded as infrastructure faults", key, len(rewards))
        if excluded:
            logger.warning(
                "eval/%s: %d of %d attempts excluded as infrastructure faults, not scored as zero",
                key,
                excluded,
                len(rewards),
            )

    logger.info("eval %s: %s", rollout_id, log_dict)
    log_dict["eval/step"] = compute_rollout_step(args, rollout_id)
    tracking.log(args, log_dict, step_key="eval/step")
    return True
