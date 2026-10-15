"""Multi-turn computer-use rollout on OSWorld through NeMo Gym.

The division of labour, and why this file is a converter rather than a loop:

- **NeMo Gym** owns the episode. One POST to its agent server's ``/run`` boots
  an OSWorld VM, runs ``DesktopEnv`` unmodified against our policy, grades with
  OSWorld's own evaluator, and returns an exact trace: per model call the
  prompt/sampled token ids and sampling logprobs, screenshots factored once
  into ``media_assets``, and a ``segment_id`` that changes whenever a dropped
  screenshot breaks the training prefix.
- **This file** turns one graded run into one Miles training sample *per
  segment*. Siblings share a ``rollout_id``, so ``_normalize_rewards_by_rollout``
  gives them one reward normalized inside the GRPO group and broadcast back.

Why per segment rather than per episode: Gym has already done the stitching
analysis the HUD example does by hand. A segment is by construction a run of
calls whose prompts extend one another, so stitching inside one is exact, and
the boundary between two is exactly where it would not be.

Two patches to Gym are required for this to receive anything; both are in
LIN-1996's workspace. Without them the policy runner accepts no
``history_policy`` and discards every token id, so ``completion_evidence``
comes back empty and every episode is written off.

Wire with:
  --custom-generate-function-path examples.experimental.osworld.rollout.generate
  --custom-rm-path examples.experimental.osworld.rollout.reward_func
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any

from PIL import Image as PILImage

from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_gate: asyncio.Semaphore | None = None


def _sandbox_gate(args: Any) -> asyncio.Semaphore:
    """Cap concurrent OSWorld VMs process-wide, independent of batch geometry."""
    global _gate
    if _gate is None:
        _gate = asyncio.Semaphore(int(getattr(args, "osworld_max_sandboxes", 8)))
    return _gate


async def _run_gym_task(args: Any, task: dict, sampling_params: dict) -> dict:
    """POST one task to Gym and return its response envelope."""
    import aiohttp  # noqa: PLC0415

    base_url = str(getattr(args, "osworld_gym_url", "http://127.0.0.1:11000")).rstrip("/")
    create_params = dict(task.get("responses_create_params") or {})
    create_params.setdefault("max_output_tokens", int(sampling_params.get("max_new_tokens", 1500)))
    create_params.setdefault("temperature", float(sampling_params.get("temperature", 1.0)))
    if sampling_params.get("top_p") is not None:
        create_params.setdefault("top_p", float(sampling_params["top_p"]))
    body = {
        "responses_create_params": create_params,
        "verifier_metadata": task.get("verifier_metadata") or {},
        # Gym emits the exact trace only for a training-purpose run, and
        # cross-checks this against the metadata carrier below.
        "rollout_purpose": "training",
    }
    body["responses_create_params"].setdefault("metadata", {})["nemo_rl_rollout_purpose"] = "training"

    timeout = aiohttp.ClientTimeout(total=float(getattr(args, "osworld_episode_timeout_s", 2400)))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{base_url}/run", json=body) as response:
            response.raise_for_status()
            return await response.json()


def _calls(response: dict) -> list[dict]:
    """One record per model call, joining the two arrays transport v3 splits it across.

    ``output`` carries the token arrays and ``model_call_metadata`` the segment
    and media identity, paired by position; v3 leaves ``completion_evidence``
    empty, so reading that instead silently yields nothing to train on.
    """
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
    """One token sequence for a segment, with loss on the sampled tokens only.

    ``loss_mask`` and ``logprobs`` cover the response region -- everything after
    the first call's prompt -- matching what Miles reads alongside
    ``response_length``, so both are shorter than ``tokens``.

    Returns ``None`` when a call's prompt does not extend its predecessor's
    prompt+generation: Gym promises that only within a segment, so a violation
    means the trace and this reader disagree and nothing here is safe to train.
    """
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


def _images(segment: list[dict], media_assets: dict[str, dict]) -> list[PILImage.Image]:
    """Decode the screenshots this segment's prompts referenced, in order."""
    images: list[PILImage.Image] = []
    seen: set[str] = set()
    for call in segment:
        for media_id in call.get("media_ids") or []:
            if media_id in seen:
                continue
            seen.add(media_id)
            part = ((media_assets.get(media_id) or {}).get("source_part") or {}).get("image_url") or {}
            url = str(part.get("url") or "")
            if "base64," not in url:
                continue
            images.append(PILImage.open(io.BytesIO(base64.b64decode(url.split("base64,", 1)[1]))).convert("RGB"))
    return images


def _vision_inputs(state: Any, images: list[PILImage.Image], tokens: list[int]) -> dict | None:
    """pixel_values/image_grid_thw for the sequence, verified against its ids.

    Matched by prefix rather than count: a screenshot taken after the segment's
    last model call never entered a prompt, so the trace can hold more images
    than the ids account for. Anything but an exact prefix match is a processing
    disagreement and fails loudly.
    """
    if not images:
        return None
    import torch  # noqa: PLC0415

    image_token_id = state.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    actual = sum(1 for token in tokens if token == image_token_id)
    if actual == 0:
        return None

    processed = state.processor.image_processor(images=images, return_tensors="pt")
    grid = processed["image_grid_thw"]
    merge = getattr(state.processor.image_processor, "merge_size", 2)
    per_image = (torch.prod(grid, dim=-1) // (merge * merge)).tolist()

    total, keep = 0, 0
    for count in per_image:
        if total == actual:
            break
        total += int(count)
        keep += 1
    if total != actual:
        raise ValueError(
            f"image token mismatch: ids have {actual}, screenshots produce {int(sum(per_image))} (no prefix matches)"
        )

    reprocessed = processed if keep == len(images) else state.processor.image_processor(
        images=images[:keep], return_tensors="pt"
    )
    return {"pixel_values": reprocessed["pixel_values"], "image_grid_thw": reprocessed["image_grid_thw"]}


def _failed(sample: Sample, reason: str, state: Any) -> Sample:
    """A structurally valid write-off that trains on nothing.

    The write-off carries one minimal image, not just a pad token: under FSDP
    every rank must invoke the vision tower the same number of times per
    microbatch, because its layers' weight all-gathers are collectives. A
    text-only sample skips them and stalls the ranks that did issue them
    (miles#2406).
    """
    logger.warning(reason)
    import torch  # noqa: PLC0415

    tokenizer = state.processor.tokenizer
    processed = state.processor.image_processor(images=[PILImage.new("RGB", (64, 64))], return_tensors="pt")
    merge = getattr(state.processor.image_processor, "merge_size", 2)
    n_image_tokens = int(torch.prod(processed["image_grid_thw"], dim=-1).sum().item()) // (merge * merge)

    sample.status = Sample.Status.FAILED
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
    sample.metadata.setdefault("reward", 0.0)
    return sample


async def generate(args: Any, sample: Sample, sampling_params) -> list[Sample]:
    assert not args.partial_rollout, "Partial rollout is not supported for OSWorld rollouts."

    # In-function on purpose: sglang_rollout pulls in sglang_router, which only
    # exists on a training node.
    from miles.rollout.sglang_rollout import GenerateState  # noqa: PLC0415

    state = GenerateState(args)
    task = (sample.metadata or {}).get("osworld_task")
    if not task:
        raise ValueError("no OSWorld task row: expected sample.metadata['osworld_task']")

    async with _sandbox_gate(args):
        try:
            envelope = await _run_gym_task(args, task, dict(sampling_params))
        except Exception as e:  # noqa: BLE001
            # One episode is one VM and a long run starts thousands; a provider
            # or timeout failure is routine and must not take the rollout down.
            logger.warning("osworld episode failed to run: %s", e, exc_info=True)
            return [_failed(sample, f"osworld episode errored: {type(e).__name__}", state)]

    # Gym puts the grade and the admission decision on the row, not inside the
    # response envelope that carries the trace.
    response = envelope.get("response") or {}
    reward = float(envelope.get("reward") or 0.0)
    if envelope.get("mask_sample") is True or envelope.get("runtime_eligible") is False:
        reason = envelope.get("runtime_admission_reason") or "unknown"
        return [_failed(sample, f"osworld episode not runtime-eligible: {reason}", state)]

    media_assets = response.get("media_assets") or {}
    try:
        calls = _calls(response)
    except ValueError as e:
        return [_failed(sample, f"osworld episode dropped: {e}", state)]
    segments = _segments(calls)
    if not segments:
        return [_failed(sample, "osworld episode carried no exact model-call evidence", state)]

    rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
    siblings: list[Sample] = []
    for index, segment in enumerate(segments):
        stitched = _stitch(segment)
        if stitched is None:
            logger.warning("osworld segment %d did not stitch; dropped", index)
            continue
        tokens, loss_mask, logprobs, response_start = stitched
        try:
            vision = _vision_inputs(state, _images(segment, media_assets), tokens)
        except ValueError as e:
            logger.warning("osworld segment %d dropped: %s", index, e)
            continue
        if vision is None:
            continue

        child = sample if index == 0 else sample.model_copy(deep=True)
        child.rollout_id = rollout_id
        child.tokens = tokens
        child.response_length = len(tokens) - response_start
        child.loss_mask = loss_mask
        child.rollout_log_probs = logprobs
        child.multimodal_train_inputs = vision
        child.response = state.tokenizer.decode(tokens[response_start:], skip_special_tokens=False)
        child.metadata = dict(child.metadata or {})
        child.metadata["reward"] = reward
        child.metadata["osworld_segment_index"] = index
        child.status = Sample.Status.COMPLETED
        siblings.append(child)

    if not siblings:
        return [_failed(sample, f"osworld episode kept no segment of {len(segments)}", state)]

    logger.info(
        "[osworld %s] reward=%.3f calls=%d segments=%d kept=%d tokens=%d",
        (task.get("verifier_metadata") or {}).get("task_id", "unknown"),
        reward,
        len(calls),
        len(segments),
        len(siblings),
        sum(len(s.tokens) for s in siblings),
    )
    return siblings


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """Pass through the reward OSWorld's own evaluator computed inside the VM."""
    if isinstance(samples, list):
        return [s.metadata.get("reward", 0.0) for s in samples]
    return samples.metadata.get("reward", 0.0)
