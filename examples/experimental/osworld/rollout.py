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

**Every episode returns exactly ``osworld_samples_per_episode`` samples**, the
real segments first and padding after. The padding trains on nothing and moves
no reward statistic, because reward normalization takes one reward per
``rollout_id``. It is **not** loss-neutral, and the recipe document states the
consequence plainly. ``loss_function`` divides by the sample count unless the
rollout-side schedule supplies ``num_rollouts``, and that schedule is the one
switched off below, while the FSDP backend's ``TRAIN_KEYS`` carries no
``rollout_mask_sums`` to fall back on. Two things follow. The constant factor
from the padded count largely cancels through AdamW's moment estimates, so it
is *not* an effective learning-rate change. What does not cancel is that the
loss is **segment-weighted rather than rollout-weighted**: an episode that
produced twelve segments carries about twelve times the weight of one that
produced a single segment, and a fixed cap does not change that. The core Miles
change is what removes it. A data-dependent sample count
cannot be trained on an image batch today: the schedule that counts
``--global-batch-size`` in rollouts rather than samples
(``miles/utils/dp_schedule.py``) is switched off for any batch carrying
``multimodal_train_inputs`` (``train_data_conversion.py:381``) and needs a
schedule config the FSDP backend never publishes, so the legacy sample-strided
split runs instead and floor-divides the shard into steps -- leaving the tail
rows without log-probs while they keep their rewards, which is an ``IndexError``
in ``get_grpo_returns``. A fixed count makes that division exact. Set the cap to
OSWorld's step cap and it is an upper bound by construction, since an episode
cannot produce more segments than it makes model calls.

**An infrastructure failure aborts the group rather than scoring zero.**
``remove_sample`` only zeroes the loss mask -- the row keeps its reward and
still moves the group's mean and standard deviation, which Miles documents
deliberately in ``filter_hub/sample_filters.py`` and pins in its tests -- and
``Sample.Status.FAILED`` is read nowhere in the framework. ``ABORTED`` is what
``apply_aborted_filter`` rejects a group on, before any reward statistic exists,
and what oversampling then refills. So a sandbox crash costs its group and a
regenerated prompt; it never teaches the policy that the task was impossible.

Wire with:
  --custom-generate-function-path examples.experimental.osworld.rollout.generate
  --custom-rm-path examples.experimental.osworld.rollout.reward_func
"""

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

# Why an episode kept no trainable segment. Reported per episode and summed over
# the run, because an episode that quietly contributes nothing is the failure
# mode this converter is most likely to have and least likely to show.
_REJECTIONS = (
    "no_exact_evidence",
    "stitch_violation",
    "image_mismatch",
    "text_only_call",
    "over_context",
)


@dataclass(frozen=True)
class EpisodeConfig:
    """Run-wide knobs this converter needs, resolved once per call site.

    They arrive through the environment rather than as flags because Miles's
    training entry parses argv strictly -- an unknown ``--osworld-*`` would be
    rejected -- and because every one of them is a launch constraint: the run is
    correct only if every episode uses the same value. The launcher forwards
    them and derives ``--global-batch-size`` from ``samples_per_episode``, so
    the two cannot be set apart.
    """

    gym_url: str
    samples_per_episode: int
    max_sandboxes: int
    episode_timeout_s: float

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
        )

_gate: asyncio.Semaphore | None = None


def _sandbox_gate(config: EpisodeConfig) -> asyncio.Semaphore:
    """Cap concurrent OSWorld VMs process-wide, independent of batch geometry."""
    global _gate
    if _gate is None:
        _gate = asyncio.Semaphore(config.max_sandboxes)
    return _gate


async def _run_gym_task(config: EpisodeConfig, task: dict, sampling_params: dict) -> dict:
    """POST one task to Gym and return its response envelope."""
    import aiohttp  # noqa: PLC0415

    base_url = config.gym_url
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

    timeout = aiohttp.ClientTimeout(total=config.episode_timeout_s)
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


def _images(segment: list[dict], media_assets: dict[str, dict]) -> tuple[list[PILImage.Image], list[bytes], list[str]]:
    """Decode this segment's screenshots in prompt order, with their bytes and ids.

    First-appearance order across the segment's calls is the prompt order
    whether Gym lists each call's whole media set or only its new ones, because
    prompts within a segment extend one another. A screenshot referenced twice
    in one prompt would collapse here, which ``verify_image_token_layout`` then
    rejects as a run-count disagreement rather than mis-binding it.

    Gym publishes its own ``content_digest`` per asset, so the bytes we decode
    are checked against the bytes it recorded. That catches a transport or
    base64 fault between Gym and here, which no amount of checking our own
    digest against itself would.
    """
    media_ids = _media_occurrences(segment)
    images: list[PILImage.Image] = []
    payloads: list[bytes] = []
    for media_id in media_ids:
        asset = media_assets.get(media_id) or {}
        source = asset.get("source_part") or {}
        # Both OpenAI shapes occur: image_url as the bare data URL, and as
        # the nested {"url": ...} object.
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
    """This segment's screenshots in prompt order, one entry per occurrence.

    Each call lists its whole prompt's media, so the last call's list is the
    stitched sequence's media. Compatibility is checked between *successive*
    calls rather than against the final list alone: every list being a prefix
    of the final one still admits an intermediate shrink -- ``[A,B]``, ``[A]``,
    ``[A,B,C]`` passes that weaker test while the middle call's prompt accounts
    for two images and claims one. Successive prefixes forbid the shrink and
    imply the final-prefix property anyway.

    This is the media half of append-compatibility, which Gym promises within a
    segment but which this converter has to confirm for itself: two calls can
    swap one screenshot for another of the same size without disturbing the
    token prefix at all, and both assets' digests would still be valid.

    Occurrences are not deduplicated. The arena is content-addressed, so one
    prompt showing the same screenshot twice lists its id twice and owes two
    token runs; collapsing them would reject a legitimate episode.
    """
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
    """Fail unless this asset is the one Gym committed to when it recorded it.

    Gym's ``content_digest`` is sha256 over the canonically serialized
    ``source_part`` -- not over the image bytes -- so recomputing it the same
    way covers the data URL, the detail and the part type together
    (``osworld_agent/trajectory.py:register_media_asset``). Its ``media_id`` is
    that digest's first 24 characters, which is also checked, because an id that
    no longer matches its own content means the asset arena was rebuilt without
    its digests.
    """
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
    """pixel_values/image_grid_thw for the sequence, verified image by image.

    The engine returns prompt ids already expanded, so each image owns one run
    of ``<|image_pad|>`` as long as that image costs. Run *i* is checked against
    image *i* positionally, which a summed total would not do: it would accept a
    wrong per-image assignment whose total agrees, and a wrong image order.

    Returns ``None`` only for a legitimate text-only call, where the segment
    referenced no screenshot at all. Any disagreement raises.
    """
    if not images:
        return None

    image_token_id = state.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    verify_image_token_layout(tokens, image_token_id, expanded_token_counts(state.processor, images))
    processed = state.processor.image_processor(images=images, return_tensors="pt")
    return {"pixel_values": processed["pixel_values"], "image_grid_thw": processed["image_grid_thw"]}


def _verify_each_call(state: Any, segment: list[dict], media_ids: list[str], images: list[PILImage.Image]) -> None:
    """Check every call's own prompt against its own claimed screenshots.

    The stitched sequence only ever exposes the *final* prompt's layout, so a
    call in the middle whose prompt holds an image its metadata does not claim
    is invisible there -- its sampled tokens would still be trained, from a
    context the trace cannot account for. Each call is therefore checked
    against its own occurrence list, which is a prefix of the segment's.
    """
    image_token_id = state.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    counts = dict(zip(media_ids, expanded_token_counts(state.processor, images), strict=True))
    for index, call in enumerate(segment):
        claimed = [str(media_id) for media_id in call.get("media_ids") or []]
        try:
            verify_image_token_layout(call.get("prompt_token_ids") or [], image_token_id, [counts[m] for m in claimed])
        except MediaBindingError as e:
            raise MediaBindingError(f"call {index}: {e}") from e


def _inert(sample: Sample, state: Any) -> Sample:
    """Shape a sample so it travels with the batch and trains on nothing.

    It carries one minimal image, not just a pad token: under FSDP every rank
    must invoke the vision tower the same number of times per microbatch,
    because its layers' weight all-gathers are collectives. A text-only sample
    skips them and stalls the ranks that did issue them (miles#2406).
    """
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
    """Give up on this episode for an infrastructure reason, costing its group.

    ``ABORTED`` is the one status Miles acts on: ``apply_aborted_filter``
    rejects the whole group before ``_post_process_rewards`` computes anything,
    and oversampling regenerates the prompt. Writing the episode off with
    ``remove_sample`` instead would keep its row, and its 0.0 would enter the
    group's mean and standard deviation as a real task failure -- teaching the
    policy that a task its sandbox never ran was impossible.
    """
    logger.warning("osworld episode aborted: %s", reason)
    sample = _inert(sample, state)
    sample.status = Sample.Status.ABORTED
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["reward"] = 0.0
    sample.metadata["osworld_abort_reason"] = reason
    sample.metadata["osworld_rejections"] = dict(counters)
    return [sample]


def _admitted_reward(envelope: dict) -> float | None:
    """The grade, only if Gym positively says it admitted and scored the episode.

    Checked positively rather than by rejecting the two failure values: a
    missing ``runtime_eligible`` or ``mask_sample`` passes a negative test, and
    a missing reward then defaults to zero, so a malformed or empty reply would
    reach training as a genuine task failure instead of as the protocol fault
    it is.
    """
    if envelope.get("runtime_eligible") is not True or envelope.get("mask_sample") is not False:
        return None
    reward = envelope.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        return None
    return float(reward)


def _graded_only(sample: Sample, reason: str, state: Any, counters: dict, reward: float) -> list[Sample]:
    """An episode Gym graded but this converter cannot train on.

    Only evaluation takes this path. Gym admitted the episode and its evaluator
    produced a score, so that score is the measurement whether or not the trace
    is trainable -- aborting it would drop a real grade for a training-only
    reason, and the aborted row is then scored as a zero by the evaluation
    metrics rather than excluded.
    """
    logger.info("osworld episode graded but not trainable: %s", reason)
    graded = _inert(sample, state)
    graded.status = Sample.Status.COMPLETED
    graded.metadata = dict(graded.metadata or {})
    graded.metadata["reward"] = reward
    graded.metadata["osworld_untrainable_reason"] = reason
    graded.metadata["osworld_rejections"] = dict(counters)
    return [graded]


def _padding(template: Sample, reward: float, rollout_id: int, index: int, state: Any) -> Sample:
    """Fill a fixed-size episode out to its sample count with an inert sibling.

    It carries the episode's own reward rather than zero, because
    ``_normalize_rewards_by_rollout`` refuses a rollout whose siblings disagree
    on one, and because the group's statistics are taken over one reward per
    rollout -- so padding changes no statistic at all.
    """
    pad = _inert(copy.deepcopy(template), state)
    pad.status = Sample.Status.COMPLETED
    pad.rollout_id = rollout_id
    pad.metadata = dict(pad.metadata or {})
    pad.metadata["reward"] = reward
    pad.metadata["osworld_segment_index"] = index
    pad.metadata["osworld_padding"] = True
    return pad


def _boundary_provenance(response: dict) -> list[dict]:
    """Gym's rewrite boundaries, kept so a segment count can be explained later.

    A boundary is the only record of *why* an episode split where it did, and
    the segmentation statistics this recipe reports are read off it rather than
    off a cached-prefix proxy.
    """
    return [
        {
            "segment_id": event.get("segment_id"),
            "reason": event.get("reason"),
            "retained_media": event.get("retained_media_count"),
            "removed_media": event.get("removed_media_count"),
            # The policy Gym actually applied, read off its own events. A policy
            # name forwarded from the launcher would only record what we asked
            # for, which is not what the checkpoint was trained under.
            "policy": event.get("policy_name"),
            "policy_version": event.get("policy_version"),
            "config_digest": event.get("config_digest"),
        }
        for event in response.get("boundary_events") or []
    ]


async def generate(input: "GenerateFnInput") -> "GenerateFnOutput":
    """One graded episode -> a fixed number of training samples, or one eval row.

    The modern signature rather than the legacy ``(args, sample, sampling_params)``
    one on purpose: only this form is handed ``evaluation``, and a held-out
    evaluation must report one result per task. Padding it to the training
    sample count would make a single success read as fifteen.
    """
    # In-function on purpose: base_types reaches sglang_router through the data
    # source, which exists only on a training node, and this module stays
    # importable on a CPU-only box so its arithmetic can be checked offline.
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
            # One episode is one VM and a long run starts thousands; a provider
            # or timeout failure is routine and must not take the rollout down.
            logger.warning("osworld episode failed to run: %s", e, exc_info=True)
            return GenerateFnOutput(samples=_aborted(sample, f"transport: {type(e).__name__}: {e}", state, counters))

    # Gym puts the grade and the admission decision on the row, not inside the
    # response envelope that carries the trace. Admission is required
    # *positively*: an envelope that does not say it was admitted, or that
    # carries no numeric grade, is a protocol fault, and defaulting its reward
    # to zero would manufacture a task failure out of a malformed reply.
    response = envelope.get("response") or {}
    graded = _admitted_reward(envelope)
    if graded is None:
        reason = envelope.get("runtime_admission_reason") or "no valid admission or reward"
        return GenerateFnOutput(samples=_aborted(sample, f"not runtime-eligible: {reason}", state, counters))
    reward = graded

    # Past this point Gym has graded the episode, so an unusable trace is this
    # converter's limitation rather than an environment fault: training drops
    # the group, evaluation keeps the grade.
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
        # Nothing downstream compares a sample against the context bound: both
        # rollout length flags are read only inside Miles's own request builder,
        # which a custom generate function never reaches, so an over-long
        # sequence would surface as a rope indexing error in the forward pass.
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
            # A model call with no screenshot is legitimate -- a text-only
            # summary turn -- but FSDP's vision-tower collectives make a
            # text-only sample unsafe, so it is counted rather than trained.
            counters["text_only_call"] += 1
            logger.info("osworld segment %d is text-only; counted and dropped", index)
            continue

        # Sample is a plain dataclass; the copy happens before any tensor is
        # attached, so it stays cheap.
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
        # Provenance: which Gym segment these tokens are, which pixels they were
        # bound to, and under what preprocessing. Without the digests and the
        # processor identity a later run cannot tell a reproduction from a
        # coincidence.
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
        # Impossible while the cap is OSWorld's step cap, since segments cannot
        # outnumber model calls -- so this means the two were configured apart,
        # and silently dropping real segments would bias training toward the
        # start of every episode.
        if input.evaluation:
            # An evaluation needs one row per task, so the extra segments are
            # dropped rather than the episode: its environment score is the
            # measurement, and the segmentation is not what is being reported.
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
    """Score the held-out set over the episodes the environment actually graded.

    Miles's default logger turns a null reward into a zero before averaging, so
    one sandbox that never ran would lower the reported task score rather than
    leave the denominator. This recipe aborts an episode precisely when its
    grade is not the policy's fault, so those rows are excluded from the score
    and reported as their own counts instead. Returning True suppresses the
    default logging for this dataset.

    Wire with --custom-eval-rollout-log-function-path.
    """
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
            # No score at all is not a score of zero; leave the mean out so a
            # wholly failed evaluation cannot read as a perfect failure.
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
