"""The Miles generate function for Harbor ``computer-1`` CUA rollouts.

One episode runs one Harbor trial in this process, pointed at a recording proxy
that keeps every policy call's exact prompt ids, sampled ids and logprobs, and
the screenshot bytes that were in the request. The recorded calls become one
training sample per segment, all sharing the episode's rollout identity and its
single reward.

Select it with::

    --custom-generate-function-path examples.experimental.harbor_cua.rollout.generate
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from examples.experimental.harbor_cua.media import MediaMismatchError, MediaProcessor, placeholder_vision_sample
from examples.experimental.harbor_cua.options import (
    UPSTREAM_AUTO,
    UPSTREAM_ROUTER,
    WRITE_OFF_ABORT_GROUP,
    HarborCuaOptions,
    add_arguments,
)
from examples.experimental.harbor_cua.recorder import CallRecorder, RecorderError
from examples.experimental.harbor_cua.samples import build_samples
from examples.experimental.harbor_cua.segments import segment_calls
from examples.experimental.harbor_cua.trial_driver import TrialOutcome, run_trial

from miles.rollout.agentic.rollout_limits import rollout_semaphore
from miles.utils.types import Sample

if TYPE_CHECKING:
    from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput

logger = logging.getLogger(__name__)

SANDBOX_LIMIT_NAME = "harbor_cua_sandbox"

_recorder: CallRecorder | None = None
_recorder_lock = asyncio.Lock()


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    # In-function on purpose: base_types reaches the chat-template registry, which needs
    # the sglang build only a training node has, and that would stop the offline tests
    # importing this module at all.
    from miles.rollout.base_types import GenerateFnOutput  # noqa: PLC0415

    args = input.args
    assert not args.partial_rollout, "Partial rollout is not supported for Harbor CUA rollouts."
    state = input.state
    sample = input.sample
    metadata = sample.metadata or {}
    instance_id = metadata.get("instance_id")
    if not instance_id:
        raise ValueError("no Harbor task: expected sample.metadata['instance_id'] to name a task directory")

    options = HarborCuaOptions.from_args(args)
    recorder = await _shared_recorder(args, options)
    token = recorder.open_rollout()
    try:
        async with rollout_semaphore(SANDBOX_LIMIT_NAME, options.max_sandboxes):
            outcome = await run_trial(
                instance_id=str(instance_id),
                api_base=recorder.base_url(token),
                session_id=token,
                sampling_params=input.sampling_params,
                max_turns=options.max_turns,
            )
    finally:
        record = recorder.take(token)

    sample.metadata = {**metadata, **outcome.metadata, "trial_dir": outcome.trial_dir}
    media = MediaProcessor(state.processor)
    sample.metadata["processor_fingerprint"] = media.fingerprint()

    write_off_mode = options.write_off_mode
    if outcome.infrastructure_failure:
        return GenerateFnOutput(
            samples=[_write_off(sample, state, f"trial failed: {outcome.exit_status}", write_off_mode)]
        )

    try:
        _check_call_accounting(record, outcome, sample.metadata)
        segmentation = segment_calls(
            record.trained_calls(),
            max_sample_tokens=options.sample_token_budget(args),
        )
        samples, stats = build_samples(
            template=sample,
            record=record,
            segmentation=segmentation,
            media=media,
            reward=outcome.reward,
            rollout_id=_rollout_id(sample),
            tokenizer=state.tokenizer,
        )
    except (MediaMismatchError, ValueError) as exc:
        return GenerateFnOutput(
            samples=[_write_off(sample, state, f"conversion refused the episode: {exc}", write_off_mode)]
        )

    sample.metadata.update(stats.as_metadata())
    sample.metadata["arena_bytes"] = record.arena.total_bytes()
    if not samples:
        return GenerateFnOutput(
            samples=[_write_off(sample, state, "episode produced no trainable segment", write_off_mode)]
        )

    if input.evaluation and len(samples) > 1:
        # The eval path extends its data with every returned sample and averages that
        # flat reward list, so N siblings would count one episode N times.
        stats.evaluation_siblings_dropped = len(samples) - 1
        samples = samples[:1]
    for sibling in samples:
        sibling.metadata.update(stats.as_metadata())
    logger.info(
        "harbor cua %s: reward=%s calls=%d segments=%d kept=%d tokens=%d",
        instance_id,
        outcome.reward,
        stats.trained_calls,
        stats.segments,
        stats.kept_segments,
        stats.trained_tokens,
    )
    return GenerateFnOutput(samples=samples)


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """Only reached if a sample left ``reward`` unset; the trial's number is the reward."""
    if isinstance(samples, list):
        return [float(s.metadata.get("reward", 0.0)) for s in samples]
    return float(samples.metadata.get("reward", 0.0))


async def _resolve_upstream(args, options: HarborCuaOptions) -> str:
    """Where the recorder forwards.

    The Miles router re-serialises a chat reply to the OpenAI schema, so the token
    evidence the recorder asks for does not survive it; one of its engines answers
    with the fields intact. ``auto`` asks the router which engines it has, which is
    the same question the built-in rollout paths ask.
    """
    if options.upstream not in (UPSTREAM_AUTO, UPSTREAM_ROUTER):
        return options.upstream
    if options.upstream == UPSTREAM_ROUTER:
        return options.router_url(args)

    from miles.rollout.inference_rollout.inference_rollout_train import get_worker_urls  # noqa: PLC0415

    urls = await get_worker_urls(args)
    if not urls:
        raise RecorderError(
            f"the router at {options.router_url(args)} reports no engines; pass "
            "--harbor-cua-upstream <engine url> or 'router' to choose explicitly"
        )
    return urls[0].rstrip("/")


async def _shared_recorder(args, options: HarborCuaOptions) -> CallRecorder:
    """One recorder per rollout process, bound to loopback.

    ``computer-1`` makes its model calls from this process rather than from the
    sandbox, so the sandbox needs no route back here and the proxy never has to
    be reachable from outside.
    """
    global _recorder
    async with _recorder_lock:
        if _recorder is None:
            upstream = await _resolve_upstream(args, options)
            recorder = CallRecorder(upstream=upstream, timeout=options.call_timeout_s)
            await recorder.start()
            logger.info("harbor cua recorder listening on 127.0.0.1:%d for %s", recorder.port, upstream)
            _recorder = recorder
        return _recorder


def _rollout_id(sample: Sample) -> int:
    rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
    assert rollout_id is not None, "Harbor CUA samples need Sample.rollout_id or Sample.index"
    return int(rollout_id)


def _check_call_accounting(record, outcome: TrialOutcome, metadata: dict[str, Any]) -> None:
    """Every Harbor episode makes at least one trainable call, so fewer means misclassified.

    ``computer-1`` queries the policy once per episode and retries a parse error by
    spending the next episode, so the count is a lower bound rather than an equality:
    reactive compaction can add a call inside one episode, and the final-answer
    fallback adds one at the end. Fewer than one per episode can only mean a call the
    agent loop made was taken for one of the compactor's. A policy call the agent acted
    on without the recorder capturing it fails outright, since an extra call can make up
    the count while that action's tokens train as masked context.
    """
    episodes = int((outcome.metadata.get("agent_metrics") or {}).get("n_episodes") or 0)
    actions = len(record.trained_calls())
    metadata["harbor_episodes"] = episodes
    metadata["recorded_trained_calls"] = actions
    metadata["rejected_calls"] = list(record.rejected)
    if record.unrecorded_policy_calls:
        raise ValueError(
            f"{len(record.unrecorded_policy_calls)} policy calls reached the agent without token evidence: "
            f"{record.unrecorded_policy_calls[:3]}"
        )
    if episodes and actions < episodes:
        raise ValueError(
            f"recorded {actions} action calls for {episodes} Harbor episodes; "
            f"{len(record.compactor_calls())} were classified as the compactor's and "
            f"{len(record.rejected)} were refused: {record.rejected[:3]}"
        )


def _write_off(sample: Sample, state, reason: str, mode: str) -> Sample:
    """A structurally valid sample that trains on nothing.

    It carries one minimal image because under FSDP the vision tower's weight
    all-gathers are collectives: a text-only sample skips them and the ranks that
    issued them block until some other phase happens to post a matching call.
    """
    logger.warning("harbor cua write-off (%s): %s", mode, reason)
    tokens, train_inputs = placeholder_vision_sample(state.processor)
    sample.status = Sample.Status.ABORTED if mode == WRITE_OFF_ABORT_GROUP else Sample.Status.FAILED
    sample.remove_sample = True
    sample.tokens = tokens
    sample.multimodal_train_inputs = train_inputs
    sample.multimodal_inputs = None
    sample.response_length = 0
    sample.response = ""
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.weight_versions = []
    sample.reward = 0.0
    sample.rollout_id = _rollout_id(sample)
    sample.metadata.setdefault("reward", 0.0)
    sample.metadata["write_off_reason"] = reason
    return sample


generate.add_arguments = add_arguments
