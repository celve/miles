"""
Generic agentic generate function for agent-environment RL training.

The agent logic is fully encapsulated in a user-provided async function
(--custom-agent-function-path). This generate function only handles:
  1. TITO session tracing (OpenAIEndpointTracer)
  2. Collecting the worker-assembled training samples (the session server
     converts records to samples, truncates and merges in the owning worker)
  3. Driver-side metadata application (agent_metadata, session_metadata)
  4. Where each agent call runs (--custom-agent-function-mode)

Agent function contract:
  async def my_agent(
      base_url: str,
      prompt: ...,
      request_kwargs: dict,
      metadata: dict,       # sample.metadata — env-specific fields
      **kwargs,
  ) -> dict | None:
      ...

  Returning None means no extra metadata to attach.
  Returning a dict merges it into every sample's metadata, so downstream
  reward models (--custom-rm-path) can read whatever the agent left there.
  Raising miles.rollout.agentic.agent_function.InfraAbort discards the sample
  (ABORTED, with metadata["exit_status"] naming the cause); any other exception
  keeps whatever the session recorded, scored by the reward model.

  Under ``subproc`` (the default) each call runs in a child process of its own:
  its arguments and return value must be picklable, and module-level state lasts
  one call. Limits shared across calls go through ``miles.rollout.agentic.rollout_limits``.
"""

import argparse
import functools
import logging
import socket
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import httpx
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

from miles.dashboard.hooks import push_agent_reports
from miles.rollout.agentic.agent_function import AGENT_FUNCTION_MODES, InfraAbort, call_agent_function
from miles.rollout.agentic.credentials import provider_env_names
from miles.rollout.agentic.remote import AgentWorkerEndpoint, forwarded_env, rollout_abort_scope
from miles.rollout.agentic.sandbox_worker import SandboxWorkerConfig
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.rollout.session.v2.metrics import SESSION_ROLLOUT_METRICS_KEY
from miles.utils.function_registry import load_function
from miles.utils.lifecycle import TrajectoryIdentity
from miles.utils.output_store.dropped import record_dropped_samples
from miles.utils.rollout_data_config import lower_limit
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# what Harbor's agent function, the CUA recipe and the sandbox providers read; secrets never go by value
_DEFAULT_WORKER_ENV = ("HARBOR_*", "AGENT_*", "LITE_*", *provider_env_names())


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    assert not input.args.partial_rollout, "Partial rollout is not supported"
    assert input.args.session_server_instances, (
        "agentic_tool_call.generate requires session_server_instances. "
        "Pass --use-session-server to start the session server."
    )
    use_v2 = input.args.use_session_server == "v2"
    collect_spec_metrics = use_v2 and input.args.sglang.common_value("speculative_algorithm") is not None
    tracer = await OpenAIEndpointTracer.create(
        input.args,
        evaluation=input.evaluation,
        sampling_params=input.sampling_params,
        extra_key=input.sample.kv_cache_namespace,
    )

    # a mixed dataset may name its own agent and a lower length limit
    agent_function_path = input.sample.agent_function_path or input.args.custom_agent_function_path
    custom_agent_function: Callable = load_function(agent_function_path)
    assert custom_agent_function is not None, f"Custom agent function {agent_function_path} not found"
    worker, sandbox, worker_env = _agent_placement(input.args)

    max_seq_len = lower_limit(input.args.max_seq_len, input.sample.max_seq_len)

    metadata = input.sample.metadata
    if max_seq_len is not None:
        _warn_max_seq_len_does_not_stop_the_agent()
        metadata = {**metadata, "max_seq_len": max_seq_len}
    if tracer.session_server_instance_id:
        metadata = {**metadata, "session_server_instance_id": tracer.session_server_instance_id}

    log_prefix = f"[session={tracer.session_id}]"

    # From the tracer, not args: with multiple instances the owning ip:port is per-session.
    # The dispatcher owns the source identity used to join subprocess phase reports.
    metadata = {
        **metadata,
        "session_server_id": tracer.session_server_id,
        "miles_lifecycle": asdict(TrajectoryIdentity.from_sample(input.sample)),
    }

    agent_metadata = None
    infra_abort: InfraAbort | None = None
    collect_failed = False
    cancelled = False
    t_start = time.monotonic()
    try:
        logger.debug(f"{log_prefix} Starting agent function call")
        agent_metadata = await call_agent_function(
            custom_agent_function,
            mode=input.args.custom_agent_function_mode,
            worker=worker,
            sandbox=sandbox,
            extra_env=worker_env,
            abort_scope=rollout_abort_scope(input.args),
            on_child_reports=lambda reports: push_agent_reports(reports, evaluation=input.evaluation),
            base_url=tracer.agent_base_url,
            prompt=input.sample.prompt,
            request_kwargs=build_chat_request_kwargs(input.sampling_params),
            metadata=metadata,
        )
        logger.debug(f"{log_prefix} Agent function returned in {time.monotonic()-t_start:.1f}s")
    except InfraAbort as e:
        infra_abort = e
        logger.warning(f"{log_prefix} Agent function discarded the sample ({e.exit_status}): {e}")
    except Exception as e:
        logger.warning(f"{log_prefix} Agent function failed: {e}", exc_info=True)
    except BaseException:
        cancelled = True
        raise
    finally:
        # Collect even if the agent failed.
        logger.debug(f"{log_prefix} Calling collect_samples...")
        collect_kwargs = {"max_seq_len": max_seq_len}
        if use_v2:
            collect_kwargs["agent_metadata"] = agent_metadata
        try:
            result = await tracer.collect_samples(input.sample, **collect_kwargs)
        # Costs this sample, not the run; a non-2xx still raises, and a StorePlanError stops the rollout.
        except (TimeoutError, httpx.TransportError) as e:
            collect_failed = True
            logger.warning(f"{log_prefix} Failed collecting samples: {e!r}", exc_info=True)
        except Exception as e:
            if infra_abort is None:
                raise
            collect_failed = True
            logger.warning(f"{log_prefix} Failed collecting samples after the agent's InfraAbort: {e!r}")
        else:
            logger.debug(
                f"{log_prefix} collect_samples done: {len(result.samples)} samples, "
                f"total_time={time.monotonic()-t_start:.1f}s"
            )
            if cancelled:
                # The session handed these samples' output store objects to us; nobody will train them.
                record_dropped_samples(result.samples)

    # The agent's cause first: the outage it reports can also be what broke the collection.
    if infra_abort is not None:
        return _aborted_output(input.sample, infra_abort.exit_status, use_v2, collect_spec_metrics)
    if collect_failed:
        return _aborted_output(input.sample, "CollectFailed", use_v2, collect_spec_metrics)
    if not result.samples:
        if result.empty_reason == "all_truncated":
            logger.warning("All samples truncated (prompt already exceeds max_seq_len)")
            return _aborted_output(input.sample, "AllTruncated", use_v2, collect_spec_metrics)
        logger.warning("No model calls recorded for sample")
        return _aborted_output(input.sample, "NoModelCalls", use_v2, collect_spec_metrics)

    session_rollout_metrics = None
    if collect_spec_metrics:
        session_rollout_metrics = result.session_metadata[SESSION_ROLLOUT_METRICS_KEY]
        if session_rollout_metrics["session_id"] != tracer.session_id:
            raise ValueError(
                "session_rollout_metrics.session_id does not match the collected session: "
                f"{session_rollout_metrics['session_id']!r} != {tracer.session_id!r}"
            )
        if session_rollout_metrics["metrics"] is None:
            raise ValueError("a successful session collect must carry metrics")

    samples = result.samples
    if collect_spec_metrics:
        for sample in samples:
            sample.metadata.pop(SESSION_ROLLOUT_METRICS_KEY, None)
            sample.metadata[SESSION_ROLLOUT_METRICS_KEY] = session_rollout_metrics
    if use_v2 and len(samples) > 1:
        # FIXME: handle sample index issues.
        rollout_id = input.sample.rollout_identity
        assert rollout_id is not None, "v2 agentic samples require input Sample.rollout_id or Sample.index"
        for sample in samples:
            sample.rollout_id = rollout_id
    if not use_v2:
        # v1: the agent's metadata is applied driver-side. Under v2 it traveled
        # through collect_samples and came back applied by the server-side
        # merge (per-sample metadata/reward on the wire) — no overlay here.
        for s in samples:
            s.metadata.update(agent_metadata or {})

    for sample in samples:
        # The tracer owns this identity; agent metadata must not replace it.
        sample.metadata["session_id"] = tracer.session_id

    # If the agent function reports wall-clock time spent outside policy generation
    # (env/tool steps), surface it on Sample.non_generation_time so throughput
    # accounting subtracts it.
    ngt = ((agent_metadata or {}).get("agent_metrics") or {}).get("total_tool_time")
    if ngt is not None:
        for s in samples:
            s.non_generation_time = ngt

    if use_v2:
        return GenerateFnOutput(samples=samples)

    (sample,) = samples
    sample.metadata.update(result.session_metadata)
    return GenerateFnOutput(samples=sample)


def _aborted_output(
    input_sample: Sample, exit_status: str, use_v2: bool, collect_spec_metrics: bool
) -> GenerateFnOutput:
    sample = deepcopy(input_sample)
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**sample.metadata, "exit_status": exit_status}
    if collect_spec_metrics:
        sample.metadata.pop(SESSION_ROLLOUT_METRICS_KEY, None)
    return GenerateFnOutput(samples=[sample] if use_v2 else sample)


def _agent_placement(
    args,
) -> tuple[AgentWorkerEndpoint | None, SandboxWorkerConfig | None, dict[str, str] | None]:
    mode = args.custom_agent_function_mode
    if mode not in ("worker", "sandbox"):
        return None, None, None
    env = forwarded_env(args.custom_agent_function_worker_env)
    if mode == "worker":
        if not args.custom_agent_function_worker_url:
            raise ValueError("--custom-agent-function-mode worker needs --custom-agent-function-worker-url")
        return AgentWorkerEndpoint(url=args.custom_agent_function_worker_url), None, env
    if not args.custom_agent_function_sandbox_template:
        raise ValueError("--custom-agent-function-mode sandbox needs --custom-agent-function-sandbox-template")
    sandbox = SandboxWorkerConfig(
        template=args.custom_agent_function_sandbox_template,
        owner=args.custom_agent_function_sandbox_owner or socket.gethostname(),
        lifetime_s=args.custom_agent_function_sandbox_timeout,
        boot_env_file=args.custom_agent_function_sandbox_env_file,
        records_dir=args.custom_agent_function_sandbox_records_dir,
        records=tuple(args.custom_agent_function_sandbox_records),
    )
    return None, sandbox, env


@functools.cache
def _warn_max_seq_len_does_not_stop_the_agent() -> None:
    """Once per process: the session server leaves the agent's control flow alone, so the limit only truncates."""
    logger.warning(
        "max_seq_len reaches the agent as metadata['max_seq_len'] and truncates the collected samples; "
        "Miles does not stop the agent at it, so an agent that does not read it runs on past it"
    )


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--custom-agent-function-mode",
        choices=AGENT_FUNCTION_MODES,
        default="subproc",
        help="Where each agent call runs: subproc, a child process of its own so one episode's "
        "blocking step or crash cannot stall the others; inline, on the rollout's event loop; worker, "
        "a subproc child on an agent worker (python -m miles.rollout.agentic.worker) at "
        "--custom-agent-function-worker-url, so the agents' CPU work leaves this host; or sandbox, the same "
        "on a worker in a sandbox created for each call from --custom-agent-function-sandbox-template.",
    )
    parser.add_argument(
        "--custom-agent-function-worker-url",
        type=str,
        default=None,
        help="The agent worker's URL for --custom-agent-function-mode worker.",
    )
    parser.add_argument(
        "--custom-agent-function-worker-env",
        nargs="*",
        default=list(_DEFAULT_WORKER_ENV),
        help="Env var names (fnmatch patterns) sent with each worker call for its agent process, so this run's "
        "runtime env stays where they are set. Paths among them must exist on the worker.",
    )
    parser.add_argument(
        "--custom-agent-function-sandbox-template",
        type=str,
        default=None,
        help="For --custom-agent-function-mode sandbox: the E2B-API sandbox template whose start command serves "
        "an agent worker once its boot env arrives (contract: miles.rollout.agentic.sandbox_worker). The "
        "service is the one E2B_API_URL names, reached through E2B_SANDBOX_URL.",
    )
    parser.add_argument(
        "--custom-agent-function-sandbox-env-file",
        type=str,
        default=None,
        help="A JSON object of strings added to each sandbox worker's boot env (by path, so secrets stay out of "
        "env dumps and logs). The E2B key is always added.",
    )
    parser.add_argument(
        "--custom-agent-function-sandbox-owner",
        type=str,
        default=None,
        help="The owner metadata on the sandbox workers, for finding the ones a run left (default: hostname).",
    )
    parser.add_argument(
        "--custom-agent-function-sandbox-timeout",
        type=int,
        default=7800,
        help="Each sandbox worker's lifetime in seconds: a sandbox this process never killed expires then. "
        "Keep it above the agent's own cap.",
    )
    parser.add_argument(
        "--custom-agent-function-sandbox-records-dir",
        type=str,
        default=None,
        help="Copy these files from the dir an agent's result names under trial_dir out of its sandbox, into "
        "<dir>/<trial name>, before the sandbox goes; the result then names the copy.",
    )
    parser.add_argument(
        "--custom-agent-function-sandbox-records",
        nargs="*",
        default=["result.json", "config.json", "trial.log", "agent/timings.jsonl"],
        help="Paths, relative to the trial dir, that --custom-agent-function-sandbox-records-dir copies.",
    )


generate.add_arguments = _add_arguments
# reads Sample.agent_function_path / max_seq_len; a wrapper that delegates here may set this too
generate.reads_sample_agent_fields = True


# Process keys to match ChatCompletionRequest input
def build_chat_request_kwargs(sampling_params: dict[str, Any]) -> dict[str, Any]:
    request_kwargs = dict(sampling_params)
    key_map = {
        "max_new_tokens": "max_tokens",
        "min_new_tokens": "min_tokens",
        "sampling_seed": "seed",
    }
    for src, dst in key_map.items():
        if src in request_kwargs:
            if dst not in request_kwargs:
                request_kwargs[dst] = request_kwargs[src]
            request_kwargs.pop(src, None)

    reserved_keys = {"model", "messages"}
    allowed_keys = set(ChatCompletionRequest.model_fields) - reserved_keys
    return {key: value for key, value in request_kwargs.items() if key in allowed_keys and value is not None}
