"""The agent-worker protocol, caller side: run one agent-function call on a worker (``miles.rollout.agentic.worker``).

The worker runs the call through ``subproc`` on its own host. This side submits it, long-polls for the child's
reports and outcome, and cancels it when its own task is cancelled. Every worker response comes within
``MAX_POLL_WAIT_S``, so a proxy that cuts slow responses (the sandbox gateway answers 504 after 30 s) sits in between.

    POST   /calls               {call_id, fn: "module:qualname", kwargs, env, scope} -> {call_id}; idempotent per call_id
    GET    /calls/{id}?after=n&wait=s   -> {state: running|done, reports: [b64 pickle, ...], next, outcome?}
    DELETE /calls/{id}          cancel: the worker's child gets SIGTERM, as a cancelled subproc call's does
    GET    /health
"""

import asyncio
import base64
import fnmatch
import functools
import json
import logging
import os
import pickle
import socket
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from miles.rollout.agentic import rollout_limits
from miles.rollout.agentic.credentials import PROVIDER_CREDENTIALS
from miles.rollout.agentic.errors import InfraAbort, RemoteTraceback

logger = logging.getLogger(__name__)

MAX_POLL_WAIT_S = 25.0
_POLL_WAIT_S = 20.0
_REQUEST_ATTEMPTS = 3
_CANCEL_TIMEOUT_S = 10.0
# never forwarded by value, whatever --custom-agent-function-worker-env says; a *_FILE path to one is fine
_SECRET_PATTERNS = ("*SECRET", "*SECRET_KEY", "*TOKEN", "*PASSWORD", "*API_KEY")
_CREDENTIAL_NAMES = frozenset(name for spec in PROVIDER_CREDENTIALS.values() for name in spec["key_env_vars"])


@dataclass
class _Inflight:
    worker: "AgentWorkerEndpoint"
    scope: str | None = None
    aborted: bool = False


# this process's calls on workers, by call_id, so a rollout abort can cancel its own
_inflight: dict[str, _Inflight] = {}
# aborts by scope; None aborts every scope
_aborts: Counter = Counter()


def rollout_abort_scope(args) -> str | None:
    """The router a rollout generates through; training and the dedicated eval engines (--eval-num-gpus) differ."""
    ip, port = args.sglang_router_ip, args.sglang_router_port
    return None if ip is None or port is None else f"{ip}:{port}"


def _reaches(abort_scope: str | None, call_scope: str | None) -> bool:
    return abort_scope is None or call_scope is None or abort_scope == call_scope


def aborts_so_far(scope: str | None = None) -> int:
    """How many rollout aborts reaching ``scope`` this process has seen, for a caller still preparing a worker."""
    return sum(n for aborted, n in _aborts.items() if _reaches(aborted, scope))


@dataclass(frozen=True)
class AgentWorkerEndpoint:
    url: str
    headers: dict[str, str] = field(default_factory=dict)


def function_path(fn: Callable) -> str:
    return f"{fn.__module__}:{fn.__qualname__}"


def forwarded_env(patterns: Sequence[str]) -> dict[str, str]:
    """This process's env vars whose names match ``patterns`` (``fnmatch``), for the worker's agent process.

    Secrets are left out: a worker reads its own credentials, or a key file a forwarded ``*_FILE`` names."""
    return {
        name: value
        for name, value in os.environ.items()
        if _matches(name, patterns)
        and name not in _CREDENTIAL_NAMES
        and (name.endswith("_FILE") or not _matches(name, _SECRET_PATTERNS))
    }


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


async def cancel_inflight_calls(scope: str | None = None) -> int:
    """Cancel the calls of ``scope`` (every call when None) this process has submitted or is submitting to a
    worker; return how many.

    The worker cancels each on its DELETE, or refuses a submit the DELETE overtook. The caller stops polling
    either way, so a call whose DELETE was lost is cancelled by the worker's orphan timer."""
    _aborts[scope] += 1
    calls = [(call_id, inflight) for call_id, inflight in _inflight.items() if _reaches(scope, inflight.scope)]
    for _, inflight in calls:
        inflight.aborted = True
    async with asyncio.TaskGroup() as group:
        for call_id, inflight in calls:
            group.create_task(_cancel_on(inflight.worker, call_id))
    return len(calls)


@functools.cache
def _caller_scope() -> str:
    # a PID alone can repeat across the hosts and PID namespaces sharing one worker
    return f"{rollout_limits.current_scope()}-{uuid.uuid4().hex[:12]}"


def encode_report(item: Any) -> str:
    return base64.b64encode(pickle.dumps(item)).decode()


def decode_report(payload: str) -> Any:
    return pickle.loads(base64.b64decode(payload))


async def call_on_worker(
    worker: AgentWorkerEndpoint,
    fn: Callable,
    kwargs: dict[str, Any],
    *,
    env: dict[str, str],
    on_child_reports: Callable[[list[Any]], None] | None = None,
    abort_scope: str | None = None,
) -> Any:
    """Run ``fn(**kwargs)`` on the worker and return its result, as ``call_agent_function``'s ``subproc`` would.

    Only a rollout abort of ``abort_scope`` (``rollout_abort_scope``), or of every scope, cancels it."""
    call_id = uuid.uuid4().hex
    body = {
        "call_id": call_id,
        "fn": function_path(fn),
        "kwargs": kwargs,
        "env": env,
        "scope": _caller_scope(),
    }
    json.dumps(body)  # a kwarg the protocol cannot carry fails here, before anything runs
    async with _client(worker) as client:
        _inflight[call_id] = inflight = _Inflight(worker=worker, scope=abort_scope)
        try:
            await _request(client, "POST", "/calls", json=body)
            return await _poll_until_done(client, call_id, inflight, on_child_reports)
        except asyncio.CancelledError:
            await asyncio.shield(_cancel(client, call_id))
            raise
        finally:
            _inflight.pop(call_id, None)


async def _poll_until_done(
    client: httpx.AsyncClient,
    call_id: str,
    inflight: _Inflight,
    on_child_reports: Callable[[list[Any]], None] | None,
) -> Any:
    after = 0
    while True:
        state = await _request(client, "GET", f"/calls/{call_id}", params={"after": after, "wait": _POLL_WAIT_S})
        if state["reports"] and on_child_reports is not None:
            try:
                on_child_reports([decode_report(r) for r in state["reports"]])
            except Exception:
                logger.exception("handling a report from the agent worker failed")
        after = state["next"]
        if state["state"] == "done":
            return _unpack_outcome(state["outcome"])
        if inflight.aborted:
            raise InfraAbort(
                "WorkerCancelled", f"the rollout aborted call {call_id} on agent worker {client.base_url}"
            )


def _unpack_outcome(outcome: dict[str, Any]) -> Any:
    kind = outcome["kind"]
    if kind == "ok":
        return outcome["value"]
    if kind == "infra_abort":
        raise InfraAbort(outcome["exit_status"], outcome.get("message"))
    if kind == "cancelled":
        raise InfraAbort("WorkerCancelled", "the agent worker cancelled the call")
    raise RuntimeError(f"{outcome['type']}: {outcome['message']}") from RemoteTraceback(
        f'\n"""\n{outcome.get("traceback", "")}"""'
    )


async def _request(client: httpx.AsyncClient, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    """One protocol request; a lost connection is retried (submit is idempotent, a poll only reads)."""
    for attempt in range(_REQUEST_ATTEMPTS):
        try:
            response = await client.request(method, path, **kwargs)
        except httpx.TransportError as exc:
            if attempt + 1 == _REQUEST_ATTEMPTS:
                raise InfraAbort("WorkerUnreachable", f"agent worker {client.base_url}: {exc!r}") from exc
            await asyncio.sleep(2**attempt)
            continue
        if response.status_code == 404 and method == "GET":
            raise InfraAbort("WorkerLost", f"agent worker {client.base_url} no longer has the call (restarted?)")
        if response.status_code >= 400:
            raise InfraAbort(
                "WorkerUnreachable",
                f"agent worker {client.base_url}: HTTP {response.status_code} {response.text[:200]}",
            )
        return response.json()
    raise AssertionError("unreachable")


async def _cancel_on(worker: "AgentWorkerEndpoint", call_id: str) -> None:
    async with _client(worker) as client:
        await _cancel(client, call_id)


async def _cancel(client: httpx.AsyncClient, call_id: str) -> None:
    try:
        await asyncio.wait_for(client.delete(f"/calls/{call_id}"), timeout=_CANCEL_TIMEOUT_S)
    except Exception as exc:
        logger.warning(f"cancelling call {call_id} on agent worker {client.base_url} failed: {exc!r}")


def _client(worker: AgentWorkerEndpoint) -> httpx.AsyncClient:
    # keepalive probes, as swe_agent_function's agent-server client sets: an idle NAT path drops a quiet connection
    socket_options = [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
        (
            socket.IPPROTO_TCP,
            getattr(socket, "TCP_KEEPIDLE", 4),
            60,
        ),  # config-access-exempt: TCP keepalive constants exist only on some platforms
        (
            socket.IPPROTO_TCP,
            getattr(socket, "TCP_KEEPINTVL", 5),
            30,
        ),  # config-access-exempt: TCP keepalive constants exist only on some platforms
        (
            socket.IPPROTO_TCP,
            getattr(socket, "TCP_KEEPCNT", 6),
            5,
        ),  # config-access-exempt: TCP keepalive constants exist only on some platforms
    ]
    return httpx.AsyncClient(
        base_url=worker.url.rstrip("/"),
        headers=worker.headers,
        transport=httpx.AsyncHTTPTransport(socket_options=socket_options),
        timeout=httpx.Timeout(MAX_POLL_WAIT_S + 30.0),
    )
