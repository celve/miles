"""The agent-worker protocol, caller side: run one agent-function call on a worker (``miles.rollout.agentic.worker``).

The worker runs the call through ``subproc`` on its own host. This side submits it, long-polls for the child's
reports and outcome, and cancels it when its own task is cancelled. Every worker response comes within
``MAX_POLL_WAIT_S``.

    POST   /calls               {call_id, fn: "module:qualname", kwargs, env, scope} -> {call_id}; idempotent per call_id
    GET    /calls/{id}?after=n&wait=s   -> {state: running|done, reports: [b64 pickle, ...], next, outcome?}
    DELETE /calls/{id}          cancel: the worker's child gets SIGTERM, as a cancelled subproc call's does
    GET    /health
"""

import asyncio
import base64
import contextlib
import fnmatch
import functools
import json
import logging
import os
import pickle
import socket
import uuid
from collections.abc import Callable, Iterator, Sequence
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
_SECRET_PATTERNS = ("*SECRET", "*SECRET_KEY", "*TOKEN", "*PASSWORD", "*API_KEY")
_CREDENTIAL_NAMES = frozenset(name for spec in PROVIDER_CREDENTIALS.values() for name in spec["key_env_vars"])

_inflight: dict[str, tuple[str | None, str | None]] = {}
_aborted: set[str] = set()


def rollout_abort_scope(args) -> str:
    return f"{args.sglang_router_ip}:{args.sglang_router_port}"


def _reaches(abort_scope: str | None, call_scope: str | None) -> bool:
    return abort_scope is None or call_scope is None or abort_scope == call_scope


def function_path(fn: Callable) -> str:
    return f"{fn.__module__}:{fn.__qualname__}"


def forwarded_env(patterns: Sequence[str]) -> dict[str, str]:
    """This process's env vars whose names match ``patterns`` (``fnmatch``), except credentials."""
    return {
        name: value
        for name, value in os.environ.items()
        if _matches(name, patterns)
        and name not in _CREDENTIAL_NAMES
        and (name.endswith("_FILE") or not _matches(name, _SECRET_PATTERNS))
    }


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def has_inflight_calls() -> bool:
    return bool(_inflight)


async def cancel_inflight_calls(scope: str | None = None) -> int:
    """Cancel the calls of ``scope`` (every call when None) this process has submitted or is submitting to a
    worker; return how many."""
    calls = [(call_id, url) for call_id, (url, call_scope) in _inflight.items() if _reaches(scope, call_scope)]
    _aborted.update(call_id for call_id, _ in calls)
    async with asyncio.TaskGroup() as group:
        for call_id, url in calls:
            if url is not None:
                group.create_task(_cancel_on(url, call_id))
    return len(calls)


@contextlib.contextmanager
def booting_call(abort_scope: str | None) -> Iterator[Callable[[], bool]]:
    """Count a call whose worker is still starting among the in-flight ones, and yield whether a rollout abort has
    reached it."""
    call_id = uuid.uuid4().hex
    _inflight[call_id] = (None, abort_scope)
    try:
        yield lambda: call_id in _aborted
    finally:
        _inflight.pop(call_id, None)
        _aborted.discard(call_id)


@functools.cache
def _caller_scope() -> str:
    return f"{rollout_limits.current_scope()}-{uuid.uuid4().hex[:12]}"


def encode_report(item: Any) -> str:
    return base64.b64encode(pickle.dumps(item)).decode()


def decode_report(payload: str) -> Any:
    return pickle.loads(base64.b64decode(payload))


async def call_on_worker(
    worker_url: str,
    fn: Callable,
    kwargs: dict[str, Any],
    *,
    env: dict[str, str],
    on_child_reports: Callable[[list[Any]], None] | None = None,
    abort_scope: str | None = None,
) -> Any:
    """Run ``fn(**kwargs)`` on the worker and return its result, as ``call_agent_function``'s ``subproc`` would."""
    call_id = uuid.uuid4().hex
    body = {
        "call_id": call_id,
        "fn": function_path(fn),
        "kwargs": kwargs,
        "env": env,
        "scope": _caller_scope(),
    }
    json.dumps(body)
    async with _client(worker_url) as client:
        _inflight[call_id] = (worker_url, abort_scope)
        try:
            await _request(client, "POST", "/calls", json=body)
            return await _poll_until_done(client, call_id, on_child_reports)
        except asyncio.CancelledError:
            await asyncio.shield(_cancel(client, call_id))
            raise
        finally:
            _inflight.pop(call_id, None)
            _aborted.discard(call_id)


async def _poll_until_done(
    client: httpx.AsyncClient, call_id: str, on_child_reports: Callable[[list[Any]], None] | None
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
        if call_id in _aborted:
            raise InfraAbort(
                "WorkerCancelled", f"the rollout aborted call {call_id} on agent worker {client.base_url}"
            )


def _unpack_outcome(outcome: dict[str, Any]) -> Any:
    kind = outcome["kind"]
    if kind == "ok":
        return outcome["value"]
    if kind == "infra_abort":
        raise InfraAbort(outcome["exit_status"], outcome["message"])
    if kind == "cancelled":
        raise InfraAbort("WorkerCancelled", "the agent worker cancelled the call")
    raise RuntimeError(f"{outcome['type']}: {outcome['message']}") from RemoteTraceback(
        f'\n"""\n{outcome["traceback"]}"""'
    )


async def _request(client: httpx.AsyncClient, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
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


async def _cancel_on(worker_url: str, call_id: str) -> None:
    async with _client(worker_url) as client:
        await _cancel(client, call_id)


async def _cancel(client: httpx.AsyncClient, call_id: str) -> None:
    try:
        await asyncio.wait_for(client.delete(f"/calls/{call_id}"), timeout=_CANCEL_TIMEOUT_S)
    except Exception as exc:
        logger.warning(f"cancelling call {call_id} on agent worker {client.base_url} failed: {exc!r}")


def _client(worker_url: str) -> httpx.AsyncClient:
    socket_options = [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
        (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPIDLE", 4), 60),
        (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPINTVL", 5), 30),
        (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPCNT", 6), 5),
    ]  # config-access-exempt: TCP keepalive constant names vary across supported host platforms
    return httpx.AsyncClient(
        base_url=worker_url.rstrip("/"),
        transport=httpx.AsyncHTTPTransport(socket_options=socket_options),
        timeout=httpx.Timeout(MAX_POLL_WAIT_S + 30.0),
    )
