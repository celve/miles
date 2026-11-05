"""Sandbox start functions, and ``--custom-agent-function-mode sandbox``, which runs each call on a sandbox of its own.

A start function (``StartFn``) starts one sandbox and returns ``(close_fn, base_url)``. In sandbox mode
(``--custom-agent-function-sandbox-start-path``) it is called as ``start(label, "")``, and ``base_url`` is an agent
worker (``miles.rollout.agentic.worker``) serving there. The call runs on it as ``worker`` mode runs it, and
``close_fn``, which copies out what the call left in the sandbox and frees it, runs however the call ends.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

from miles.rollout.agentic.errors import InfraAbort
from miles.rollout.agentic.remote import booting_call, call_on_worker
from miles.rollout.agentic.rollout_limits import UNSHARED_ENV_VAR

logger = logging.getLogger(__name__)

# A provider's "start one sandbox" hook: (task_id, tasks_dir) -> (close_fn, base_url).
StartFn = Callable[[str, str], tuple[Callable[[], None], str]]

_HEALTH_DEADLINE_S = 120.0
_HEALTH_POLL_S = 0.2


async def create_once(start_fn: StartFn, task_id: str, tasks_dir: str, *, logger: logging.Logger) -> tuple[Any, str]:
    """One sandbox-create attempt, safe against cancellation mid-create."""
    result: list[tuple[Any, str]] = []
    done = threading.Event()

    def _start() -> tuple[Any, str]:
        try:
            result.append(start_fn(task_id, tasks_dir))
        finally:
            done.set()
        return result[0]

    try:
        return await asyncio.to_thread(_start)
    except asyncio.CancelledError:

        def _reap() -> None:
            done.wait()
            for close_fn, _url in result:
                try:
                    close_fn()
                    logger.info(f"Closed sandbox orphaned by cancelled episode: {task_id}")
                except Exception as e:
                    logger.warning(f"Failed to close orphaned sandbox for {task_id}: {e}")

        # not a daemon: a subproc agent call waits for it before its process exits
        threading.Thread(target=_reap, name=f"tb2-sandbox-reap-{task_id}", daemon=False).start()
        raise


async def call_in_sandbox(
    start_fn: StartFn,
    fn: Callable,
    kwargs: dict[str, Any],
    *,
    env: dict[str, str],
    on_child_reports: Callable[[list[Any]], None] | None = None,
    abort_scope: str | None = None,
) -> Any:
    """Run ``fn(**kwargs)`` on the agent worker in a sandbox ``start_fn`` starts for this call, as ``worker`` mode
    would, and close the sandbox after."""
    label = str((kwargs.get("metadata") or {}).get("instance_id") or fn.__name__)
    close_fn = None
    try:
        with booting_call(abort_scope) as aborted:
            close_fn, worker_url = await _start(start_fn, label)
            await _wait_healthy(worker_url, label, aborted)
        return await call_on_worker(
            worker_url,
            fn,
            kwargs,
            env={**env, UNSHARED_ENV_VAR: "sandbox"},
            on_child_reports=on_child_reports,
            abort_scope=abort_scope,
        )
    finally:
        if close_fn is not None:
            await _close(close_fn, label)


async def _start(start_fn: StartFn, label: str) -> tuple[Callable[[], None], str]:
    try:
        return await create_once(start_fn, label, "", logger=logger)
    except Exception as exc:
        raise InfraAbort("SandboxWorkerUnavailable", f"starting the sandbox worker for {label}: {exc!r}") from exc


async def _wait_healthy(worker_url: str, label: str, aborted: Callable[[], bool]) -> None:
    deadline = time.monotonic() + _HEALTH_DEADLINE_S
    async with httpx.AsyncClient(base_url=worker_url, timeout=10) as client:
        while True:
            if aborted():
                raise InfraAbort(
                    "WorkerCancelled", f"a rollout abort landed while the sandbox worker for {label} booted"
                )
            if await _is_healthy(client):
                return
            if time.monotonic() > deadline:
                raise InfraAbort(
                    "SandboxWorkerUnavailable",
                    f"the sandbox worker for {label} at {worker_url} not healthy after {_HEALTH_DEADLINE_S:.0f}s",
                )
            await asyncio.sleep(_HEALTH_POLL_S)


async def _is_healthy(client: httpx.AsyncClient) -> bool:
    try:
        return (await client.get("/health")).status_code == 200
    except httpx.TransportError:
        return False


async def _close(close_fn: Callable[[], None], label: str) -> None:
    try:
        await asyncio.to_thread(close_fn)
    except Exception as exc:
        logger.warning(f"closing the sandbox worker for {label} failed: {exc!r}")
