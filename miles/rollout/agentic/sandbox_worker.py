"""``--custom-agent-function-mode sandbox``: each call on an agent worker of its own, in a fresh sandbox.

The sandbox comes from ``--custom-agent-function-sandbox-template`` on an E2B-API sandbox service. Its start command
waits for ``BOOT_CONFIG`` (``{"env": {...}}``), adds that env, and serves ``miles.rollout.agentic.worker`` on
``WORKER_PORT``. The call reaches it through the service's gateway (``E2B_SANDBOX_URL``, header routing), runs as
``worker`` mode runs it, and the sandbox is killed when the call ends, however it ends.
"""

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from miles.rollout.agentic.credentials import PROVIDER_CREDENTIALS, resolve_provider_api_key
from miles.rollout.agentic.errors import InfraAbort
from miles.rollout.agentic.rollout_limits import UNSHARED_ENV_VAR
from miles.rollout.agentic.remote import AgentWorkerEndpoint, aborts_so_far, call_on_worker

logger = logging.getLogger(__name__)

BOOT_CONFIG = "/tmp/agent_worker_boot.json"
WORKER_PORT = 11100
_HEALTH_DEADLINE_S = 120.0
_KILL_ATTEMPTS = 3
# matches a worker's subproc agent processes (agent_function._CHILD_BOOTSTRAP), but not this probe's own command line
_AGENT_PROCESS_PROBE = "grep -qsa '[_]child_main' /proc/[0-9]*/cmdline"
# Harbor's convention: an agent's result names its trial dir here, which goes with the sandbox
RECORDS_KEY = "trial_dir"


@dataclass(frozen=True)
class SandboxWorkerConfig:
    template: str
    owner: str
    lifetime_s: int
    boot_env_file: str | None = None
    records_dir: str | None = None
    records: tuple[str, ...] = ()


def boot_env(config: SandboxWorkerConfig) -> dict[str, str]:
    """The env a sandbox's worker boots with: the service key, for sandboxes its agent creates, plus the boot file's."""
    spec = PROVIDER_CREDENTIALS["e2b"]
    (key_env,) = spec["key_env_vars"]
    env = {key_env: resolve_provider_api_key(key_env, spec["file_env_var"], spec["default_path"])}
    if config.boot_env_file:
        extra = json.loads(Path(config.boot_env_file).expanduser().read_text())
        if not isinstance(extra, dict) or not all(isinstance(v, str) for v in extra.values()):
            raise ValueError(f"{config.boot_env_file} must hold a JSON object of strings")
        env.update(extra)
    return env


def _gateway_url() -> str:
    url = os.getenv("E2B_SANDBOX_URL", "").strip().rstrip("/")
    if not url:
        raise ValueError("--custom-agent-function-mode sandbox reaches its workers through E2B_SANDBOX_URL; set it")
    return url


async def call_in_sandbox(
    config: SandboxWorkerConfig,
    fn: Callable,
    kwargs: dict[str, Any],
    *,
    env: dict[str, str],
    on_child_reports: Callable[[list[Any]], None] | None = None,
) -> Any:
    """Run ``fn(**kwargs)`` on a worker in a sandbox created for this call, as ``worker`` mode would."""
    label = str((kwargs.get("metadata") or {}).get("instance_id") or fn.__name__)
    # a rollout abort while the sandbox boots finds no submitted call to cancel, so this call checks for one itself
    aborts = aborts_so_far()
    async with open_worker(config, label, boot_env(config), aborts=aborts) as (endpoint, sandbox):
        _raise_if_aborted(aborts, label)
        # every call has a host of its own, so the rollout's flock-based limits would each limit one agent
        env = {**env, UNSHARED_ENV_VAR: "sandbox"}
        value = await call_on_worker(endpoint, fn, kwargs, env=env, on_child_reports=on_child_reports)
        return await _copy_records(config, sandbox, value)


@asynccontextmanager
async def open_worker(
    config: SandboxWorkerConfig, label: str, env: dict[str, str], aborts: int | None = None
) -> AsyncIterator[tuple[AgentWorkerEndpoint, Any]]:
    from e2b import AsyncSandbox  # an optional dependency, needed only in this mode

    gateway = _gateway_url()
    t0 = time.monotonic()
    try:
        # the lifetime outlasts the call, so a sandbox this process never got to kill still expires
        sandbox = await AsyncSandbox.create(
            config.template,
            timeout=config.lifetime_s,
            api_key=env["E2B_API_KEY"],
            metadata={"owner": config.owner, "role": "agent-worker", "call": label[:63]},
        )
    except Exception as exc:
        raise InfraAbort(
            "SandboxWorkerUnavailable", f"creating a {config.template} sandbox for {label}: {exc!r}"
        ) from exc
    try:
        endpoint = AgentWorkerEndpoint(
            url=gateway, headers={"E2b-Sandbox-Id": sandbox.sandbox_id, "E2b-Sandbox-Port": str(WORKER_PORT)}
        )
        try:
            await sandbox.files.write(BOOT_CONFIG, json.dumps({"env": env}))
            await _wait_healthy(endpoint, label, aborts)
        except InfraAbort:
            raise
        except Exception as exc:
            raise InfraAbort(
                "SandboxWorkerUnavailable", f"booting sandbox {sandbox.sandbox_id} for {label}: {exc!r}"
            ) from exc
        logger.info(
            f"agent worker sandbox {sandbox.sandbox_id} for {label} serving after {time.monotonic() - t0:.1f}s"
        )
        yield endpoint, sandbox
    finally:
        # a cancelled call is cancelled once, so this still runs to the end
        await _wait_for_agent_exit(sandbox, label)
        await _kill(sandbox, label)


def _raise_if_aborted(aborts: int | None, label: str) -> None:
    if aborts is not None and aborts_so_far() != aborts:
        raise InfraAbort("WorkerCancelled", f"a rollout abort landed while the sandbox worker for {label} booted")


async def _wait_healthy(endpoint: AgentWorkerEndpoint, label: str, aborts: int | None) -> None:
    deadline = time.monotonic() + _HEALTH_DEADLINE_S
    async with httpx.AsyncClient(base_url=endpoint.url, headers=endpoint.headers, timeout=10) as client:
        while True:
            _raise_if_aborted(aborts, label)
            try:
                if (await client.get("/health")).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise InfraAbort(
                    "SandboxWorkerUnavailable",
                    f"agent worker in sandbox {endpoint.headers['E2b-Sandbox-Id']} not healthy after "
                    f"{_HEALTH_DEADLINE_S:.0f}s",
                )
            await asyncio.sleep(0.2)


async def _wait_for_agent_exit(sandbox: Any, label: str) -> None:
    """Wait until the worker's agent processes have exited, so a cancelled agent's own cleanup (closing the desktop it
    created) finishes before the kill: the worker reports a cancelled call done while its child is still cleaning up.
    """
    # deferred: agent_function imports this module
    from miles.rollout.agentic.agent_function import _CANCEL_GRACE_S, _LEFTOVER_CLEANUP_S

    # the worker's subproc SIGKILLs its child by then, so nothing is left to wait for after it
    deadline_s = int(_CANCEL_GRACE_S + _LEFTOVER_CLEANUP_S) + 10
    try:
        await sandbox.commands.run(
            f"timeout {deadline_s} bash -c 'while {_AGENT_PROCESS_PROBE}; do sleep 0.5; done'", timeout=deadline_s + 30
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"agent processes in sandbox {sandbox.sandbox_id} for {label} not seen to exit: {exc!r}")


async def _kill(sandbox: Any, label: str) -> None:
    for attempt in range(_KILL_ATTEMPTS):
        try:
            await sandbox.kill()
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"killing agent worker sandbox {sandbox.sandbox_id} for {label} (attempt {attempt + 1}): {exc!r}"
            )
            await asyncio.sleep(2 * (attempt + 1))
    logger.error(f"agent worker sandbox {sandbox.sandbox_id} for {label} left to expire at its timeout")


async def _copy_records(config: SandboxWorkerConfig, sandbox: Any, value: Any) -> Any:
    """Copy ``config.records`` from the dir the result names out of the sandbox, and name the copy instead."""
    if not config.records_dir or not isinstance(value, dict) or not value.get(RECORDS_KEY):
        return value
    remote = str(value[RECORDS_KEY])
    local = Path(config.records_dir) / Path(remote).name
    for name in config.records:
        try:
            data = await sandbox.files.read(f"{remote}/{name}", format="bytes")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"record {name} not copied from sandbox {sandbox.sandbox_id}: {exc!r}")
            continue
        (local / name).parent.mkdir(parents=True, exist_ok=True)
        (local / name).write_bytes(bytes(data))
    return {**value, RECORDS_KEY: str(local)}
