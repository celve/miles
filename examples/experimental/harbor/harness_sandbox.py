"""One Harbor agent server per trial, each in its own CPU sandbox on the sandbox service, reached through its gateway.

The template's contract: its start command waits for ``BOOT_CONFIG`` (``{"env": {...}}``), adds that env, and
serves ``miles_agent_server.py`` on ``SERVER_PORT``; softlash/LIN-2511 ``scripts/harness`` builds one. The gateway
504s a response whose headers take over 30 s and cuts one over ~90 s, so a trial is submitted to ``/runs`` and
long-polled rather than held on ``/run``.
"""

import asyncio
import json
import logging
import os
import socket
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from miles.rollout.agentic.agent_function import InfraAbort
from miles.rollout.agentic.credentials import PROVIDER_CREDENTIALS, resolve_provider_api_key

logger = logging.getLogger(__name__)

BOOT_CONFIG = "/tmp/harness_boot.json"
SERVER_PORT = 8000
# below the gateway's 30 s header cap, with room for the round trip
_POLL_WAIT_S = 25.0
_HEALTH_DEADLINE_S = 120.0
_TRANSIENT_ATTEMPTS = 5
_TRANSIENT_STATUSES = {502, 503, 504}
_KILL_ATTEMPTS = 3
# what a trial leaves that analysis reads (LIN-2510's ab_report), small enough to copy before the kill
_TRIAL_RECORD_FILES = ("result.json", "config.json", "trial.log", "agent/timings.jsonl")


@dataclass(frozen=True)
class SandboxServer:
    sandbox_id: str
    base_url: str
    headers: dict[str, str]
    sandbox: Any = field(repr=False, compare=False)


def _owner() -> str:
    return os.getenv("AGENT_SERVER_SANDBOX_OWNER", "").strip() or socket.gethostname()


def _e2b_api_key() -> str:
    spec = PROVIDER_CREDENTIALS["e2b"]
    (key_env,) = spec["key_env_vars"]
    return os.getenv(key_env, "").strip() or resolve_provider_api_key(
        key_env, spec["file_env_var"], spec["default_path"]
    )


def _gateway_url() -> str:
    # header routing goes through the same gateway the e2b SDK reaches envd by
    url = os.getenv("E2B_SANDBOX_URL", "").strip().rstrip("/")
    if not url:
        raise ValueError("set E2B_SANDBOX_URL to the sandbox service's gateway to reach per-trial agent servers")
    return url


@asynccontextmanager
async def open_server(template: str, label: str, lifetime_s: int) -> AsyncIterator[SandboxServer]:
    """A booted agent server in a fresh sandbox from ``template``, killed when the block exits, however it exits."""
    from e2b import AsyncSandbox

    api_key = _e2b_api_key()
    gateway = _gateway_url()
    t0 = time.monotonic()
    try:
        # the lifetime outlasts the trial cap, so a sandbox this process never got to kill still expires
        sandbox = await AsyncSandbox.create(
            template,
            timeout=lifetime_s,
            api_key=api_key,
            metadata={"owner": _owner(), "role": "harness", "trial": label[:63]},
        )
    except Exception as exc:
        raise InfraAbort("HarnessSandboxUnavailable", f"creating a {template} sandbox for {label}: {exc!r}") from exc
    try:
        server = SandboxServer(
            sandbox_id=sandbox.sandbox_id,
            base_url=gateway,
            headers={"E2b-Sandbox-Id": sandbox.sandbox_id, "E2b-Sandbox-Port": str(SERVER_PORT)},
            sandbox=sandbox,
        )
        try:
            await sandbox.files.write(BOOT_CONFIG, json.dumps({"env": {"E2B_API_KEY": api_key}}))
            await _wait_healthy(server)
        except InfraAbort:
            raise
        except Exception as exc:
            raise InfraAbort(
                "HarnessSandboxUnavailable", f"booting sandbox {sandbox.sandbox_id} for {label}: {exc!r}"
            ) from exc
        logger.info(f"harness sandbox {sandbox.sandbox_id} for {label} serving after {time.monotonic() - t0:.1f}s")
        yield server
    finally:
        # a cancelled trial (its cap, or SIGTERM) is cancelled once, so this still runs to the end
        await _kill(sandbox, label)


async def _wait_healthy(server: SandboxServer) -> None:
    deadline = time.monotonic() + _HEALTH_DEADLINE_S
    async with httpx.AsyncClient(base_url=server.base_url, headers=server.headers, timeout=10) as client:
        while True:
            try:
                if (await client.get("/health")).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise InfraAbort(
                    "HarnessSandboxUnavailable",
                    f"agent server in sandbox {server.sandbox_id} not healthy after {_HEALTH_DEADLINE_S:.0f}s",
                )
            await asyncio.sleep(0.2)


async def _kill(sandbox, label: str) -> None:
    for attempt in range(_KILL_ATTEMPTS):
        try:
            await sandbox.kill()
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"killing harness sandbox {sandbox.sandbox_id} for {label} (attempt {attempt + 1}): {exc!r}"
            )
            await asyncio.sleep(2 * (attempt + 1))
    logger.error(f"harness sandbox {sandbox.sandbox_id} for {label} left to expire at its timeout")


async def run_job(server: SandboxServer, request: dict[str, Any]) -> dict[str, Any]:
    """Submit ``request`` to the server's ``/runs`` and long-poll until it is done; return the ``/run``-shaped reply."""
    run_id = uuid.uuid4().hex
    timeout = httpx.Timeout(10, read=_POLL_WAIT_S + 10)
    async with httpx.AsyncClient(base_url=server.base_url, headers=server.headers, timeout=timeout) as client:
        # a resent submit finds the run its run_id already started
        await _call(server, client, "POST", "/runs", json={**request, "run_id": run_id})
        while True:
            status = await _call(server, client, "GET", f"/runs/{run_id}", params={"wait": _POLL_WAIT_S})
            if status.get("state") == "done":
                return status.get("result") or {}


async def _call(server: SandboxServer, client: httpx.AsyncClient, method: str, path: str, **kwargs) -> dict[str, Any]:
    where = f"agent server in sandbox {server.sandbox_id}"
    for attempt in range(_TRANSIENT_ATTEMPTS):
        try:
            response = await client.request(method, path, **kwargs)
        except httpx.TransportError as exc:
            error: Exception = exc
        else:
            if response.status_code == 404 and method == "GET":
                raise InfraAbort("AgentServerLost", f"{where} has no run at {path}")
            if response.status_code not in _TRANSIENT_STATUSES:
                try:
                    response.raise_for_status()
                    return response.json()
                except (httpx.HTTPError, ValueError) as exc:
                    raise InfraAbort("AgentServerUnreachable", f"{where}: {method} {path}: {exc!r}") from exc
            error = httpx.HTTPStatusError(f"{response.status_code}", request=response.request, response=response)
        await asyncio.sleep(min(2**attempt, 10))
    raise InfraAbort("AgentServerUnreachable", f"{where}: {method} {path}: {error!r}") from error


async def copy_trial_record(server: SandboxServer, trial_dir: str, local_root: Path) -> Path:
    """Copy the trial's record files out of its sandbox into ``local_root/<trial name>``; missing ones are skipped."""
    local = local_root / Path(trial_dir).name
    for name in _TRIAL_RECORD_FILES:
        try:
            data = await server.sandbox.files.read(f"{trial_dir}/{name}", format="bytes")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"trial record {name} not copied from sandbox {server.sandbox_id}: {exc!r}")
            continue
        (local / name).parent.mkdir(parents=True, exist_ok=True)
        (local / name).write_bytes(bytes(data))
    return local
