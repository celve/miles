"""An agent worker: runs agent-function calls through ``subproc`` on this host, for rollouts on another one.

A rollout with ``--custom-agent-function-mode worker`` sends each call here (protocol: ``miles.rollout.agentic.remote``),
so the agent's processes, and the CPU they burn, live on this host instead of the GPU box. The agent function must be
importable here under the name the rollout loaded it by, and the paths in its forwarded env must exist here.

    python -m miles.rollout.agentic.worker --host 0.0.0.0 --port 11100
"""

import argparse
import asyncio
import fnmatch
import importlib
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

from miles.rollout.agentic import rollout_limits
from miles.rollout.agentic.agent_function import call_agent_function
from miles.rollout.agentic.errors import InfraAbort
from miles.rollout.agentic.remote import MAX_POLL_WAIT_S, encode_report

logger = logging.getLogger(__name__)

_REAP_INTERVAL_S = 10.0


@dataclass
class _Call:
    reports: list[str] = field(default_factory=list)
    outcome: dict[str, Any] | None = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    last_polled: float = field(default_factory=time.monotonic)
    done_at: float | None = None
    task: asyncio.Task | None = None

    def add_reports(self, items: list[Any]) -> None:
        self.reports.extend(encode_report(item) for item in items)
        self.changed.set()

    def finish(self, outcome: dict[str, Any]) -> None:
        self.outcome = outcome
        self.done_at = time.monotonic()
        self.changed.set()


@dataclass(frozen=True)
class WorkerConfig:
    orphan_timeout_s: float = 120.0
    retention_s: float = 3600.0
    # env var names (fnmatch) whose value on this host wins over a call's, such as paths that differ here
    keep_env: tuple[str, ...] = ()

    def call_env(self, forwarded: dict[str, str]) -> dict[str, str]:
        return {k: v for k, v in forwarded.items() if not any(fnmatch.fnmatchcase(k, p) for p in self.keep_env)}


def _resolve(path: str) -> Any:
    module_name, _, qualname = path.partition(":")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)  # config-access-exempt: resolves a dotted import path
    return obj


async def _run(call: _Call, fn_path: str, kwargs: dict[str, Any], env: dict[str, str]) -> None:
    try:
        value = await call_agent_function(
            _resolve(fn_path), mode="subproc", on_child_reports=call.add_reports, extra_env=env, **kwargs
        )
        json.dumps(value)  # the protocol carries JSON; a result it cannot is this call's error, below
        call.finish({"kind": "ok", "value": value})
    except InfraAbort as exc:
        call.finish({"kind": "infra_abort", "exit_status": exc.exit_status, "message": str(exc)})
    except asyncio.CancelledError:
        call.finish({"kind": "cancelled"})
    except Exception as exc:
        call.finish(
            {"kind": "error", "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        )


def create_app(config: WorkerConfig | None = None) -> web.Application:
    config = config or WorkerConfig()
    calls: dict[str, _Call] = {}

    async def submit(request: web.Request) -> web.Response:
        body = await request.json()
        call_id = str(body["call_id"])
        if call_id not in calls:
            call = calls[call_id] = _Call()
            # the caller's rollout-limits scope: its limits are per rollout, wherever its calls run
            env = {**config.call_env(body.get("env", {})), rollout_limits.SCOPE_ENV_VAR: str(body["scope"])}
            call.task = asyncio.create_task(_run(call, body["fn"], body["kwargs"], env))
        return web.json_response({"call_id": call_id})

    async def poll(request: web.Request) -> web.Response:
        call = calls.get(request.match_info["call_id"])
        if call is None:
            raise web.HTTPNotFound(text="unknown call")
        after = int(request.query.get("after", 0))
        wait = min(max(float(request.query.get("wait", 0)), 0.0), MAX_POLL_WAIT_S)
        call.last_polled = time.monotonic()
        call.changed.clear()
        if len(call.reports) <= after and call.outcome is None:
            try:
                await asyncio.wait_for(call.changed.wait(), timeout=wait)
            except TimeoutError:
                pass
        call.last_polled = time.monotonic()
        return web.json_response(
            {
                "state": "running" if call.outcome is None else "done",
                "reports": call.reports[after:],
                "next": len(call.reports),
                "outcome": call.outcome,
            }
        )

    async def cancel(request: web.Request) -> web.Response:
        call_id = request.match_info["call_id"]
        call = calls.get(call_id)
        if call is None:
            # a cancel can overtake its own submit; the tombstone makes that submit start nothing
            calls[call_id] = tombstone = _Call()
            tombstone.finish({"kind": "cancelled"})
        elif call.task is not None and not call.task.done():
            call.task.cancel()
        return web.json_response({"cancelled": True})

    async def health(request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "running": sum(c.outcome is None for c in calls.values())})

    async def reap(app: web.Application) -> None:
        while True:
            await asyncio.sleep(_REAP_INTERVAL_S)
            now = time.monotonic()
            for call_id, call in list(calls.items()):
                if call.outcome is None and now - call.last_polled > config.orphan_timeout_s:
                    logger.warning(f"cancelling call {call_id}: no poll for {config.orphan_timeout_s:.0f}s")
                    call.task.cancel()
                elif call.done_at is not None and now - call.done_at > config.retention_s:
                    del calls[call_id]

    async def start_reaper(app: web.Application):
        task = asyncio.create_task(reap(app))
        yield
        task.cancel()
        for call in calls.values():
            if call.task is not None and not call.task.done():
                call.task.cancel()

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.add_routes(
        [
            web.post("/calls", submit),
            web.get("/calls/{call_id}", poll),
            web.delete("/calls/{call_id}", cancel),
            web.get("/health", health),
        ]
    )
    app.cleanup_ctx.append(start_reaper)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11100)
    parser.add_argument(
        "--orphan-timeout",
        type=float,
        default=WorkerConfig.orphan_timeout_s,
        help="Cancel a call nobody has polled for this many seconds (its caller is gone).",
    )
    parser.add_argument(
        "--keep-env",
        nargs="*",
        default=[],
        help="Env var names (fnmatch patterns) whose value on this host wins over the one a call forwards, "
        "e.g. HARBOR_TASKS_DIR when the task dirs live elsewhere here.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = WorkerConfig(orphan_timeout_s=args.orphan_timeout, keep_env=tuple(args.keep_env))
    web.run_app(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
