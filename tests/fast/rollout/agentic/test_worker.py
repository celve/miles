"""A ``worker`` agent call runs in a ``subproc`` child on an agent worker yet behaves like the awaited coroutine."""

import asyncio
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from aiohttp import web

from miles.rollout.agentic import remote
from miles.rollout.agentic.agent_function import InfraAbort, call_agent_function
from miles.rollout.agentic.remote import AgentWorkerEndpoint
from miles.rollout.agentic.rollout_limits import current_scope
from miles.rollout.agentic.worker import WorkerConfig, create_app
from miles.utils.misc import call_agent_abort_hook
from miles.utils.test_utils import agent_function_stubs as stubs


class _Served:
    """An aiohttp app on a free local port, on a loop of its own, as a separate worker host would be."""

    def __init__(self, app: web.Application):
        self._loop = asyncio.new_event_loop()
        self._runner = web.AppRunner(app)
        self._loop.run_until_complete(self._runner.setup())
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        self._loop.run_until_complete(site.start())
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop).result(timeout=60)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)


@pytest.fixture
def serve():
    served: list[_Served] = []

    def start(config: WorkerConfig | None = None, app: web.Application | None = None) -> AgentWorkerEndpoint:
        served.append(_Served(app or create_app(config)))
        return AgentWorkerEndpoint(url=served[-1].url)

    yield start
    for s in served:
        s.close()


def _on_worker(endpoint, fn, extra_env=None, on_child_reports=None, **kwargs):
    return asyncio.run(
        call_agent_function(
            fn, mode="worker", worker=endpoint, extra_env=extra_env, on_child_reports=on_child_reports, **kwargs
        )
    )


def _wait_for(path: Path, timeout_s: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    return path.exists()


def test_the_call_runs_in_a_child_process_on_the_worker(serve):
    reported = _on_worker(serve(), stubs.report_pid)

    assert reported["pid"] != os.getpid()


def test_the_child_gets_the_forwarded_env_except_what_the_worker_keeps(serve, monkeypatch):
    monkeypatch.setenv("LIN2510_KEPT", "the worker's own")
    endpoint = serve(WorkerConfig(keep_env=("LIN2510_KEPT",)))

    seen = _on_worker(
        endpoint,
        stubs.read_env,
        extra_env={"LIN2510_KEPT": "the caller's", "LIN2510_FORWARDED": "the caller's"},
        names=["LIN2510_KEPT", "LIN2510_FORWARDED"],
    )

    assert seen == {"LIN2510_KEPT": "the worker's own", "LIN2510_FORWARDED": "the caller's"}


def test_the_child_shares_the_callers_rollout_limits(serve):
    assert _on_worker(serve(), stubs.report_limits_scope) == current_scope()


def test_reports_reach_the_caller_in_order(serve):
    received: list = []

    result = _on_worker(
        serve(), stubs.report_then_return, on_child_reports=received.extend, items=[{"n": 1}, {"n": 2}]
    )

    assert result == "done"
    assert received == [{"n": 1}, {"n": 2}]


def test_an_infra_abort_keeps_its_cause_across_the_worker(serve):
    with pytest.raises(InfraAbort) as raised:
        _on_worker(serve(), stubs.abort_for_infra)

    assert raised.value.exit_status == "SandboxUnavailable"


def test_an_agent_error_arrives_with_its_traceback(serve):
    with pytest.raises(RuntimeError, match="ValueError: agent failed on purpose") as raised:
        _on_worker(serve(), stubs.fail)

    assert "agent_function_stubs.py" in str(raised.value.__cause__)


def test_cancelling_the_caller_runs_the_agent_cleanup_on_the_worker(serve, tmp_path):
    endpoint = serve()
    pid_file, cleaned_up = tmp_path / "pid", tmp_path / "cleaned_up"

    async def cancel_once_running():
        task = asyncio.create_task(
            call_agent_function(
                stubs.wait_until_cancelled,
                mode="worker",
                worker=endpoint,
                pid_file=str(pid_file),
                cleaned_up=str(cleaned_up),
            )
        )
        while not pid_file.exists():
            await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_once_running())

    assert _wait_for(cleaned_up)


def test_a_rollout_abort_cancels_the_calls_running_on_workers(serve, tmp_path):
    """Both rollout aborts and the hot-restart takeover go through call_agent_abort_hook."""
    endpoint = serve()
    pid_file, cleaned_up = tmp_path / "pid", tmp_path / "cleaned_up"

    async def abort_while_running():
        task = asyncio.create_task(
            call_agent_function(
                stubs.wait_until_cancelled,
                mode="worker",
                worker=endpoint,
                pid_file=str(pid_file),
                cleaned_up=str(cleaned_up),
            )
        )
        while not pid_file.exists():
            await asyncio.sleep(0.1)
        await call_agent_abort_hook(SimpleNamespace(custom_agent_function_path=None))
        with pytest.raises(InfraAbort) as raised:
            await task
        return raised.value.exit_status

    assert asyncio.run(abort_while_running()) == "WorkerCancelled"
    assert _wait_for(cleaned_up)


def test_secrets_are_not_forwarded_by_value(monkeypatch):
    monkeypatch.setenv("HARBOR_ADMIN_SECRET", "s3cret")
    monkeypatch.setenv("HARBOR_ADMIN_SECRET_FILE", "/secrets/harbor_admin")
    monkeypatch.setenv("E2B_API_KEY", "e2b_value")
    monkeypatch.setenv("HARBOR_TASKS_DIR", "/tasks")

    env = remote.forwarded_env(["HARBOR_*", "E2B_*"])

    assert env["HARBOR_ADMIN_SECRET_FILE"] == "/secrets/harbor_admin"
    assert env["HARBOR_TASKS_DIR"] == "/tasks"
    assert "HARBOR_ADMIN_SECRET" not in env
    assert "E2B_API_KEY" not in env


def test_a_call_whose_caller_vanished_is_cancelled(serve, tmp_path, monkeypatch):
    monkeypatch.setattr("miles.rollout.agentic.worker._REAP_INTERVAL_S", 0.2)
    endpoint = serve(WorkerConfig(orphan_timeout_s=1.0))
    pid_file, cleaned_up = tmp_path / "pid", tmp_path / "cleaned_up"

    httpx.post(
        f"{endpoint.url}/calls",
        json={
            "call_id": "orphan",
            "fn": remote.function_path(stubs.wait_until_cancelled),
            "kwargs": {"pid_file": str(pid_file), "cleaned_up": str(cleaned_up)},
            "env": {},
            "scope": "orphan-test",
        },
    ).raise_for_status()

    assert _wait_for(pid_file)
    assert _wait_for(cleaned_up, timeout_s=30)


def test_a_resubmitted_call_runs_once(serve, tmp_path):
    endpoint = serve()
    body = {
        "call_id": "same",
        "fn": remote.function_path(stubs.report_pid),
        "kwargs": {},
        "env": {},
        "scope": "resubmit-test",
    }

    for _ in range(2):
        httpx.post(f"{endpoint.url}/calls", json=body).raise_for_status()
    done = httpx.get(f"{endpoint.url}/calls/same", params={"wait": 20}).json()

    assert done["state"] == "done"
    assert done["outcome"]["kind"] == "ok"
    assert httpx.get(f"{endpoint.url}/health").json()["running"] == 0


def test_a_worker_that_lost_the_call_discards_the_sample(serve):
    async def accept(request):
        return web.json_response({"call_id": "x"})

    forgetful = web.Application()
    forgetful.add_routes([web.post("/calls", accept)])

    with pytest.raises(InfraAbort) as raised:
        _on_worker(serve(app=forgetful), stubs.report_pid)

    assert raised.value.exit_status == "WorkerLost"


def test_an_unreachable_worker_discards_the_sample(monkeypatch):
    monkeypatch.setattr(remote, "_REQUEST_ATTEMPTS", 1)

    with pytest.raises(InfraAbort) as raised:
        _on_worker(AgentWorkerEndpoint(url="http://127.0.0.1:9"), stubs.report_pid)

    assert raised.value.exit_status == "WorkerUnreachable"


def test_a_kwarg_the_protocol_cannot_carry_fails_before_the_call(serve):
    with pytest.raises(TypeError):
        _on_worker(serve(), stubs.report_pid, payload=object())


def test_the_worker_side_imports_without_torch_sglang_or_ray():
    """The sandbox and CPU-box images carry no ML stack; agents that report lifecycle phases still pull ray in."""
    code = textwrap.dedent("""
        import sys

        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("torch", "sglang", "ray"):
                    raise ImportError("blocked: " + name)

        sys.meta_path.insert(0, Block())
        import miles.rollout.agentic.worker
        """)
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[4])
