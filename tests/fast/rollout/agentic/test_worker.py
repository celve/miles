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


def test_the_childs_rollout_limits_are_the_callers_and_no_other_hosts(serve):
    endpoint = serve()

    scopes = {_on_worker(endpoint, stubs.report_limits_scope) for _ in range(2)}

    assert len(scopes) == 1
    (scope,) = scopes
    assert scope.startswith(f"{current_scope()}-") and scope != current_scope()


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
            await asyncio.wait_for(task, timeout=60)
        return raised.value.exit_status

    assert asyncio.run(abort_while_running()) == "WorkerCancelled"
    assert _wait_for(cleaned_up)


def test_a_rollout_abort_leaves_the_eval_engines_calls_running(serve, tmp_path):
    """--eval-num-gpus evaluates beside training in one process; training's oversampling abort must spare eval."""
    endpoint = serve()
    files = {name: (tmp_path / f"{name}.pid", tmp_path / f"{name}.cleaned_up") for name in ("train", "eval")}

    def start(name, port):
        pid_file, cleaned_up = files[name]
        return asyncio.create_task(
            call_agent_function(
                stubs.wait_until_cancelled,
                mode="worker",
                worker=endpoint,
                abort_scope=f"127.0.0.1:{port}",
                pid_file=str(pid_file),
                cleaned_up=str(cleaned_up),
            )
        )

    async def abort_training():
        train, evaluation = start("train", 30000), start("eval", 31000)
        while not all(pid.exists() for pid, _ in files.values()):
            await asyncio.sleep(0.1)
        await call_agent_abort_hook(
            SimpleNamespace(custom_agent_function_path=None, sglang_router_ip="127.0.0.1", sglang_router_port=30000)
        )
        with pytest.raises(InfraAbort):
            await asyncio.wait_for(train, timeout=60)
        await asyncio.sleep(2)
        still_running = not evaluation.done()
        await call_agent_abort_hook(SimpleNamespace(custom_agent_function_path=None), every_scope=True)
        with pytest.raises(InfraAbort):
            await asyncio.wait_for(evaluation, timeout=60)
        return still_running

    assert asyncio.run(abort_training())
    assert _wait_for(files["train"][1]) and _wait_for(files["eval"][1])


def test_the_rollout_abort_entrypoint_cancels_a_trial_waiting_on_its_environment(serve, tmp_path, monkeypatch):
    """The oversampling abort awaits pending groups after the hook: the worker call must end, not run to its cap."""
    from miles.rollout.inference_rollout import inference_rollout_train

    async def no_engines(args):
        return []

    monkeypatch.setattr(inference_rollout_train, "get_worker_urls", no_engines)
    monkeypatch.setattr(inference_rollout_train.SampleOwnershipRecorder, "log_dropped_samples", lambda **kw: None)
    endpoint = serve()
    pid_file, cleaned_up = tmp_path / "pid", tmp_path / "cleaned_up"

    async def group():
        # as generate does: a discarded call becomes an aborted sample of its group
        try:
            await call_agent_function(
                stubs.wait_until_cancelled,
                mode="worker",
                worker=endpoint,
                pid_file=str(pid_file),
                cleaned_up=str(cleaned_up),
            )
        except InfraAbort as exc:
            return [SimpleNamespace(exit_status=exc.exit_status)]

    async def abort_mid_trial():
        pending = {asyncio.create_task(group())}
        while not pid_file.exists():
            await asyncio.sleep(0.1)
        state = SimpleNamespace(
            args=SimpleNamespace(partial_rollout=False, custom_agent_function_path=None), aborted=False
        )
        await asyncio.wait_for(inference_rollout_train.abort(state, pending, rollout_id=0), timeout=60)

    asyncio.run(abort_mid_trial())

    assert _wait_for(cleaned_up)


def test_a_cancel_that_overtakes_its_submit_keeps_the_call_from_starting(serve, tmp_path):
    """A rollout abort can DELETE a call whose POST has not reached the worker yet."""
    endpoint = serve()
    pid_file, cleaned_up = tmp_path / "pid", tmp_path / "cleaned_up"

    httpx.delete(f"{endpoint.url}/calls/late").raise_for_status()
    httpx.post(
        f"{endpoint.url}/calls",
        json={
            "call_id": "late",
            "fn": remote.function_path(stubs.wait_until_cancelled),
            "kwargs": {"pid_file": str(pid_file), "cleaned_up": str(cleaned_up)},
            "env": {},
            "scope": "late-submit-test",
        },
    ).raise_for_status()
    state = httpx.get(f"{endpoint.url}/calls/late", params={"wait": 5}).json()

    assert state["state"] == "done" and state["outcome"] == {"kind": "cancelled"}
    time.sleep(2)
    assert not pid_file.exists()
    assert httpx.get(f"{endpoint.url}/health").json()["running"] == 0


def test_an_abort_during_the_submit_keeps_the_trial_from_starting(serve, tmp_path):
    app = create_app()

    @web.middleware
    async def slow_submit(request, handler):
        if request.method == "POST":
            await asyncio.sleep(2)
        return await handler(request)

    app.middlewares.append(slow_submit)
    endpoint = serve(app=app)
    pid_file, cleaned_up = tmp_path / "pid", tmp_path / "cleaned_up"

    async def abort_during_submit():
        task = asyncio.create_task(
            call_agent_function(
                stubs.wait_until_cancelled,
                mode="worker",
                worker=endpoint,
                pid_file=str(pid_file),
                cleaned_up=str(cleaned_up),
            )
        )
        await asyncio.sleep(0.5)
        assert await remote.cancel_inflight_calls() == 1
        with pytest.raises(InfraAbort) as raised:
            await asyncio.wait_for(task, timeout=60)
        return raised.value.exit_status

    assert asyncio.run(abort_during_submit()) == "WorkerCancelled"
    time.sleep(2)
    assert not pid_file.exists()


def test_an_aborted_call_whose_cancel_is_lost_stops_polling_and_is_reaped(serve, tmp_path, monkeypatch):
    """The DELETE never arrives; the caller still stops, and the worker's orphan timer ends the child."""
    monkeypatch.setattr("miles.rollout.agentic.worker._REAP_INTERVAL_S", 0.2)
    monkeypatch.setattr(remote, "_POLL_WAIT_S", 1.0)
    endpoint = serve(WorkerConfig(orphan_timeout_s=2.0))
    pid_file, cleaned_up = tmp_path / "pid", tmp_path / "cleaned_up"

    async def lose_the_cancel(worker, call_id):
        return None

    monkeypatch.setattr(remote, "_cancel_on", lose_the_cancel)

    async def abort_with_a_lost_cancel():
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
        await remote.cancel_inflight_calls()
        with pytest.raises(InfraAbort) as raised:
            await asyncio.wait_for(task, timeout=30)
        return raised.value.exit_status

    assert asyncio.run(abort_with_a_lost_cancel()) == "WorkerCancelled"
    assert _wait_for(cleaned_up, timeout_s=30)


def test_secrets_are_not_forwarded_by_value(monkeypatch):
    monkeypatch.setenv("HARBOR_ADMIN_SECRET", "s3cret")
    monkeypatch.setenv("HARBOR_ADMIN_SECRET_FILE", "/secrets/harbor_admin")
    monkeypatch.setenv("E2B_API_KEY", "e2b_value")
    monkeypatch.setenv("HARBOR_TASKS_DIR", "/tasks")
    monkeypatch.setenv("AGENT_MAX_OUTPUT_TOKENS", "2048")
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-value")
    monkeypatch.setenv("MODAL_PROFILE", "team")

    env = remote.forwarded_env(["HARBOR_*", "E2B_*", "AGENT_*", "MODAL_*"])

    assert env["HARBOR_ADMIN_SECRET_FILE"] == "/secrets/harbor_admin"
    assert env["HARBOR_TASKS_DIR"] == "/tasks"
    # a token budget is a setting, not a credential
    assert env["AGENT_MAX_OUTPUT_TOKENS"] == "2048"
    assert "HARBOR_ADMIN_SECRET" not in env
    assert "E2B_API_KEY" not in env
    assert "MODAL_TOKEN_ID" not in env
    assert env["MODAL_PROFILE"] == "team"


def test_the_default_forwarded_names_carry_every_providers_endpoint_but_no_key():
    from miles.rollout.generate_hub.agentic_tool_call import _DEFAULT_WORKER_ENV

    for name in ("E2B_API_URL", "E2B_SANDBOX_URL", "E2B_DOMAIN", "E2B_API_KEY_FILE", "MODAL_PROFILE"):
        assert name in _DEFAULT_WORKER_ENV
    assert "E2B_API_KEY" not in _DEFAULT_WORKER_ENV


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
