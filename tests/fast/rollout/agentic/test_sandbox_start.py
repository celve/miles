import argparse
import asyncio
import threading
from types import SimpleNamespace

import pytest
from aiohttp import web

from miles.rollout.agentic import remote, sandbox_start
from miles.rollout.agentic.agent_function import InfraAbort, call_agent_function
from miles.rollout.agentic.rollout_limits import UNSHARED_ENV_VAR
from miles.rollout.agentic.worker import create_app
from miles.rollout.generate_hub import agentic_tool_call
from miles.utils.misc import call_agent_abort_hook
from miles.utils.test_utils import agent_function_stubs as stubs

from .test_worker import _rollout_args, _serve_in_thread, _wait_for


def _worker_app(sandboxes: SimpleNamespace) -> web.Application:
    app = create_app()

    @web.middleware
    async def gate(request, handler):
        sandboxes.requests.append((request.method, request.path))
        if request.path == "/health" and not sandboxes.healthy.is_set():
            raise web.HTTPServiceUnavailable()
        return await handler(request)

    app.middlewares.append(gate)
    return app


@pytest.fixture
def sandboxes():
    state = SimpleNamespace(
        started=[], closed=[], requests=[], servers=[], healthy=threading.Event(), creating=threading.Event()
    )
    state.healthy.set()
    state.creating.set()

    def start(task_id, tasks_dir):
        assert state.creating.wait(30)
        url, close = _serve_in_thread(_worker_app(state))
        state.servers.append(close)
        state.started.append((task_id, tasks_dir))
        return (lambda: state.closed.append(task_id)), url

    state.start = start
    yield state
    for close in state.servers:
        close()


def _in_sandbox(fn, start, **kwargs):
    return asyncio.run(call_agent_function(fn, mode="sandbox", sandbox_start=start, **kwargs))


def test_the_call_runs_on_the_worker_its_start_function_started_which_is_then_closed(sandboxes):
    seen = _in_sandbox(
        stubs.read_env,
        sandboxes.start,
        extra_env={"SANDBOX_TEST_FORWARDED": "the caller's"},
        names=["SANDBOX_TEST_FORWARDED"],
        metadata={"instance_id": "task-7"},
    )

    assert seen == {"SANDBOX_TEST_FORWARDED": "the caller's"}
    assert sandboxes.started == [("task-7", "")]
    assert sandboxes.closed == ["task-7"]
    assert ("POST", "/calls") in sandboxes.requests


def test_an_infra_abort_keeps_its_cause_and_the_sandbox_still_closes(sandboxes):
    with pytest.raises(InfraAbort) as raised:
        _in_sandbox(stubs.abort_for_infra, sandboxes.start)

    assert raised.value.exit_status == "SandboxUnavailable"
    assert sandboxes.closed == ["abort_for_infra"]


def test_a_sandbox_that_cannot_be_started_discards_the_sample():
    def start(task_id, tasks_dir):
        raise RuntimeError("500: snapshot not ready")

    with pytest.raises(InfraAbort) as raised:
        _in_sandbox(stubs.report_pid, start)

    assert raised.value.exit_status == "SandboxWorkerUnavailable"


def test_a_worker_that_never_turns_healthy_discards_the_sample_and_closes_its_sandbox(sandboxes, monkeypatch):
    sandboxes.healthy.clear()
    monkeypatch.setattr(sandbox_start, "_HEALTH_DEADLINE_S", 0.5)

    with pytest.raises(InfraAbort) as raised:
        _in_sandbox(stubs.report_pid, sandboxes.start)

    assert raised.value.exit_status == "SandboxWorkerUnavailable"
    assert sandboxes.closed == ["report_pid"]
    assert ("POST", "/calls") not in sandboxes.requests


def test_cancelling_the_caller_runs_the_agent_cleanup_and_closes_the_sandbox(sandboxes, tmp_path):
    pid_file, cleaned_up = tmp_path / "pid", tmp_path / "cleaned_up"

    async def cancel_once_running():
        task = asyncio.create_task(
            call_agent_function(
                stubs.wait_until_cancelled,
                mode="sandbox",
                sandbox_start=sandboxes.start,
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

    assert sandboxes.closed == ["wait_until_cancelled"]
    assert _wait_for(cleaned_up)


@pytest.mark.parametrize("phase", ["create", "health"])
def test_a_rollout_abort_while_the_sandbox_boots_stops_the_call_before_it_starts(sandboxes, phase):
    args = _rollout_args()
    held = sandboxes.creating if phase == "create" else sandboxes.healthy
    held.clear()

    async def abort_while_booting():
        task = asyncio.create_task(
            call_agent_function(
                stubs.report_pid,
                mode="sandbox",
                sandbox_start=sandboxes.start,
                abort_scope=remote.rollout_abort_scope(args),
            )
        )
        await asyncio.sleep(0.5)
        await call_agent_abort_hook(args)
        held.set()
        with pytest.raises(InfraAbort) as raised:
            await task
        return raised.value

    abort = asyncio.run(abort_while_booting())

    assert abort.exit_status == "WorkerCancelled"
    assert ("POST", "/calls") not in sandboxes.requests
    assert sandboxes.closed == ["report_pid"]


def test_a_rollout_abort_leaves_another_rollouts_booting_call_to_run(sandboxes):
    sandboxes.healthy.clear()

    async def abort_another_rollout():
        task = asyncio.create_task(
            call_agent_function(
                stubs.report_pid,
                mode="sandbox",
                sandbox_start=sandboxes.start,
                abort_scope=remote.rollout_abort_scope(_rollout_args(router_port=30001)),
            )
        )
        await asyncio.sleep(0.5)
        await call_agent_abort_hook(_rollout_args(router_port=30000))
        sandboxes.healthy.set()
        return await task

    assert asyncio.run(abort_another_rollout())["pid"]
    assert sandboxes.closed == ["report_pid"]


def test_a_call_that_starts_after_an_abort_runs(sandboxes):
    asyncio.run(remote.cancel_inflight_calls())

    assert _in_sandbox(stubs.report_pid, sandboxes.start)["pid"]


@pytest.mark.parametrize(
    "stub, kwargs",
    [
        ("hold_semaphore", {"name": "create", "limit": 2, "hold_s": 0.0}),
        ("hold_lock", {"name": "build", "hold_s": 0.0}),
    ],
)
def test_rollout_wide_limits_refuse_rather_than_limit_one_sandbox(sandboxes, stub, kwargs):
    with pytest.raises(RuntimeError, match="cannot be shared from a sandbox agent"):
        _in_sandbox(getattr(stubs, stub), sandboxes.start, **kwargs)


def test_the_agent_sees_that_its_limits_are_unshared(sandboxes):
    assert _in_sandbox(stubs.read_env, sandboxes.start, names=[UNSHARED_ENV_VAR]) == {UNSHARED_ENV_VAR: "sandbox"}


def test_sandbox_mode_needs_its_start_function():
    with pytest.raises(ValueError, match="sandbox-start-path"):
        asyncio.run(call_agent_function(stubs.report_pid, mode="sandbox"))


def _parsed(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    agentic_tool_call.generate.add_arguments(parser)
    args = parser.parse_args(list(argv))
    args.sglang_router_ip, args.sglang_router_port = "127.0.0.1", 30000
    return args


def test_the_start_path_reaches_the_call(monkeypatch):
    monkeypatch.setenv("HARBOR_TASKS_DIR", "/scratch/tasks")
    args = _parsed(
        "--custom-agent-function-mode=sandbox",
        "--custom-agent-function-sandbox-start-path=miles.utils.test_utils.agent_function_stubs.report_pid",
    )

    options = agentic_tool_call._worker_options(args)

    assert options["sandbox_start"] is stubs.report_pid
    assert options["abort_scope"] == "127.0.0.1:30000"
    assert options["extra_env"]["HARBOR_TASKS_DIR"] == "/scratch/tasks"


def test_sandbox_mode_without_a_start_path_is_refused():
    with pytest.raises(ValueError, match="sandbox-start-path"):
        agentic_tool_call._worker_options(_parsed("--custom-agent-function-mode=sandbox"))
