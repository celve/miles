"""A ``sandbox`` agent call runs on a worker of its own, which is killed when the call ends.

A fake ``e2b`` stands in for the sandbox service; a real worker app plays the sandbox's worker behind its gateway.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import types

import pytest
from aiohttp import web

from miles.rollout.agentic import sandbox_worker
from miles.rollout.agentic.agent_function import InfraAbort, call_agent_function
from miles.rollout.agentic.sandbox_worker import SandboxWorkerConfig
from miles.rollout.agentic.worker import create_app
from miles.utils.test_utils import agent_function_stubs as stubs

from .test_worker import _Served


# bash: the cmdline files of every live descendant of $1
_DESCENDANTS = r"""
_descendant_cmdlines() {
  local todo=("$ROOT") pid child
  while [ ${#todo[@]} -gt 0 ]; do
    pid=${todo[0]}; todo=("${todo[@]:1}")
    for child in $(cat /proc/$pid/task/*/children 2>/dev/null); do echo /proc/$child/cmdline; todo+=("$child"); done
  done
}
export -f _descendant_cmdlines; export ROOT=$0
"""


class FakeAsyncSandbox:
    created: list = []
    killed: list = []
    create_error: Exception | None = None
    files_on_disk: dict = {}

    def __init__(self, template: str, kwargs: dict):
        self.sandbox_id = f"sbx-{len(FakeAsyncSandbox.created)}"
        self.template, self.kwargs, self.written = template, kwargs, {}
        sandbox = self

        class Files:
            async def write(self, path, data):
                sandbox.written[path] = data

            async def read(self, path, format="text"):
                if path not in FakeAsyncSandbox.files_on_disk:
                    raise FileNotFoundError(path)
                return FakeAsyncSandbox.files_on_disk[path]

        self.files = Files()

        class Commands:
            async def run(self, cmd, timeout=60, **kwargs):
                # the probe, scoped to this test's own descendants: the host may hold other tests' agent processes
                assert sandbox_worker._AGENT_PROCESS_PROBE in cmd
                cmd = cmd.replace("/proc/[0-9]*/cmdline", "$(_descendant_cmdlines)")
                proc = await asyncio.create_subprocess_exec("bash", "-c", _DESCENDANTS + cmd, str(os.getpid()))
                await asyncio.wait_for(proc.wait(), timeout)

        self.commands = Commands()

    @classmethod
    async def create(cls, template, **kwargs):
        if cls.create_error is not None:
            raise cls.create_error
        sandbox = cls(template, kwargs)
        cls.created.append(sandbox)
        return sandbox

    async def kill(self):
        FakeAsyncSandbox.killed.append(self.sandbox_id)
        FakeAsyncSandbox.on_kill(self)

    on_kill = staticmethod(lambda sandbox: None)


class _HeaderRecorder:
    """The sandbox's worker app, recording the routing headers each request reached it with."""

    def __init__(self):
        self.headers: list[dict] = []
        self.requests: list[tuple[str, str]] = []
        app = create_app()

        @web.middleware
        async def record(request, handler):
            self.headers.append({k: v for k, v in request.headers.items() if k.startswith("E2b-")})
            self.requests.append((request.method, request.path))
            return await handler(request)

        app.middlewares.append(record)
        self.served = _Served(app)


@pytest.fixture
def service(monkeypatch):
    worker = _HeaderRecorder()
    e2b = types.ModuleType("e2b")
    e2b.AsyncSandbox = FakeAsyncSandbox
    monkeypatch.setitem(sys.modules, "e2b", e2b)
    FakeAsyncSandbox.created, FakeAsyncSandbox.killed = [], []
    FakeAsyncSandbox.create_error, FakeAsyncSandbox.files_on_disk = None, {}
    monkeypatch.setattr(FakeAsyncSandbox, "on_kill", staticmethod(lambda sandbox: None))
    monkeypatch.setenv("E2B_SANDBOX_URL", worker.served.url)
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    yield worker
    worker.served.close()


def _config(**overrides) -> SandboxWorkerConfig:
    return SandboxWorkerConfig(**{"template": "agent-worker", "owner": "run-7", "lifetime_s": 600, **overrides})


def _in_sandbox(fn, config=None, extra_env=None, **kwargs):
    return asyncio.run(
        call_agent_function(fn, mode="sandbox", sandbox=config or _config(), extra_env=extra_env, **kwargs)
    )


def test_the_call_runs_on_a_worker_in_its_own_sandbox_which_is_then_killed(service):
    seen = _in_sandbox(stubs.read_env, extra_env={"LIN2511_FORWARDED": "the caller's"}, names=["LIN2511_FORWARDED"])

    assert seen == {"LIN2511_FORWARDED": "the caller's"}
    (sandbox,) = FakeAsyncSandbox.created
    assert sandbox.template == "agent-worker"
    assert sandbox.kwargs["timeout"] == 600
    assert sandbox.kwargs["metadata"] == {"owner": "run-7", "role": "agent-worker", "call": "read_env"}
    assert json.loads(sandbox.written[sandbox_worker.BOOT_CONFIG]) == {"env": {"E2B_API_KEY": "test-key"}}
    assert FakeAsyncSandbox.killed == ["sbx-0"]
    assert service.headers and all(
        h == {"E2b-Sandbox-Id": "sbx-0", "E2b-Sandbox-Port": str(sandbox_worker.WORKER_PORT)} for h in service.headers
    )


def test_the_boot_env_file_joins_the_key(service, tmp_path):
    env_file = tmp_path / "boot.json"
    env_file.write_text(json.dumps({"SESSION_FORWARD": "spec"}))

    _in_sandbox(stubs.report_pid, config=_config(boot_env_file=str(env_file)))

    (sandbox,) = FakeAsyncSandbox.created
    assert json.loads(sandbox.written[sandbox_worker.BOOT_CONFIG]) == {
        "env": {"E2B_API_KEY": "test-key", "SESSION_FORWARD": "spec"}
    }


def test_a_malformed_boot_env_file_fails_before_any_sandbox(service, tmp_path):
    env_file = tmp_path / "boot.json"
    env_file.write_text(json.dumps({"N": 1}))

    with pytest.raises(ValueError, match="JSON object of strings"):
        _in_sandbox(stubs.report_pid, config=_config(boot_env_file=str(env_file)))
    assert FakeAsyncSandbox.created == []


def test_an_infra_abort_keeps_its_cause_and_the_sandbox_still_goes(service):
    with pytest.raises(InfraAbort) as raised:
        _in_sandbox(stubs.abort_for_infra)

    assert raised.value.exit_status == "SandboxUnavailable"
    assert FakeAsyncSandbox.killed == ["sbx-0"]


def test_a_sandbox_that_cannot_be_created_discards_the_sample(service):
    FakeAsyncSandbox.create_error = RuntimeError("500: snapshot not ready")

    with pytest.raises(InfraAbort) as raised:
        _in_sandbox(stubs.report_pid)

    assert raised.value.exit_status == "SandboxWorkerUnavailable"


def test_a_worker_that_never_answers_discards_the_sample_and_kills_its_sandbox(service, monkeypatch):
    monkeypatch.setenv("E2B_SANDBOX_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(sandbox_worker, "_HEALTH_DEADLINE_S", 0.5)

    with pytest.raises(InfraAbort) as raised:
        _in_sandbox(stubs.report_pid)

    assert raised.value.exit_status == "SandboxWorkerUnavailable"
    assert FakeAsyncSandbox.killed == ["sbx-0"]


def test_cancelling_the_caller_kills_the_sandbox(service, tmp_path):
    pid_file, cleaned_up = tmp_path / "pid", tmp_path / "cleaned_up"

    async def cancel_once_running():
        task = asyncio.create_task(
            call_agent_function(
                stubs.wait_until_cancelled,
                mode="sandbox",
                sandbox=_config(),
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

    assert FakeAsyncSandbox.killed == ["sbx-0"]


@pytest.mark.parametrize("cancel", ["task", "rollout_abort"])
def test_a_cancelled_agents_cleanup_finishes_before_the_kill(service, tmp_path, monkeypatch, cancel):
    """Harbor closes the desktop it created in its cancellation cleanup; killing the sandbox first would leak it."""
    from miles.rollout.agentic import remote

    pid_file, cleaned_up = tmp_path / "pid", tmp_path / "cleaned_up"
    seen_at_kill: list[bool] = []
    monkeypatch.setattr(
        FakeAsyncSandbox, "on_kill", staticmethod(lambda sandbox: seen_at_kill.append(cleaned_up.exists()))
    )

    async def cancel_once_running():
        task = asyncio.create_task(
            call_agent_function(
                stubs.clean_up_slowly_when_cancelled,
                mode="sandbox",
                sandbox=_config(),
                pid_file=str(pid_file),
                cleaned_up=str(cleaned_up),
                cleanup_s=1.0,
            )
        )
        while not pid_file.exists():
            await asyncio.sleep(0.1)
        if cancel == "task":
            task.cancel()
        else:
            assert await remote.cancel_inflight_calls() == 1
        with pytest.raises((asyncio.CancelledError, InfraAbort)):
            await task

    asyncio.run(cancel_once_running())

    assert seen_at_kill == [True]


@pytest.mark.parametrize("phase", ["create", "health"])
def test_a_rollout_abort_while_the_sandbox_boots_stops_the_call_before_it_starts(service, monkeypatch, phase):
    """The abort hook runs once and the rollout then drains its tasks, so a call still booting must not start after."""
    from miles.rollout.agentic import remote

    gate = asyncio.Event()
    real_create, real_wait = FakeAsyncSandbox.create, sandbox_worker._wait_healthy

    async def create_held(template, **kwargs):
        if phase == "create":
            await gate.wait()
        return await real_create(template, **kwargs)

    async def wait_held(endpoint, label, aborts):
        if phase == "health":
            await gate.wait()
        return await real_wait(endpoint, label, aborts)

    monkeypatch.setattr(FakeAsyncSandbox, "create", create_held)
    monkeypatch.setattr(sandbox_worker, "_wait_healthy", wait_held)

    async def abort_while_booting():
        task = asyncio.create_task(call_agent_function(stubs.report_pid, mode="sandbox", sandbox=_config()))
        await asyncio.sleep(0.3)
        assert await remote.cancel_inflight_calls() == 0
        gate.set()
        with pytest.raises(InfraAbort) as raised:
            await task
        return raised.value

    abort = asyncio.run(abort_while_booting())

    assert abort.exit_status == "WorkerCancelled"
    assert ("POST", "/calls") not in service.requests
    assert FakeAsyncSandbox.killed == ["sbx-0"]


def test_a_call_that_starts_after_an_abort_runs(service):
    from miles.rollout.agentic import remote

    asyncio.run(remote.cancel_inflight_calls())

    assert _in_sandbox(stubs.report_pid)["pid"]


@pytest.mark.parametrize(
    "stub, kwargs",
    [
        ("hold_semaphore", {"name": "create", "limit": 2, "hold_s": 0.0}),
        ("hold_lock", {"name": "build", "hold_s": 0.0}),
    ],
)
def test_rollout_wide_limits_refuse_rather_than_limit_one_sandbox(service, stub, kwargs):
    """Each sandbox has its own /tmp, so a flock there would limit one agent and silently stop coordinating the rest."""
    with pytest.raises(RuntimeError, match="cannot be shared from a sandbox agent"):
        _in_sandbox(getattr(stubs, stub), **kwargs)


def test_the_trial_records_are_copied_out_before_the_kill(service, tmp_path):
    FakeAsyncSandbox.files_on_disk = {
        "/opt/t/x/result.json": b'{"reward": 1}',
        "/opt/t/x/agent/timings.jsonl": b"{}\n",
    }
    config = _config(
        records_dir=str(tmp_path / "records"), records=("result.json", "agent/timings.jsonl", "trial.log")
    )

    out = _in_sandbox(stubs.return_value, config=config, value={"trial_dir": "/opt/t/x", "reward": 1.0})

    local = tmp_path / "records" / "x"
    assert out == {"trial_dir": str(local), "reward": 1.0}
    assert (local / "result.json").read_bytes() == b'{"reward": 1}'
    assert (local / "agent" / "timings.jsonl").read_bytes() == b"{}\n"
    assert not (local / "trial.log").exists()
    assert FakeAsyncSandbox.killed == ["sbx-0"]


def test_without_a_records_dir_the_result_is_untouched(service):
    assert _in_sandbox(stubs.return_value, value={"trial_dir": "/opt/t/x"}) == {"trial_dir": "/opt/t/x"}


def test_sandbox_mode_needs_its_template():
    with pytest.raises(ValueError, match="sandbox-template"):
        asyncio.run(call_agent_function(stubs.report_pid, mode="sandbox"))


def test_the_launch_args_reach_the_sandbox_config(monkeypatch):
    import argparse

    from miles.rollout.generate_hub import agentic_tool_call

    parser = argparse.ArgumentParser()
    agentic_tool_call.generate.add_arguments(parser)
    monkeypatch.setenv("HARBOR_TASKS_DIR", "/scratch/tasks")
    args = parser.parse_args(
        [
            "--custom-agent-function-mode=sandbox",
            "--custom-agent-function-sandbox-template=agent-worker",
            "--custom-agent-function-sandbox-owner=run-7",
            "--custom-agent-function-sandbox-env-file=/tmp/boot.json",
            "--custom-agent-function-sandbox-records-dir=/tmp/records",
        ]
    )

    worker, sandbox, env = agentic_tool_call._agent_placement(args)

    assert worker is None and env["HARBOR_TASKS_DIR"] == "/scratch/tasks"
    assert sandbox == SandboxWorkerConfig(
        template="agent-worker",
        owner="run-7",
        lifetime_s=7800,
        boot_env_file="/tmp/boot.json",
        records_dir="/tmp/records",
        records=("result.json", "config.json", "trial.log", "agent/timings.jsonl"),
    )


def test_the_agent_process_probe_matches_a_subproc_child():
    """The probe the provider runs in the sandbox, against a process whose command line is subproc's child bootstrap."""
    from miles.rollout.agentic.agent_function import _CHILD_BOOTSTRAP

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", _CHILD_BOOTSTRAP])
    try:
        time.sleep(0.2)
        assert subprocess.run(["bash", "-c", sandbox_worker._AGENT_PROCESS_PROBE]).returncode == 0
    finally:
        child.kill()
        child.wait()
