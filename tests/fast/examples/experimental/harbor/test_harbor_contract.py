"""Contract tests against the REAL harbor package: what the agent function
builds must be something Harbor's own models accept.

Skipped wherever harbor is not installed (CPU CI today); they run in any
environment provisioned per the example README. They catch the drift the
faked-package unit tests cannot: an EnvironmentType value or AgentName that
stopped existing, a TrialConfig/AgentConfig field that moved. (Pydantic's
default extra="ignore" means a RENAMED optional field can still slip through;
required-field and enum drift is caught.)
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

harbor = pytest.importorskip("harbor")

import harbor_agent_function as haf  # noqa: E402
from examples.experimental.harbor_cua.qwen_computer import Qwen3VLComputer1  # noqa: E402
from harbor.agents.computer_1.computer_1 import Computer1Chat, EpisodeLoggingPaths  # noqa: E402
from harbor.agents.factory import AgentFactory  # noqa: E402
from harbor.models.agent.name import AgentName  # noqa: E402
from harbor.models.environment_type import EnvironmentType  # noqa: E402
from harbor.models.trial.config import TrialConfig  # noqa: E402
from harbor.trial.trial import Trial  # noqa: E402

from miles.rollout.agentic.agent_function import InfraAbort  # noqa: E402

_SESSION_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_tokens": 512}


@pytest.fixture
def tasks_dir(tmp_path, monkeypatch):
    (tmp_path / "task-1").mkdir()
    monkeypatch.setenv("HARBOR_TASKS_DIR", str(tmp_path))
    monkeypatch.setenv("HARBOR_ENV_TYPE", "e2b")
    return tmp_path


@pytest.fixture
def chat_server():
    """A chat-completions endpoint that records each request it answers."""
    requests: list[tuple[str, dict, dict]] = []
    reply = json.dumps(
        {
            "id": "r",
            "object": "chat.completion",
            "created": 0,
            "model": "model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"analysis": "", "plan": "", "action": {"type": "wait"}}',
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            requests.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", requests
    server.shutdown()


def _computer_1(session_url: str, logs_dir):
    cfg = haf.build_trial_config(
        {"instance_id": "task-1", "agent_name": "computer-1", "max_seq_len": 4096}, session_url, _SESSION_SAMPLING
    )
    return AgentFactory.create_agent_from_config(cfg.agent, logs_dir=logs_dir)


def test_harbor_builds_the_qwen3_vl_computer_1_with_max_seq_len_set(tasks_dir, tmp_path):
    agent = _computer_1("http://trainer:30000/sessions/s1/v1", tmp_path)

    assert isinstance(agent, Qwen3VLComputer1)
    assert agent._enable_images is True


def test_every_call_computer_1_makes_carries_miles_sampling(tasks_dir, tmp_path, chat_server):
    """The agent loop, both compactor summaries and the final-answer fallback each reach the session."""
    base_url, requests = chat_server
    agent = _computer_1(f"{base_url}/sessions/s1/v1", tmp_path)
    agent._chat = chat = Computer1Chat(agent._llm)

    async def every_kind_of_call():
        await agent._query_litellm(chat, "Begin.", EpisodeLoggingPaths(None, None, None))
        assert await agent._compactor._build_summary_from_history(chat, "the task") is not None
        assert await agent._compactor._build_short_summary("the task", "the screen") is not None
        await agent._litellm_extract_text_fallback("the task")

    asyncio.run(every_kind_of_call())

    assert len(requests) == 4
    for path, headers, body in requests:
        assert path == "/sessions/s1/v1/chat/completions"
        assert headers["authorization"] == "Bearer dummy"
        assert {key: body.get(key) for key in _SESSION_SAMPLING} == _SESSION_SAMPLING


@pytest.mark.parametrize("env_type", ["docker", "daytona", "e2b", "modal"])
def test_the_pass_through_environment_types_exist(env_type):
    assert EnvironmentType(env_type).value == env_type


@pytest.mark.parametrize("agent_name", sorted(haf.HARNESS_BINDINGS))
def test_every_binding_names_a_real_harbor_agent(agent_name):
    assert AgentName(agent_name).value == agent_name


@pytest.mark.parametrize("agent_name", [*sorted(haf.HARNESS_BINDINGS), "some-other-openai-chat-agent"])
def test_the_built_config_is_a_valid_trial_config(tasks_dir, agent_name):
    """Real pydantic validation of everything build_trial_config assembles."""
    cfg = haf.build_trial_config(
        {"instance_id": "task-1", "agent_name": agent_name, "max_seq_len": 4096},
        "http://trainer:30000/sessions/s1/v1",
        {"temperature": 0.8, "max_tokens": 512},
    )
    assert isinstance(cfg, TrialConfig)
    assert cfg.environment.type == EnvironmentType.E2B


def test_resource_enforcement_policy_uses_harbor_model(tasks_dir, monkeypatch):
    monkeypatch.setenv("HARBOR_CPU_ENFORCEMENT_POLICY", "request")
    monkeypatch.setenv("HARBOR_MEMORY_ENFORCEMENT_POLICY", "limit")

    cfg = haf.build_trial_config(
        {"instance_id": "task-1", "agent_name": "mini-swe-agent"},
        "http://trainer:30000/sessions/s1/v1",
        {},
    )

    assert cfg.environment.cpu_enforcement_policy.value == "request"
    assert cfg.environment.memory_enforcement_policy.value == "limit"


def test_the_trial_entrypoints_exist():
    assert callable(Trial.create) and callable(Trial.run)


def test_the_result_fields_the_mapping_reads_still_exist():
    """trial_result_to_metadata reads these via getattr(default None): a rename
    would silently drop metrics/timings, so lock the names here."""
    from harbor.models.trial.result import TrialResult

    fields = set(TrialResult.model_fields)
    assert {
        "started_at",
        "finished_at",
        "environment_setup",
        "agent_setup",
        "agent_execution",
        "verifier",
        "exception_info",
        "verifier_result",
        "agent_result",
    } <= fields


def test_the_built_config_survives_the_agent_server_json_round_trip(tasks_dir, monkeypatch):
    """The agent server rebuilds the config from /run's JSON; every field the bindings set must come back."""
    monkeypatch.setenv("HARBOR_AGENT_IMPORT_PATH", "lin2399_agent:LiteOSWorldComputer1")
    monkeypatch.setenv("HARBOR_AGENT_KWARGS", json.dumps({"settle_sec": 2.0, "keep_screenshots": 2}))
    monkeypatch.setenv("HARBOR_VERIFIER_IMPORT_PATH", "harbor_liteosworld.verifier:LiteOSWorldVerifier")
    monkeypatch.setenv("HARBOR_ENV_KWARGS", json.dumps({"prebuilt_template_id": "lin2139-liteosworld"}))
    cfg = haf.build_trial_config(
        {"instance_id": "task-1", "agent_name": "computer-1"}, "http://trainer:30000/sessions/s1/v1", _SESSION_SAMPLING
    )

    request = haf._agent_server_request(cfg, {"instance_id": "task-1", "agent_name": "computer-1"}, "http://t/v1")
    received = TrialConfig.model_validate(json.loads(json.dumps(request["trial_config"])))

    assert received == cfg
    assert received.agent.import_path == "lin2399_agent:LiteOSWorldComputer1"
    assert received.verifier.import_path == "harbor_liteosworld.verifier:LiteOSWorldVerifier"
    assert received.environment.kwargs == {"prebuilt_template_id": "lin2139-liteosworld"}


def _recorded(cfg, *, exception=None, agent_started=False, reward=None):
    from datetime import UTC, datetime

    from harbor.models.task.id import LocalTaskId
    from harbor.models.trial.result import AgentInfo, ExceptionInfo, TimingInfo, TrialResult
    from harbor.models.verifier.result import VerifierResult

    now = datetime.now(UTC)
    stamp = TimingInfo(started_at=now, finished_at=now)
    return TrialResult(
        task_name="task-1",
        trial_name=cfg.trial_name,
        trial_uri="file:///srv/trials/task-1",
        task_id=LocalTaskId(path=cfg.task.path),
        task_checksum="0",
        config=cfg,
        agent_info=AgentInfo(name="computer-1", version="1"),
        exception_info=(
            ExceptionInfo(exception_type=exception, exception_message="boom", exception_traceback="", occurred_at=now)
            if exception
            else None
        ),
        verifier_result=VerifierResult(rewards={"reward": reward}) if reward is not None else None,
        started_at=now,
        finished_at=now,
        environment_setup=stamp,
        agent_setup=stamp if agent_started else None,
        agent_execution=stamp if agent_started else None,
    )


@pytest.fixture
def agent_server(monkeypatch):
    """A real HTTP /run that answers with the TrialResult JSON a test puts in ``reply``."""
    reply: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            body = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("AGENT_SERVER_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(haf, "_agent_server_client", None, raising=False)
    yield reply
    server.shutdown()
    server.server_close()


def _run_remote(reply, result):
    reply.update(
        {"reward": 0.0, "exit_status": "", "trial_result": result.model_dump(mode="json"), "trial_dir": "/srv/x"}
    )
    return asyncio.run(
        haf.run(
            "http://trainer:30000/sessions/s1",
            [],
            _SESSION_SAMPLING,
            {"instance_id": "task-1", "agent_name": "computer-1"},
        )
    )


def _built(tasks_dir):
    return haf.build_trial_config(
        {"instance_id": "task-1", "agent_name": "computer-1"}, "http://trainer:30000/sessions/s1/v1", _SESSION_SAMPLING
    )


def test_a_serialized_sandbox_failure_is_discarded_through_the_agent_server(tasks_dir, agent_server):
    with pytest.raises(InfraAbort) as raised:
        _run_remote(agent_server, _recorded(_built(tasks_dir), exception="EnvironmentStartTimeoutError"))

    assert raised.value.exit_status == "SandboxUnavailable"


def test_a_serialized_lost_desktop_is_discarded_through_the_agent_server(tasks_dir, agent_server):
    with pytest.raises(InfraAbort) as raised:
        _run_remote(agent_server, _recorded(_built(tasks_dir), exception="RuntimeRequestError", agent_started=True))

    assert raised.value.exit_status == "SandboxLost"


def test_a_serialized_policy_timeout_scores_zero_through_the_agent_server(tasks_dir, agent_server):
    out = _run_remote(agent_server, _recorded(_built(tasks_dir), exception="AgentTimeoutError", agent_started=True))

    assert out["reward"] == 0.0 and out["exit_status"] == "TimeLimitExceeded"


def test_a_serialized_verdict_keeps_its_reward_through_the_agent_server(tasks_dir, agent_server):
    out = _run_remote(agent_server, _recorded(_built(tasks_dir), agent_started=True, reward=1.0))

    assert out == {**out, "reward": 1.0, "exit_status": "Submitted", "trial_dir": "/srv/x"}
