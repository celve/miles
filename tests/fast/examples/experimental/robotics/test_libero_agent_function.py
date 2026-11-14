"""``libero_agent_function.run`` against a fake simulator server and a fake policy, both real HTTP servers.

The sandbox is the one stand-in: ``libero_sandbox.Sandbox`` is replaced by an object whose ``url`` is the
fake simulator's, so everything from the reset to the reward goes through the code a rollout runs.
"""

import asyncio
import base64
import json
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import libero_agent_function as laf
import libero_probe
import make_libero_data
import pytest

from miles.rollout.agentic.errors import InfraAbort
from miles.utils.types import Sample

EPISODE = {"suite": "libero_goal", "task_id": 3, "trial_id": 7}
STATE = {"arm": {"x": 44.3, "y": 0.9, "z": 25.9, "roll": 3.14, "pitch": 0.06, "yaw": 0.0}, "gripper": {"width": 7.7}}
ACTION = {"arm": {"x": 0, "y": 0, "z": -5, "roll": 0, "pitch": 0, "yaw": 0.0}, "gripper": {"width": -8}}
FRAMES = {"agentview": base64.b64encode(b"agentview-png").decode(), "wrist": base64.b64encode(b"wrist-png").decode()}


def answer(action=ACTION, thinking="the drawer is to the right") -> str:
    return f"<think>{thinking}</think>Moving down.\n```json\n{json.dumps(action)}\n```"


@contextmanager
def serve(handle):
    """A local HTTP server that answers POSTs with ``handle(path, body) -> (status, payload)``."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            status, payload = handle(self.path, body)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


class FakeSimulator:
    """Ends the episode on action ``done_at`` (1-based) with ``success``; answers 500 on action ``fail_at``."""

    def __init__(self, *, done_at=3, success=True, fail_at=None):
        self.done_at, self.success, self.fail_at = done_at, success, fail_at
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, path, body):
        self.requests.append((path, body))
        acts = sum(1 for p, _ in self.requests if p == "/act")
        if path == "/act" and acts == self.fail_at:
            return 500, {"error": "Traceback: MuJoCo died"}
        done = path == "/act" and acts == self.done_at
        return 200, {
            "state": STATE,
            "images": None if done else FRAMES,
            "done": done,
            "success": done and self.success,
            "outcome": "done" if done else ("reached" if path == "/act" else None),
            "executed": 9 if path == "/act" else 0,
            "actions": [],
            "steps": 9 * acts,
            "max_steps": 300,
            "instruction": "open the middle drawer of the cabinet",
            "seconds": 0.1,
        }


class FakePolicy:
    """An OpenAI chat-completions endpoint that gives ``answers`` in order and records each request."""

    def __init__(self, answers, *, finish_reasons=None, prompt_tokens=1000):
        self.answers = answers
        self.finish_reasons = finish_reasons or ["stop"] * len(answers)
        self.prompt_tokens = prompt_tokens
        self.requests: list[dict] = []

    def __call__(self, path, body):
        assert path == "/v1/chat/completions"
        self.requests.append(body)
        turn = len(self.requests) - 1
        return 200, {
            "id": f"c{turn}",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.answers[turn], "reasoning_content": "r"},
                    "finish_reason": self.finish_reasons[turn],
                }
            ],
            "usage": {"prompt_tokens": self.prompt_tokens * (turn + 1), "completion_tokens": 100, "total_tokens": 0},
        }


class FakeSandbox:
    instances: list["FakeSandbox"] = []

    def __init__(self, *, template, max_lifetime_s, labels):
        self.template, self.max_lifetime_s, self.labels = template, max_lifetime_s, labels
        self.url = FakeSandbox.sim_url
        self.killed = False
        FakeSandbox.instances.append(self)

    def kill(self):
        self.killed = True

    def log_tail(self, lines=40):
        return "simulator log tail"


@pytest.fixture
def episode(monkeypatch, tmp_path):
    """Run one episode: ``await episode(policy, simulator, metadata=..., request_kwargs=...)``."""
    monkeypatch.setenv("MILES_ROLLOUT_LIMITS_SCOPE", f"test-{tmp_path.name}")
    monkeypatch.setattr(laf.libero_sandbox, "Sandbox", FakeSandbox)
    monkeypatch.setenv("LIBERO_TEMPLATE", "libero-sim-test")
    FakeSandbox.instances = []

    async def play(policy, simulator, *, metadata=None, request_kwargs=None):
        with serve(simulator) as sim_url, serve(policy) as policy_url:
            FakeSandbox.sim_url = sim_url
            return await laf.run(
                base_url=policy_url,
                prompt=[{"role": "user", "content": "unused"}],
                request_kwargs=(
                    request_kwargs if request_kwargs is not None else {"max_tokens": 512, "temperature": 1.0}
                ),
                metadata=metadata or dict(EPISODE),
            )

    return play


async def test_success_scores_one_and_sends_pears_conversation(episode):
    policy, simulator = FakePolicy([answer()] * 3), FakeSimulator(done_at=3)

    result = await episode(policy, simulator)

    assert result["reward"] == 1.0
    assert result["agent_metrics"]["end_reason"] == "env" and result["agent_metrics"]["turns"] == 3
    assert simulator.requests[0] == ("/reset", EPISODE)
    assert [body for path, body in simulator.requests[1:]] == [{"action": ACTION}] * 3
    (sandbox,) = FakeSandbox.instances
    assert sandbox.killed and sandbox.template == "libero-sim-test" and sandbox.labels["suite"] == "libero_goal"

    last = policy.requests[-1]["messages"]
    assert [m["role"] for m in last] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert last[0]["content"].endswith("## TASK\n\nopen the middle drawer of the cabinet\n")
    assert "## ENVIRONMENT\n\n- agentview: a fixed camera" in last[0]["content"]
    user = last[1]["content"]
    assert json.loads(user[0]["text"].split("\n\n")[0]) == STATE
    assert user[0]["text"].endswith(laf.REMINDER)
    assert [part.get("text") for part in user[1::2]] == ["[agentview]", "[wrist]"]
    assert [part["image_url"]["url"] for part in user[2::2]] == [
        "data:image/png;base64," + FRAMES["agentview"],
        "data:image/png;base64," + FRAMES["wrist"],
    ]
    # the assistant turn goes back whole, reasoning included, for the session server's prefix match
    assert last[2]["reasoning_content"] == "r" and last[2]["content"] == answer()


async def test_done_without_success_scores_zero(episode):
    result = await episode(FakePolicy([answer()] * 2), FakeSimulator(done_at=2, success=False))

    assert result["reward"] == 0.0 and result["agent_metrics"]["end_reason"] == "env"


@pytest.mark.parametrize(
    "bad",
    [
        "I will move down.",
        "```json\n{not json}\n```",
        answer({"arm": ACTION["arm"]}),
        answer({**ACTION, "gripper": {"width": True}}),
        answer({**ACTION, "arm": {**ACTION["arm"], "x": "1"}}),
        f"<think>```json\n{json.dumps(ACTION)}\n```</think>no block after the thinking",
    ],
)
async def test_an_answer_without_a_valid_action_ends_the_episode_with_zero(episode, bad):
    simulator = FakeSimulator()

    result = await episode(FakePolicy([answer(), bad]), simulator)

    assert result["reward"] == 0.0 and result["agent_metrics"]["end_reason"] == "parse"
    assert [path for path, _ in simulator.requests] == ["/reset", "/act"]
    assert FakeSandbox.instances[0].killed


async def test_the_last_block_after_the_thinking_is_the_action(episode):
    other = {**ACTION, "gripper": {"width": 8}}
    text = f"<think>```json\n{json.dumps(other)}\n```</think>```json\n{json.dumps(other)}\n```\n```json\n{json.dumps(ACTION)}\n```"
    simulator = FakeSimulator(done_at=1)

    await episode(FakePolicy([text]), simulator)

    assert simulator.requests[1] == ("/act", {"action": ACTION})


async def test_a_simulator_failure_discards_the_sample(episode):
    simulator = FakeSimulator(fail_at=2)

    with pytest.raises(InfraAbort) as raised:
        await episode(FakePolicy([answer()] * 3), simulator)

    assert raised.value.exit_status == "SimulatorLost"
    assert "MuJoCo died" in str(raised.value) and "simulator log tail" in str(raised.value)
    assert FakeSandbox.instances[0].killed


async def test_a_sandbox_that_does_not_start_discards_the_sample(episode, monkeypatch):
    def refuse(**kwargs):
        raise RuntimeError("template not found")

    monkeypatch.setattr(laf.libero_sandbox, "Sandbox", refuse)

    with pytest.raises(InfraAbort) as raised:
        await episode(FakePolicy([answer()]), FakeSimulator())

    assert raised.value.exit_status == "SandboxStartFailed"


async def test_a_sandbox_started_after_the_episode_was_cancelled_is_killed(monkeypatch, tmp_path):
    monkeypatch.setenv("MILES_ROLLOUT_LIMITS_SCOPE", f"test-{tmp_path.name}")
    entered, release = threading.Event(), threading.Event()

    class SlowSandbox(FakeSandbox):
        def __init__(self, **kwargs):
            entered.set()
            release.wait(10)
            super().__init__(**kwargs)

    monkeypatch.setattr(laf.libero_sandbox, "Sandbox", SlowSandbox)
    monkeypatch.setenv("LIBERO_TEMPLATE", "libero-sim-test")
    FakeSandbox.instances, FakeSandbox.sim_url = [], "http://127.0.0.1:9"
    task = asyncio.ensure_future(laf.run(base_url="http://127.0.0.1:9", prompt=None, metadata=dict(EPISODE)))
    await asyncio.to_thread(entered.wait, 10)

    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(100):
        if FakeSandbox.instances and FakeSandbox.instances[0].killed:
            break
        await asyncio.sleep(0.05)

    (sandbox,) = FakeSandbox.instances
    assert sandbox.killed


async def test_a_reply_cut_at_its_token_cap_ends_the_episode(episode):
    result = await episode(FakePolicy([answer(), answer()], finish_reasons=["stop", "length"]), FakeSimulator())

    assert result["reward"] == 0.0 and result["agent_metrics"]["end_reason"] == "length"


async def test_the_context_budget_caps_each_reply_and_a_reply_past_it_is_not_acted_on(episode, monkeypatch):
    monkeypatch.setenv("LIBERO_TURN_RESERVE_TOKENS", "500")
    policy, simulator = FakePolicy([answer()] * 5, prompt_tokens=1000), FakeSimulator(done_at=5)

    result = await episode(policy, simulator, metadata={**EPISODE, "max_seq_len": 3000})

    # 3000 budget - 500 reserve: the first reply gets 2500 capped at 512; after it 1100 tokens are used, so
    # the second gets 1400 capped at 512; after the second 2100 are used, which leaves 400, so it gets 400
    assert [request["max_tokens"] for request in policy.requests] == [512, 512, 400]
    # the third exchange totals 3100, past the budget: its action is not carried out
    assert [path for path, _ in simulator.requests] == ["/reset", "/act", "/act"]
    assert result["agent_metrics"]["end_reason"] == "budget" and result["reward"] == 0.0


async def test_the_env_budget_lowers_max_seq_len_but_cannot_raise_it(episode, monkeypatch):
    monkeypatch.setenv("LIBERO_TURN_RESERVE_TOKENS", "0")
    policy = FakePolicy([answer()] * 2)

    monkeypatch.setenv("LIBERO_CONTEXT_TOKENS", "9000")
    await episode(policy, FakeSimulator(done_at=1), metadata={**EPISODE, "max_seq_len": 600}, request_kwargs={})
    monkeypatch.setenv("LIBERO_CONTEXT_TOKENS", "300")
    await episode(policy, FakeSimulator(done_at=1), metadata={**EPISODE, "max_seq_len": 600}, request_kwargs={})

    assert [request["max_tokens"] for request in policy.requests] == [600, 300]


async def test_the_turn_cap_ends_the_episode(episode, monkeypatch):
    monkeypatch.setenv("LIBERO_MAX_TURNS", "2")

    result = await episode(FakePolicy([answer()] * 3), FakeSimulator(done_at=3))

    assert result["agent_metrics"]["end_reason"] == "max_turns" and result["agent_metrics"]["turns"] == 2


async def test_metadata_without_an_episode_is_refused(episode):
    with pytest.raises(ValueError, match="suite, task_id and trial_id"):
        await episode(FakePolicy([answer()]), FakeSimulator(), metadata={"suite": "libero_goal"})
    assert FakeSandbox.instances == []


async def test_reward_func_reads_the_agents_reward():
    samples = [Sample(metadata={"reward": 1.0}), Sample(metadata={})]

    assert await laf.reward_func(None, samples) == [1.0, 0.0]
    assert await laf.reward_func(None, samples[0]) == 1.0


def test_heldout_trials_are_disjoint_from_training():
    train = make_libero_data.rows(["libero_goal"], [0, 1], range(40), "train")
    heldout = make_libero_data.rows(["libero_goal"], [0, 1], range(40, 50), "heldout")

    def keys(rows):
        return {(r["metadata"]["suite"], r["metadata"]["task_id"], r["metadata"]["trial_id"]) for r in rows}

    assert len(train) == 80 and len(heldout) == 20 and not keys(train) & keys(heldout)


@pytest.mark.parametrize("urls", [["http://a:1"], ["http://a:1", "http://a:1"]])
def test_the_probe_needs_two_distinct_servers(monkeypatch, urls):
    monkeypatch.setattr(sys, "argv", ["libero_probe.py", *[arg for url in urls for arg in ("--url", url)]])

    with pytest.raises(SystemExit) as exited:
        libero_probe.main()

    assert exited.value.code == 2
