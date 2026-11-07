"""The video-generation env's episode loop, against a scripted chat endpoint and the fake generator."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import examples.experimental.video_gen_agent.env as env
import pytest
from examples.experimental.video_gen_agent import reward
from examples.experimental.video_gen_agent.generator import Clip, ClipRequest
from examples.experimental.video_gen_agent.reward import reward_func, stub_score

from miles.rollout.agentic.agent_function import InfraAbort
from miles.utils.types import Sample

REQUEST = "A red fox runs through fresh snow at dawn."


class _CountingGenerator:
    requests: list[ClipRequest] = []

    def __init__(self):
        self.clip_dir = Path(os.environ["AGENT_VIDEO_GEN_FAKE_DIR"])

    async def generate(self, request: ClipRequest) -> Clip:
        _CountingGenerator.requests.append(request)
        path = self.clip_dir / f"{len(_CountingGenerator.requests)}.mp4"
        path.write_bytes(json.dumps(request.prompt).encode())
        return Clip(path=str(path))


class _HangingGenerator:
    async def generate(self, request: ClipRequest) -> Clip:
        await asyncio.sleep(3600)


@pytest.fixture
def options(monkeypatch, tmp_path):
    _CountingGenerator.requests = []
    monkeypatch.setenv("AGENT_VIDEO_GEN_GENERATOR", f"{__name__}._CountingGenerator")
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path / "clips"))
    monkeypatch.setenv("AGENT_VIDEO_GEN_SUBMITTED_DIR", str(tmp_path / "kept"))
    (tmp_path / "clips").mkdir()
    return monkeypatch


def _call(name: str, arguments) -> dict:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": raw}}


def _script(monkeypatch, *replies: list[dict] | str) -> list[list[dict]]:
    """Answer each chat request with the next reply: a list of tool calls, or plain text."""
    sent: list[list[dict]] = []

    async def fake_post(url, payload):
        assert url == "http://session/v1/chat/completions"
        assert payload["tools"] == env.TOOLS and payload["temperature"] == 0.5
        sent.append(json.loads(json.dumps(payload["messages"])))
        reply = replies[len(sent) - 1]
        if isinstance(reply, str):
            return {"choices": [{"message": {"role": "assistant", "content": reply, "tool_calls": None}}]}
        return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": reply}}]}

    monkeypatch.setattr(env, "post", fake_post)
    return sent


async def _run() -> dict:
    prompt = [{"role": "user", "content": REQUEST}]
    return await env.run_agent("http://session", prompt, {"temperature": 0.5}, {})


def _tool_replies(messages: list[dict]) -> list:
    return [m["content"] for m in messages if m["role"] == "tool"]


@pytest.mark.asyncio
async def test_generate_then_submit_records_the_clip_and_states_the_budget(options):
    sent = _script(
        options,
        [_call("generate", {"prompt": "fox in snow", "negative_prompt": "blur", "seed": 7, "seconds": 30})],
        [_call("submit", {"clip_id": "clip_1"})],
    )

    result = await _run()

    assert sent[0][0]["role"] == "system" and "832x480 at 16 fps" in sent[0][0]["content"]
    assert "3 generations and 8 turns" in sent[0][0]["content"]
    assert sent[0][1:] == [{"role": "user", "content": REQUEST}]
    assert _tool_replies(sent[1]) == [
        "Generated clip_1: 5 s (cut to the 5 s limit), seed 7. Generations left: 2. Turns left: 7."
    ]
    (request,) = _CountingGenerator.requests
    assert request == ClipRequest("fox in snow", "blur", 7, 5.0, 832, 480, 16)
    video_gen = result["video_gen"]
    submitted_path = Path(video_gen["submitted"].pop("path"))
    assert video_gen["submitted"] == {"clip_id": "clip_1", "prompt": "fox in snow"}
    assert submitted_path.parent == Path(os.environ["AGENT_VIDEO_GEN_SUBMITTED_DIR"])
    assert submitted_path.read_bytes() == (Path(os.environ["AGENT_VIDEO_GEN_FAKE_DIR"]) / "1.mp4").read_bytes()
    assert video_gen["request"] == REQUEST and video_gen["turns"] == 2
    assert [c["clip_id"] for c in video_gen["clips"]] == ["clip_1"]
    assert result["agent_metrics"]["total_tool_time"] >= 0
    assert "reward" not in result


@pytest.mark.asyncio
async def test_budget_is_enforced_and_told_to_the_agent(options):
    sent = _script(
        options,
        [_call("generate", {"prompt": f"take {i}"}) for i in range(4)],
        [_call("submit", {"clip_id": "clip_9"})],
        [_call("submit", {"clip_id": "clip_3"}), _call("generate", {"prompt": "after submit"})],
    )

    result = await _run()

    assert [r.seed for r in _CountingGenerator.requests] == [0, 1, 2]
    replies = _tool_replies(sent[2])
    assert replies[2].endswith("Generations left: 0. Turns left: 7.")
    assert (
        replies[3]
        == "Error: no generations left. Submit one of clip_1, clip_2, clip_3. Generations left: 0. Turns left: 7."
    )
    assert replies[4].startswith("Error: unknown clip_id 'clip_9'; generated clips: clip_1, clip_2, clip_3.")
    assert result["video_gen"]["submitted"]["clip_id"] == "clip_3"
    assert len(sent) == 3


@pytest.mark.asyncio
async def test_turn_cap_without_submit_ends_the_episode_with_no_clip(options):
    options.setenv("AGENT_VIDEO_GEN_MAX_TURNS", "3")
    sent = _script(options, "I will think about it.", "Still thinking.", [_call("generate", {"prompt": "x"})])

    result = await _run()

    assert len(sent) == 3
    assert sent[1][-1] == {"role": "user", "content": "Call generate or submit. Generations left: 3. Turns left: 2."}
    assert result["video_gen"]["submitted"] is None and result["video_gen"]["turns"] == 3
    sample = Sample(metadata=result)
    assert await reward_func(SimpleNamespace(video_reward_pool="judge"), [sample], judge_pool=None) == [0.0]


@pytest.mark.asyncio
async def test_bad_calls_are_answered_without_spending_a_generation(options):
    sent = _script(
        options,
        [
            _call("generate", "{not json"),
            _call("generate", {"seconds": 2}),
            _call("generate", {"prompt": "x", "seed": "7"}),
            _call("generate", {"prompt": "x", "seconds": -1}),
            _call("render", {}),
            _call("submit", {}),
            _call("submit", {"clip_id": []}),
            _call("submit", {"clip_id": {"id": "clip_1"}}),
        ],
        [_call("generate", {"prompt": "x"})],
        [_call("submit", {"clip_id": "clip_1"})],
    )

    await _run()

    replies = _tool_replies(sent[1])
    assert [r.split(".")[0] for r in replies] == [
        "Error: the arguments of generate are not valid JSON",
        "Error: generate needs a non-empty prompt",
        "Error: seed must be an integer",
        "Error: seconds must be a positive number",
        "Error: unknown tool 'render'; the tools are generate and submit",
        "Error: unknown clip_id None; generated clips: none yet",
        "Error: unknown clip_id []; generated clips: none yet",
        "Error: unknown clip_id {'id': 'clip_1'}; generated clips: none yet",
    ]
    assert all(r.endswith("Generations left: 3. Turns left: 7.") for r in replies)
    assert len(_CountingGenerator.requests) == 1


@pytest.mark.asyncio
async def test_generator_outage_discards_the_sample(options):
    options.setenv("AGENT_VIDEO_GEN_GENERATOR", "examples.experimental.video_gen_agent.generator.FakeClipGenerator")
    options.setenv("AGENT_VIDEO_GEN_FAKE_UNAVAILABLE", "1")
    _script(options, [_call("generate", {"prompt": "x"})])

    with pytest.raises(InfraAbort) as raised:
        await _run()

    assert raised.value.exit_status == "GeneratorUnavailable"


@pytest.mark.asyncio
async def test_generator_timeout_discards_the_sample(options):
    options.setenv("AGENT_VIDEO_GEN_GENERATOR", f"{__name__}._HangingGenerator")
    options.setenv("AGENT_VIDEO_GEN_TIMEOUT_S", "0.05")
    _script(options, [_call("generate", {"prompt": "x"})])

    with pytest.raises(InfraAbort) as raised:
        await _run()

    assert raised.value.exit_status == "GeneratorUnavailable"


@pytest.mark.asyncio
async def test_clip_observations_return_the_clip_as_a_video_part(options):
    options.setenv("AGENT_VIDEO_GEN_CLIP_OBSERVATIONS", "1")
    sent = _script(
        options, [_call("generate", {"prompt": "x", "seconds": 2})], [_call("submit", {"clip_id": "clip_1"})]
    )

    await _run()

    (reply,) = _tool_replies(sent[1])
    clip_path = Path(os.environ["AGENT_VIDEO_GEN_FAKE_DIR"]) / "1.mp4"
    assert reply == [
        {"type": "text", "text": "Generated clip_1: 2 s, seed 0. Generations left: 2. Turns left: 7."},
        {"type": "video_url", "video_url": {"url": f"file://{clip_path}"}},
    ]


@pytest.mark.asyncio
async def test_submitted_clip_is_copied_out_by_content_hash(options, tmp_path):
    kept = tmp_path / "kept"
    _script(options, [_call("generate", {"prompt": "x"})], [_call("submit", {"clip_id": "clip_1"})])

    first = (await _run())["video_gen"]["submitted"]["path"]
    _script(options, [_call("generate", {"prompt": "x"})], [_call("submit", {"clip_id": "clip_1"})])
    second = (await _run())["video_gen"]["submitted"]["path"]

    assert Path(first).parent == kept and first == second
    assert Path(first).read_bytes() == b'"x"'
    assert [p.name for p in kept.iterdir()] == [Path(first).name]


def test_options_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("AGENT_VIDEO_GEN_GENERATOR", "some.generator")
    monkeypatch.setenv("AGENT_VIDEO_GEN_SUBMITTED_DIR", "/kept")
    monkeypatch.setenv("AGENT_VIDEO_GEN_MAX_SECONDS", "2.5")
    monkeypatch.setenv("AGENT_VIDEO_GEN_FPS", "24")
    monkeypatch.setenv("AGENT_VIDEO_GEN_CLIP_OBSERVATIONS", "1")

    assert env.VideoGenEnvOptions.from_env() == env.VideoGenEnvOptions(
        generator="some.generator", submitted_dir="/kept", max_seconds=2.5, fps=24, clip_observations=True
    )
    monkeypatch.setenv("AGENT_VIDEO_GEN_MAX_GENERATIONS", "4")
    with pytest.raises(ValueError, match="MAX_GENERATIONS is at most 3, got 4"):
        env.VideoGenEnvOptions.from_env()
    monkeypatch.setenv("AGENT_VIDEO_GEN_MAX_GENERATIONS", "0")
    with pytest.raises(ValueError, match="MAX_GENERATIONS must be positive"):
        env.VideoGenEnvOptions.from_env()
    monkeypatch.delenv("AGENT_VIDEO_GEN_MAX_GENERATIONS")
    monkeypatch.delenv("AGENT_VIDEO_GEN_SUBMITTED_DIR")
    with pytest.raises(ValueError, match="SUBMITTED_DIR must name"):
        env.VideoGenEnvOptions.from_env()
    monkeypatch.delenv("AGENT_VIDEO_GEN_GENERATOR")
    with pytest.raises(ValueError, match="GENERATOR must name"):
        env.VideoGenEnvOptions.from_env()


class _StubPool:
    """A reward pool client over the stub scorer, recording what it is asked to score."""

    def __init__(self, broken: bool = False):
        self.calls: list[tuple[list[str], list[str]]] = []
        self.broken = broken

    async def score(self, outputs, prompts):
        self.calls.append((list(outputs), list(prompts)))
        if self.broken:
            raise RuntimeError("judge down")
        return reward.StubClipScorer().score_batch(outputs, prompts)


_ARGS = SimpleNamespace(video_reward_pool="judge")


@pytest.mark.asyncio
async def test_reward_scores_each_distinct_clip_once_on_the_pool(tmp_path):
    clips = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for clip in clips:
        clip.write_bytes(clip.name.encode())
    pool = _StubPool()

    def sample(path):
        submitted = {"clip_id": "clip_1", "path": str(path), "prompt": "x"} if path else None
        return Sample(metadata={"video_gen": {"request": REQUEST, "submitted": submitted}})

    rewards = await reward_func(
        _ARGS, [sample(clips[0]), sample(clips[1]), sample(clips[0]), sample(None)], judge_pool=pool
    )

    assert pool.calls == [([str(clips[0]), str(clips[1])], [REQUEST, REQUEST])]
    assert rewards == [stub_score(str(clips[0])), stub_score(str(clips[1])), stub_score(str(clips[0])), 0.0]
    assert 0 <= rewards[0] < 1


@pytest.mark.asyncio
async def test_reward_pool_failure_is_a_missing_reward():
    submitted = {"clip_id": "clip_1", "path": "/nowhere.mp4", "prompt": "x"}
    samples = [Sample(metadata={"video_gen": {"request": REQUEST, "submitted": submitted}}), Sample(metadata={})]

    assert await reward_func(_ARGS, samples, judge_pool=_StubPool(broken=True)) == [None, 0.0]


@pytest.mark.asyncio
async def test_an_episode_without_a_submission_never_calls_the_pool():
    pool = _StubPool()

    assert await reward_func(_ARGS, [Sample(metadata={})], judge_pool=pool) == [0.0]
    assert pool.calls == []


@pytest.mark.asyncio
async def test_reward_names_an_undeclared_pool():
    with pytest.raises(AssertionError, match="--video-reward-pool judge is not a declared pool"):
        await reward_func(_ARGS, [Sample(metadata={})], other_pool=_StubPool())
