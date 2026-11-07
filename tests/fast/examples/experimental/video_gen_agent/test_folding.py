"""Clip folding, recall and multi-shot extend in the env's episode loop, on real mp4 clips and a scripted planner."""

import json
import os
from pathlib import Path

import av
import examples.experimental.video_gen_agent.env as env
import numpy as np
import pytest
from examples.experimental.video_gen_agent.generator import Clip, ClipRequest
from PIL import Image

REQUEST = "A paper boat drifts down a rainy street."
_SIZE = {"AGENT_VIDEO_GEN_WIDTH": "64", "AGENT_VIDEO_GEN_HEIGHT": "48", "AGENT_VIDEO_GEN_FPS": "8"}


def write_clip(path: Path, shade: int, *, frames: int = 9) -> str:
    """A 64x48 mp4 whose frame ``i`` is uniformly ``shade + 10 * i``, so a decoded frame names its index."""
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=8)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(frames):
            frame = np.full((48, 64, 3), shade + 10 * index, dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())
    return str(path)


def frame_shades(path: str) -> list[int]:
    with av.open(path) as container:
        return [round(frame.to_ndarray(format="rgb24").mean()) for frame in container.decode(video=0)]


class _Mp4Generator:
    """One real clip per request, ``shade`` 20 for the first, 40 for the second, ...; it records each request."""

    requests: list[ClipRequest] = []

    def __init__(self):
        self.clip_dir = Path(os.environ["AGENT_VIDEO_GEN_FAKE_DIR"])

    async def generate(self, request: ClipRequest) -> Clip:
        _Mp4Generator.requests.append(request)
        index = len(_Mp4Generator.requests)
        return Clip(path=write_clip(self.clip_dir / f"{index}.mp4", 20 * index))


@pytest.fixture
def options(monkeypatch, tmp_path):
    _Mp4Generator.requests = []
    (tmp_path / "clips").mkdir()
    for name, value in {
        "AGENT_VIDEO_GEN_GENERATOR": f"{__name__}._Mp4Generator",
        "AGENT_VIDEO_GEN_FAKE_DIR": str(tmp_path / "clips"),
        "AGENT_VIDEO_GEN_SUBMITTED_DIR": str(tmp_path / "kept"),
        "AGENT_VIDEO_GEN_CLIP_OBSERVATIONS": "1",
        **_SIZE,
    }.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def _call(name: str, arguments: dict) -> dict:
    return {"id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def _script(monkeypatch, *turns: list[dict]) -> list[dict]:
    """Answer each chat request with the next turn's tool calls; returns the payloads sent."""
    sent: list[dict] = []

    async def fake_post(url, payload):
        sent.append(json.loads(json.dumps(payload)))
        usage = {"prompt_tokens": 100 * len(sent)}
        message = {"role": "assistant", "content": "", "tool_calls": turns[len(sent) - 1]}
        return {"choices": [{"message": message}], "usage": usage}

    monkeypatch.setattr(env, "post", fake_post)
    return sent


async def _run() -> dict:
    return await env.run_agent("http://session", [{"role": "user", "content": REQUEST}], {}, {})


def _media(messages: list[dict]) -> list[tuple[str, str]]:
    return [
        (part["type"], part[part["type"]]["url"].removeprefix("file://"))
        for m in messages
        if isinstance(m["content"], list)
        for part in m["content"]
        if part["type"] in ("video_url", "image_url")
    ]


def _clip(index: int) -> str:
    return str(Path(os.environ["AGENT_VIDEO_GEN_FAKE_DIR"]) / f"{index}.mp4")


@pytest.mark.asyncio
async def test_only_the_last_k_clips_stay_in_view_and_older_ones_fold_to_a_note(options):
    options.setenv("AGENT_VIDEO_GEN_KEEP_CLIPS", "1")
    sent = _script(
        options,
        [_call("generate", {"prompt": "a boat"})],
        [_call("generate", {"prompt": "a boat in rain"})],
        [_call("submit", {"clip_id": "clip_2"})],
    )

    result = await _run()

    assert [tool["function"]["name"] for tool in sent[0]["tools"]] == ["generate", "submit", "recall"]
    assert "Only your last 1 clips stay in view" in sent[0]["messages"][0]["content"]
    assert _media(sent[1]["messages"]) == [("video_url", _clip(1))]
    assert _media(sent[2]["messages"]) == [("video_url", _clip(2))]
    folded = sent[2]["messages"][3]
    assert folded["role"] == "tool" and folded["content"] == (
        "Generated clip_1: 5 s, seed 0. Generations left: 2. Turns left: 7. "
        'clip_1 is folded out of view; it was generated from the prompt "a boat". Call recall to watch it again.'
    )
    # The planner's own turns, its critique among them, are never folded.
    assert sent[2]["messages"][:3] == sent[1]["messages"][:3]
    assert sent[2]["messages"][4] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [_call("generate", {"prompt": "a boat in rain"})],
    }
    record = result["video_gen"]
    assert record["folded_clips"] == 1 and record["submitted"]["clip_id"] == "clip_2"
    assert [c["path"] for c in record["clips"]] == [_clip(1), _clip(2)]
    assert [t["prompt_tokens"] for t in record["turn_stats"]] == [100, 200, 300]
    assert all(t["seconds"] >= 0 for t in record["turn_stats"])


@pytest.mark.asyncio
async def test_recall_shows_a_folded_clip_at_its_own_path_and_folds_the_older_view(options):
    options.setenv("AGENT_VIDEO_GEN_KEEP_CLIPS", "1")
    sent = _script(
        options,
        [_call("generate", {"prompt": "a boat"})],
        [_call("generate", {"prompt": "a boat in rain"})],
        [_call("recall", {"clip_id": "clip_2"}), _call("recall", {"clip_id": "clip_1"})],
        [_call("submit", {"clip_id": "clip_1"})],
    )

    result = await _run()

    already, recalled = [m["content"] for m in sent[3]["messages"][-2:]]
    assert already == "Error: clip_2 is already in view. Generations left: 1. Turns left: 5."
    assert recalled[0]["text"].startswith("Recalled clip_1.")
    assert _media(sent[3]["messages"]) == [("video_url", _clip(1))]
    assert "clip_2 is folded out of view" in sent[3]["messages"][5]["content"]
    assert result["video_gen"]["recalls"] == 1 and result["video_gen"]["folded_clips"] == 2


@pytest.mark.asyncio
async def test_a_keyframe_fold_shows_the_clips_last_frame_from_a_file_written_once(options):
    options.setenv("AGENT_VIDEO_GEN_KEEP_CLIPS", "1")
    options.setenv("AGENT_VIDEO_GEN_FOLD", "keyframe")
    sent = _script(
        options,
        [_call("generate", {"prompt": "a boat"})],
        [_call("generate", {"prompt": "a boat in rain"})],
        [_call("generate", {"prompt": "a boat at night"})],
        [_call("submit", {"clip_id": "clip_3"})],
    )

    await _run()

    frame = f"{_clip(1)}.last.png"
    assert _media(sent[2]["messages"]) == [("image_url", frame), ("video_url", _clip(2))]
    assert _media(sent[3]["messages"])[0] == ("image_url", frame)
    assert sent[2]["messages"][3]["content"][0]["text"].endswith(
        "clip_1 is folded to its last frame; call recall to watch it again."
    )
    # Frame 8 of clip 1 is shade 20 + 80.
    assert abs(np.asarray(Image.open(frame)).mean() - 100) < 3
    assert sorted(p.name for p in Path(frame).parent.iterdir() if p.suffix == ".png") == [
        "1.mp4.last.png",
        "2.mp4.last.png",
    ]


@pytest.mark.asyncio
async def test_extend_continues_a_clip_from_its_last_frame_on_its_own_budget(options):
    options.setenv("AGENT_VIDEO_GEN_MAX_EXTENDS", "1")
    options.setenv("AGENT_VIDEO_GEN_MAX_GENERATIONS", "1")
    sent = _script(
        options,
        [_call("generate", {"prompt": "a boat", "seconds": 1})],
        [
            _call("extend", {"clip_id": "clip_9", "prompt": "x"}),
            _call("extend", {"clip_id": "clip_1", "prompt": "it sinks"}),
        ],
        [_call("extend", {"clip_id": "clip_2", "prompt": "again"}), _call("generate", {"prompt": "again"})],
        [_call("submit", {"clip_id": "clip_2"})],
    )

    result = await _run()

    assert [tool["function"]["name"] for tool in sent[0]["tools"]] == ["generate", "submit", "extend"]
    assert "You have 1 extends." in sent[0]["messages"][0]["content"]
    first, second = _Mp4Generator.requests
    assert first.image_path is None
    assert second.image_path == f"{_clip(1)}.last.png" and second.prompt == "it sinks" and second.seed == 1
    unknown, extended = [m["content"] for m in sent[2]["messages"][-2:]]
    assert unknown.startswith("Error: unknown clip_id 'clip_9'; generated clips: clip_1.")
    assert (
        extended[0]["text"]
        == "Generated clip_2 extending clip_1: 5 s, seed 1. Generations left: 0. Extends left: 0. Turns left: 6."
    )
    no_extends, no_generations = [m["content"] for m in sent[3]["messages"][-2:]]
    assert no_extends == "Error: no extends left. Generations left: 0. Extends left: 0. Turns left: 5."
    assert no_generations.startswith("Error: no generations left.")
    assert [c["parent"] for c in result["video_gen"]["clips"]] == [None, "clip_1"]


@pytest.mark.asyncio
async def test_submitting_a_shot_submits_its_chain_joined_without_the_repeated_frames(options):
    options.setenv("AGENT_VIDEO_GEN_MAX_EXTENDS", "2")
    _script(
        options,
        [_call("generate", {"prompt": "a boat"})],
        [_call("extend", {"clip_id": "clip_1", "prompt": "it sinks"})],
        [_call("extend", {"clip_id": "clip_2", "prompt": "a fish swims by"})],
        [_call("submit", {"clip_id": "clip_3"})],
    )

    submitted = (await _run())["video_gen"]["submitted"]

    assert submitted["shots"] == ["clip_1", "clip_2", "clip_3"] and submitted["prompt"] == "a fish swims by"
    assert Path(submitted["path"]).parent == Path(os.environ["AGENT_VIDEO_GEN_SUBMITTED_DIR"])
    shades = frame_shades(submitted["path"])
    expected = frame_shades(_clip(1)) + frame_shades(_clip(2))[1:] + frame_shades(_clip(3))[1:]
    assert len(shades) == 9 + 8 + 8 and np.abs(np.subtract(shades, expected)).max() <= 2


@pytest.mark.asyncio
async def test_an_unextended_clip_is_submitted_as_it_is(options):
    options.setenv("AGENT_VIDEO_GEN_MAX_EXTENDS", "2")
    _script(options, [_call("generate", {"prompt": "a boat"})], [_call("submit", {"clip_id": "clip_1"})])

    submitted = (await _run())["video_gen"]["submitted"]

    assert "shots" not in submitted
    assert Path(submitted["path"]).read_bytes() == Path(_clip(1)).read_bytes()


@pytest.mark.asyncio
async def test_with_folding_and_extend_off_the_requests_are_unchanged(options):
    sent = _script(
        options,
        [_call("generate", {"prompt": "a boat"}), _call("recall", {"clip_id": "clip_1"})],
        [_call("extend", {"clip_id": "clip_1", "prompt": "x"})],
        [_call("submit", {"clip_id": "clip_1"})],
    )

    await _run()

    assert all(payload["tools"] == env.TOOLS for payload in sent)
    assert sent[1]["messages"][-1]["content"].startswith(
        "Error: unknown tool 'recall'; the tools are generate and submit."
    )
    assert sent[2]["messages"][-1]["content"].startswith(
        "Error: unknown tool 'extend'; the tools are generate and submit."
    )
    assert _media(sent[2]["messages"]) == [("video_url", _clip(1))]


def test_folding_needs_clip_observations_and_a_known_fold(monkeypatch):
    monkeypatch.setenv("AGENT_VIDEO_GEN_GENERATOR", "some.generator")
    monkeypatch.setenv("AGENT_VIDEO_GEN_SUBMITTED_DIR", "/kept")
    monkeypatch.setenv("AGENT_VIDEO_GEN_KEEP_CLIPS", "2")
    with pytest.raises(ValueError, match="KEEP_CLIPS folds clip observations"):
        env.VideoGenEnvOptions.from_env()
    monkeypatch.setenv("AGENT_VIDEO_GEN_CLIP_OBSERVATIONS", "1")
    monkeypatch.setenv("AGENT_VIDEO_GEN_FOLD", "summary")
    with pytest.raises(ValueError, match="FOLD is one of text, keyframe"):
        env.VideoGenEnvOptions.from_env()
    monkeypatch.setenv("AGENT_VIDEO_GEN_FOLD", "keyframe")
    monkeypatch.setenv("AGENT_VIDEO_GEN_MAX_EXTENDS", "-1")
    with pytest.raises(ValueError, match="MAX_EXTENDS must not be negative"):
        env.VideoGenEnvOptions.from_env()
    monkeypatch.setenv("AGENT_VIDEO_GEN_MAX_EXTENDS", "4")
    options = env.VideoGenEnvOptions.from_env()
    assert (options.keep_clips, options.fold, options.max_extends) == (2, "keyframe", 4)
