import json
import os

import pytest
from examples.experimental.video_gen_agent.multiturn import fold_sweep
from fastapi import FastAPI, Request

from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


@pytest.fixture
def engine():
    """Counts 1000 prompt tokens per clip, 100 per image and 10 per message; records each request."""
    app, requests = FastAPI(), []

    @app.post("/flush_cache")
    async def flush():
        requests.append("flush")
        return {}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        requests.append(payload)
        parts = [p["type"] for m in payload["messages"] if isinstance(m["content"], list) for p in m["content"]]
        tokens = 1000 * parts.count("video_url") + 100 * parts.count("image_url") + 10 * len(payload["messages"])
        message = {"role": "assistant", "content": "x"}
        return {"choices": [{"message": message}], "usage": {"prompt_tokens": tokens}}

    server = UvicornThreadServer(app, host="127.0.0.1", port=find_available_port(34500))
    server.start()
    yield server.url, requests
    server.stop()


async def test_each_k_replays_the_same_episode_and_folding_cuts_the_prompt(engine, tmp_path, monkeypatch):
    url, requests = engine
    monkeypatch.delenv("AGENT_VIDEO_GEN_KEEP_CLIPS", raising=False)
    clips = fold_sweep.make_clips(str(tmp_path), 4, width=64, height=48, frames=9, fps=8)

    full = await fold_sweep.run_episode(url, clips, keep=0, fold="text", max_seconds=1)
    kept = await fold_sweep.run_episode(url, clips, keep=1, fold="text", max_seconds=1)
    keyframes = await fold_sweep.run_episode(url, clips, keep=1, fold="keyframe", max_seconds=1)

    # Turn t of N=4 shots sees t-1 clips; K=1 keeps one of them, and a keyframe fold leaves an image per folded clip.
    assert [t["prompt_tokens"] for t in full] == [20, 1040, 2060, 3080, 4100]
    assert [t["prompt_tokens"] for t in kept] == [20, 1040, 1060, 1080, 1100]
    assert [t["prompt_tokens"] for t in keyframes] == [20, 1040, 1160, 1280, 1400]
    assert [r for r in requests if r == "flush"] == ["flush"] * 3
    chats = [r for r in requests if r != "flush"]
    assert all(r["max_tokens"] == 1 and r["temperature"] == 0 for r in chats)
    assert [tool["function"]["name"] for tool in chats[0]["tools"]] == ["generate", "submit", "extend"]
    summary = fold_sweep.summarize(kept)
    assert summary["peak_prompt_tokens"] == 1100 and summary["prompt_tokens_total"] == 4300
    assert json.loads(json.dumps(summary))["turns"] == kept
    assert "AGENT_VIDEO_GEN_KEEP_CLIPS" not in os.environ
