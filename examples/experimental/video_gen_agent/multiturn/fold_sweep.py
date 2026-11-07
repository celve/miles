"""K sweep for clip folding: one fixed multi-shot episode, replayed through the env against a VLM engine for each K.

The planner's turns are scripted (generate, then extend the newest clip until N shots, then submit), and the generator
hands out the same N clips every time, so the only thing that changes between runs is K. Each turn's request is the
env's own, folds included, sent to ``--engine-url``'s ``/v1/chat/completions``; the engine's reply is read for its
prompt token count and timed, then replaced by the scripted turn. The engine's prefix cache is flushed before each
episode, so every K starts cold and later turns reuse the prefix as a run would.

Example:
    python -m examples.experimental.video_gen_agent.multiturn.fold_sweep --engine-url http://127.0.0.1:30000 \\
        --make-clips 8 --clip-dir /tmp/sweep_clips --keep 0 1 2 3 --out /tmp/sweep.json
"""

import argparse
import asyncio
import contextlib
import json
import os
import tempfile
from pathlib import Path

import av
import examples.experimental.video_gen_agent.env as env
import httpx
import numpy as np
from examples.experimental.video_gen_agent.generator import Clip, ClipRequest

_CLIPS_ENV = "FOLD_SWEEP_CLIPS"


class ReplayClipGenerator:
    """Hands out ``FOLD_SWEEP_CLIPS``'s clips in order, one per call."""

    def __init__(self):
        self.clips = os.environ[_CLIPS_ENV].split(os.pathsep)
        self.calls = 0

    async def generate(self, request: ClipRequest) -> Clip:
        self.calls += 1
        return Clip(path=self.clips[self.calls - 1])


def make_clips(clip_dir: str, count: int, *, width: int, height: int, frames: int, fps: int) -> list[str]:
    """Synthetic clips of a square sliding over a different background each; the engine's cost depends on shape only."""
    Path(clip_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = Path(clip_dir) / f"shot_{index}_{width}x{height}_{frames}f_{fps}fps.mp4"
        if not path.exists():
            with av.open(str(path), mode="w", format="mp4") as container:
                stream = container.add_stream("libx264", rate=fps)
                stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
                for step in range(frames):
                    frame = np.full((height, width, 3), (40 * index) % 256, dtype=np.uint8)
                    left = (step * width // frames) % (width - height // 4)
                    frame[height // 3 : height // 3 + height // 4, left : left + height // 4] = 255
                    container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
                container.mux(stream.encode())
        paths.append(str(path))
    return paths


def _scripted_turn(turn: int, shots: int) -> dict:
    if turn == 1:
        name, arguments = "generate", {"prompt": "shot 1"}
    elif turn <= shots:
        name, arguments = "extend", {"clip_id": f"clip_{turn - 1}", "prompt": f"shot {turn}"}
    else:
        name, arguments = "submit", {"clip_id": f"clip_{shots}"}
    call = {"id": f"call_{turn}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
    return {"role": "assistant", "content": "", "tool_calls": [call]}


async def run_episode(engine_url: str, clips: list[str], *, keep: int, fold: str, max_seconds: float) -> list[dict]:
    """One episode at K=``keep``; returns each turn's prompt tokens and engine seconds."""
    shots = len(clips)
    with tempfile.TemporaryDirectory() as submitted_dir:
        options = {
            "AGENT_VIDEO_GEN_GENERATOR": f"{__name__}.ReplayClipGenerator",
            "AGENT_VIDEO_GEN_SUBMITTED_DIR": submitted_dir,
            "AGENT_VIDEO_GEN_CLIP_OBSERVATIONS": "1",
            "AGENT_VIDEO_GEN_KEEP_CLIPS": str(keep),
            "AGENT_VIDEO_GEN_FOLD": fold,
            "AGENT_VIDEO_GEN_MAX_EXTENDS": str(shots - 1),
            "AGENT_VIDEO_GEN_MAX_TURNS": str(shots + 1),
            "AGENT_VIDEO_GEN_MAX_SECONDS": str(max_seconds),
            _CLIPS_ENV: os.pathsep.join(clips),
        }
        with _environment(options):
            async with httpx.AsyncClient(base_url=engine_url, timeout=None) as client:
                (await client.post("/flush_cache")).raise_for_status()
                record = (await _replay(engine_url, client, shots))["video_gen"]
    assert record["submitted"] and len(record["submitted"].get("shots", [1])) == shots, record["submitted"]
    return record["turn_stats"]


async def _replay(engine_url: str, client: httpx.AsyncClient, shots: int) -> dict:
    turns = 0

    async def post(url, payload):
        nonlocal turns
        turns += 1
        response = await client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        body["choices"][0]["message"] = _scripted_turn(turns, shots)
        return body

    original_post, env.post = env.post, post
    try:
        prompt = [{"role": "user", "content": "A multi-shot video."}]
        return await env.run_agent(engine_url, prompt, {"max_tokens": 1, "temperature": 0}, {})
    finally:
        env.post = original_post


@contextlib.contextmanager
def _environment(values: dict[str, str]):
    saved = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name)
            else:
                os.environ[name] = value


def summarize(turn_stats: list[dict]) -> dict:
    tokens = [t["prompt_tokens"] for t in turn_stats]
    seconds = [t["seconds"] for t in turn_stats]
    return {
        "prompt_tokens_per_turn": float(np.mean(tokens)),
        "peak_prompt_tokens": max(tokens),
        "prompt_tokens_total": sum(tokens),
        "seconds_per_turn": float(np.mean(seconds)),
        "seconds_total": sum(seconds),
        "turns": turn_stats,
    }


async def _main(args) -> None:
    clips = args.clips or make_clips(
        args.clip_dir, args.make_clips, width=args.width, height=args.height, frames=args.frames, fps=args.fps
    )
    results = {}
    for keep in args.keep:
        runs = [
            await run_episode(args.engine_url, clips, keep=keep, fold=args.fold, max_seconds=args.max_seconds)
            for _ in range(args.repeats)
        ]
        # The median run per turn, so one slow request does not move a K's numbers.
        turn_stats = [
            {key: float(np.median([run[turn][key] for run in runs])) for key in ("prompt_tokens", "seconds")}
            for turn in range(len(runs[0]))
        ]
        results[str(keep)] = summarize(turn_stats)
        row = results[str(keep)]
        print(
            f"K={keep or 'off'}: {row['prompt_tokens_per_turn']:.0f} prompt tokens/turn (peak {row['peak_prompt_tokens']:.0f}), "
            f"{row['seconds_per_turn']:.2f} s/turn, {row['seconds_total']:.1f} s/episode",
            flush=True,
        )
    meta = {"shots": len(clips), "fold": args.fold, "repeats": args.repeats, "clips": clips}
    Path(args.out).write_text(json.dumps({"meta": meta, "by_keep": results}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine-url", required=True)
    parser.add_argument(
        "--clips", nargs="*", help="the episode's shots, in order; default: --make-clips synthetic ones"
    )
    parser.add_argument("--make-clips", type=int, default=8)
    parser.add_argument("--clip-dir", default="/tmp/fold_sweep_clips")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max-seconds", type=float, default=2.0)
    parser.add_argument("--keep", type=int, nargs="+", default=[0, 1, 2, 3], help="K values; 0 folds nothing")
    parser.add_argument("--fold", choices=["text", "keyframe"], default="text")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
