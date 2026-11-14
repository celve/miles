"""Run ``libero_agent_function.run`` on real ``libero-sim`` sandboxes, outside training: a smoke or a measurement.

The policy is a served model (``--policy-url``, an OpenAI-compatible server such as SGLang's, without
``/v1``), or by default a local scripted stub that answers every turn with the next pose change of
``libero_probe.SCRIPT`` in PEAR's answer format and checks each request carries PEAR's conversation (two
512x512 PNG frames per observation). Episodes run concurrently, each in its own sandbox, as a rollout
runs them; ``--samples`` episodes of a row make a group.

    export E2B_API_URL=... E2B_SANDBOX_URL=...   # the sandbox service, as in libero_sandbox
    python libero_episodes.py --episodes 4                       # the scripted smoke, no GPU
    AGENT_MODEL_NAME=<served name> python libero_episodes.py --policy-url http://<host>:30000 \
        --rows data/libero.heldout.jsonl --limit 8 --samples 8 --concurrency 16 --out episodes.jsonl
"""

import argparse
import asyncio
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import libero_agent_function
import libero_probe
from PIL import Image


def _check_request(body: dict) -> None:
    messages = body["messages"]
    assert messages[0]["role"] == "system" and "## TASK" in messages[0]["content"], "no PEAR system prompt"
    observation = messages[-1]["content"]
    urls = [part["image_url"]["url"] for part in observation if part["type"] == "image_url"]
    assert len(urls) == 2, f"{len(urls)} frames in the last observation"
    for url in urls:
        image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        assert image.format == "PNG" and image.size == (512, 512), f"frame {image.format} {image.size}"


def _scripted_policy(script: list[dict]):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            _check_request(body)
            turn = sum(1 for message in body["messages"] if message["role"] == "assistant")
            action = script[turn % len(script)]
            text = f"<think>scripted turn {turn}</think>\n```json\n{json.dumps(action)}\n```"
            data = json.dumps(
                {
                    "id": f"smoke-{turn}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 600 * (turn + 1), "completion_tokens": 40, "total_tokens": 0},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


async def _episode(policy_url: str, metadata: dict, request_kwargs: dict, limit: asyncio.Semaphore) -> dict:
    async with limit:
        started = time.monotonic()
        try:
            result = await libero_agent_function.run(
                base_url=policy_url, prompt=None, request_kwargs=request_kwargs, metadata=metadata
            )
        except Exception as exc:
            return {"metadata": metadata, "error": f"{type(exc).__name__}: {exc}"[:2000]}
    metrics = result["agent_metrics"]
    return {
        "metadata": metadata,
        "reward": result["reward"],
        "end_reason": metrics["end_reason"],
        "turns": metrics["turns"],
        "outcomes": result["libero"]["outcomes"],
        "gen_s_mean": round(metrics["total_gen_time"] / max(1, len(metrics["gen_times"])), 2),
        "native_steps": metrics["native_steps"],
        "sandbox_start_s": round(metrics["sandbox_start_time"], 1),
        "reset_s": round(metrics["reset_time"], 2),
        "act_s_mean": round(sum(metrics["sim_times"][1:]) / max(1, len(metrics["sim_times"]) - 1), 2),
        "wall_s": round(time.monotonic() - started, 1),
    }


def _episodes(args) -> list[dict]:
    if args.rows:
        with open(args.rows) as rows:
            metadata = [json.loads(line)["metadata"] for line in rows if line.strip()][: args.limit]
    else:
        metadata = [{"suite": args.suite, "task_id": args.task, "trial_id": trial} for trial in range(args.episodes)]
    return [dict(row) for row in metadata for _ in range(args.samples)]


def _summary(results: list[dict]) -> str:
    done = [r for r in results if "error" not in r]
    groups: dict[str, list[float]] = {}
    for r in done:
        groups.setdefault(json.dumps(r["metadata"], sort_keys=True), []).append(r["reward"])
    mixed = sum(1 for rewards in groups.values() if 0 < sum(rewards) < len(rewards))
    rate = sum(r["reward"] for r in done) / max(1, len(done))
    ends: dict[str, int] = {}
    for r in done:
        ends[r["end_reason"]] = ends.get(r["end_reason"], 0) + 1
    return (
        f"{len(done)}/{len(results)} episodes completed; success {rate:.3f}; "
        f"{mixed}/{len(groups)} groups with mixed rewards; end reasons {ends}"
    )


async def _main(args) -> int:
    server, url = (None, args.policy_url) if args.policy_url else _scripted_policy(libero_probe.SCRIPT)
    request_kwargs = {"max_tokens": args.max_tokens, "temperature": args.temperature, "top_p": args.top_p}
    limit = asyncio.Semaphore(args.concurrency)
    try:
        results = await asyncio.gather(*(_episode(url, m, request_kwargs, limit) for m in _episodes(args)))
    finally:
        if server is not None:
            server.shutdown()
    with open(args.out, "w") if args.out else open("/dev/null", "w") as out:
        for result in results:
            print(json.dumps(result))
            out.write(json.dumps(result) + "\n")
    print(_summary(results))
    return 1 if any("error" in r for r in results) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy-url", default=None, help="a served model's OpenAI root, without /v1")
    parser.add_argument("--rows", default=None, help="rows from make_libero_data.py; else --suite/--task/--episodes")
    parser.add_argument("--limit", type=int, default=None, help="the first N rows")
    parser.add_argument("--episodes", type=int, default=2, help="without --rows: trials 0..N-1")
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1, help="episodes per row, a group")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4096, help="one reply's cap")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--out", default=None, help="write one JSON line per episode here")
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
