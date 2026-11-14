"""Run ``libero_agent_function.run`` end to end on real ``libero-sim`` sandboxes, with a scripted policy and no GPU.

A local OpenAI-compatible stub stands in for the session server: it answers every turn with the next
pose change of ``libero_probe.SCRIPT`` in PEAR's answer format, and checks each request carries PEAR's
conversation (two 512x512 PNG frames per observation). Episodes run concurrently, each in its own
sandbox, as a rollout would run them.

    export E2B_API_URL=... E2B_SANDBOX_URL=...   # the sandbox service, as in libero_sandbox
    python libero_smoke.py --episodes 4 --suite libero_goal --task 0
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


async def _episode(policy_url: str, metadata: dict) -> dict:
    started = time.monotonic()
    try:
        result = await libero_agent_function.run(
            base_url=policy_url, prompt=None, request_kwargs={"max_tokens": 256}, metadata=metadata
        )
    except Exception as exc:
        return {"metadata": metadata, "error": f"{type(exc).__name__}: {exc}"[:2000]}
    metrics = result["agent_metrics"]
    return {
        "metadata": metadata,
        "reward": result["reward"],
        "end_reason": metrics["end_reason"],
        "turns": metrics["turns"],
        "native_steps": metrics["native_steps"],
        "sandbox_start_s": round(metrics["sandbox_start_time"], 1),
        "reset_s": round(metrics["reset_time"], 2),
        "act_s_mean": round(sum(metrics["sim_times"][1:]) / max(1, len(metrics["sim_times"]) - 1), 2),
        "wall_s": round(time.monotonic() - started, 1),
    }


async def _main(args) -> int:
    server, url = _scripted_policy(libero_probe.SCRIPT)
    try:
        episodes = [{"suite": args.suite, "task_id": args.task, "trial_id": trial} for trial in range(args.episodes)]
        results = await asyncio.gather(*(_episode(url, metadata) for metadata in episodes))
    finally:
        server.shutdown()
    for result in results:
        print(json.dumps(result))
    failed = [r for r in results if "error" in r]
    print(f"{len(results) - len(failed)}/{len(results)} episodes completed")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=2, help="concurrent episodes, trials 0..N-1")
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task", type=int, default=0)
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
