"""A reverse proxy that lets stock sglang serve NeMo Gym's exact-trace contract.

Gym's ``exact_trace.py`` reads ``prompt_token_ids``, ``generation_token_ids`` and
``generation_log_probs`` off the chat *message* -- the shape NeMo-RL's vLLM proxy
returns -- and never asks for them in the request. sglang has all three, but on
the *choice* and only when asked:

  request:  return_prompt_token_ids=True -> choice.prompt_token_ids
            return_token_ids=True        -> choice.response_token_ids
            return_meta_info=True        -> choice.meta_info.output_token_logprobs
                                            as (logprob, token_id, text) triples

So this proxy adds the flags on the way out and lifts the three fields onto the
message on the way back. Point Gym's ``POLICY_BASE_URL`` at it and sglang needs
no patch; Gym needs no fork.

    python gym_sglang_proxy.py --upstream http://127.0.0.1:30000 --port 30001

Verified against sglang 0.5.22.dev41 serving Qwen3-VL-4B-Instruct: four-turn
screenshot conversations came back with one logprob per sampled token and a
prompt that extended the previous turn's prompt+generation exactly.
"""

from __future__ import annotations

import argparse
import json

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

TRACE_FLAGS = {
    "return_prompt_token_ids": True,
    "return_token_ids": True,
    "return_meta_info": True,
}

_HOP_BY_HOP = {"content-length", "content-encoding", "transfer-encoding", "connection"}


def _lift_onto_message(choice: dict) -> None:
    message = choice.get("message")
    if not isinstance(message, dict):
        return

    prompt_ids = choice.get("prompt_token_ids")
    if isinstance(prompt_ids, list):
        message.setdefault("prompt_token_ids", prompt_ids)

    generation_ids = choice.get("response_token_ids")
    triples = (choice.get("meta_info") or {}).get("output_token_logprobs") or []
    if not isinstance(generation_ids, list) and triples:
        generation_ids = [triple[1] for triple in triples]
    if isinstance(generation_ids, list):
        message.setdefault("generation_token_ids", generation_ids)

    logprobs = [triple[0] for triple in triples]
    if not logprobs:
        content = (choice.get("logprobs") or {}).get("content") or []
        logprobs = [entry["logprob"] for entry in content if "logprob" in entry]
    if logprobs:
        message.setdefault("generation_log_probs", logprobs)


def build_app(upstream: str, timeout: float) -> FastAPI:
    app = FastAPI()
    client = httpx.AsyncClient(base_url=upstream.rstrip("/"), timeout=timeout)

    @app.on_event("shutdown")
    async def _close() -> None:
        await client.aclose()

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def _forward(path: str, request: Request) -> Response:
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP | {"host"}}
        is_chat = request.method == "POST" and path.endswith("chat/completions")
        streaming = False

        if is_chat:
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                streaming = bool(payload.get("stream"))
                if not streaming:
                    payload.update(TRACE_FLAGS)
                    payload.setdefault("logprobs", True)
                    body = json.dumps(payload).encode()

        if streaming or not is_chat:
            upstream_request = client.build_request(
                request.method, f"/{path}", content=body, headers=headers, params=request.query_params
            )
            response = await client.send(upstream_request, stream=True)
            passthrough = {k: v for k, v in response.headers.items() if k.lower() not in _HOP_BY_HOP}
            return StreamingResponse(
                response.aiter_raw(),
                status_code=response.status_code,
                headers=passthrough,
                background=response.aclose,
            )

        response = await client.request(
            request.method, f"/{path}", content=body, headers=headers, params=request.query_params
        )
        if response.status_code != 200:
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers={k: v for k, v in response.headers.items() if k.lower() not in _HOP_BY_HOP},
            )

        data = response.json()
        for choice in data.get("choices", []):
            if isinstance(choice, dict):
                _lift_onto_message(choice)
        return JSONResponse(data)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:30000")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    uvicorn.run(build_app(args.upstream, args.timeout), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
