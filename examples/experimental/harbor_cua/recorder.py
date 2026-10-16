"""A recording reverse proxy between Harbor's litellm client and SGLang.

Harbor's ``computer-1`` never reads token ids back: ``Computer1Chat.chat()``
calls ``litellm.acompletion`` directly and builds an ``LLMResponse`` without
them, and ``rollout_details`` is a hardcoded empty list. So the exact evidence is
taken here instead, at the moment each call is made -- which also means no
Harbor fork and no dependence on litellm preserving provider-specific fields.

The flag mapping is LIN-1996's, measured against sglang serving Qwen3-VL-4B:

  request:  return_prompt_token_ids=True -> choice.prompt_token_ids
            return_token_ids=True        -> choice.response_token_ids
            return_meta_info=True        -> choice.meta_info.output_token_logprobs
                                            as (logprob, token_id, text) triples

Each rollout gets its own URL prefix because litellm appends only
``/chat/completions`` to ``api_base`` and offers no dependable header hook.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import secrets
from dataclasses import dataclass
from urllib.parse import unquote

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from examples.experimental.harbor_cua.calls import (
    CallEvidenceError,
    MediaOccurrence,
    RecordedCall,
    RolloutRecord,
)

logger = logging.getLogger(__name__)

TRACE_FLAGS = {
    "return_prompt_token_ids": True,
    "return_token_ids": True,
    "return_meta_info": True,
}

_HOP_BY_HOP = {"content-length", "content-encoding", "transfer-encoding", "connection", "host"}
_DATA_URL_PREFIX = "data:"


class RecorderError(RuntimeError):
    """The recorder could not serve or keep a call."""


@dataclass(frozen=True)
class ParsedChoice:
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    output_logprobs: tuple[float, ...]
    finish_reason: str | None
    meta_info: dict


def parse_media(messages: list) -> tuple[list[tuple[bytes, str]], list[str]]:
    """Every ``image_url`` part's bytes, in occurrence order.

    Harbor sends screenshots as inline ``data:`` URLs, so the bytes in the
    request are already immutable; a remote URL would not be, and is refused
    rather than fetched behind inference's back.
    """
    media: list[tuple[bytes, str]] = []
    problems: list[str] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = (part.get("image_url") or {}).get("url") or ""
            if not url.startswith(_DATA_URL_PREFIX):
                problems.append(f"image part is not an inline data URL: {url[:40]!r}")
                continue
            header, _, payload = url[len(_DATA_URL_PREFIX) :].partition(",")
            if "base64" not in header:
                problems.append(f"image data URL is not base64: {header!r}")
                continue
            try:
                media.append((base64.b64decode(unquote(payload)), header.split(";")[0] or "image/webp"))
            except (binascii.Error, ValueError) as exc:
                problems.append(f"image data URL did not decode: {exc}")
    return media, problems


def parse_choice(choice: dict, envelope: dict | None = None) -> ParsedChoice:
    """Read one choice's token evidence, wherever the server put it.

    sglang answers on the choice, and a proxy in front of it may hoist the ids to
    the envelope instead -- which is where Harbor's own reader looks for the
    prompt ids. Accepting both is what keeps the recorder working behind the
    Miles router as well as against an engine directly.
    """
    envelope = envelope or {}
    metadata = envelope.get("metadata") if isinstance(envelope.get("metadata"), dict) else {}
    meta_info = choice.get("meta_info") or envelope.get("meta_info") or metadata.get("meta_info") or {}
    triples = meta_info.get("output_token_logprobs") or []

    output_ids = (
        choice.get("response_token_ids") or envelope.get("response_token_ids") or metadata.get("response_token_ids")
    )
    if not isinstance(output_ids, list) and triples:
        output_ids = [triple[1] for triple in triples]

    logprobs = [triple[0] for triple in triples]
    if not logprobs:
        content = (choice.get("logprobs") or {}).get("content") or []
        logprobs = [entry["logprob"] for entry in content if "logprob" in entry]

    prompt_ids = (
        choice.get("prompt_token_ids") or envelope.get("prompt_token_ids") or metadata.get("prompt_token_ids")
    )
    if not isinstance(prompt_ids, list):
        raise CallEvidenceError(
            "the engine returned no prompt_token_ids; the recorder asks for them with "
            "return_prompt_token_ids, so an empty field means the engine did not honour it"
        )
    return ParsedChoice(
        prompt_token_ids=tuple(prompt_ids),
        output_token_ids=tuple(output_ids or ()),
        output_logprobs=tuple(logprobs),
        finish_reason=choice.get("finish_reason"),
        meta_info=dict(meta_info),
    )


class CallRecorder:
    """Keeps one ``RolloutRecord`` per open rollout, keyed by its URL token."""

    def __init__(self, *, upstream: str, timeout: float = 1800.0) -> None:
        self._upstream = upstream.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._upstream, timeout=timeout)
        self._records: dict[str, RolloutRecord] = {}
        self._app = self._build_app()
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None
        self._port: int | None = None

    async def start(self) -> None:
        if self._server is not None:
            return
        config = uvicorn.Config(self._app, host="127.0.0.1", port=0, log_level="info", access_log=True)
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        for _ in range(600):
            sockets = getattr(self._server, "servers", None)
            if sockets:
                self._port = sockets[0].sockets[0].getsockname()[1]
                return
            await asyncio.sleep(0.05)
        raise RecorderError("the recorder did not bind a port within 30s")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
        await self._client.aclose()
        self._server, self._task, self._port = None, None, None

    @property
    def port(self) -> int:
        if self._port is None:
            raise RecorderError("the recorder is not started")
        return self._port

    def open_rollout(self) -> str:
        token = secrets.token_hex(8)
        self._records[token] = RolloutRecord()
        return token

    def base_url(self, token: str) -> str:
        return f"http://127.0.0.1:{self.port}/r/{token}/v1"

    def take(self, token: str) -> RolloutRecord:
        return self._records.pop(token)

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/r/{token}/v1/chat/completions")
        async def _chat(token: str, request: Request) -> Response:
            return await self._forward_chat(token, request)

        @app.api_route("/r/{token}/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def _passthrough(token: str, path: str, request: Request) -> Response:
            return await self._passthrough(path, request)

        return app

    async def _forward_chat(self, token: str, request: Request) -> Response:
        record = self._records.get(token)
        if record is None:
            return JSONResponse({"error": {"message": f"unknown rollout {token}"}}, status_code=404)

        body = await request.body()
        payload = json.loads(body or b"{}")
        if payload.get("stream"):
            # Streaming would hand back tokens a chunk at a time with no prompt ids;
            # Harbor's computer-1 never streams, so this is a misconfiguration.
            return JSONResponse({"error": {"message": "the recorder does not serve streaming"}}, status_code=400)

        auxiliary = bool(request.headers.get("x-session-id") or payload.get("session_id"))
        media_bytes, media_problems = parse_media(payload.get("messages") or [])
        payload.update(TRACE_FLAGS)
        payload.setdefault("logprobs", True)

        upstream = await self._client.post(
            "/v1/chat/completions",
            content=json.dumps(payload).encode(),
            headers={k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP},
        )
        if upstream.status_code != 200:
            record.rejected.append(f"engine returned {upstream.status_code} for call {len(record.calls)}")
            return Response(content=upstream.content, status_code=upstream.status_code)

        data = upstream.json()
        self._keep(record, data, auxiliary=auxiliary, media_bytes=media_bytes, media_problems=media_problems)
        for choice in data.get("choices", []):
            if isinstance(choice, dict):
                choice.pop("meta_info", None)
        return JSONResponse(data)

    def _keep(
        self,
        record: RolloutRecord,
        data: dict,
        *,
        auxiliary: bool,
        media_bytes: list[tuple[bytes, str]],
        media_problems: list[str],
    ) -> None:
        index = len(record.calls)
        if media_problems:
            record.rejected.append(f"call {index}: " + "; ".join(media_problems))
            return
        choices = data.get("choices") or []
        if not choices:
            record.rejected.append(f"call {index}: the engine returned no choices")
            return
        try:
            parsed = parse_choice(choices[0], data)
            occurrences: tuple[MediaOccurrence, ...] = tuple(
                record.arena.put(payload, media_type) for payload, media_type in media_bytes
            )
            record.calls.append(
                RecordedCall(
                    index=index,
                    auxiliary=auxiliary,
                    prompt_token_ids=parsed.prompt_token_ids,
                    output_token_ids=parsed.output_token_ids,
                    output_logprobs=parsed.output_logprobs,
                    media=occurrences,
                    finish_reason=parsed.finish_reason,
                    meta_info=parsed.meta_info,
                    model=data.get("model"),
                )
            )
        except CallEvidenceError as exc:
            # The keys are in the message because the usual cause is a proxy that
            # re-serialises the reply and drops the fields the recorder asked for.
            record.rejected.append(
                f"call {index}: {exc} (choice keys {sorted(choices[0])}, envelope keys {sorted(data)}, "
                f"metadata {str(data.get('metadata'))[:200]})"
            )

    async def _passthrough(self, path: str, request: Request) -> Response:
        upstream_request = self._client.build_request(
            request.method,
            f"/v1/{path}",
            content=await request.body(),
            headers={k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP},
            params=request.query_params,
        )
        response = await self._client.send(upstream_request, stream=True)
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items() if k.lower() not in _HOP_BY_HOP},
            background=response.aclose,
        )
