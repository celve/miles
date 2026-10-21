"""Screenshot agent and reward for ``tests/e2e/sglang/test_session_server_images.py``."""

import base64
import hashlib
import io
import json
import os

import httpx
import numpy as np
from PIL import Image, ImageDraw

from miles.utils.test_utils.openai_stream_client import stream_chat_completions

JOURNAL_PATH_ENV = "MILES_SESSION_IMAGE_E2E_JOURNAL"
NUM_IMAGE_TURNS = 3
STREAMED_TURN = 1
FRAME_SIZE = (640, 448)
_BUTTON_LABELS = ("Open", "Save", "Cancel", "Next", "Back", "Submit", "Delete", "Search")


def frame_png(seed: int, step: int) -> bytes:
    rng = np.random.default_rng([seed, step])
    width, height = FRAME_SIZE
    image = Image.new("RGB", FRAME_SIZE, _color(rng, 200, 256))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 40), fill=_color(rng, 0, 120))
    draw.text((12, 12), f"Screen {step + 1}", fill=(255, 255, 255))
    for label in rng.choice(_BUTTON_LABELS, size=3, replace=False):
        x, y = int(rng.integers(20, width - 140)), int(rng.integers(60, height - 60))
        draw.rectangle((x, y, x + 120, y + 40), fill=_color(rng, 0, 160), outline=(0, 0, 0))
        draw.text((x + 12, y + 12), str(label), fill=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def episode_frames(frame_seed: int) -> list[bytes]:
    return [frame_png(frame_seed, step) for step in range(NUM_IMAGE_TURNS)]


def image_id(payload: bytes) -> str:
    # The session server's ImageStore key for these bytes.
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


async def run_agent(base_url, prompt, request_kwargs, metadata, **kwargs):
    messages = list(prompt)
    entry = {"frame_seed": metadata["frame_seed"], "turns": 0, "image_ids": []}
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            for step, payload in enumerate(episode_frames(metadata["frame_seed"])):
                messages.append(
                    {"role": "user", "content": [_image_part(payload), {"type": "text", "text": _ask(step)}]}
                )
                payload_json = {**request_kwargs, "messages": messages}
                choice = (await _chat(client, base_url, payload_json, stream=step == STREAMED_TURN))["choices"][0]
                messages.append({"role": "assistant", "content": choice["message"]["content"] or ""})
                entry["turns"] += 1
                entry["image_ids"].append(image_id(payload))
                entry["finish_reason"] = choice["finish_reason"]
                if choice["finish_reason"] == "length":
                    break
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"[:4000]
        raise
    finally:
        _append_journal(entry)
    return {"image_turns": entry["turns"], "sent_image_ids": entry["image_ids"]}


async def reward_by_index(args, samples, **kwargs) -> list[float]:
    return [float(sample.index % 2) for sample in samples]


async def _chat(client: httpx.AsyncClient, base_url: str, payload: dict, *, stream: bool) -> dict:
    url = f"{base_url}/v1/chat/completions"
    if stream:
        return await stream_chat_completions(client, url, payload, label=url)
    response = await client.post(url, json=payload)
    assert response.status_code == 200, f"{url} failed ({response.status_code}): {response.text}"
    return response.json()


def _image_part(payload: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(payload).decode()}"}}


def _ask(step: int) -> str:
    return f"Screen {step + 1} of {NUM_IMAGE_TURNS}. Name the next UI action in one short sentence."


def _color(rng: np.random.Generator, low: int, high: int) -> tuple[int, int, int]:
    return tuple(int(channel) for channel in rng.integers(low, high, size=3))


def _append_journal(entry: dict) -> None:
    # One write per line under O_APPEND, so concurrent episode processes never interleave.
    fd = os.open(os.environ[JOURNAL_PATH_ENV], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, (json.dumps(entry) + "\n").encode())
    finally:
        os.close(fd)
