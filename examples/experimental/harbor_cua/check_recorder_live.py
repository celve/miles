"""Prove the recorder's evidence against a live engine, before any training.

Sends a multi-turn screenshot conversation shaped like Harbor's generic
``computer-1`` path through the recorder to a running sglang server, then checks
what came back:

* the engine returned prompt ids, sampled ids and one logprob per sampled token;
* turn k+1's prompt extends turn k's prompt plus output, so segments can be long;
* each image-pad run in the prompt is exactly as long as that occurrence's
  processed token count, in order -- not merely equal in total;
* the engine's effective prompt ids equal the ids this machine renders and
  expands for the same messages, elementwise;
* a request carrying ``X-Session-ID`` is classified as the compactor's;
* dropping an early screenshot closes a segment and the next call opens another.

    python -m examples.experimental.harbor_cua.check_recorder_live \\
        --upstream http://127.0.0.1:30000 \\
        --model /personal/lin2045/models/Qwen3-VL-4B-Instruct
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import pathlib
from dataclasses import dataclass, field

import httpx
from PIL import Image, ImageDraw

from examples.experimental.harbor_cua.media import (
    MediaProcessor,
    first_difference,
    pad_token_runs,
    render_prompt_ids,
    verify_occurrence_runs,
)
from examples.experimental.harbor_cua.recorder import CallRecorder
from examples.experimental.harbor_cua.samples import build_samples
from examples.experimental.harbor_cua.segments import segment_calls
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample

SYSTEM_TEXT = (
    "You drive a 1024x900 desktop. Reply with one JSON object: "
    '{"analysis": "...", "plan": "...", "action": {"type": "click", "x": 0, "y": 0}}'
)


@dataclass
class Checks:
    results: dict[str, object] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def record(self, name: str, ok: bool, detail: object = None) -> None:
        self.results[name] = {"ok": ok, "detail": detail}
        if not ok:
            self.failures.append(f"{name}: {detail}")


def screenshot(index: int, size: tuple[int, int] = (1024, 900)) -> bytes:
    """A distinct frame per turn, so two occurrences never share bytes by accident."""
    image = Image.new("RGB", size, (240, 240 - index * 7, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle([40 + index * 30, 60 + index * 20, 240 + index * 30, 210 + index * 20], fill=(200, 40 + index * 20, 40))
    draw.text((60, 760), f"frame {index}", fill=(10, 10, 10))
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=80)
    return buffer.getvalue()


def data_url(payload: bytes) -> str:
    return f"data:image/webp;base64,{base64.b64encode(payload).decode()}"


def user_turn(text: str, image: bytes) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_url(image), "detail": "auto"}},
        ],
    }


async def main_async(args: argparse.Namespace) -> int:
    checks = Checks()
    recorder = CallRecorder(upstream=args.upstream, timeout=args.timeout)
    await recorder.start()
    token = recorder.open_rollout()
    processor = load_processor(args.model, trust_remote_code=True)
    tokenizer = load_tokenizer(args.model, trust_remote_code=True)
    media = MediaProcessor(processor)

    try:
        images = [screenshot(index) for index in range(args.turns)]
        messages: list[dict] = [{"role": "system", "content": SYSTEM_TEXT}, user_turn("Begin.", images[0])]
        sent: list[list[dict]] = []

        async with httpx.AsyncClient(timeout=args.timeout) as client:
            for turn in range(args.turns):
                sent.append(json.loads(json.dumps(messages)))
                reply = await _post(client, recorder.base_url(token), args.model_name, messages)
                messages = messages + [
                    {"role": "assistant", "content": reply},
                    user_turn(f"Observation {turn + 1}.", images[min(turn + 1, args.turns - 1)]),
                ]

            # The compactor's own call, identifiable only by the header LiteLLM.call sends.
            await _post(
                client,
                recorder.base_url(token),
                args.model_name,
                [{"role": "user", "content": "Summarise the session in one line."}],
                headers={"X-Session-ID": token},
            )

            # A trim: the first screenshot goes, which is what closes a segment.
            trimmed = [messages[0]] + [_strip_images(message) for message in messages[1:-2]] + messages[-2:]
            sent.append(json.loads(json.dumps(trimmed)))
            await _post(client, recorder.base_url(token), args.model_name, trimmed)

        record = recorder.take(token)
        _check_evidence(checks, record)
        _check_append_compatibility(checks, record)
        _check_media(checks, record, media)
        _check_render_equivalence(checks, record, processor, sent, media)
        _check_segments(checks, record, media, tokenizer)
        if args.fixture_out:
            _write_fixture(pathlib.Path(args.fixture_out), record, media)
    finally:
        await recorder.stop()

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(checks.results, handle, indent=2, default=str)
    for name, result in checks.results.items():
        print(f"{'PASS' if result['ok'] else 'FAIL'}  {name}")
    print(json.dumps({"stats": checks.results.get("samples_valid", {}).get("detail")}, default=str)[:600])
    if checks.failures:
        print("FAILED:\n  " + "\n  ".join(failure[:400] for failure in checks.failures))
        return 1
    print("All live checks passed.")
    return 0


async def _post(client, base_url: str, model: str, messages: list[dict], headers: dict | None = None) -> str:
    response = await client.post(
        f"{base_url}/chat/completions",
        json={"model": model, "messages": messages, "max_tokens": 64, "temperature": 1.0},
        headers=headers or {},
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"] or ""


def _strip_images(message: dict) -> dict:
    content = message.get("content")
    if not isinstance(content, list):
        return message
    kept = [part for part in content if part.get("type") != "image_url"]
    if not kept:
        kept = [{"type": "text", "text": "(earlier screenshot removed to save context)"}]
    return {**message, "content": kept}


def _check_evidence(checks: Checks, record) -> None:
    calls = record.calls
    checks.record("calls_recorded", bool(calls), {"calls": len(calls), "rejected": record.rejected})
    checks.record(
        "fields_present",
        all(call.prompt_token_ids and call.output_token_ids for call in calls),
        [
            {"index": call.index, "prompt": len(call.prompt_token_ids), "output": len(call.output_token_ids)}
            for call in calls
        ],
    )
    checks.record(
        "logprobs_aligned",
        all(len(call.output_logprobs) == len(call.output_token_ids) for call in calls),
        [len(call.output_logprobs) for call in calls],
    )
    compactor = [call.index for call in record.compactor_calls()]
    checks.record("compactor_classified", len(compactor) == 1, {"compactor_call_indices": compactor})
    checks.record(
        "requests_retained",
        all(call.request.get("messages") for call in record.calls),
        {"first_request_sampling": record.calls[0].request.get("sampling") if record.calls else None},
    )


def _check_append_compatibility(checks: Checks, record) -> None:
    actions = record.trained_calls()
    compatible = []
    for previous, current in zip(actions, actions[1:], strict=False):
        held = previous.prompt_token_ids + previous.output_token_ids
        compatible.append(current.prompt_token_ids[: len(held)] == held)
    checks.record(
        "append_compatible_turns",
        compatible[:-1] and all(compatible[:-1]),
        {"per_transition": compatible, "note": "the last transition is the deliberate trim"},
    )


def _check_media(checks: Checks, record, media: MediaProcessor) -> None:
    details = []
    ok = True
    for call in record.trained_calls():
        assets = media.process_all(record.arena, call.media)
        try:
            verify_occurrence_runs(call.prompt_token_ids, media.image_pad_id, assets, where=f"call {call.index}")
            details.append({"index": call.index, "runs": pad_token_runs(call.prompt_token_ids, media.image_pad_id)})
        except Exception as exc:  # noqa: BLE001
            ok = False
            details.append({"index": call.index, "error": str(exc)})
    checks.record("media_runs_match_per_occurrence", ok, details)
    checks.record("processor_fingerprint", True, media.fingerprint())


def _check_render_equivalence(checks: Checks, record, processor, sent: list[list[dict]], media) -> None:
    """The strongest check: the ids inference used, against the ids we render here."""
    details = []
    ok = True
    for call, messages in zip(record.trained_calls(), sent, strict=False):
        images = [Image.open(io.BytesIO(record.arena.get(o.asset_id))).convert("RGB") for o in call.media]
        try:
            rendered = render_prompt_ids(processor, messages, images)
        except Exception as exc:  # noqa: BLE001
            ok = False
            details.append({"index": call.index, "error": f"local render failed: {exc}"})
            continue
        index = first_difference(rendered, list(call.prompt_token_ids))
        equal = index is None and len(rendered) == len(call.prompt_token_ids)
        ok = ok and equal
        details.append(
            {
                "index": call.index,
                "equal": equal,
                "engine_len": len(call.prompt_token_ids),
                "rendered_len": len(rendered),
                "first_difference": index,
            }
        )
    checks.record("effective_ids_equal_local_render", ok, details)


def _check_segments(checks: Checks, record, media, tokenizer) -> None:
    segmentation = segment_calls(record.trained_calls())
    samples, stats = build_samples(
        template=Sample(index=0),
        record=record,
        segmentation=segmentation,
        media=media,
        reward=1.0,
        rollout_id=0,
        tokenizer=tokenizer,
    )
    checks.record(
        "trim_closes_a_segment",
        len(segmentation.segments) >= 2,
        {"segments": [s.call_indices for s in segmentation.segments], "boundaries": segmentation.boundaries},
    )
    checks.record(
        "samples_valid",
        all(
            len(s.loss_mask) == s.response_length and len(s.rollout_log_probs) == s.response_length for s in samples
        ),
        {"samples": len(samples), "stats": stats.as_metadata()},
    )
    for sample in samples:
        sample.validate()


def _write_fixture(root: pathlib.Path, record, media) -> None:
    """A captured call and its media, for the session-server track to validate against.

    The request keeps its asset ids rather than base64, and the bytes go beside it, so
    a consumer can process the same pixels with the same processor and compare against
    the engine's own effective ids.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "media").mkdir(exist_ok=True)
    for asset_id, payload in record.arena.assets.items():
        (root / "media" / f"{asset_id.replace(':', '_')}.webp").write_bytes(payload)
    calls = []
    for call in record.trained_calls():
        assets = media.process_all(record.arena, call.media)
        calls.append(
            {
                "index": call.index,
                "request": call.request,
                "media": [
                    {"asset_id": occurrence.asset_id, "media_type": occurrence.media_type, "tokens": asset.token_count}
                    for occurrence, asset in zip(call.media, assets, strict=True)
                ],
                "prompt_token_ids": list(call.prompt_token_ids),
                "output_token_ids": list(call.output_token_ids),
                "output_logprobs": list(call.output_logprobs),
                "finish_reason": call.finish_reason,
                "meta_info": {k: v for k, v in call.meta_info.items() if k != "output_token_logprobs"},
            }
        )
    (root / "calls.json").write_text(
        json.dumps({"processor_fingerprint": media.fingerprint(), "calls": calls}, indent=2, default=str)
    )
    print(f"fixture written to {root}: {len(calls)} calls, {len(record.arena.assets)} images")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True, help="local checkpoint path, for the processor and tokenizer")
    parser.add_argument("--model-name", default=None, help="the name the engine serves, if it differs from --model")
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--json-out", default=None, help="write the full check record here")
    parser.add_argument("--fixture-out", default=None, help="write a captured call and its media here")
    args = parser.parse_args()
    args.model_name = args.model_name or args.model
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
