"""The recording proxy: what it asks the engine for, and what it keeps."""

import base64
import json

import httpx
import pytest

from examples.experimental.harbor_cua.recorder import CallRecorder, parse_media
from tests.fast.examples.experimental.harbor_cua.conftest import png_bytes

def data_url(payload: bytes, media_type: str = "image/webp") -> str:
    return f"data:{media_type};base64,{base64.b64encode(payload).decode()}"


def chat_response(*, prompt_ids, output_ids, logprobs=None):
    triples = [(logprob, token, "") for logprob, token in zip(logprobs or [-0.5] * len(output_ids), output_ids, strict=True)]
    return {
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "{}"},
                "finish_reason": "stop",
                "prompt_token_ids": list(prompt_ids),
                "response_token_ids": list(output_ids),
                "meta_info": {"output_token_logprobs": triples, "weight_version": "3"},
            }
        ],
    }


def canned(recorder, response, *, status=200, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(status, json=response)

    recorder._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://upstream")
    return recorder


def client(recorder):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=recorder._app), base_url="http://recorder")


def messages_with(*images):
    content = [{"type": "text", "text": "act"}]
    content += [{"type": "image_url", "image_url": {"url": data_url(image)}} for image in images]
    return [{"role": "user", "content": content}]


async def post_chat(recorder, token, messages, headers=None, extra=None):
    async with client(recorder) as http:
        return await http.post(
            f"/r/{token}/v1/chat/completions",
            json={"model": "m", "messages": messages, **(extra or {})},
            headers=headers or {},
        )


@pytest.fixture
def recorder():
    return CallRecorder(upstream="http://upstream")


async def test_the_trace_flags_and_logprobs_are_added_outbound(recorder):
    seen = []
    canned(recorder, chat_response(prompt_ids=[1, 2], output_ids=[3]), seen=seen)
    token = recorder.open_rollout()

    await post_chat(recorder, token, messages_with(png_bytes((1, 1, 1))))

    assert seen[0]["return_prompt_token_ids"] is True
    assert seen[0]["return_token_ids"] is True
    assert seen[0]["return_meta_info"] is True
    assert seen[0]["logprobs"] is True


async def test_a_call_is_kept_with_its_tokens_logprobs_and_media(recorder):
    canned(recorder, chat_response(prompt_ids=[1, 2, 3], output_ids=[7, 8], logprobs=[-0.1, -0.2]))
    token = recorder.open_rollout()
    first, second = png_bytes((1, 0, 0)), png_bytes((2, 0, 0))

    await post_chat(recorder, token, messages_with(first, second))
    record = recorder.take(token)

    assert len(record.calls) == 1
    call = record.calls[0]
    assert call.prompt_token_ids == (1, 2, 3)
    assert call.output_token_ids == (7, 8)
    assert call.output_logprobs == (-0.1, -0.2)
    assert [record.arena.get(occurrence.asset_id) for occurrence in call.media] == [first, second]
    assert call.finish_reason == "stop"
    assert call.meta_info["weight_version"] == "3"
    assert record.rejected == []


async def test_repeated_images_dedupe_as_assets_but_stay_distinct_occurrences(recorder):
    canned(recorder, chat_response(prompt_ids=[1], output_ids=[2]))
    token = recorder.open_rollout()
    same = png_bytes((4, 4, 4))

    await post_chat(recorder, token, messages_with(same, same))
    record = recorder.take(token)

    assert len(record.calls[0].media) == 2
    assert len(record.arena.assets) == 1


async def test_meta_info_is_not_forwarded_but_token_ids_are(recorder):
    canned(recorder, chat_response(prompt_ids=[1], output_ids=[2]))
    token = recorder.open_rollout()

    response = await post_chat(recorder, token, messages_with(png_bytes((3, 3, 3))))

    choice = response.json()["choices"][0]
    assert "meta_info" not in choice
    assert choice["prompt_token_ids"] == [1]
    assert choice["response_token_ids"] == [2]


async def test_a_session_id_header_marks_the_call_auxiliary(recorder):
    canned(recorder, chat_response(prompt_ids=[1], output_ids=[2]))
    token = recorder.open_rollout()

    await post_chat(recorder, token, messages_with(png_bytes((5, 5, 5))), headers={"X-Session-ID": "abc"})
    record = recorder.take(token)

    assert record.calls[0].auxiliary is True
    assert record.action_calls() == []
    assert len(record.auxiliary_calls()) == 1


async def test_a_call_without_prompt_token_ids_is_refused_not_guessed(recorder):
    response = chat_response(prompt_ids=[1], output_ids=[2])
    response["choices"][0].pop("prompt_token_ids")
    canned(recorder, response)
    token = recorder.open_rollout()

    await post_chat(recorder, token, messages_with(png_bytes((6, 6, 6))))
    record = recorder.take(token)

    assert record.calls == []
    assert "return_prompt_token_ids" in record.rejected[0]


async def test_a_remote_image_url_is_refused_rather_than_fetched(recorder):
    canned(recorder, chat_response(prompt_ids=[1], output_ids=[2]))
    token = recorder.open_rollout()
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.webp"}}]}]

    await post_chat(recorder, token, messages)
    record = recorder.take(token)

    assert record.calls == []
    assert "not an inline data URL" in record.rejected[0]


async def test_streaming_is_refused(recorder):
    canned(recorder, chat_response(prompt_ids=[1], output_ids=[2]))
    token = recorder.open_rollout()

    response = await post_chat(recorder, token, messages_with(png_bytes((7, 7, 7))), extra={"stream": True})

    assert response.status_code == 400


async def test_an_unknown_rollout_token_is_rejected(recorder):
    canned(recorder, chat_response(prompt_ids=[1], output_ids=[2]))

    response = await post_chat(recorder, "nope", messages_with(png_bytes((8, 8, 8))))

    assert response.status_code == 404


async def test_an_engine_error_is_passed_through_and_noted(recorder):
    canned(recorder, {"error": {"message": "boom"}}, status=500)
    token = recorder.open_rollout()

    response = await post_chat(recorder, token, messages_with(png_bytes((9, 9, 9))))
    record = recorder.take(token)

    assert response.status_code == 500
    assert record.calls == []
    assert "engine returned 500" in record.rejected[0]


async def test_calls_are_numbered_in_the_order_they_arrive(recorder):
    canned(recorder, chat_response(prompt_ids=[1], output_ids=[2]))
    token = recorder.open_rollout()

    for _ in range(3):
        await post_chat(recorder, token, messages_with(png_bytes((1, 2, 3))))
    record = recorder.take(token)

    assert [call.index for call in record.calls] == [0, 1, 2]


def test_media_parsing_keeps_occurrence_order_across_messages():
    first, second = png_bytes((1, 1, 1)), png_bytes((2, 2, 2))
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(first)}}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(second)}}]},
    ]

    media, problems = parse_media(messages)

    assert [payload for payload, _ in media] == [first, second]
    assert problems == []
