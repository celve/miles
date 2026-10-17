"""End-to-end image trajectories through session server v2.

The mock backend stands in for SGLang's native ``/generate``: it expands the
pre-tokenized placeholders with the engine's **own** helper
(``BaseMultimodalProcessor._expand_input_ids``) and reports the result as
``prompt_token_ids``, so a disagreement between the session server's expansion
and the engine's fails these tests rather than silently training on the wrong
ids.
"""

from __future__ import annotations

import base64
import io
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import requests
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from tests.ci.ci_register import register_cpu_ci
from tests.fast.router.test_sessions import _create_session, _post_chat

from miles.rollout.session.config import compute_session_server_config
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_and_merge_input_sample
from miles.rollout.session.server import SessionServer
from miles.utils.http_utils import find_available_port
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

# 105.0 s measured for the whole file on a devbox, not on a CI runner; each case
# starts its own server because the flags under test are launch-scoped.
register_cpu_ci(est_time=140, suite="stage-b-cpu", labels=["rollout"])

CHECKPOINT = "Qwen/Qwen3-VL-2B-Instruct"
IMAGE_TOKEN = "<|image_pad|>"


@pytest.fixture(scope="module")
def vlm():
    """One processor and tokenizer for the whole module; loading them is the slow part."""
    processor = load_processor(CHECKPOINT, trust_remote_code=True)
    if processor is None or not hasattr(processor, "_get_num_multimodal_tokens"):
        pytest.skip(f"{CHECKPOINT} processor unavailable")
    tokenizer = load_tokenizer(CHECKPOINT, trust_remote_code=True)
    return SimpleNamespace(
        processor=processor,
        tokenizer=tokenizer,
        pad_id=tokenizer.convert_tokens_to_ids(IMAGE_TOKEN),
        merge=getattr(processor.image_processor, "merge_size", 2),
    )


def png(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def image_part(payload: bytes) -> dict:
    url = f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}"
    return {"type": "image_url", "image_url": {"url": url}}


def user_turn(payload: bytes | None, text: str) -> dict:
    if payload is None:
        return {"role": "user", "content": [{"type": "text", "text": text}]}
    return {"role": "user", "content": [image_part(payload), {"type": "text", "text": text}]}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


class ScriptedGenerateBackend:
    """A mock router in front of SGLang, serving ``/generate`` the way the image path uses it."""

    def __init__(self, vlm, replies: list[str], *, count_bias: int = 0, finish_reason: str = "stop"):
        self.vlm = vlm
        self.replies = replies
        # A nonzero bias makes the engine expand differently from the server, which
        # is the failure the exactness gate exists to catch.
        self.count_bias = count_bias
        self.finish_reason = finish_reason
        self.calls: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.routing_keys: list[str | None] = []
        # Every endpoint a request-time engine lookup would read; none may be hit.
        self.metadata_calls: list[str] = []
        self.host = "127.0.0.1"
        self.port = find_available_port(32000)
        self.app = FastAPI()
        self._server = UvicornThreadServer(self.app, host=self.host, port=self.port)

        @self.app.get("/health")
        async def health():
            return JSONResponse(content={"status": "ok"})

        @self.app.get("/get_server_info")
        @self.app.get("/list_workers")
        @self.app.get("/workers")
        async def metadata(request: Request):
            self.metadata_calls.append(request.url.path)
            return JSONResponse(content={"detail": "not expected"}, status_code=500)

        @self.app.post("/generate")
        async def generate(request: Request):
            payload = self._record(request, await request.json())
            return JSONResponse(content=self._reply(payload))

        @self.app.post("/v1/chat/completions")
        async def chat(request: Request):
            payload = self._record(request, await request.json())
            return JSONResponse(content=self._chat_reply(payload))

    def _record(self, request: Request, payload: dict) -> dict:
        self.calls.append(payload)
        self.paths.append(request.url.path)
        self.routing_keys.append(request.headers.get("X-SMG-Routing-Key"))
        return payload

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self):
        self._server.start()

    def stop(self):
        self._server.stop()

    def _counts(self, image_data: list[str]) -> list[int]:
        sizes = []
        for url in image_data:
            raw = base64.b64decode(url.split(",", 1)[1])
            with Image.open(io.BytesIO(raw)) as handle:
                sizes.append((handle.height, handle.width))
        counts = self.vlm.processor._get_num_multimodal_tokens(image_sizes=sizes).num_image_tokens
        return [int(count) + self.count_bias for count in counts]

    def _output_ids(self, index: int) -> list[int]:
        text = self.replies[min(index, len(self.replies) - 1)]
        return self.vlm.tokenizer.encode(text, add_special_tokens=False)

    def _reply(self, payload: dict) -> dict:
        from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

        index = len(self.calls) - 1
        image_data = payload.get("image_data") or []
        effective = BaseMultimodalProcessor._expand_input_ids(
            payload["input_ids"], self._counts(image_data), self.vlm.pad_id
        )
        output_ids = self._output_ids(index)
        return {
            "text": self.vlm.tokenizer.decode(output_ids),
            "prompt_token_ids": effective,
            "meta_info": {
                "id": f"gen-{index}",
                "finish_reason": {"type": self.finish_reason, "matched": None},
                "prompt_tokens": len(effective),
                "completion_tokens": len(output_ids),
                "cached_tokens": 0,
                "weight_version": "w0",
                "output_token_logprobs": [[-i / 128, tid, None] for i, tid in enumerate(output_ids)],
            },
        }

    def _chat_reply(self, payload: dict) -> dict:
        """The text route, for a text-only turn inside an image session."""
        index = len(self.calls) - 1
        output_ids = self._output_ids(index)
        return {
            "id": f"chat-{index}",
            "object": "chat.completion",
            "created": 0,
            "model": "scripted",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.vlm.tokenizer.decode(output_ids)},
                    "finish_reason": "stop",
                    "meta_info": {
                        "id": f"chat-{index}",
                        "completion_tokens": len(output_ids),
                        "weight_version": "w0",
                        "output_token_logprobs": [[-i / 128, tid, None] for i, tid in enumerate(output_ids)],
                    },
                }
            ],
        }


# What `resolve_session_image_engine` reads from SGLang 0.5.22's own launch defaults.
ENGINE_SETTINGS = {
    "model_path": CHECKPOINT,
    "revision": None,
    "mm_process_config": None,
    "image_processor_backend": "auto",
    "sampling_defaults": "model",
    "tool_call_parser": None,
    "reasoning_parser": None,
}


def image_server_args(engine: dict | None = None, **extra) -> dict[str, Any]:
    return {
        "miles_router_timeout": 30,
        "hf_checkpoint": CHECKPOINT,
        "chat_template_path": None,
        "apply_chat_template_kwargs": None,
        "tito_model": "default",
        "use_rollout_routing_replay": False,
        "use_rollout_indexer_replay": False,
        "use_sampling_support_replay": False,
        "rollout_top_logprobs_num": 0,
        "rollout_sampling_logprobs_mode": "selected",
        "sglang_speculative_algorithm": None,
        "num_layers": None,
        "moe_router_topk": None,
        "save_debug_trajectory_data": None,
        "lora_rank": 0,
        "lora_adapter_path": None,
        "use_session_server": "v2",
        "session_image_support": "qwen3_vl",
        "session_image_engine": {**ENGINE_SETTINGS, **(engine or {})},
        "session_image_text_only_leaf": "drop",
        "session_server_instance_id": uuid.uuid4().hex,
        "pause_generation_mode": "retract",
        "session_sample_picker_path": "miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
        "session_sample_postprocessor_path": "miles.rollout.session.v2.postprocessor_hub.default_postprocess",
        **extra,
    }


def build_server(args_values: dict[str, Any], *, backend_url: str) -> tuple[SessionServer, int]:
    port = find_available_port(31000)
    config = compute_session_server_config(
        SimpleNamespace(**args_values),
        host="127.0.0.1",
        port=port,
        instance_id=args_values["session_server_instance_id"],
        backend_url=backend_url,
    )
    return SessionServer(config), port


@contextmanager
def image_router(
    vlm,
    replies: list[str],
    *,
    count_bias: int = 0,
    extra_args: dict | None = None,
    engine: dict | None = None,
    finish_reason: str = "stop",
):
    backend = ScriptedGenerateBackend(vlm, replies, count_bias=count_bias, finish_reason=finish_reason)
    backend.start()
    try:
        server_obj, port = build_server(image_server_args(engine, **(extra_args or {})), backend_url=backend.url)
        server = UvicornThreadServer(server_obj.app, host="127.0.0.1", port=port)
        server.start()
        try:
            yield SimpleNamespace(url=f"http://127.0.0.1:{port}", backend=backend)
        finally:
            server.stop()
    finally:
        backend.stop()


def collect(url: str, session_id: str, body: dict | None = None) -> list[Sample]:
    response = requests.post(f"{url}/sessions/{session_id}/samples", json=body or {}, timeout=20.0)
    assert response.status_code == 200, response.content
    return decode_samples_and_merge_input_sample(response.content, Sample(), fields=COMPUTED_FIELDS_V2).samples


def chat(env, session_id: str, messages: list[dict]) -> requests.Response:
    return _post_chat(env.url, session_id, {"messages": messages, "max_tokens": 16, "temperature": 1.0})


def image_token_runs(tokens: list[int], pad_id: int) -> list[int]:
    runs, run = [], 0
    for token in tokens:
        if token == pad_id:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return runs


def expected_counts(vlm, payloads: list[bytes]) -> list[int]:
    sizes = []
    for payload in payloads:
        with Image.open(io.BytesIO(payload)) as handle:
            sizes.append((handle.height, handle.width))
    return [int(c) for c in vlm.processor._get_num_multimodal_tokens(image_sizes=sizes).num_image_tokens]


SHOT1 = png(120, 80, (10, 20, 30))
SHOT2 = png(160, 120, (40, 80, 120))
# Same processed size as SHOT1, different pixels: identical placeholder tokens.
SHOT1_TWIN = png(120, 80, (200, 190, 180))


class TestAppend:
    def test_two_image_turns_make_one_sample_carrying_both_images(self, vlm):
        with image_router(vlm, ["clicked", "done"]) as env:
            session_id = _create_session(env.url)
            first = [user_turn(SHOT1, "step 1")]
            assert chat(env, session_id, first).status_code == 200
            second = first + [assistant("clicked"), user_turn(SHOT2, "step 2")]
            assert chat(env, session_id, second).status_code == 200

            samples = collect(env.url, session_id)
            assert len(samples) == 1
            sample = samples[0]
            counts = expected_counts(vlm, [SHOT1, SHOT2])
            assert image_token_runs(sample.tokens, vlm.pad_id) == counts
            assert sample.multimodal_train_inputs["image_grid_thw"].shape == (2, 3)
            assert len(sample.loss_mask) == sample.response_length
            assert len(sample.rollout_log_probs) == sample.response_length
            assert len(sample.tokens) >= sample.response_length

    def test_the_engine_is_sent_the_unexpanded_render_and_every_image(self, vlm):
        with image_router(vlm, ["clicked"]) as env:
            session_id = _create_session(env.url)
            assert chat(env, session_id, [user_turn(SHOT1, "step 1")]).status_code == 200
            call = env.backend.calls[0]
            assert image_token_runs(call["input_ids"], vlm.pad_id) == [1]
            assert len(call["image_data"]) == 1
            assert call["mm_content_hashes"][0].startswith("sha256:")
            assert call["return_prompt_token_ids"] is True

    def test_pixels_follow_occurrence_order(self, vlm):
        with image_router(vlm, ["a", "b"]) as env:
            session_id = _create_session(env.url)
            first = [user_turn(SHOT1, "one")]
            chat(env, session_id, first)
            chat(env, session_id, first + [assistant("a"), user_turn(SHOT2, "two")])
            sample = collect(env.url, session_id)[0]
            grid = sample.multimodal_train_inputs["image_grid_thw"]
            own = expected_counts(vlm, [SHOT1, SHOT2])
            per_image = (torch.prod(grid, dim=-1) // (vlm.merge**2)).tolist()
            assert per_image == own


class TestRepeatedAndReordered:
    def test_the_same_screenshot_twice_is_two_occurrences(self, vlm):
        with image_router(vlm, ["ok"]) as env:
            session_id = _create_session(env.url)
            messages = [{"role": "user", "content": [image_part(SHOT1), image_part(SHOT1)]}]
            assert chat(env, session_id, messages).status_code == 200
            sample = collect(env.url, session_id)[0]
            counts = expected_counts(vlm, [SHOT1, SHOT1])
            assert image_token_runs(sample.tokens, vlm.pad_id) == counts
            assert sample.multimodal_train_inputs["image_grid_thw"].shape == (2, 3)

    def test_reordering_images_is_a_different_prompt(self, vlm):
        with image_router(vlm, ["ok", "ok"]) as env:
            session_id = _create_session(env.url)
            forward = [{"role": "user", "content": [image_part(SHOT1), image_part(SHOT2)]}]
            reverse = [{"role": "user", "content": [image_part(SHOT2), image_part(SHOT1)]}]
            assert chat(env, session_id, forward).status_code == 200
            assert chat(env, session_id, reverse).status_code == 200
            samples = collect(env.url, session_id, {})
            # Two roots with different prompts both survive the retry picker.
            assert len(samples) == 2


class TestRewriteAndReplacement:
    def test_dropping_an_old_screenshot_opens_a_second_sample(self, vlm):
        with image_router(vlm, ["a", "b", "c"]) as env:
            session_id = _create_session(env.url)
            first = [user_turn(SHOT1, "one")]
            chat(env, session_id, first)
            second = first + [assistant("a"), user_turn(SHOT2, "two")]
            chat(env, session_id, second)
            # The harness drops shot1 and replays its action as text: no stored path
            # is a prefix of this, so it becomes a new root.
            rewritten = [
                user_turn(None, "one"),
                assistant("a"),
                user_turn(SHOT2, "two"),
                assistant("b"),
                user_turn(SHOT1_TWIN, "three"),
            ]
            assert chat(env, session_id, rewritten).status_code == 200

            samples = collect(env.url, session_id)
            assert len(samples) == 2
            by_images = sorted(s.multimodal_train_inputs["image_grid_thw"].shape[0] for s in samples)
            assert by_images == [2, 2]

    def test_a_same_grid_different_image_sibling_is_not_a_retry(self, vlm):
        """Identical placeholder tokens, different pixels: the earlier leaf must survive."""
        with image_router(vlm, ["a", "b"]) as env:
            session_id = _create_session(env.url)
            assert chat(env, session_id, [user_turn(SHOT1, "look")]).status_code == 200
            assert chat(env, session_id, [user_turn(SHOT1_TWIN, "look")]).status_code == 200
            first, second = env.backend.calls[0], env.backend.calls[1]
            assert first["input_ids"] == second["input_ids"], "this case only bites when the ids match"
            assert first["mm_content_hashes"] != second["mm_content_hashes"]
            assert len(collect(env.url, session_id)) == 2


class TestRetry:
    def test_an_identical_resend_is_dropped(self, vlm):
        with image_router(vlm, ["a", "a"]) as env:
            session_id = _create_session(env.url)
            messages = [user_turn(SHOT1, "look")]
            assert chat(env, session_id, messages).status_code == 200
            assert chat(env, session_id, messages).status_code == 200
            assert len(collect(env.url, session_id)) == 1


class TestTruncation:
    def test_a_cut_inside_an_image_drops_that_image(self, vlm):
        with image_router(vlm, ["a", "b"]) as env:
            session_id = _create_session(env.url)
            first = [user_turn(SHOT1, "one")]
            chat(env, session_id, first)
            chat(env, session_id, first + [assistant("a"), user_turn(SHOT2, "two")])
            whole = collect(env.url, session_id)[0]
            assert whole.multimodal_train_inputs["image_grid_thw"].shape[0] == 2

            # A budget that lands inside the second image's placeholder run.
            counts = expected_counts(vlm, [SHOT1, SHOT2])
            budget = len(whole.tokens) - counts[1] // 2
            trimmed = collect(env.url, session_id, {"max_seq_len": budget})
            assert len(trimmed) == 1
            assert trimmed[0].multimodal_train_inputs["image_grid_thw"].shape[0] == 1


class TestMixedTextOnly:
    def test_a_text_only_leaf_is_dropped_by_default(self, vlm):
        with image_router(vlm, ["a", "summary"]) as env:
            session_id = _create_session(env.url)
            assert chat(env, session_id, [user_turn(SHOT1, "look")]).status_code == 200
            assert chat(env, session_id, [user_turn(None, "summarize")]).status_code == 200
            samples = collect(env.url, session_id)
            assert len(samples) == 1
            assert samples[0].multimodal_train_inputs is not None

    def test_keep_emits_it_without_a_placeholder_image(self, vlm):
        with image_router(vlm, ["a", "summary"], extra_args={"session_image_text_only_leaf": "keep"}) as env:
            session_id = _create_session(env.url)
            assert chat(env, session_id, [user_turn(SHOT1, "look")]).status_code == 200
            assert chat(env, session_id, [user_turn(None, "summarize")]).status_code == 200
            samples = collect(env.url, session_id)
            assert len(samples) == 2
            text_only = [s for s in samples if s.multimodal_train_inputs is None]
            assert len(text_only) == 1
            assert vlm.pad_id not in text_only[0].tokens

    def test_a_text_only_turn_uses_the_chat_route(self, vlm):
        with image_router(vlm, ["a"]) as env:
            session_id = _create_session(env.url)
            assert chat(env, session_id, [user_turn(None, "no pictures")]).status_code == 200
            assert "image_data" not in env.backend.calls[0]
            assert "messages" in env.backend.calls[0]


class TestExactnessGate:
    def test_an_engine_that_expands_differently_fails_the_request(self, vlm):
        with image_router(vlm, ["a"], count_bias=1) as env:
            session_id = _create_session(env.url)
            response = chat(env, session_id, [user_turn(SHOT1, "look")])
            assert response.status_code >= 400
            assert b"effective prompt ids differ" in response.content


class TestRefusals:
    def test_an_image_is_refused_when_support_is_off(self, vlm):
        with image_router(vlm, ["a"], extra_args={"session_image_support": "off"}) as env:
            session_id = _create_session(env.url)
            response = chat(env, session_id, [user_turn(SHOT1, "look")])
            assert response.status_code == 400
            assert b"--session-image-support" in response.content

    def test_a_remote_image_url_is_refused(self, vlm):
        with image_router(vlm, ["a"]) as env:
            session_id = _create_session(env.url)
            messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/a.png"}}]}]
            response = chat(env, session_id, messages)
            assert response.status_code == 400
            assert b"inline base64 data URL" in response.content

    def test_n_above_one_is_refused_on_an_image_turn(self, vlm):
        with image_router(vlm, ["a"]) as env:
            session_id = _create_session(env.url)
            response = _post_chat(
                env.url, session_id, {"messages": [user_turn(SHOT1, "look")], "max_tokens": 8, "n": 2}
            )
            assert response.status_code == 400
            assert b"is not supported for an image-bearing request" in response.content


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "click a point",
            "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
        },
    }
]


class TestEngineSettings:
    """Matching ids prove the grid agrees, not that the pixels behind it do.

    So the engines' own image settings decide whether the server may start. They come
    from the args Miles launches the engines with, so nothing asks an engine anything.
    """

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            ({"mm_process_config": {"image": {"do_normalize": False}}}, "mm_process_config"),
            ({"model_path": "Qwen/Qwen3-VL-4B-Instruct"}, "the two processors would disagree"),
            ({"image_processor_backend": "torchvision"}, "image_processor_backend"),
            ({"revision": "abc123"}, "pin revision"),
            ({"tool_call_parser": "auto"}, "name the parser explicitly"),
        ],
        ids=["preprocessing-override", "other-checkpoint", "image-backend", "pinned-revision", "auto-parser"],
    )
    def test_an_unsupported_engine_setting_refuses_to_start(self, vlm, override, message):
        with pytest.raises(ValueError, match=message):
            build_server(image_server_args(override), backend_url="http://127.0.0.1:9")

    def test_settings_that_state_nothing_refuse_to_start(self, vlm):
        with pytest.raises(ValueError, match="needs the policy engines' image settings"):
            build_server(image_server_args(session_image_engine=None), backend_url="http://127.0.0.1:9")

    def test_the_server_starts_with_nothing_behind_it(self, vlm):
        """Miles starts this server before any engine, so building it must contact nothing."""
        server_obj, _ = build_server(image_server_args(), backend_url="http://127.0.0.1:9")
        assert server_obj.app is not None

    def test_what_bound_the_pixels_is_recorded_on_the_sample(self, vlm):
        with image_router(vlm, ["a"]) as env:
            session_id = _create_session(env.url)
            assert chat(env, session_id, [user_turn(SHOT1, "look")]).status_code == 200
            provenance = collect(env.url, session_id)[0].metadata["image_provenance"]
            assert provenance["processor"]["checkpoint"] == CHECKPOINT
            assert provenance["processor"]["image_token_id"] == vlm.pad_id
            assert provenance["engine"] == ENGINE_SETTINGS


class TestRouting:
    """An image turn reaches the engines the way a text turn does: through the router."""

    def test_an_image_turn_posts_generate_to_the_router(self, vlm):
        with image_router(vlm, ["a"]) as env:
            session_id = _create_session(env.url)
            assert chat(env, session_id, [user_turn(SHOT1, "look")]).status_code == 200
            assert env.backend.paths == ["/generate"]
            assert env.backend.routing_keys == [session_id]

    def test_the_first_image_turn_asks_no_engine_about_itself(self, vlm):
        with image_router(vlm, ["a", "b", "c"]) as env:
            session_id = _create_session(env.url)
            first = [user_turn(SHOT1, "look")]
            assert chat(env, session_id, first).status_code == 200
            second = [*first, assistant("a"), user_turn(None, "and now?")]
            assert chat(env, session_id, second).status_code == 200
            assert env.backend.paths == ["/generate", "/generate"]
            assert env.backend.metadata_calls == []


class TestSamplingParameters:
    """The engine's own converter owns the mapping; these pin the cases that were wrong."""

    @staticmethod
    def params(env) -> dict:
        return env.backend.calls[0]["sampling_params"]

    def test_max_completion_tokens_beats_max_tokens(self, vlm):
        with image_router(vlm, ["a"]) as env:
            session_id = _create_session(env.url)
            _post_chat(
                env.url,
                session_id,
                {
                    "messages": [user_turn(SHOT1, "look")],
                    "max_tokens": 128,
                    "max_completion_tokens": 32,
                    "temperature": 1.0,
                },
            )
            assert self.params(env)["max_new_tokens"] == 32

    def test_seed_becomes_sampling_seed(self, vlm):
        with image_router(vlm, ["a"]) as env:
            session_id = _create_session(env.url)
            _post_chat(
                env.url,
                session_id,
                {"messages": [user_turn(SHOT1, "look")], "max_tokens": 8, "temperature": 1.0, "seed": 42},
            )
            params = self.params(env)
            assert params["sampling_seed"] == 42
            assert "seed" not in params

    def test_controls_the_chat_route_honours_are_carried(self, vlm):
        with image_router(vlm, ["a"]) as env:
            session_id = _create_session(env.url)
            _post_chat(
                env.url,
                session_id,
                {
                    "messages": [user_turn(SHOT1, "look")],
                    "max_tokens": 8,
                    "temperature": 1.0,
                    "min_tokens": 2,
                    "ignore_eos": True,
                    "stop_regex": "STOP",
                },
            )
            params = self.params(env)
            assert params["min_new_tokens"] == 2
            assert params["ignore_eos"] is True
            assert params["stop_regex"] == "STOP"


class TestToolCallsAndFinishReason:
    def test_tools_without_an_engine_parser_are_refused(self, vlm):
        """Otherwise an image turn returns raw tool syntax where a text turn returns tool_calls."""
        with image_router(vlm, ["a"]) as env:
            session_id = _create_session(env.url)
            response = _post_chat(
                env.url, session_id, {"messages": [user_turn(SHOT1, "look")], "max_tokens": 8, "tools": TOOLS}
            )
            assert response.status_code == 400
            assert b"no --tool-call-parser" in response.content

    def test_a_parsed_call_is_served_when_the_engine_has_a_parser(self, vlm):
        reply = '<tool_call>\n{"name": "click", "arguments": {"x": 3}}\n</tool_call>'
        with image_router(vlm, [reply], engine={"tool_call_parser": "qwen25"}) as env:
            session_id = _create_session(env.url)
            response = _post_chat(
                env.url, session_id, {"messages": [user_turn(SHOT1, "look")], "max_tokens": 64, "tools": TOOLS}
            )
            assert response.status_code == 200, response.content
            choice = response.json()["choices"][0]
            assert choice["finish_reason"] == "tool_calls"
            calls = choice["message"]["tool_calls"]
            assert [call["function"]["name"] for call in calls] == ["click"]
            assert calls[0]["id"].startswith("call_") and len(calls[0]["id"]) > 10

    def test_tool_choice_none_leaves_the_text_alone(self, vlm):
        reply = '<tool_call>\n{"name": "click", "arguments": {"x": 3}}\n</tool_call>'
        with image_router(vlm, [reply], engine={"tool_call_parser": "qwen25"}) as env:
            session_id = _create_session(env.url)
            response = _post_chat(
                env.url,
                session_id,
                {
                    "messages": [user_turn(SHOT1, "look")],
                    "max_tokens": 64,
                    "tools": TOOLS,
                    "tool_choice": "none",
                },
            )
            assert response.status_code == 200, response.content
            choice = response.json()["choices"][0]
            assert choice["message"].get("tool_calls") in (None, [])
            assert choice["finish_reason"] == "stop"

    def test_a_length_cut_is_not_relabelled_as_tool_calls(self, vlm):
        """Relabelling would clear TrajectoryNode.truncated and let the tree extend a closed path."""
        reply = '<tool_call>\n{"name": "click", "arguments": {"x": 3}}\n</tool_call>'
        with image_router(vlm, [reply], engine={"tool_call_parser": "qwen25"}, finish_reason="length") as env:
            session_id = _create_session(env.url)
            response = _post_chat(
                env.url, session_id, {"messages": [user_turn(SHOT1, "look")], "max_tokens": 4, "tools": TOOLS}
            )
            assert response.status_code == 200, response.content
            assert response.json()["choices"][0]["finish_reason"] == "length"
            tree = requests.get(f"{env.url}/sessions/{session_id}", timeout=10).json()["metadata"]["tree"]
            assert tree["nodes"][0]["truncated"] is True


class TestClientLogprobs:
    def test_a_client_that_asks_for_logprobs_gets_the_chat_shape(self, vlm):
        with image_router(vlm, ["clicked"]) as env:
            session_id = _create_session(env.url)
            response = _post_chat(
                env.url,
                session_id,
                {"messages": [user_turn(SHOT1, "look")], "max_tokens": 16, "temperature": 1.0, "logprobs": True},
            )
            assert response.status_code == 200, response.content
            content = response.json()["choices"][0]["logprobs"]["content"]
            assert content, "logprobs content must not be empty"
            assert {"token", "logprob", "bytes", "top_logprobs"} <= set(content[0])
            assert env.backend.calls[0]["return_text_in_logprobs"] is True

    def test_a_client_that_does_not_ask_gets_none(self, vlm):
        with image_router(vlm, ["clicked"]) as env:
            session_id = _create_session(env.url)
            response = chat(env, session_id, [user_turn(SHOT1, "look")])
            assert response.status_code == 200
            assert "logprobs" not in response.json()["choices"][0]
            assert "return_text_in_logprobs" not in env.backend.calls[0]


class TestModelSamplingDefaults:
    """An omitted field must take the same default on both of a session's routes."""

    def test_omitted_fields_take_the_checkpoints_generation_config(self, vlm):
        with image_router(vlm, ["a"]) as env:
            session_id = _create_session(env.url)
            # No temperature/top_p/top_k: the engine declares sampling_defaults="model",
            # so the checkpoint's generation config supplies them, as on the chat route.
            response = _post_chat(env.url, session_id, {"messages": [user_turn(SHOT1, "look")], "max_tokens": 8})
            assert response.status_code == 200, response.content
            params = env.backend.calls[-1]["sampling_params"]
            assert (params["temperature"], params["top_p"], params["top_k"]) == (0.7, 0.8, 20)

    def test_an_engine_that_does_not_use_model_defaults_gets_none(self, vlm):
        with image_router(vlm, ["a"], engine={"sampling_defaults": "openai"}) as env:
            session_id = _create_session(env.url)
            response = _post_chat(env.url, session_id, {"messages": [user_turn(SHOT1, "look")], "max_tokens": 8})
            assert response.status_code == 200, response.content
            params = env.backend.calls[-1]["sampling_params"]
            assert params["temperature"] == 1.0 and params["top_p"] == 1.0

    def test_an_explicit_value_still_wins(self, vlm):
        """`repetition_penalty`, because narrowing top_p trips the session's own sampling guard."""
        with image_router(vlm, ["a"]) as env:
            session_id = _create_session(env.url)
            response = _post_chat(
                env.url,
                session_id,
                {
                    "messages": [user_turn(SHOT1, "look")],
                    "max_tokens": 8,
                    "temperature": 1.0,
                    "repetition_penalty": 1.3,
                },
            )
            assert response.status_code == 200, response.content
            params = env.backend.calls[-1]["sampling_params"]
            assert params["repetition_penalty"] == 1.3
            assert params["temperature"] == 1.0
