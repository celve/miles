from __future__ import annotations

import base64
import io
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests
import safetensors.numpy
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from tests.ci.ci_register import register_cpu_ci
from tests.fast.fixtures.args_fixtures import sglang_router_args
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config
from tests.fast.router.test_sessions import _create_session, _post_chat

from miles.backends.megatron_utils.megatron_config import MegatronConfig
from miles.rollout.session.config import compute_session_server_config
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_and_merge_input_sample
from miles.rollout.session.server import SessionServer
from miles.utils.http_utils import find_available_port
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

register_cpu_ci(est_time=50, suite="stage-b-cpu", labels=["rollout"])

CHECKPOINT = "Qwen/Qwen3-VL-2B-Instruct"
IMAGE_TOKEN = "<|image_pad|>"


@pytest.fixture(scope="module")
def vlm():
    processor = load_processor(CHECKPOINT, trust_remote_code=True)
    if processor is None:
        pytest.skip(f"{CHECKPOINT} processor unavailable")
    tokenizer = load_tokenizer(CHECKPOINT, trust_remote_code=True)
    return SimpleNamespace(
        processor=processor, tokenizer=tokenizer, pad_id=tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
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


class TokenSpaceChatBackend:
    def __init__(self, vlm, replies: list[str], *, count_bias: int = 0, drop_images: bool = False):
        self.vlm = vlm
        self.replies = replies
        self.count_bias = count_bias
        self.drop_images = drop_images
        self.calls: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.routing_keys: list[str | None] = []
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

        @self.app.post("/v1/chat/completions")
        async def chat(request: Request):
            payload = await request.json()
            self.calls.append(payload)
            self.paths.append(request.url.path)
            self.routing_keys.append(request.headers.get("X-SMG-Routing-Key"))
            return JSONResponse(content=self._reply(payload))

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self):
        self._server.start()

    def stop(self):
        self._server.stop()

    def _images(self, messages: list[dict]) -> list[str]:
        from sglang.srt.parser.jinja_template_utils import process_content_for_template_format

        image_data: list = []
        for message in messages:
            process_content_for_template_format(
                {**message, "content": message.get("content") or ""}, "openai", image_data, [], [], []
            )
        return [] if self.drop_images else [image.url for image in image_data]

    def _counts(self, image_urls: list[str]) -> list[int]:
        sizes = []
        for url in image_urls:
            with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as handle:
                sizes.append((handle.height, handle.width))
        counts = self.vlm.processor._get_num_multimodal_tokens(image_sizes=sizes).num_image_tokens
        return [int(count) + self.count_bias for count in counts]

    def _reply(self, payload: dict) -> dict:
        from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

        index = len(self.calls) - 1
        effective = list(payload["input_ids"])
        start = payload.get("mm_token_expansion_start_len") or 0
        image_urls = self._images(payload["messages"]) if start else []
        if image_urls:
            expanded = BaseMultimodalProcessor._expand_input_ids(
                effective[start:], self._counts(image_urls), self.vlm.pad_id
            )
            effective = effective[:start] + expanded
        output_ids = self.vlm.tokenizer.encode(
            self.replies[min(index, len(self.replies) - 1)], add_special_tokens=False
        )
        choices = []
        for choice_index in range(payload.get("n") or 1):
            choice = {
                "index": choice_index,
                "message": {
                    "role": "assistant",
                    "content": self.vlm.tokenizer.decode(output_ids, skip_special_tokens=True),
                },
                "finish_reason": "stop",
                "meta_info": {
                    "id": f"chat-{index}",
                    "completion_tokens": len(output_ids),
                    "weight_version": "w0",
                    "output_token_logprobs": [[-i / 128, tid, None] for i, tid in enumerate(output_ids)],
                },
            }
            if payload.get("return_prompt_token_ids") or payload.get("return_token_ids"):
                choice["prompt_token_ids"] = effective
            choices.append(choice)
        return {"id": f"chat-{index}", "object": "chat.completion", "created": 0, "model": "m", "choices": choices}


def image_server_args(media_dir: Path, **extra) -> dict[str, Any]:
    return {
        "miles_router_timeout": 30,
        "hf_checkpoint": CHECKPOINT,
        "chat_template_path": None,
        "apply_chat_template_kwargs": None,
        "tito_model": "qwen3_vl",
        "use_rollout_routing_replay": False,
        "use_rollout_indexer_replay": False,
        "rollout_indexer_topk_num_streams": None,
        "use_sampling_support_replay": False,
        "use_output_store": False,
        "rollout_top_logprobs_num": 0,
        "rollout_sampling_logprobs_mode": "selected",
        "sglang": make_sglang_config(speculative_algorithm=None),
        "num_layers": None,
        "raw_megatron": MegatronConfig(trainers=[], base_args={}),
        "save_debug_trajectory_data": None,
        "lora_rank": 0,
        "lora_adapter_path": None,
        "lora_train_only": False,
        "use_session_server": "v2",
        "use_session_images": True,
        "session_image_text_only_samples": "drop",
        "shared_media_dir": str(media_dir),
        "run_uuid": "run0",
        "session_message_matcher": "strict",
        "session_server_instance_id": uuid.uuid4().hex,
        "pause_generation_mode": "retract",
        "session_sample_picker_path": "miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
        "session_sample_postprocessor_path": "miles.rollout.session.v2.postprocessor_hub.default_postprocess",
        **sglang_router_args("manual"),
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
def image_router(vlm, media_dir: Path, replies: list[str], *, count_bias: int = 0, drop_images: bool = False, **extra):
    backend = TokenSpaceChatBackend(vlm, replies, count_bias=count_bias, drop_images=drop_images)
    backend.start()
    try:
        server_obj, port = build_server(image_server_args(media_dir, **extra), backend_url=backend.url)
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


def ref_bytes(sample: Sample) -> list[bytes]:
    return [Path(path).read_bytes() for path in sample.multimodal_inputs["images"]]


SHOT1 = png(120, 80, (10, 20, 30))
SHOT2 = png(160, 120, (40, 80, 120))
SHOT1_TWIN = png(120, 80, (200, 190, 180))


class TestAppend:
    def test_two_image_turns_make_one_sample_carrying_both_refs(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["clicked", "done"]) as env:
            session_id = _create_session(env.url)
            first = [user_turn(SHOT1, "step 1")]
            assert chat(env, session_id, first).status_code == 200
            second = first + [assistant("clicked"), user_turn(SHOT2, "step 2")]
            assert chat(env, session_id, second).status_code == 200

            samples = collect(env.url, session_id)
            assert len(samples) == 1
            sample = samples[0]
            assert image_token_runs(sample.tokens, vlm.pad_id) == expected_counts(vlm, [SHOT1, SHOT2])
            assert ref_bytes(sample) == [SHOT1, SHOT2]
            assert all(
                path.startswith(str((tmp_path / "run0").resolve())) for path in sample.multimodal_inputs["images"]
            )
            assert sample.multimodal_train_inputs is None
            assert len(sample.loss_mask) == sample.response_length
            assert len(sample.rollout_log_probs) == sample.response_length

    def test_the_engine_is_sent_the_unexpanded_render_and_the_clients_images(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["clicked"]) as env:
            session_id = _create_session(env.url)
            assert chat(env, session_id, [user_turn(SHOT1, "step 1")]).status_code == 200
            call = env.backend.calls[0]
            assert image_token_runs(call["input_ids"], vlm.pad_id) == [1]
            assert call["messages"][0]["content"][0] == image_part(SHOT1)
            assert call["return_prompt_token_ids"] is True

    def test_expansion_starts_at_the_first_placeholder(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["clicked"]) as env:
            session_id = _create_session(env.url)
            assert chat(env, session_id, [user_turn(SHOT1, "step 1")]).status_code == 200
            assert chat(env, session_id, [user_turn(None, "no pictures")]).status_code == 200
            image_call, text_call = env.backend.calls
            assert image_call["mm_token_expansion_start_len"] == image_call["input_ids"].index(vlm.pad_id)
            assert "mm_token_expansion_start_len" not in text_call
            assert "return_prompt_token_ids" not in text_call

    def test_a_zero_expansion_start_loses_the_images(self, vlm, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "miles.rollout.session.v2.core.SessionCoreV2._image_turn_fields",
            lambda self, input_ids: {"return_prompt_token_ids": True, "mm_token_expansion_start_len": 0},
        )
        with image_router(vlm, tmp_path, ["clicked"]) as env:
            session_id = _create_session(env.url)
            response = chat(env, session_id, [user_turn(SHOT1, "step 1")])
            assert response.status_code >= 400
            assert b"effective prompt ids differ" in response.content

    def test_the_samples_reply_carries_no_pixels(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["a"]) as env:
            session_id = _create_session(env.url)
            chat(env, session_id, [user_turn(SHOT1, "one")])
            response = requests.post(f"{env.url}/sessions/{session_id}/samples", json={}, timeout=20.0)
            assert response.status_code == 200
            names = safetensors.numpy.load(response.content).keys()
            assert not [name for name in names if name.startswith("multimodal_train_inputs.")]


class TestTemplate:
    def test_turns_after_a_model_stop_match_the_checkpoints_template(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["clicked<|im_end|>", "done<|im_end|>"]) as env:
            session_id = _create_session(env.url)
            first = [user_turn(SHOT1, "step 1")]
            assert chat(env, session_id, first).status_code == 200
            second = first + [assistant("clicked"), user_turn(SHOT2, "step 2")]
            assert chat(env, session_id, second).status_code == 200

            im_end = vlm.tokenizer.convert_tokens_to_ids("<|im_end|>")
            accumulated = env.backend.calls[1]["input_ids"]
            assert vlm.tokenizer.decode(accumulated[accumulated.index(im_end) + 1]) == "\n"
            assert collect(env.url, session_id)[0].metadata["tito_session_mismatch"] == []

    def test_image_labels_count_across_the_whole_session(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["clicked<|im_end|>", "done<|im_end|>"]) as env:
            session_id = _create_session(env.url)
            first = [user_turn(SHOT1, "step 1")]
            body = {"max_tokens": 16, "temperature": 1.0}
            labelled = {**body, "messages": first, "chat_template_kwargs": {"add_vision_id": True}}
            assert _post_chat(env.url, session_id, labelled).status_code == 200
            second = first + [assistant("clicked"), user_turn(SHOT2, "step 2")]
            unlabelled = {**body, "messages": second, "chat_template_kwargs": {"add_vision_id": False}}
            assert _post_chat(env.url, session_id, unlabelled).status_code == 200

            assert "Picture 2: " in vlm.tokenizer.decode(env.backend.calls[1]["input_ids"])
            assert collect(env.url, session_id)[0].metadata["tito_session_mismatch"] == []


class TestRepeatedAndReordered:
    def test_the_same_screenshot_twice_is_two_refs_to_one_file(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["a", "b"]) as env:
            session_id = _create_session(env.url)
            first = [user_turn(SHOT1, "one")]
            chat(env, session_id, first)
            chat(env, session_id, first + [assistant("a"), user_turn(SHOT1, "again")])
            sample = collect(env.url, session_id)[0]
            refs = sample.multimodal_inputs["images"]
            assert len(refs) == 2 and refs[0] == refs[1]
            assert image_token_runs(sample.tokens, vlm.pad_id) == expected_counts(vlm, [SHOT1, SHOT1])
            assert len([path for path in tmp_path.rglob("*") if path.is_file()]) == 1

    def test_reordering_images_is_a_different_prompt(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["ok", "ok"]) as env:
            session_id = _create_session(env.url)
            forward = [{"role": "user", "content": [image_part(SHOT1), image_part(SHOT2)]}]
            reverse = [{"role": "user", "content": [image_part(SHOT2), image_part(SHOT1)]}]
            assert chat(env, session_id, forward).status_code == 200
            assert chat(env, session_id, reverse).status_code == 200
            assert sorted(ref_bytes(s) for s in collect(env.url, session_id)) == sorted(
                [[SHOT1, SHOT2], [SHOT2, SHOT1]]
            )


class TestRewriteAndRetry:
    def test_a_rewrite_gives_each_branch_its_own_refs(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["a", "b", "c"]) as env:
            session_id = _create_session(env.url)
            first = [user_turn(SHOT1, "one")]
            chat(env, session_id, first)
            chat(env, session_id, first + [assistant("a"), user_turn(SHOT2, "two")])
            rewritten = [
                user_turn(None, "one"),
                assistant("a"),
                user_turn(SHOT2, "two"),
                assistant("b"),
                user_turn(SHOT1_TWIN, "three"),
            ]
            assert chat(env, session_id, rewritten).status_code == 200
            samples = collect(env.url, session_id)
            assert sorted(ref_bytes(s) for s in samples) == sorted([[SHOT1, SHOT2], [SHOT2, SHOT1_TWIN]])
            for sample in samples:
                assert image_token_runs(sample.tokens, vlm.pad_id) == expected_counts(vlm, ref_bytes(sample))

    def test_a_same_grid_different_image_sibling_is_not_a_retry(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["a", "b"]) as env:
            session_id = _create_session(env.url)
            assert chat(env, session_id, [user_turn(SHOT1, "look")]).status_code == 200
            assert chat(env, session_id, [user_turn(SHOT1_TWIN, "look")]).status_code == 200
            first, second = env.backend.calls
            assert first["input_ids"] == second["input_ids"]
            assert first["messages"] != second["messages"]
            assert len(collect(env.url, session_id)) == 2

    def test_an_identical_resend_is_dropped(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["a", "a"]) as env:
            session_id = _create_session(env.url)
            messages = [user_turn(SHOT1, "look")]
            assert chat(env, session_id, messages).status_code == 200
            assert chat(env, session_id, messages).status_code == 200
            assert len(collect(env.url, session_id)) == 1


class TestTruncation:
    def test_a_cut_inside_an_image_drops_that_ref(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["a", "b"]) as env:
            session_id = _create_session(env.url)
            first = [user_turn(SHOT1, "one")]
            chat(env, session_id, first)
            chat(env, session_id, first + [assistant("a"), user_turn(SHOT2, "two")])
            whole = collect(env.url, session_id)[0]
            budget = len(whole.tokens) - expected_counts(vlm, [SHOT1, SHOT2])[1] // 2
            trimmed = collect(env.url, session_id, {"max_seq_len": budget})
            assert len(trimmed) == 1
            assert ref_bytes(trimmed[0]) == [SHOT1]


class TestMixedTextOnly:
    def test_a_text_only_sample_is_dropped_by_default(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["a", "summary"]) as env:
            session_id = _create_session(env.url)
            assert chat(env, session_id, [user_turn(SHOT1, "look")]).status_code == 200
            assert chat(env, session_id, [user_turn(None, "summarize")]).status_code == 200
            samples = collect(env.url, session_id)
            assert len(samples) == 1
            assert samples[0].multimodal_inputs is not None

    def test_keep_emits_it_without_refs(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["a", "summary"], session_image_text_only_samples="keep") as env:
            session_id = _create_session(env.url)
            assert chat(env, session_id, [user_turn(SHOT1, "look")]).status_code == 200
            assert chat(env, session_id, [user_turn(None, "summarize")]).status_code == 200
            samples = collect(env.url, session_id)
            assert len(samples) == 2
            text_only = [s for s in samples if s.multimodal_inputs is None]
            assert len(text_only) == 1
            assert vlm.pad_id not in text_only[0].tokens


class TestExactnessGate:
    def test_an_engine_that_expands_differently_fails_the_request(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["a"], count_bias=1) as env:
            session_id = _create_session(env.url)
            response = chat(env, session_id, [user_turn(SHOT1, "look")])
            assert response.status_code >= 400
            assert b"effective prompt ids differ" in response.content

    def test_an_engine_that_drops_the_images_fails_the_request(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["a"], drop_images=True) as env:
            session_id = _create_session(env.url)
            response = chat(env, session_id, [user_turn(SHOT1, "look")])
            assert response.status_code >= 400
            assert b"effective prompt ids differ" in response.content
            assert collect(env.url, session_id) == []


class TestRefusals:
    def test_an_image_is_refused_when_support_is_off(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["a"], use_session_images=False) as env:
            session_id = _create_session(env.url)
            response = chat(env, session_id, [user_turn(SHOT1, "look")])
            assert response.status_code == 400
            assert b"--use-session-images" in response.content

    def test_a_remote_image_url_is_refused(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["a"]) as env:
            session_id = _create_session(env.url)
            messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/a.png"}}]}]
            response = chat(env, session_id, messages)
            assert response.status_code == 400
            assert b"inline base64 data URL" in response.content

    def test_no_shared_media_dir_refuses_to_start(self, vlm, tmp_path):
        args = image_server_args(tmp_path, shared_media_dir=None)
        with pytest.raises(ValueError, match="--shared-media-dir"):
            build_server(args, backend_url="http://127.0.0.1:1")

    def test_the_server_starts_with_nothing_behind_it(self, vlm, tmp_path):
        server_obj, _ = build_server(image_server_args(tmp_path), backend_url="http://127.0.0.1:9")
        assert server_obj.app is not None


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


class TestAgentCompatibility:
    def test_an_image_turn_posts_the_chat_route_through_the_router(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["a", "b"]) as env:
            session_id = _create_session(env.url)
            first = [user_turn(SHOT1, "look")]
            assert chat(env, session_id, first).status_code == 200
            assert chat(env, session_id, [*first, assistant("a"), user_turn(None, "and now?")]).status_code == 200
            assert env.backend.paths == ["/v1/chat/completions", "/v1/chat/completions"]
            assert env.backend.routing_keys == [session_id, session_id]
            assert env.backend.metadata_calls == []

    def test_an_image_turn_sends_what_a_text_turn_sends_plus_the_expansion_fields(self, vlm, tmp_path):
        controls = {
            "max_tokens": 16,
            "temperature": 1.0,
            "n": 2,
            "seed": 7,
            "logit_bias": {"11": -5},
            "response_format": {"type": "json_object"},
            "tools": TOOLS,
            "tool_choice": "required",
        }
        with image_router(vlm, tmp_path, ["a", "b"]) as env:
            text_id, image_id = _create_session(env.url), _create_session(env.url)
            assert _post_chat(env.url, text_id, {"messages": [user_turn(None, "look")], **controls}).status_code == 200
            assert (
                _post_chat(env.url, image_id, {"messages": [user_turn(SHOT1, "look")], **controls}).status_code == 200
            )
            text_body, image_body = env.backend.calls
            differing = {
                key for key in text_body.keys() | image_body.keys() if text_body.get(key) != image_body.get(key)
            }
            assert differing == {"messages", "input_ids", "return_prompt_token_ids", "mm_token_expansion_start_len"}
            assert image_body["tools"] == TOOLS

    def test_the_readback_is_neither_returned_nor_recorded_on_any_choice(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["a"]) as env:
            session_id = _create_session(env.url)
            body = {"messages": [user_turn(SHOT1, "look")], "max_tokens": 16, "n": 2}
            choices = _post_chat(env.url, session_id, body).json()["choices"]
            assert len(choices) == 2
            assert not any("prompt_token_ids" in choice for choice in choices)
            record = requests.get(f"{env.url}/sessions/{session_id}", timeout=10).json()["records"][0]
            assert "return_prompt_token_ids" not in record["request"]
            assert "mm_token_expansion_start_len" not in record["request"]
            assert not any("prompt_token_ids" in choice for choice in record["response"]["choices"])

    def test_a_client_that_asks_for_prompt_ids_gets_the_engines_on_every_choice(self, vlm, tmp_path):
        with image_router(vlm, tmp_path, ["a"]) as env:
            session_id = _create_session(env.url)
            body = {"messages": [user_turn(SHOT1, "look")], "max_tokens": 16, "n": 2, "return_prompt_token_ids": True}
            choices = _post_chat(env.url, session_id, body).json()["choices"]
            counts = expected_counts(vlm, [SHOT1])
            assert [image_token_runs(choice["prompt_token_ids"], vlm.pad_id) for choice in choices] == [counts, counts]
