"""An env episode with clip observations through the session server in token space, on Qwen3.5-0.8B's processor:
the planner generates, sees the clip as a ``video_url`` tool part that SGLang's token-space processor expands, and
submits; the collected sample carries the clip for the trainer's rebuild."""

import asyncio
import json
import os
from pathlib import Path

import examples.experimental.video_gen_agent.env as env
import pytest
import torch
from examples.experimental.video_gen_agent.generator import Clip, ClipRequest
from tests.fast.rollout.session.test_token_space_images import TokenSpaceEngine, _samples
from tests.fast.rollout.session.test_token_space_videos import _make_core, _write_clip

from miles.utils.media_refs import collect_media_refs

MODEL = "Qwen/Qwen3.5-0.8B"
VIDEO_PAD = "<|video_pad|>"


@pytest.fixture(scope="module")
def model_path():
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(MODEL, local_files_only=True, allow_patterns=["*.json", "*.jinja", "*.txt"])
    except Exception as e:  # noqa: BLE001 - no cache
        pytest.skip(f"{MODEL} is not in the HF cache: {e}")
    if not (Path(path) / "video_preprocessor_config.json").exists():
        pytest.skip(f"{MODEL}'s cached snapshot has no processor configs")
    return path


@pytest.fixture(scope="module")
def tokenizer(model_path):
    from transformers import AutoTokenizer

    from miles.utils.chat_template_utils.tito_tokenizer import TEMPLATE_DIR, Qwen35TITOTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.chat_template = (TEMPLATE_DIR / Qwen35TITOTokenizer.FIXED_TEMPLATE.template).read_text(encoding="utf-8")
    return tok


@pytest.fixture(scope="module")
def processor_core(model_path):
    pytest.importorskip("sglang.srt.multimodal.token_space_mm_process.processor")
    _skip_unless_sglang_decodes_video()
    from miles.backends.training_utils.data.token_space_media_process import (
        RolloutEngineProcessorArgs,
        build_trainer_processor,
    )

    engine_args = RolloutEngineProcessorArgs(
        model_path=model_path,
        tokenizer_path=model_path,
        tokenizer_mode="auto",
        tokenizer_backend="huggingface",
        revision=None,
        model_config_parser="auto",
        model_override_args_json="{}",
        image_processor_backend="pil",
        disable_fast_image_processor=False,
        mm_process_config_json="{}",
        token_space_preprocess_cache_dir=None,
    )
    return build_trainer_processor(
        engine_args, mm_processor_worker_num=1, mm_io_worker_num=0, device=torch.device("cpu")
    ).core


def _skip_unless_sglang_decodes_video():
    """SGLang decodes clips with torchcodec (which needs FFmpeg) or decord; macOS hosts often have neither."""
    import tempfile

    from sglang.srt.utils.video_decoder import VideoDecoderWrapper

    with tempfile.TemporaryDirectory() as folder:
        try:
            VideoDecoderWrapper(_write_clip(Path(folder) / "probe.mp4", (0, 0, 0)), device="cpu")
        except Exception as e:  # noqa: BLE001 - any missing backend or library
            pytest.skip(f"SGLang cannot decode video here: {e!r}"[:200])


class _PlannerEngine(TokenSpaceEngine):
    """Expands media as SGLang does, and answers as a planner: generate a clip, then submit it."""

    async def do_proxy(self, request, path, *, body, headers):
        result = await super().do_proxy(request, path, body=body, headers=headers)
        response = json.loads(result["response_body"])
        name, arguments = (
            ("generate", {"prompt": "a red square"}) if len(self.bodies) == 1 else ("submit", {"clip_id": "clip_1"})
        )
        call = {"id": f"call-{len(self.bodies)}", "type": "function"}
        response["choices"][0]["message"]["tool_calls"] = [
            call | {"function": {"name": name, "arguments": json.dumps(arguments)}}
        ]
        result["response_body"] = json.dumps(response)
        return result


class _Mp4Generator:
    """Writes a real mp4 per request under the env's fake dir."""

    def __init__(self):
        self.clip_dir = Path(os.environ["AGENT_VIDEO_GEN_FAKE_DIR"])

    async def generate(self, request: ClipRequest) -> Clip:
        return Clip(path=_write_clip(self.clip_dir / f"{request.seed}.mp4", (200, 10, 10)))


def test_a_generated_clip_is_observed_in_token_space_and_reaches_the_trainers_sample(
    monkeypatch, tmp_path, tokenizer, processor_core, model_path
):
    monkeypatch.setenv("AGENT_VIDEO_GEN_GENERATOR", f"{__name__}._Mp4Generator")
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_VIDEO_GEN_SUBMITTED_DIR", str(tmp_path / "kept"))
    monkeypatch.setenv("AGENT_VIDEO_GEN_CLIP_OBSERVATIONS", "1")
    engine = _PlannerEngine(processor_core, tokenizer)
    core, registry = _make_core("v2", model_path, tokenizer, engine, tmp_path / "media")
    session_id = registry.create_session()

    async def post(url, payload):
        reply = await core.chat_completions(session_id, method="POST", query="", headers={}, body=json.dumps(payload))
        assert reply.status_code == 200, reply.body
        return json.loads(reply.body)

    monkeypatch.setattr(env, "post", post)
    prompt = [{"role": "user", "content": "A red square slides right."}]
    result = asyncio.run(env.run_agent("http://session", prompt, {"max_tokens": 8}, {}))

    clip = result["video_gen"]["clips"][0]["path"]
    assert result["video_gen"]["submitted"]["clip_id"] == "clip_1"
    pad_id = tokenizer.convert_tokens_to_ids(VIDEO_PAD)
    first, second = engine.bodies
    assert collect_media_refs(first["messages"], None) is None
    tool_reply = second["messages"][-1]
    assert tool_reply["role"] == "tool" and {"type": "video_url", "video_url": {"url": f"file://{clip}"}} in (
        tool_reply["content"]
    )
    start = second["mm_token_expansion_start_len"]
    assert second["input_ids"][start:].count(pad_id) == 1 and engine.prompts[1][start:].count(pad_id) > 1
    (sample,) = _samples(core, session_id, "v2")
    assert sample.multimodal_inputs == {"videos": [str(Path(clip).resolve())]}
    assert sample.tokens == engine.prompts[-1] + engine.output_ids
