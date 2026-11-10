"""A keyframe episode through the session server in token space, on Qwen3.5-0.8B's processor: each generated image
reaches the planner as an ``image_url`` tool part that SGLang's token-space processor expands, and the collected
sample carries the images for the trainer's rebuild along with the submitted video."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import examples.experimental.video_gen_agent.keyframe.env as env
import pytest
import torch
from tests.fast.examples.experimental.video_gen_agent.multiturn.test_clip_observations import (
    _ScriptedPlannerEngine,
    model_path,
    tokenizer,
)
from tests.fast.rollout.session.test_token_space_images import TokenSpaceEngine, _samples
from tests.fast.rollout.session.test_token_space_videos import _make_core

from miles.utils.chat_template_utils.tito_tokenizer import TEMPLATE_DIR, Qwen38SmallTITOTokenizer
from miles.utils.media_refs import collect_media_refs

IMAGE_PAD = "<|image_pad|>"
_FAKES = "examples.experimental.video_gen_agent.generator"
__all__ = ["model_path", "tokenizer"]


QWEN38 = "Qwen/Qwen3.8-27B"


def _processor_core(model_path):
    pytest.importorskip("sglang.srt.multimodal.token_space_mm_process.processor")
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


@pytest.fixture(scope="module")
def processor_core(model_path):
    return _processor_core(model_path)


@pytest.fixture(scope="module")
def qwen38():
    """Qwen3.8-27B's tokenizer under the qwen38small fixed template, and its token-space processor."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    try:
        path = snapshot_download(QWEN38, local_files_only=True, allow_patterns=["*.json", "*.jinja", "*.txt"])
    except Exception as e:  # noqa: BLE001 - no cache
        pytest.skip(f"{QWEN38} is not in the HF cache: {e}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.chat_template = (TEMPLATE_DIR / Qwen38SmallTITOTokenizer.FIXED_TEMPLATE.template).read_text(
        encoding="utf-8"
    )
    return SimpleNamespace(path=path, tokenizer=tokenizer, core=_processor_core(path))


def _set_fakes(monkeypatch, tmp_path):
    for name, value in {
        "AGENT_VIDEO_GEN_GENERATOR": f"{_FAKES}.FakeClipGenerator",
        "AGENT_VIDEO_GEN_IMAGE_GENERATOR": f"{_FAKES}.FakeImageGenerator",
        "AGENT_VIDEO_GEN_FAKE_DIR": str(tmp_path / "generated"),
        "AGENT_VIDEO_GEN_SUBMITTED_DIR": str(tmp_path / "kept"),
    }.items():
        monkeypatch.setenv(name, value)


def _run_episode(monkeypatch, engine, core, registry):
    session_id = registry.create_session()

    async def post(url, payload):
        reply = await core.chat_completions(session_id, method="POST", query="", headers={}, body=json.dumps(payload))
        assert reply.status_code == 200, reply.body
        return json.loads(reply.body)

    monkeypatch.setattr(env, "post", post)
    prompt = [{"role": "user", "content": "A square slides right."}]
    return session_id, asyncio.run(env.run_agent("http://session", prompt, {"max_tokens": 8}, {}))


def test_each_image_is_observed_in_token_space_and_the_sample_carries_them_and_the_video(
    monkeypatch, tmp_path, tokenizer, processor_core, model_path
):
    _set_fakes(monkeypatch, tmp_path)
    calls = [
        ("generate_image", {"prompt": "a red square"}),
        ("generate_image", {"prompt": "a blue square"}),
        ("generate_video", {"image_id": "image_2", "prompt": "it slides right"}),
    ]
    engine = _ScriptedPlannerEngine(processor_core, tokenizer, calls)
    core, registry = _make_core("v2", model_path, tokenizer, engine, tmp_path / "media")
    session_id, result = _run_episode(monkeypatch, engine, core, registry)

    images = [str(Path(image["path"]).resolve()) for image in result["video_gen"]["images"]]
    assert result["video_gen"]["submitted"]["image_id"] == "image_2"
    sent = [collect_media_refs(body["messages"], None) or {} for body in engine.bodies]
    assert [refs.get("images") for refs in sent] == [None, images[:1], images]
    pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    # The second image doubles the expanded image tokens of the first.
    assert engine.prompts[2].count(pad_id) == 2 * engine.prompts[1].count(pad_id) > 2
    (sample,) = _samples(core, session_id, "v2")
    assert sample.multimodal_inputs == {"images": images}
    assert sample.tokens == engine.prompts[-1] + engine.output_ids


def test_a_qwen38_planner_sees_each_image_under_the_counted_check(monkeypatch, tmp_path, qwen38):
    _set_fakes(monkeypatch, tmp_path)
    calls = [
        ("generate_image", {"prompt": "a red square"}),
        ("generate_image", {"prompt": "a blue square"}),
        ("generate_video", {"image_id": "image_1", "prompt": "it slides right"}),
    ]
    engine = _ScriptedPlannerEngine(qwen38.core, qwen38.tokenizer, calls)
    core, registry = _make_core(
        "v2", qwen38.path, qwen38.tokenizer, engine, tmp_path / "media", tito_model="qwen38small"
    )
    session_id, result = _run_episode(monkeypatch, engine, core, registry)

    images = [str(Path(image["path"]).resolve()) for image in result["video_gen"]["images"]]
    assert result["video_gen"]["submitted"]["image_id"] == "image_1" and len(images) == 2
    (sample,) = _samples(core, session_id, "v2")
    assert sample.multimodal_inputs == {"images": images}
    assert sample.tokens == engine.prompts[-1] + engine.output_ids


class _BatchingPlannerEngine(TokenSpaceEngine):
    """Answers the n-th request with the n-th scripted batch of tool calls."""

    def __init__(self, core, tokenizer, batches: list[list[tuple[str, dict]]]):
        super().__init__(core, tokenizer)
        self.batches = batches

    async def do_proxy(self, request, path, *, body, headers):
        result = await super().do_proxy(request, path, body=body, headers=headers)
        response = json.loads(result["response_body"])
        response["choices"][0]["message"]["tool_calls"] = [
            {"id": f"call-{len(self.bodies)}-{i}", "type": "function"}
            | {"function": {"name": name, "arguments": json.dumps(arguments)}}
            for i, (name, arguments) in enumerate(self.batches[len(self.bodies) - 1])
        ]
        result["response_body"] = json.dumps(response)
        return result


def test_a_video_batched_with_its_image_is_made_only_after_the_planner_has_seen_the_image(
    monkeypatch, tmp_path, tokenizer, processor_core, model_path
):
    _set_fakes(monkeypatch, tmp_path)
    video = ("generate_video", {"image_id": "image_1", "prompt": "it slides right"})
    engine = _BatchingPlannerEngine(
        processor_core, tokenizer, [[("generate_image", {"prompt": "a red square"}), video], [video]]
    )
    core, registry = _make_core("v2", model_path, tokenizer, engine, tmp_path / "media")

    session_id, result = _run_episode(monkeypatch, engine, core, registry)

    (image,) = [str(Path(image["path"]).resolve()) for image in result["video_gen"]["images"]]
    assert len(engine.bodies) == 2 and result["video_gen"]["submitted"]["image_id"] == "image_1"
    (sample,) = _samples(core, session_id, "v2")
    assert sample.multimodal_inputs == {"images": [image]}
