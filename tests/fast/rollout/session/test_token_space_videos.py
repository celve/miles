"""Video clips through the session server with --multimodal-token-space-processing.

The video counterpart of test_token_space_images.py on Qwen3.5-9B: the stub engine expands each clip with
SGLang's token-space processor on CPU, per-frame timestamps included, and the session server runs the
counted expansion check of ``--tito-model qwen35``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.rollout.session.test_token_space_images import TokenSpaceEngine, _chat, _samples

from miles.rollout.session.core import SessionCore
from miles.rollout.session.errors import UpstreamResponseError
from miles.rollout.session.image_expansion import build_image_token_counter
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.v2.core import SessionCoreV2
from miles.rollout.session.v2.session_state import SessionRegistryV2
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.media_refs import collect_media_refs

MODEL = "Qwen/Qwen3.5-9B"
VIDEO_PAD = "<|video_pad|>"


@pytest.fixture(scope="module")
def model_path():
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(MODEL, local_files_only=True)
    except Exception as e:  # noqa: BLE001 - no cache
        pytest.skip(f"{MODEL} is not in the HF cache: {e}")


@pytest.fixture(scope="module")
def tokenizer(model_path):
    """The Qwen3.5 tokenizer with the qwen35 fixed template, as ``--chat-template-path`` installs it."""
    from transformers import AutoTokenizer

    from miles.utils.chat_template_utils.tito_tokenizer import TEMPLATE_DIR, Qwen35TITOTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.chat_template = (TEMPLATE_DIR / Qwen35TITOTokenizer.FIXED_TEMPLATE.template).read_text(encoding="utf-8")
    return tok


@pytest.fixture(scope="module")
def engine_args(model_path):
    pytest.importorskip("sglang.srt.multimodal.token_space_mm_process.processor")
    from miles.backends.training_utils.data.token_space_media_process import RolloutEngineProcessorArgs

    return RolloutEngineProcessorArgs(
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


@pytest.fixture(scope="module")
def trainer_processor(engine_args):
    from miles.backends.training_utils.data.token_space_media_process import build_trainer_processor

    return build_trainer_processor(
        engine_args, mm_processor_worker_num=1, mm_io_worker_num=0, device=torch.device("cpu")
    )


@pytest.fixture(scope="module")
def processor_core(trainer_processor):
    return trainer_processor.core


def _write_clip(path, color: tuple[int, int, int], *, frames: int = 33, fps: int = 16) -> str:
    """An mp4 of a square sliding over a ``color`` background, the shape Wan2.2 emits at a smaller size."""
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 192))
    for index in range(frames):
        frame = np.full((192, 320, 3), color, dtype=np.uint8)
        frame[40:100, 8 * index : 8 * index + 60] = 255
        writer.write(frame)
    writer.release()
    return str(path)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    folder = tmp_path_factory.mktemp("clips")
    return [
        _write_clip(folder / f"clip{i}.mp4", color)
        for i, color in enumerate([(200, 10, 10), (10, 200, 10), (10, 10, 200)])
    ]


def _clip_part(path: str) -> dict:
    return {"type": "video_url", "video_url": {"url": path}}


def _tool_call(index: int) -> list[dict]:
    return [
        {"id": f"call-{index}", "type": "function", "function": {"name": "generate", "arguments": '{"prompt": "a"}'}}
    ]


class ToolCallingEngine(TokenSpaceEngine):
    """The token-space engine replying with a ``generate`` tool call, and an optional rewrite of the prompt
    it returns, standing in for an engine that expands a clip differently."""

    def __init__(self, *args, rewrite_prompt=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewrite_prompt = rewrite_prompt

    async def do_proxy(self, request, path, *, body, headers):
        result = await super().do_proxy(request, path, body=body, headers=headers)
        response = json.loads(result["response_body"])
        choice = response["choices"][0]
        choice["message"]["tool_calls"] = _tool_call(len(self.bodies))
        if self.rewrite_prompt is not None and "prompt_token_ids" in choice:
            choice["prompt_token_ids"] = self.rewrite_prompt(choice["prompt_token_ids"])
        result["response_body"] = json.dumps(response)
        return result


def _make_core(version: str, model_path, tokenizer, engine, media_dir, *, tito_model: str = "qwen35"):
    config = make_session_server_config(
        use_session_server=version,
        hf_checkpoint=model_path,
        tito_model=tito_model,
        instance_id=uuid.uuid4().hex,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
        multimodal_token_space_processing=True,
        run_media_dir=str(media_dir),
    )
    tito = get_tito_tokenizer(tokenizer, tokenizer_type=tito_model)
    counter = build_image_token_counter(config, tokenizer)
    assert counter is not None, f"{tito_model} turns run the counted check"
    if version == "v2":
        registry = SessionRegistryV2(tokenizer, tito_tokenizer=tito)
        core = SessionCoreV2(engine, registry, config, config.instance_id, image_token_counter=counter)
    else:
        registry = SessionRegistry(tokenizer, tito_tokenizer=tito)
        core = SessionCore(engine, registry, config, config.instance_id, image_token_counter=counter)
    return core, registry


def _episode(clips: list[str]) -> list[list[dict]]:
    """The message lists of an episode: the first clip in the user turn, each later one in a tool result."""
    turns = [[{"role": "user", "content": [{"type": "text", "text": "make a clip"}, _clip_part(clips[0])]}]]
    for index, clip in enumerate(clips[1:]):
        turns.append(
            [
                {"role": "assistant", "content": "ok", "tool_calls": _tool_call(index + 1)},
                {"role": "tool", "tool_call_id": f"call-{index + 1}", "content": [_clip_part(clip)]},
            ]
        )
    return turns


def _run_episode(core, registry, clips: list[str]):
    session_id = registry.create_session()
    messages: list[dict] = []
    for turn in _episode(clips):
        messages = messages + turn
        response = asyncio.run(_chat(core, session_id, messages))
        assert response.status_code == 200, response.body
    return session_id


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_clips_flow_through_the_session_server_in_token_space(
    version, model_path, tokenizer, processor_core, trainer_processor, clips, tmp_path
):
    from miles.backends.training_utils.data.token_space_media_process import (
        compute_multimodal_train_inputs,
        resolved_multimodal_train_inputs,
    )

    pad_id = tokenizer.convert_tokens_to_ids(VIDEO_PAD)
    engine = ToolCallingEngine(processor_core, tokenizer)
    core, registry = _make_core(version, model_path, tokenizer, engine, tmp_path)
    session_id = _run_episode(core, registry, clips)

    checkpoint = []
    for body, prompt in zip(engine.bodies, engine.prompts, strict=True):
        start = body["mm_token_expansion_start_len"]
        # Each turn resends the previous turn's expanded prompt and reply, and one placeholder for its new clip.
        assert body["input_ids"][:start] == checkpoint
        assert body["input_ids"][start:].count(pad_id) == 1
        # The engine's expansion of the new clip leaves the prefix as it was: a pure extension.
        assert prompt[:start] == checkpoint
        assert prompt[start:].count(pad_id) > 1
        checkpoint = prompt + engine.output_ids
    paths = collect_media_refs(engine.bodies[-1]["messages"], None)["videos"]
    assert paths == [str(Path(clip).resolve()) for clip in clips], "a clip already on a shared path is not copied"
    stamps = [
        _frame_timestamps(prompt, body["mm_token_expansion_start_len"], tokenizer)
        for body, prompt in zip(engine.bodies, engine.prompts, strict=True)
    ]
    assert len(stamps[0]) > 1 and stamps == [stamps[0]] * 3, "each clip's timestamps count from its own start"

    (sample,) = _samples(core, session_id, version)
    assert sample.multimodal_inputs == {"videos": paths}
    assert sample.tokens == engine.prompts[-1] + engine.output_ids
    assert sample.loss_mask.count(1) == 3 * len(engine.output_ids), "only the three sampled replies are trained"

    pending = compute_multimodal_train_inputs(
        trainer_processor,
        [sample.multimodal_inputs],
        [torch.tensor(sample.tokens)],
        mm_processor_worker_num=1,
        with_sglang_placement=False,
        device=torch.device("cpu"),
    )
    (train_inputs,) = resolved_multimodal_train_inputs(pending.multimodal_train_inputs)
    assert torch.equal(train_inputs["pixel_values_videos"], engine.features[-1]["pixel_values_videos"])
    assert torch.equal(train_inputs["video_grid_thw"], engine.features[-1]["video_grid_thw"])
    assert train_inputs["video_grid_thw"].shape[0] == 3


def _frame_timestamps(prompt: list[int], start: int, tokenizer) -> list[str]:
    """The timestamp text of each frame of the clip expanded after ``start``."""
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    position, stamps = prompt.index(vision_start, start) + 1, []
    while prompt[position] != vision_end:
        frame_start = prompt.index(vision_start, position)
        stamps.append(tokenizer.decode(prompt[position:frame_start]))
        position = prompt.index(vision_end, frame_start) + 1
    return stamps


def _unexpanded(tokenizer):
    """The engine ran the clip's placeholder as it was sent, as SGLang does on a first turn without sglang#47402."""
    pad_id = tokenizer.convert_tokens_to_ids(VIDEO_PAD)
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    def rewrite(prompt):
        last_pad = len(prompt) - 1 - prompt[::-1].index(pad_id)
        return prompt[: prompt.index(vision_start) + 1] + [pad_id] + prompt[prompt.index(vision_end, last_pad) + 1 :]

    return rewrite


def _one_frame_short(tokenizer):
    pad_id = tokenizer.convert_tokens_to_ids(VIDEO_PAD)

    def rewrite(prompt):
        last_pad = len(prompt) - 1 - prompt[::-1].index(pad_id)
        return prompt[:last_pad] + prompt[last_pad + 1 :]

    return rewrite


def _timestamps_going_back(tokenizer):
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    def rewrite(prompt):
        second_frame = prompt.index(vision_end, prompt.index(vision_start)) + 1
        frame_start = prompt.index(vision_start, second_frame)
        return (
            prompt[:second_frame] + tokenizer.encode("<0.0 seconds>", add_special_tokens=False) + prompt[frame_start:]
        )

    return rewrite


@pytest.mark.parametrize(
    "make_rewrite",
    [_unexpanded, _one_frame_short, _timestamps_going_back],
    ids=["unexpanded", "short-frame", "going-back"],
)
def test_a_clip_expanded_out_of_shape_fails_the_counted_check(
    make_rewrite, model_path, tokenizer, processor_core, clips, tmp_path
):
    engine = ToolCallingEngine(processor_core, tokenizer, rewrite_prompt=make_rewrite(tokenizer))
    core, registry = _make_core("v2", model_path, tokenizer, engine, tmp_path)
    session_id = registry.create_session()
    with pytest.raises(UpstreamResponseError, match="expanded prompt differs"):
        asyncio.run(_chat(core, session_id, _episode(clips)[0]))
    assert registry.get_session(session_id).tree.nodes == [], "nothing was committed"
