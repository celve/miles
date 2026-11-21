"""SFT on recorded keyframe episodes: the set's selection, and the render of an episode's recorded messages, which must
be the very sample the session server builds when the trained planner plays the same turns itself."""

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import examples.experimental.video_gen_agent.keyframe.sft_data as sft_data
import examples.experimental.video_gen_agent.keyframe.sft_rollout as sft_rollout
import pytest
import torch
from examples.experimental.video_gen_agent.keyframe.env import tools_for
from examples.experimental.video_gen_agent.keyframe.run_qwen35_9b_keyframe import ScriptArgs, _train_args
from examples.experimental.video_gen_agent.keyframe.sft_heldout_loss import _expand, _row_loss
from examples.experimental.video_gen_agent.keyframe.sft_rollout import IMAGE_PAD, render_episode
from tests.fast.examples.experimental.video_gen_agent.keyframe.test_image_observations import (
    _run_episode,
    _set_fakes,
    model_path,
    processor_core,
    qwen38,
    tokenizer,
)
from tests.fast.examples.experimental.video_gen_agent.keyframe.test_launcher import _config
from tests.fast.rollout.session.test_token_space_images import TokenSpaceEngine, _samples
from tests.fast.rollout.session.test_token_space_videos import _make_core

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.inference_rollout.compatibility import load_rollout_function
from miles.utils.async_utils import maybe_await
from miles.utils.chat_template_utils.template import normalize_tool_arguments
from miles.utils.chat_template_utils.tito_tokenizer import TEMPLATE_DIR, Qwen35TITOTokenizer, resolve_fixed_chat_template
from miles.utils.media_refs import collect_media_refs, load_media_for_processor
from miles.utils.types import Sample

__all__ = ["model_path", "processor_core", "qwen38", "tokenizer"]

# Two images in one turn, a turn with no call (the env nudges), an unknown tool (an error reply), then the video.
TURNS = [
    [("generate_image", {"prompt": "a red square"}), ("generate_image", {"prompt": "a blue square"})],
    [],
    [("generate_audio", {"prompt": "a hum"})],
    [("generate_video", {"image_id": "image_2", "prompt": "it slides right"})],
]
QWEN35_KWARGS = resolve_fixed_chat_template("qwen35")[1]


class _CanonicalPlannerEngine(TokenSpaceEngine):
    """Answers the n-th request with the n-th scripted turn after a thought, emitting the tokens a planner would: that
    reply as its own template renders it, through its end-of-turn token."""

    def __init__(self, core, tokenizer, turns, template_kwargs):
        super().__init__(core, tokenizer)
        self.tokenizer, self.turns, self.template_kwargs = tokenizer, turns, template_kwargs

    async def do_proxy(self, request, path, *, body, headers):
        payload = json.loads(body)
        turn = len(self.bodies)
        calls = [
            {"id": f"call-{turn + 1}-{i}", "type": "function", "function": {"name": name, "arguments": arguments}}
            for i, (name, arguments) in enumerate(self.turns[turn])
        ]
        reply = {"role": "assistant", "content": "" if calls else "Let me look first.", "reasoning_content": f"Step {turn + 1}."}
        if calls:
            reply["tool_calls"] = calls
        messages = normalize_tool_arguments(payload["messages"], "dict")
        render = dict(tools=payload["tools"], tokenize=False, **self.template_kwargs)
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, **render)
        text = self.tokenizer.apply_chat_template([*messages, reply], **render)[len(prompt) :]
        self.output_ids = self.tokenizer.encode(text[: text.rindex("<|im_end|>") + 10], add_special_tokens=False)
        result = await super().do_proxy(request, path, body=body, headers=headers)
        response = json.loads(result["response_body"])
        if calls:
            reply["tool_calls"] = [c | {"function": c["function"] | {"arguments": json.dumps(c["function"]["arguments"])}} for c in calls]
        response["choices"][0]["message"] = reply
        result["response_body"] = json.dumps(response)
        return result


def _episode(monkeypatch, tmp_path, core, tokenizer, path, tito_model):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _set_fakes(monkeypatch, tmp_path)
    engine = _CanonicalPlannerEngine(core, tokenizer, TURNS, resolve_fixed_chat_template(tito_model)[1])
    session, registry = _make_core("v2", path, tokenizer, engine, tmp_path / "media", tito_model=tito_model)
    session_id, result = _run_episode(monkeypatch, engine, session, registry)
    return session, session_id, result


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def get_samples(self, n):
        return [[Sample(prompt=row["messages"], metadata={"tools": row["tools"]})] for row in self.rows[:n]]


def _sft_sample(monkeypatch, model_path, processor_core, result) -> Sample:
    """The episode's recorded messages, as sft_data writes them, through the run's rollout hook on a running loop."""
    row = json.loads(json.dumps({"messages": result["video_gen"]["messages"], "tools": tools_for(result["video_gen"]["video_mode"])}))
    monkeypatch.setattr(sft_rollout, "_processor_core", lambda args: processor_core)
    args = SimpleNamespace(
        hf_checkpoint=model_path,
        chat_template_path=str(TEMPLATE_DIR / Qwen35TITOTokenizer.FIXED_TEMPLATE.template),
        apply_chat_template_kwargs=QWEN35_KWARGS,
        rollout_batch_size=1,
    )
    hook = load_rollout_function(
        RolloutFnConstructorInput(args=args, data_source=_Rows([row])),
        "examples.experimental.video_gen_agent.keyframe.sft_rollout.KeyframeSftRollout",
    )

    async def executor():
        return await maybe_await(hook(RolloutFnTrainInput(rollout_id=0)))

    ((sample,),) = asyncio.run(executor()).samples
    return sample


def test_a_recorded_episode_trains_on_the_sample_the_session_server_built(
    monkeypatch, tmp_path, tokenizer, processor_core, model_path
):
    session, session_id, result = _episode(monkeypatch, tmp_path, processor_core, tokenizer, model_path, "qwen35")
    (served,) = _samples(session, session_id, "v2")

    sample = _sft_sample(monkeypatch, model_path, processor_core, result)

    assert sample.tokens == served.tokens and sample.loss_mask == served.loss_mask
    assert sample.response_length == served.response_length and sample.multimodal_inputs == served.multimodal_inputs
    assert sample.tokens.count(tokenizer.convert_tokens_to_ids(IMAGE_PAD)) > 2
    # Decoded back, the trained spans are the four replies, each through its end-of-turn token, and nothing else.
    tokens = sample.tokens[-sample.response_length :]
    trained = tokenizer.decode([t for t, m in zip(tokens, sample.loss_mask, strict=True) if m])
    assert [f"Step {n}." in trained for n in (1, 2, 3, 4)] == [True] * 4 and trained.count("<|im_end|>") == 4
    assert "Error" not in trained and "Call generate_image or generate_video" not in trained


def test_a_qwen38_planners_episode_trains_on_what_the_9b_would_have_been_trained_on(
    monkeypatch, tmp_path, tokenizer, processor_core, model_path, qwen38
):
    student_session, student_id, _ = _episode(
        monkeypatch, tmp_path / "student", processor_core, tokenizer, model_path, "qwen35"
    )
    (student,) = _samples(student_session, student_id, "v2")
    # The same turns played by the 27B under its own template and tokens: only its recorded messages carry over.
    _, _, teacher = _episode(monkeypatch, tmp_path / "teacher", qwen38.core, qwen38.tokenizer, qwen38.path, "qwen38small")

    sample = _sft_sample(monkeypatch, model_path, processor_core, teacher)

    assert sample.tokens == student.tokens and sample.loss_mask == student.loss_mask


def test_the_heldout_check_scores_the_tokens_training_sees(monkeypatch, tmp_path, tokenizer, processor_core, model_path):
    from transformers import AutoProcessor

    _, _, result = _episode(monkeypatch, tmp_path, processor_core, tokenizer, model_path, "qwen35")
    sample = _sft_sample(monkeypatch, model_path, processor_core, result)
    messages = json.loads(json.dumps(result["video_gen"]["messages"]))
    images = collect_media_refs(messages, None)["images"]
    processor = AutoProcessor.from_pretrained(model_path)
    grid = processor.image_processor(images=load_media_for_processor({"images": images})["images"], return_tensors="pt")
    counts = (grid["image_grid_thw"].prod(-1) // processor.image_processor.merge_size**2).tolist()

    tokens, mask = render_episode(tokenizer, messages, tools_for(result["video_gen"]["video_mode"]), **QWEN35_KWARGS)
    tokens, mask = _expand(tokens, mask, counts, tokenizer.convert_tokens_to_ids(IMAGE_PAD))

    assert tokens == sample.tokens and mask[-sample.response_length :] == sample.loss_mask


def test_the_heldout_check_runs_a_real_forward_on_an_image_bearing_row(
    monkeypatch, tmp_path, tokenizer, processor_core, model_path
):
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    _, _, result = _episode(monkeypatch, tmp_path, processor_core, tokenizer, model_path, "qwen35")
    row = json.loads(json.dumps({"messages": result["video_gen"]["messages"], "tools": tools_for("preview")}))
    config = AutoConfig.from_pretrained(model_path)
    # Full attention only: the linear-attention kernel has no CPU path, and M-RoPE, the failing part, is in every layer.
    config.text_config.update({"num_hidden_layers": 1, "layer_types": ["full_attention"]})
    config.vision_config.update({"depth": 1})
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(config).eval()

    loss, trained = _row_loss(model, AutoProcessor.from_pretrained(model_path), tokenizer, row, QWEN35_KWARGS)

    sample = _sft_sample(monkeypatch, model_path, processor_core, result)
    assert math.isfinite(loss) and loss > 0 and trained == sum(sample.loss_mask)


def test_a_template_that_rewrites_an_earlier_turn_is_refused(tokenizer):
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]
    messages += [{"role": "assistant", "content": "c"}]

    # Without preserve_thinking the template drops the first turn's thought once a later user turn follows.
    with pytest.raises(ValueError, match="rewrites history"):
        render_episode(tokenizer, [messages[0], messages[1] | {"reasoning_content": "t"}, *messages[2:]], None)


def _dump(path: Path, groups: dict[int, list[tuple[float, bool]]]):
    samples = [
        {
            "group_index": group,
            "reward": reward,
            "metadata": {
                "id": f"req-{group}-{i}",
                "video_reward": {"scores": {}, "fired": {"motion": gated}},
                "video_gen": {"video_mode": "reference", "messages": [{"role": "user", "content": f"{group}-{i}"}]},
            },
        }
        for group, episodes in groups.items()
        for i, (reward, gated) in enumerate(episodes)
    ]
    (path / "rollout_data").mkdir(parents=True)
    torch.save({"samples": samples}, path / "rollout_data" / "0.pt")


def test_each_group_keeps_its_top_episode_only_when_it_is_ungated_and_stands_out(tmp_path):
    _dump(
        tmp_path,
        {
            0: [(0.9, False), (0.5, False), (0.5, False), (0.5, False)],  # kept
            1: [(0.9, True), (0.5, False), (0.5, False), (0.5, False)],  # top gated
            2: [(0.6, False), (0.6, False), (0.6, False), (0.6, False)],  # flat
            3: [(0.7, False), (0.7, False), (0.7, False), (0.7, False), (0.0, False)],  # top within half a std
        },
    )

    kept = sft_data.select(sft_data._episodes(tmp_path), margin=0.5)

    assert [episode["request_id"] for episode in kept] == ["req-0-0"]


def test_groups_of_two_runs_never_merge_though_both_start_at_group_zero(tmp_path):
    for run in ("a", "b"):
        _dump(tmp_path / run, {0: [(0.9, False), (0.5, False), (0.5, False), (0.5, False)]})
    for i, episode in enumerate(episodes := [e for run in ("a", "b") for e in sft_data._episodes(tmp_path / run)]):
        episode["request_id"] = f"{'ab'[i // 4]}-{i % 4}"

    kept = sft_data.select(episodes, margin=0.5)

    assert [episode["request_id"] for episode in kept] == ["a-0", "b-0"]


def test_the_set_writes_the_kept_messages_with_their_tools_and_lists_request_ids_only(monkeypatch, tmp_path):
    _dump(tmp_path / "dump", {0: [(0.9, False), (0.1, False)], 1: [(0.1, False), (0.8, False)]})
    monkeypatch.setattr("sys.argv", ["sft_data", str(tmp_path / "dump"), "--out", str(tmp_path / "set")])

    sft_data.main()

    rows = [json.loads(line) for split in ("train", "heldout") for line in (tmp_path / "set" / f"{split}.jsonl").open()]
    assert sorted(row["request_id"] for row in rows) == ["req-0-0", "req-1-1"]
    assert all(row["tools"] == tools_for("reference") and row["messages"][0]["role"] == "user" for row in rows)
    manifest = (tmp_path / "set" / "manifest.json").read_text()
    assert "0-0" in manifest and '"content"' not in manifest


def test_an_sft_run_trains_every_gpu_on_the_set_with_the_planners_template(monkeypatch, tmp_path):
    args = _config(monkeypatch, tmp_path, sft_data="/sft", num_gpus_per_node=8, video_mode="h3-ref")

    assert args.loss_type == "sft_loss" and args.debug_train_only and args.num_epoch == 2
    assert "--lr 1e-05 " in _train_args(ScriptArgs(run_id="test", sft_data="/sft", video_mode="h3-ref"))
    assert args.prompt_data == "/sft/train.jsonl" and args.tool_key == "tools"
    assert args.chat_template_path.endswith("qwen3.5_fixed.jinja") and args.apply_chat_template_kwargs == QWEN35_KWARGS
    assert args.rollout_function_path.path.endswith("keyframe.sft_rollout.KeyframeSftRollout")
    assert args.actor_num_gpus_per_node == 8 and args.save_hf.endswith("/test/hf")


def test_an_rl_run_starts_from_the_sft_run_and_keeps_it_as_its_reference(monkeypatch, tmp_path):
    sft = tmp_path / "sft"
    sft.mkdir()
    # The parser reads the model config from --hf-checkpoint, which _config writes at the base planner's path.
    (sft / "hf").symlink_to(tmp_path / "Qwen3.5-9B")

    args = _config(monkeypatch, tmp_path, init_checkpoint=str(sft), num_gpus_per_node=8, video_mode="h3-ref")

    assert args.hf_checkpoint == f"{sft}/hf" and args.ref_load == f"{sft}/checkpoints"
    assert args.loss_type != "sft_loss" and not args.debug_train_only
