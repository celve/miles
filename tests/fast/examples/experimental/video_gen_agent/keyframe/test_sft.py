"""SFT on recorded keyframe episodes: the set's selection, and the render of an episode's recorded messages, which must
be the very sample the session server builds when the trained planner plays the same turns itself."""

import json
from pathlib import Path

import examples.experimental.video_gen_agent.keyframe.sft_data as sft_data
import pytest
import torch
from examples.experimental.video_gen_agent.keyframe.env import tools_for
from examples.experimental.video_gen_agent.keyframe.sft_rollout import IMAGE_PAD, expand_images, render_episode
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

from miles.utils.chat_template_utils.template import normalize_tool_arguments
from miles.utils.chat_template_utils.tito_tokenizer import resolve_fixed_chat_template

__all__ = ["model_path", "processor_core", "qwen38", "tokenizer"]

CALLS = [
    ("generate_image", {"prompt": "a red square"}),
    ("generate_image", {"prompt": "a blue square"}),
    ("generate_video", {"image_id": "image_2", "prompt": "it slides right"}),
]
QWEN35_KWARGS = resolve_fixed_chat_template("qwen35")[1]


class _CanonicalPlannerEngine(TokenSpaceEngine):
    """Answers the n-th request with the n-th scripted call after a thought, emitting the tokens a planner would: that
    reply as its own template renders it, through its end-of-turn token."""

    def __init__(self, core, tokenizer, calls, template_kwargs):
        super().__init__(core, tokenizer)
        self.tokenizer, self.calls, self.template_kwargs = tokenizer, calls, template_kwargs

    async def do_proxy(self, request, path, *, body, headers):
        payload = json.loads(body)
        turn = len(self.bodies)
        name, arguments = self.calls[turn]
        call = {"id": f"call-{turn + 1}", "type": "function", "function": {"name": name, "arguments": arguments}}
        reply = {"role": "assistant", "content": "", "reasoning_content": f"Step {turn + 1}.", "tool_calls": [call]}
        messages = normalize_tool_arguments(payload["messages"], "dict")
        render = dict(tools=payload["tools"], tokenize=False, **self.template_kwargs)
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, **render)
        text = self.tokenizer.apply_chat_template([*messages, reply], **render)[len(prompt) :]
        self.output_ids = self.tokenizer.encode(text[: text.rindex("<|im_end|>") + 10], add_special_tokens=False)
        result = await super().do_proxy(request, path, body=body, headers=headers)
        response = json.loads(result["response_body"])
        reply["tool_calls"] = [call | {"function": {"name": name, "arguments": json.dumps(arguments)}}]
        response["choices"][0]["message"] = reply
        result["response_body"] = json.dumps(response)
        return result


def _episode(monkeypatch, tmp_path, core, tokenizer, path, tito_model):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _set_fakes(monkeypatch, tmp_path)
    engine = _CanonicalPlannerEngine(core, tokenizer, CALLS, resolve_fixed_chat_template(tito_model)[1])
    session, registry = _make_core("v2", path, tokenizer, engine, tmp_path / "media", tito_model=tito_model)
    session_id, result = _run_episode(monkeypatch, engine, session, registry)
    return session, session_id, result


def _sft_sample(tokenizer, processor_core, result):
    messages = json.loads(json.dumps(result["video_gen"]["messages"]))
    tokens, mask = render_episode(tokenizer, messages, tools_for(result["video_gen"]["video_mode"]), **QWEN35_KWARGS)
    images = [str(Path(image["path"]).resolve()) for image in result["video_gen"]["images"]]
    return expand_images(processor_core, tokens, mask, images, tokenizer.convert_tokens_to_ids(IMAGE_PAD))


def test_a_recorded_episode_renders_to_the_sample_the_session_server_built(
    monkeypatch, tmp_path, tokenizer, processor_core, model_path
):
    session, session_id, result = _episode(monkeypatch, tmp_path, processor_core, tokenizer, model_path, "qwen35")
    (sample,) = _samples(session, session_id, "v2")

    tokens, mask = _sft_sample(tokenizer, processor_core, result)

    assert tokens == sample.tokens
    assert mask[-sample.response_length :] == sample.loss_mask and not any(mask[: -sample.response_length])
    assert tokens.count(tokenizer.convert_tokens_to_ids(IMAGE_PAD)) > 2 and sum(mask) > 3 * 5
    # Decoded back, the trained spans are the three replies, each through its end-of-turn token.
    trained = tokenizer.decode([t for t, m in zip(tokens, mask, strict=True) if m])
    assert [f"Step {n}." in trained for n in (1, 2, 3)] == [True] * 3 and trained.count("<|im_end|>") == 3


def test_a_qwen38_planners_episode_renders_to_what_the_9b_would_have_been_trained_on(
    monkeypatch, tmp_path, tokenizer, processor_core, model_path, qwen38
):
    student_session, student_id, _ = _episode(
        monkeypatch, tmp_path / "student", processor_core, tokenizer, model_path, "qwen35"
    )
    (student_sample,) = _samples(student_session, student_id, "v2")
    # The same turns played by the 27B under its own template and tokens: only its recorded messages carry over.
    _, _, teacher = _episode(monkeypatch, tmp_path / "teacher", qwen38.core, qwen38.tokenizer, qwen38.path, "qwen38small")

    tokens, _ = _sft_sample(tokenizer, processor_core, teacher)

    assert tokens == student_sample.tokens


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
            3: [(0.62, False), (0.6, False), (0.58, False), (0.4, False)],  # top within half a std
        },
    )

    kept = sft_data.select(sft_data._episodes(tmp_path), margin=0.5)

    assert [episode["request_id"] for episode in kept] == ["req-0-0"]


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

    assert args.loss_type == "sft_loss" and args.debug_train_only and args.num_epoch == 2 and args.lr == 1e-5
    assert args.prompt_data == "/sft/train.jsonl" and args.tool_key == "tools" and args.tito_model == "qwen35"
    assert args.rollout_function_path.endswith("keyframe.sft_rollout.generate_rollout")
    assert args.actor_num_gpus_per_node == 8 and args.save_hf.endswith("/test/hf")


def test_an_rl_run_starts_from_the_sft_run_and_keeps_it_as_its_reference(monkeypatch, tmp_path):
    sft = tmp_path / "sft"
    sft.mkdir()
    # The parser reads the model config from --hf-checkpoint, which _config writes at the base planner's path.
    (sft / "hf").symlink_to(tmp_path / "Qwen3.5-9B")

    args = _config(monkeypatch, tmp_path, init_checkpoint=str(sft), num_gpus_per_node=8, video_mode="h3-ref")

    assert args.hf_checkpoint == f"{sft}/hf" and args.ref_load == f"{sft}/checkpoints"
    assert args.loss_type != "sft_loss" and not args.debug_train_only
