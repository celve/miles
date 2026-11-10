"""The keyframe env's episode loop, against a scripted chat endpoint and the fake image and clip generators."""

import asyncio
import json
from pathlib import Path

import examples.experimental.video_gen_agent.keyframe.env as env
import pytest
from examples.experimental.video_gen_agent.generator import Clip, ClipRequest
from examples.experimental.video_gen_agent.reward import reward_func, stub_score
from PIL import Image

from miles.rollout.agentic.agent_function import InfraAbort
from miles.utils.types import Sample

REQUEST = "A red fox runs through fresh snow at dawn."
_FAKES = "examples.experimental.video_gen_agent.generator"


class _HangingGenerator:
    async def generate(self, request: ClipRequest) -> Clip:
        await asyncio.sleep(3600)


class _BrokenGenerator:
    async def generate(self, request: ClipRequest) -> Clip:
        raise KeyError("fps")


@pytest.fixture
def options(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_VIDEO_GEN_GENERATOR", f"{_FAKES}.FakeClipGenerator")
    monkeypatch.setenv("AGENT_VIDEO_GEN_IMAGE_GENERATOR", f"{_FAKES}.FakeImageGenerator")
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("AGENT_VIDEO_GEN_SUBMITTED_DIR", str(tmp_path / "kept"))
    return monkeypatch


def _call(name: str, arguments) -> dict:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": raw}}


def _script(monkeypatch, *replies) -> list[list[dict]]:
    """Answer each chat request with the next reply: a list of tool calls, plain text, or ``"length"``."""
    sent: list[list[dict]] = []

    async def fake_post(url, payload):
        assert url == "http://session/v1/chat/completions"
        assert payload["tools"] == env.TOOLS and payload["temperature"] == 0.5
        sent.append(json.loads(json.dumps(payload["messages"])))
        reply = replies[len(sent) - 1]
        if reply == "length":
            return {"choices": [{"message": {"role": "assistant", "content": "..."}, "finish_reason": "length"}]}
        if isinstance(reply, str):
            return {"choices": [{"message": {"role": "assistant", "content": reply, "tool_calls": None}}]}
        return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": reply}}]}

    monkeypatch.setattr(env, "post", fake_post)
    return sent


async def _run() -> dict:
    return await env.run_agent("http://session", [{"role": "user", "content": REQUEST}], {"temperature": 0.5}, {})


def _tool_replies(messages: list[dict]) -> list:
    return [m["content"] for m in messages if m["role"] == "tool"]


def _clip_request(path: str) -> dict:
    return json.loads(Path(path).read_bytes())


async def test_images_are_shown_then_the_video_from_the_chosen_image_is_submitted(options, tmp_path):
    options.setenv("AGENT_VIDEO_GEN_VIDEO_MODE", "i2v")
    options.setenv("AGENT_VIDEO_GEN_IMAGE_MODEL", "Qwen/Qwen-Image-2.1-Turbo")
    sent = _script(
        options,
        [_call("generate_image", {"prompt": "fox, wide shot", "seed": 4})],
        [_call("generate_image", {"prompt": "fox, close-up", "negative_prompt": "blur"})],
        [_call("generate_video", {"image_id": "image_1", "prompt": "the fox runs left, camera pans", "seed": 9})],
    )

    result = await _run()

    system = sent[0][0]["content"]
    assert "832x480" in system and "2 seconds at 16 fps" in system and "3 images, 1 video and 6 turns" in system
    first, second = _tool_replies(sent[2])
    assert first[0] == {"type": "text", "text": "Generated image_1 from seed 4. Images left: 2. Turns left: 5."}
    assert second[0]["text"] == "Generated image_2 from seed 1. Images left: 1. Turns left: 4."
    record = result["video_gen"]
    image_1, image_2 = record["images"]
    assert first[1] == {"type": "image_url", "image_url": {"url": f"file://{image_1['path']}"}}
    with Image.open(image_2["path"]) as image:
        assert image.size == (832, 480)
        assert json.loads(image.text["request"])["negative_prompt"] == "blur"
    clip = _clip_request(record["video"]["path"])
    assert clip["image_path"] == image_1["path"] and clip["prompt"] == "the fox runs left, camera pans"
    assert (clip["seconds"], clip["width"], clip["height"], clip["fps"], clip["seed"]) == (2.0, 832, 480, 16, 9)
    (kept,) = (tmp_path / "kept").iterdir()
    assert kept.read_bytes() == Path(record["video"]["path"]).read_bytes()
    assert record["submitted"] == {
        "clip_id": "video_1",
        "path": str(kept),
        "prompt": "the fox runs left, camera pans",
        "prompt_words": 6,
        "image_id": "image_1",
        "image_prompt": "fox, wide shot",
    }
    assert record["request"] == REQUEST and record["turns"] == 3 and len(record["turn_stats"]) == 3
    assert (record["video_mode"], record["image_model"]) == ("i2v", "Qwen/Qwen-Image-2.1-Turbo")


async def test_preview_mode_is_the_default_and_makes_the_video_from_text(options):
    options.setenv("AGENT_VIDEO_GEN_VIDEO_MODEL", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    sent = _script(
        options,
        [_call("generate_image", {"prompt": "a fox in snow"})],
        [_call("generate_video", {"image_id": "image_1", "prompt": "it runs"})],
    )

    result = await _run()

    clip = _clip_request(result["video_gen"]["video"]["path"])
    assert clip["image_path"] is None and clip["prompt"] == "a fox in snow it runs"
    assert result["video_gen"]["submitted"]["prompt"] == "a fox in snow it runs"
    assert "without starting from it" in sent[0][0]["content"]
    record = result["video_gen"]
    assert (record["video_mode"], record["video_model"]) == ("preview", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")


async def test_a_video_from_an_image_made_in_the_same_turn_waits_until_the_planner_has_seen_it(options):
    sent = _script(
        options,
        [
            _call("generate_image", {"prompt": "fox"}),
            _call("generate_video", {"image_id": "image_1", "prompt": "runs"}),
        ],
        [_call("generate_video", {"image_id": "image_1", "prompt": "runs"})],
    )

    result = await _run()

    image_reply, video_reply = _tool_replies(sent[1])
    assert image_reply[1]["type"] == "image_url"
    assert video_reply == (
        "Error: you have not seen image_1 yet; look at it, then call generate_video. Images left: 2. Turns left: 5."
    )
    assert len(sent) == 2 and result["video_gen"]["submitted"]["image_id"] == "image_1"


async def test_the_video_ends_the_episode_and_later_calls_in_its_turn_are_not_run(options):
    sent = _script(
        options,
        [_call("generate_image", {"prompt": "fox"})],
        [
            _call("generate_video", {"image_id": "image_1", "prompt": "runs"}),
            _call("generate_image", {"prompt": "never made"}),
        ],
    )

    result = await _run()

    assert len(sent) == 2 and result["video_gen"]["turns"] == 2
    assert [i["prompt"] for i in result["video_gen"]["images"]] == ["fox"]
    assert result["video_gen"]["submitted"]["image_id"] == "image_1"


async def test_an_image_past_the_budget_is_refused(options):
    options.setenv("AGENT_VIDEO_GEN_MAX_IMAGES", "1")
    sent = _script(
        options,
        [_call("generate_image", {"prompt": "fox"}), _call("generate_image", {"prompt": "another"})],
        [_call("generate_video", {"image_id": "image_1", "prompt": "runs"})],
    )

    result = await _run()

    assert _tool_replies(sent[1])[1] == (
        "Error: no images left. Call generate_video with one of image_1. Images left: 0. Turns left: 5."
    )
    assert len(result["video_gen"]["images"]) == 1


async def test_an_unknown_image_names_the_images_made_and_the_turn_counts(options):
    sent = _script(
        options,
        [_call("generate_video", {"image_id": "image_1", "prompt": "runs"})],
        [_call("generate_image", {"prompt": "fox"})],
        [_call("generate_video", {"image_id": "image_7", "prompt": "runs"})],
        [_call("generate_video", {"image_id": "image_1", "prompt": "runs"})],
    )

    result = await _run()

    assert _tool_replies(sent[1]) == [
        "Error: unknown image_id 'image_1'; generated images: none yet; call generate_image first. "
        "Images left: 3. Turns left: 5."
    ]
    assert _tool_replies(sent[3])[-1] == (
        "Error: unknown image_id 'image_7'; generated images: image_1. Images left: 2. Turns left: 3."
    )
    assert result["video_gen"]["submitted"]["image_id"] == "image_1" and result["video_gen"]["turns"] == 4


@pytest.mark.parametrize(
    "name, arguments, error",
    [
        ("generate_image", "{not json", "Error: the arguments of generate_image are not valid JSON."),
        ("generate_image", {"prompt": " "}, "Error: generate_image needs a non-empty prompt."),
        ("generate_image", {"prompt": "fox", "seed": True}, "Error: seed must be an integer."),
        ("generate_image", {"prompt": "fox", "negative_prompt": 3}, "Error: negative_prompt must be a string."),
        ("submit", {"clip_id": "clip_1"}, "Error: unknown tool 'submit'; the tools are generate_image and"),
    ],
)
async def test_a_bad_call_gets_an_error_reply_and_the_episode_goes_on(options, name, arguments, error):
    options.setenv("AGENT_VIDEO_GEN_MAX_TURNS", "2")
    sent = _script(options, [_call(name, arguments)], "I give up.")

    await _run()

    assert _tool_replies(sent[1])[0].startswith(error)


async def test_a_bad_video_call_spends_no_video(options):
    sent = _script(
        options,
        [_call("generate_image", {"prompt": "fox"})],
        [_call("generate_video", {"image_id": "image_1"})],
        [_call("generate_video", {"image_id": "image_1", "prompt": "runs"})],
    )

    result = await _run()

    assert _tool_replies(sent[2])[-1].startswith("Error: generate_video needs a non-empty prompt.")
    assert result["video_gen"]["submitted"]["clip_id"] == "video_1"


async def test_an_episode_out_of_turns_ends_without_a_video_and_scores_0(options):
    options.setenv("AGENT_VIDEO_GEN_MAX_TURNS", "2")
    sent = _script(options, "Let me think.", [_call("generate_image", {"prompt": "fox"})])

    result = await _run()

    assert sent[1][-1] == {
        "role": "user",
        "content": "Call generate_image or generate_video. Images left: 3. Turns left: 1.",
    }
    assert result["video_gen"]["submitted"] is None and result["video_gen"]["video"] is None
    sample = Sample(metadata={"video_gen": result["video_gen"]})
    args = type("Args", (), {"video_reward_pool": "judge"})
    assert await reward_func(args, [sample], judge_pool=None) == [0.0]


async def test_a_turn_cut_at_the_token_limit_ends_the_episode(options):
    sent = _script(options, [_call("generate_image", {"prompt": "fox"})], "length")

    result = await _run()

    assert len(sent) == 2 and result["video_gen"]["submitted"] is None and result["video_gen"]["turns"] == 2


async def test_the_submitted_clip_is_what_the_reward_scores(options):
    _script(
        options,
        [_call("generate_image", {"prompt": "fox"})],
        [_call("generate_video", {"image_id": "image_1", "prompt": "runs"})],
    )
    result = await _run()
    sample = Sample(metadata={"video_gen": result["video_gen"]})
    seen = []

    class _Pool:
        async def score(self, outputs, prompts):
            seen.append((outputs, prompts))
            return [stub_score(path) for path in outputs]

    args = type("Args", (), {"video_reward_pool": "judge"})

    (score,) = await reward_func(args, [sample], judge_pool=_Pool())

    path = result["video_gen"]["submitted"]["path"]
    assert seen == [([path], [REQUEST])] and score == stub_score(path)


@pytest.mark.parametrize("generator", ["GENERATOR", "IMAGE_GENERATOR"])
async def test_a_generator_outage_aborts_the_group(options, generator):
    options.setenv(f"AGENT_VIDEO_GEN_{generator}", f"{__name__}._HangingGenerator")
    options.setenv("AGENT_VIDEO_GEN_TIMEOUT_S", "0.05")
    _script(
        options,
        [_call("generate_image", {"prompt": "fox"})],
        [_call("generate_video", {"image_id": "image_1", "prompt": "runs"})],
    )

    with pytest.raises(InfraAbort) as aborted:
        await _run()

    assert aborted.value.exit_status == "GeneratorUnavailable"


async def test_an_unavailable_generator_aborts_the_group(options):
    options.setenv("AGENT_VIDEO_GEN_FAKE_UNAVAILABLE", "1")
    _script(options, [_call("generate_image", {"prompt": "fox"})])

    with pytest.raises(InfraAbort) as aborted:
        await _run()

    assert aborted.value.exit_status == "GeneratorUnavailable"


@pytest.mark.parametrize(
    "variable, value",
    [
        ("AGENT_VIDEO_GEN_GENERATOR", f"{__name__}._BrokenGenerator"),
        ("AGENT_VIDEO_GEN_MAX_IMAGES", "4"),
        ("AGENT_VIDEO_GEN_VIDEO_MODE", "t2v"),
        ("AGENT_VIDEO_GEN_IMAGE_GENERATOR", ""),
    ],
)
async def test_an_env_fault_aborts_the_group_rather_than_scoring_0(options, variable, value):
    options.setenv(variable, value)
    _script(
        options,
        [_call("generate_image", {"prompt": "fox"})],
        [_call("generate_video", {"image_id": "image_1", "prompt": "runs"})],
    )

    with pytest.raises(InfraAbort) as aborted:
        await _run()

    assert aborted.value.exit_status == "EnvError"


async def test_no_prompt_is_capped_by_default(options):
    sent = _script(
        options,
        [_call("generate_image", {"prompt": " ".join(["fox"] * 300)})],
        [_call("generate_video", {"image_id": "image_1", "prompt": " ".join(["runs"] * 300)})],
    )

    result = await _run()

    assert "words" not in sent[0][0]["content"]
    assert result["video_gen"]["submitted"]["prompt_words"] == 600


async def test_a_prompt_over_the_word_cap_gets_an_error_reply_and_spends_a_turn_but_no_generation(options):
    options.setenv("AGENT_VIDEO_GEN_MAX_PROMPT_WORDS", "120")
    words = lambda n: " ".join(["fox"] * n)  # noqa: E731
    sent = _script(
        options,
        [_call("generate_image", {"prompt": words(121)})],
        [_call("generate_image", {"prompt": words(119)})],
        [_call("generate_video", {"image_id": "image_1", "prompt": "runs fast"})],
        [_call("generate_video", {"image_id": "image_1", "prompt": "runs"})],
    )

    result = await _run()

    assert "Each prompt is at most 120 words, and the chosen image's prompt and your video" in sent[0][0]["content"]
    assert _tool_replies(sent[1]) == [
        "Error: the generate_image prompt has 121 words; the limit is 120. Images left: 3. Turns left: 5."
    ]
    # In preview mode the generator reads the image's prompt and the video prompt together.
    assert _tool_replies(sent[3])[-1] == (
        "Error: image_1's prompt and the generate_video prompt have 121 words together; the limit is 120. "
        "Images left: 2. Turns left: 3."
    )
    record = result["video_gen"]
    assert len(record["images"]) == 1 and record["submitted"]["prompt"] == f"{words(119)} runs"
    assert len(_clip_request(record["video"]["path"])["prompt"].split()) == 120 and record["turns"] == 4


async def test_in_i2v_mode_the_cap_is_on_the_video_prompt_alone_and_is_an_option(options):
    options.setenv("AGENT_VIDEO_GEN_VIDEO_MODE", "i2v")
    options.setenv("AGENT_VIDEO_GEN_MAX_PROMPT_WORDS", "3")
    sent = _script(
        options,
        [_call("generate_image", {"prompt": "a red fox"})],
        [_call("generate_video", {"image_id": "image_1", "prompt": "it runs very fast"})],
        [_call("generate_video", {"image_id": "image_1", "prompt": "it runs fast"})],
    )

    result = await _run()

    assert "Each prompt is at most 3 words. An episode" in sent[0][0]["content"]
    assert _tool_replies(sent[2])[-1].startswith("Error: the generate_video prompt has 4 words; the limit is 3.")
    assert result["video_gen"]["submitted"]["prompt"] == "it runs fast"
