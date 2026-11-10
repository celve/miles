"""Reference mode: the video is made from one or more watched images as references (LIN-2830)."""

import json
import os
from pathlib import Path

import examples.experimental.video_gen_agent.keyframe.env as env
import pytest
from examples.experimental.video_gen_agent.keyframe import reference_model
from tests.fast.examples.experimental.video_gen_agent.keyframe.test_env import (  # noqa: F401
    _call,
    _clip_request,
    _run,
    _script,
    _tool_replies,
    options,
)


@pytest.fixture
def reference(options):  # noqa: F811
    options.setenv("AGENT_VIDEO_GEN_VIDEO_MODE", "reference")
    return options


async def test_the_video_is_made_from_several_watched_images_as_references(reference, tmp_path):
    sent = _script(
        reference,
        [_call("generate_image", {"prompt": "a corgi"})],
        [_call("generate_image", {"prompt": "a sunny park"})],
        [_call("generate_video", {"image_ids": ["image_2", "image_1"], "prompt": "it runs at the camera"})],
        tools=env.REFERENCE_TOOLS,
    )

    result = await _run()

    assert "ids of one or more images whose subjects and look the video should show" in sent[0][0]["content"]
    record = result["video_gen"]
    image_1, image_2 = record["images"]
    clip = _clip_request(record["video"]["path"])
    assert clip["reference_paths"] == [image_2["path"], image_1["path"]] and clip["image_path"] is None
    assert clip["prompt"] == "a sunny park a corgi it runs at the camera"
    (kept,) = (tmp_path / "kept").iterdir()
    assert record["submitted"] == {
        "clip_id": "video_1",
        "path": str(kept),
        "prompt": "a sunny park a corgi it runs at the camera",
        "prompt_words": 9,
        "image_ids": ["image_2", "image_1"],
        "image_prompts": ["a sunny park", "a corgi"],
    }
    assert record["video"]["image_ids"] == ["image_2", "image_1"] and record["video_mode"] == "reference"


@pytest.mark.parametrize(
    "arguments, error",
    [
        ({"image_id": "image_1", "prompt": "runs"}, "image_ids must be a non-empty list"),
        ({"image_ids": "image_1", "prompt": "runs"}, "image_ids must be a non-empty list"),
        ({"image_ids": [], "prompt": "runs"}, "image_ids must be a non-empty list"),
        ({"image_ids": ["image_1", 2], "prompt": "runs"}, "image_ids must be a non-empty list"),
        ({"image_ids": ["image_1", "image_1"], "prompt": "runs"}, "image_ids names an image twice"),
        ({"image_ids": ["image_1", "image_7"], "prompt": "runs"}, "unknown image_ids entry 'image_7'; generated"),
        ({"image_ids": ["image_1"]}, "generate_video needs a non-empty prompt"),
    ],
)
async def test_a_bad_image_list_gets_an_error_reply_and_spends_no_video(reference, arguments, error):
    sent = _script(
        reference,
        [_call("generate_image", {"prompt": "fox"})],
        [_call("generate_video", arguments)],
        [_call("generate_video", {"image_ids": ["image_1"], "prompt": "runs"})],
        tools=env.REFERENCE_TOOLS,
    )

    result = await _run()

    assert _tool_replies(sent[2])[-1].startswith(f"Error: {error}")
    assert result["video_gen"]["submitted"]["image_ids"] == ["image_1"]


async def test_every_reference_must_have_been_seen(reference):
    sent = _script(
        reference,
        [_call("generate_image", {"prompt": "fox"})],
        [
            _call("generate_image", {"prompt": "snow"}),
            _call("generate_video", {"image_ids": ["image_1", "image_2"], "prompt": "runs"}),
        ],
        [_call("generate_video", {"image_ids": ["image_1", "image_2"], "prompt": "runs"})],
        tools=env.REFERENCE_TOOLS,
    )

    result = await _run()

    assert _tool_replies(sent[2])[-1].startswith("Error: you have not seen image_2 yet")
    assert result["video_gen"]["submitted"]["image_ids"] == ["image_1", "image_2"]


async def test_the_word_cap_counts_every_chosen_images_prompt(reference):
    reference.setenv("AGENT_VIDEO_GEN_MAX_PROMPT_WORDS", "5")
    sent = _script(
        reference,
        [_call("generate_image", {"prompt": "a red fox"})],
        [_call("generate_image", {"prompt": "deep snow"})],
        [_call("generate_video", {"image_ids": ["image_1", "image_2"], "prompt": "it runs"})],
        [_call("generate_video", {"image_ids": ["image_1"], "prompt": "it runs"})],
        tools=env.REFERENCE_TOOLS,
    )

    result = await _run()

    assert "the chosen images' prompts and your video prompt together too" in sent[0][0]["content"]
    assert _tool_replies(sent[3])[-1].startswith(
        "Error: the prompts of image_1, image_2 and the generate_video prompt have 7 words together; the limit is 5."
    )
    assert result["video_gen"]["submitted"]["prompt"] == "a red fox it runs"


def test_the_served_directory_links_the_checkpoint_and_names_the_reference_pipeline(tmp_path):
    src = tmp_path / "Wan2.1-VACE-14B-diffusers"
    for component in ("transformer", "vae", "scheduler"):
        (src / component).mkdir(parents=True)
    (src / "transformer" / "config.json").write_text("{}")
    (src / "model_index.json").write_text(json.dumps({"_class_name": "WanVACEPipeline", "vae": ["diffusers", "x"]}))
    (src / "scheduler" / "scheduler_config.json").write_text(json.dumps({"flow_shift": 3.0, "solver_order": 2}))
    dst = tmp_path / "served"
    argv = ["--src", str(src), "--dst", str(dst), "--flow-shift", "5", "--lora", "/l/a.safetensors:transformer"]

    reference_model.main(argv)
    reference_model.main(argv)

    assert os.readlink(dst / "transformer") == str(src / "transformer") and os.readlink(dst / "vae") == str(src / "vae")
    index = json.loads((dst / "model_index.json").read_text())
    assert index == {
        "_class_name": "WanVACEReferencePipeline",
        "vae": ["diffusers", "x"],
        "lora_paths": [["/l/a.safetensors", "transformer"]],
    }
    assert json.loads((dst / "scheduler" / "scheduler_config.json").read_text()) == {"flow_shift": 5.0, "solver_order": 2}
    pipeline = Path(reference_model.__file__).with_name("reference_pipeline.py").read_text()
    assert (dst / "pipeline.py").read_text() == pipeline and "class WanVACEReferencePipeline(" in pipeline
    assert json.loads((src / "model_index.json").read_text())["_class_name"] == "WanVACEPipeline"


def test_a_lora_names_the_expert_it_merges_into():
    with pytest.raises(SystemExit):
        reference_model.main(["--src", "s", "--dst", "d", "--flow-shift", "5", "--lora", "/l/a.safetensors"])
