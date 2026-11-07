"""The env's generator wrapper against fake multimodal_gen servers behind the real Miles router."""

import os
from dataclasses import replace

import pytest
from examples.experimental.video_gen_agent.generator import ClipRequest, GeneratorUnavailable
from examples.experimental.video_gen_agent.multiturn import generator
from examples.experimental.video_gen_agent.multiturn.generator import RouterClipGenerator, generator_frames
from tests.fast.examples.experimental.video_gen_agent.phase0.test_reward import generator_router  # noqa: F401

_HEIGHT, _WIDTH = 32, 48


def _request(prompt: str, *, seconds: float, fps: int = 4) -> ClipRequest:
    return ClipRequest(
        prompt=prompt, negative_prompt="blur", seed=7, seconds=seconds, width=_WIDTH, height=_HEIGHT, fps=fps
    )


@pytest.fixture
def clip_generator(monkeypatch, tmp_path, generator_router):  # noqa: F811
    (host, port), seen = generator_router
    monkeypatch.setenv(generator.URL_ENV, f"http://{host}:{port}")
    monkeypatch.setenv(generator.CLIP_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(generator.STEPS_ENV, "3")
    monkeypatch.setenv(generator.GUIDANCE_ENV, "1.0")
    return RouterClipGenerator(), seen, tmp_path


@pytest.mark.parametrize(("seconds", "fps", "frames"), [(2, 24, 49), (2, 4, 9), (1, 16, 17), (0.1, 4, 5)])
def test_frames_round_to_the_vae_stride(seconds, fps, frames):
    assert generator_frames(_request("a", seconds=seconds, fps=fps)) == frames


async def test_a_request_becomes_one_clip_with_the_envs_shape_and_the_launchers_schedule(clip_generator):
    clip_generator, seen, clip_dir = clip_generator

    clip = await clip_generator.generate(_request("a boat", seconds=2))

    (body,) = seen
    assert (body["prompt"], body["negative_prompt"], body["seed"]) == ("a boat", "blur", 7)
    assert (body["num_frames"], body["fps"], body["width"], body["height"]) == (9, 4, _WIDTH, _HEIGHT)
    assert (body["num_inference_steps"], body["guidance_scale"]) == (3, 1.0)
    assert os.path.dirname(clip.path) == str(clip_dir) and os.path.getsize(clip.path) > 0
    assert "image_path" not in body


async def test_an_extension_sends_the_frame_it_continues_from(clip_generator):
    clip_generator, seen, _ = clip_generator

    await clip_generator.generate(replace(_request("it sinks", seconds=2), image_path="/clips/a.mp4.last.png"))

    (body,) = seen
    assert body["image_path"] == ["/clips/a.mp4.last.png"] and body["prompt"] == "it sinks"


async def test_a_failing_generator_is_an_outage_not_a_policy_failure(clip_generator):
    clip_generator, _, _ = clip_generator

    with pytest.raises(GeneratorUnavailable, match="router failed"):
        await clip_generator.generate(_request("an outage", seconds=2))
