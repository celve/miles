"""The keyframe env's image generator wrapper against a fake multimodal_gen image engine."""

import os

import pytest
from examples.experimental.video_gen_agent.generator import GeneratorUnavailable, ImageRequest
from examples.experimental.video_gen_agent.keyframe import generator
from examples.experimental.video_gen_agent.keyframe.generator import RouterImageGenerator
from PIL import Image
from tests.fast.diffusion.test_frozen_generator import _fake_image_engine

from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


@pytest.fixture
def image_generator(monkeypatch, tmp_path):
    seen = []
    app = _fake_image_engine(channels=4, height=32, width=48, seen=seen)
    server = UvicornThreadServer(app, host="127.0.0.1", port=find_available_port(35000))
    server.start()
    monkeypatch.setenv(generator.URL_ENV, server.url)
    monkeypatch.setenv(generator.IMAGE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(generator.STEPS_ENV, "8")
    yield RouterImageGenerator(), seen, tmp_path
    server.stop()


async def test_a_request_becomes_one_png_with_the_envs_size_and_the_launchers_schedule(image_generator):
    image_generator, seen, image_dir = image_generator

    image = await image_generator.generate(
        ImageRequest(prompt="a kite", negative_prompt="", seed=3, width=48, height=32)
    )

    (body,) = seen
    assert (body["prompt"], body["seed"], body["width"], body["height"]) == ("a kite", 3, 48, 32)
    assert body["num_inference_steps"] == 8 and "guidance_scale" not in body and "negative_prompt" not in body
    assert os.path.dirname(image.path) == str(image_dir)
    with Image.open(image.path) as png:
        assert png.size == (48, 32)


async def test_a_failing_image_generator_is_an_outage_not_a_policy_failure(monkeypatch, tmp_path):
    monkeypatch.setenv(generator.URL_ENV, f"http://127.0.0.1:{find_available_port(36000)}")
    monkeypatch.setenv(generator.IMAGE_DIR_ENV, str(tmp_path))

    with pytest.raises(GeneratorUnavailable, match="image generator's router failed"):
        await RouterImageGenerator().generate(
            ImageRequest(prompt="a", negative_prompt="", seed=0, width=48, height=32)
        )
