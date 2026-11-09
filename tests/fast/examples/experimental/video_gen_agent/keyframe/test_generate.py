"""The keyframe generate hook: it hands both generators' routers to the env and is otherwise agentic_tool_call's."""

import argparse
import os
from types import SimpleNamespace

import pytest
from examples.experimental.video_gen_agent.keyframe import generate as hook
from examples.experimental.video_gen_agent.keyframe.generator import URL_ENV as IMAGE_URL_ENV
from examples.experimental.video_gen_agent.multiturn.generator import URL_ENV as VIDEO_URL_ENV

from miles.rollout.generate_hub import agentic_tool_call

_ROUTERS = {"policy": ("10.0.0.1", 3000), "image": ("10.0.0.2", 3001), "video": ("10.0.0.3", 3002)}


def _input(**models):
    args = SimpleNamespace(
        video_generator_model=models.get("video", "video"),
        image_generator_model=models.get("image", "image"),
        sglang_model_routers=_ROUTERS,
        sglang_router_ip="10.0.0.1",
        sglang_router_port=3000,
    )
    return SimpleNamespace(args=args)


@pytest.fixture
def no_router_env(monkeypatch):
    monkeypatch.delenv(VIDEO_URL_ENV, raising=False)
    monkeypatch.delenv(IMAGE_URL_ENV, raising=False)
    return monkeypatch


async def test_the_env_gets_both_routers_and_the_episode_runs_as_agentic_tool_call(no_router_env):
    calls = []

    async def fake_generate(input):
        calls.append((input, os.environ[VIDEO_URL_ENV], os.environ[IMAGE_URL_ENV]))
        return "output"

    no_router_env.setattr(agentic_tool_call, "generate", fake_generate)
    input = _input()

    assert await hook.generate(input) == "output"
    assert calls == [(input, "http://10.0.0.3:3002", "http://10.0.0.2:3001")]


async def test_a_model_name_with_no_router_is_refused_rather_than_sent_to_the_policy(no_router_env):
    with pytest.raises(ValueError, match="no --sglang-config model is named 'images'"):
        await hook.generate(_input(image="images"))


async def test_a_different_router_than_the_env_already_has_is_refused(no_router_env):
    no_router_env.setenv(IMAGE_URL_ENV, "http://10.0.0.9:1")

    with pytest.raises(ValueError, match="the 'image' model's router is http://10.0.0.2:3001"):
        await hook.generate(_input())


def test_the_hook_takes_agentic_tool_calls_arguments_and_both_generator_models():
    parser = argparse.ArgumentParser()
    hook.generate.add_arguments(parser)

    defaults = parser.parse_args([])
    args = parser.parse_args(["--image-generator-model", "qwen", "--custom-agent-function-mode", "inline"])

    assert (defaults.video_generator_model, defaults.image_generator_model) == ("video", "image")
    assert (args.image_generator_model, args.custom_agent_function_mode) == ("qwen", "inline")
    assert hook.generate.reads_sample_agent_fields
