"""The multi-turn generate hook: it hands the generator's router to the env and is otherwise agentic_tool_call's."""

import argparse
import os
from types import SimpleNamespace

import pytest
from examples.experimental.video_gen_agent.multiturn import generate as hook
from examples.experimental.video_gen_agent.multiturn.generator import URL_ENV

from miles.rollout.generate_hub import agentic_tool_call


def _input(router):
    args = SimpleNamespace(video_generator_model="generator", sglang_model_routers={"generator": router})
    return SimpleNamespace(args=args)


async def test_the_env_gets_the_generators_router_and_the_episode_runs_as_agentic_tool_call(monkeypatch):
    monkeypatch.delenv(URL_ENV, raising=False)
    calls = []

    async def fake_generate(input):
        calls.append((input, os.environ[URL_ENV]))
        return "output"

    monkeypatch.setattr(agentic_tool_call, "generate", fake_generate)
    input = _input(("10.0.0.2", 3001))

    assert await hook.generate(input) == "output"
    assert calls == [(input, "http://10.0.0.2:3001")]


async def test_a_different_router_than_the_env_already_has_is_refused(monkeypatch):
    monkeypatch.setenv(URL_ENV, "http://10.0.0.9:1")

    with pytest.raises(ValueError, match="generator's router is http://10.0.0.2:3001"):
        await hook.generate(_input(("10.0.0.2", 3001)))


def test_the_hook_takes_agentic_tool_calls_arguments_and_the_generator_model():
    parser = argparse.ArgumentParser()
    hook.generate.add_arguments(parser)

    args = parser.parse_args(["--video-generator-model", "wan", "--custom-agent-function-mode", "inline"])

    assert (args.video_generator_model, args.custom_agent_function_mode) == ("wan", "inline")
    assert hook.generate.reads_sample_agent_fields
