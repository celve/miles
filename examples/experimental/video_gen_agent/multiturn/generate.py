"""``--custom-generate-function-path examples.experimental.video_gen_agent.multiturn.generate.generate``: the env's
``agentic_tool_call.generate``, with the frozen generator's router address handed to the env, which never sees args.

``--video-generator-model`` names the ``--sglang-config`` model that serves the generator (default ``generator``).
"""

import argparse
import os

from examples.experimental.video_gen_agent.multiturn.generator import URL_ENV
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub import agentic_tool_call
from miles.rollout.sglang_rollout import get_model_url


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    url = get_model_url(input.args, input.args.video_generator_model, "")
    # The router address is fixed for the run; worker and sandbox modes forward AGENT_* vars to the env.
    if os.environ.setdefault(URL_ENV, url) != url:
        raise ValueError(f"{URL_ENV} is {os.environ[URL_ENV]}, but the generator's router is {url}")
    return await agentic_tool_call.generate(input)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    agentic_tool_call.generate.add_arguments(parser)
    parser.add_argument(
        "--video-generator-model",
        type=str,
        default="generator",
        help="The --sglang-config model that serves the frozen generator.",
    )


generate.add_arguments = _add_arguments
generate.reads_sample_agent_fields = agentic_tool_call.generate.reads_sample_agent_fields
