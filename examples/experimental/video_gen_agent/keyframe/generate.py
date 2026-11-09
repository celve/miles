"""``--custom-generate-function-path examples.experimental.video_gen_agent.keyframe.generate.generate``: the env's
``agentic_tool_call.generate``, with both frozen generators' router addresses handed to the env, which never sees args.

``--video-generator-model`` and ``--image-generator-model`` name the ``--sglang-config`` models that serve them
(default ``video`` and ``image``).
"""

import argparse
import os

from examples.experimental.video_gen_agent.keyframe.generator import URL_ENV as IMAGE_URL_ENV
from examples.experimental.video_gen_agent.multiturn import generate as multiturn
from examples.experimental.video_gen_agent.multiturn.generator import URL_ENV as VIDEO_URL_ENV
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub import agentic_tool_call
from miles.rollout.sglang_rollout import get_model_url


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    for env, model in (
        (VIDEO_URL_ENV, input.args.video_generator_model),
        (IMAGE_URL_ENV, input.args.image_generator_model),
    ):
        # get_model_url falls back to the policy's router for a name it does not know.
        if model not in (input.args.sglang_model_routers or {}):
            raise ValueError(f"no --sglang-config model is named {model!r}")
        url = get_model_url(input.args, model, "")
        # The router addresses are fixed for the run; worker and sandbox modes forward AGENT_* vars to the env.
        if os.environ.setdefault(env, url) != url:
            raise ValueError(f"{env} is {os.environ[env]}, but the {model!r} model's router is {url}")
    return await agentic_tool_call.generate(input)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    multiturn.generate.add_arguments(parser)
    parser.set_defaults(video_generator_model="video")
    parser.add_argument(
        "--image-generator-model",
        type=str,
        default="image",
        help="The --sglang-config model that serves the frozen image generator.",
    )


generate.add_arguments = _add_arguments
generate.reads_sample_agent_fields = agentic_tool_call.generate.reads_sample_agent_fields
