"""The frozen image generator behind the keyframe env's ``ImageGenerator`` interface:
``AGENT_VIDEO_GEN_IMAGE_GENERATOR=examples.experimental.video_gen_agent.keyframe.generator.RouterImageGenerator``.
The video generator is the multi-turn recipe's ``RouterClipGenerator``, which makes image-to-video clips too.

Options come from ``AGENT_VIDEO_GEN_*`` env vars, as the env's own do:

  AGENT_VIDEO_GEN_IMAGE_GENERATOR_URL        the image model's router; ``generate.generate`` sets it (required)
  AGENT_VIDEO_GEN_IMAGE_DIR                  where images are written, a path the engines and trainer can read (required)
  AGENT_VIDEO_GEN_IMAGE_NUM_INFERENCE_STEPS  denoising steps (default: the model's)
  AGENT_VIDEO_GEN_IMAGE_GUIDANCE_SCALE       CFG scale (default: the model's)
"""

import os

import httpx

from examples.experimental.video_gen_agent.generator import GeneratorUnavailable, Image, ImageRequest
from miles.diffusion.api_client import DiffusionApiClient

URL_ENV = "AGENT_VIDEO_GEN_IMAGE_GENERATOR_URL"
IMAGE_DIR_ENV = "AGENT_VIDEO_GEN_IMAGE_DIR"
STEPS_ENV = "AGENT_VIDEO_GEN_IMAGE_NUM_INFERENCE_STEPS"
GUIDANCE_ENV = "AGENT_VIDEO_GEN_IMAGE_GUIDANCE_SCALE"


class RouterImageGenerator:
    def __init__(self):
        self.client = DiffusionApiClient(server_url=os.environ[URL_ENV])
        self.image_dir = os.environ[IMAGE_DIR_ENV]
        self.num_inference_steps = int(os.environ[STEPS_ENV]) if os.environ.get(STEPS_ENV) else None
        self.guidance_scale = float(os.environ[GUIDANCE_ENV]) if os.environ.get(GUIDANCE_ENV) else None

    async def generate(self, request: ImageRequest) -> Image:
        try:
            path = await self.client.generate_image(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or None,
                seed=request.seed,
                width=request.width,
                height=request.height,
                output_dir=self.image_dir,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
            )
        except httpx.HTTPError as error:
            raise GeneratorUnavailable(f"the image generator's router failed: {error!r}") from error
        return Image(path=path)
