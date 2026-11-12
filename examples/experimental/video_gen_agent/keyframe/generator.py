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
    """With ``url``, a second image model's router (the eval arms' image editor) at its own default steps and CFG;
    without it, the env's image model as its env vars set it."""

    def __init__(self, url: str | None = None):
        self.client = DiffusionApiClient(server_url=url or os.environ[URL_ENV])
        self.image_dir = os.environ[IMAGE_DIR_ENV]
        steps, guidance = (None, None) if url else (os.environ.get(STEPS_ENV), os.environ.get(GUIDANCE_ENV))
        self.num_inference_steps = int(steps) if steps else None
        self.guidance_scale = float(guidance) if guidance else None

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
                extra_sampling_params={"image_path": list(request.image_paths)} if request.image_paths else None,
            )
        except httpx.HTTPError as error:
            raise GeneratorUnavailable(f"the image generator's router failed: {error!r}") from error
        return Image(path=path)
