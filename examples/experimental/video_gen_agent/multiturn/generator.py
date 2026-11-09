"""Item 1's frozen generator behind the env's ``ClipGenerator`` interface:
``AGENT_VIDEO_GEN_GENERATOR=examples.experimental.video_gen_agent.multiturn.generator.RouterClipGenerator``.

Options come from ``AGENT_VIDEO_GEN_*`` env vars, as the env's own do:

  AGENT_VIDEO_GEN_GENERATOR_URL          the generator model's router; ``generate.generate`` sets it (required)
  AGENT_VIDEO_GEN_CLIP_DIR               where clips are written, a path the engines and trainer can read (required)
  AGENT_VIDEO_GEN_NUM_INFERENCE_STEPS    denoising steps (default: the generator's)
  AGENT_VIDEO_GEN_GUIDANCE_SCALE         CFG scale (default: the generator's)
  AGENT_VIDEO_GEN_GUIDANCE_SCALE_2       Wan2.2-A14B's low-noise expert's CFG scale (default: the generator's)
"""

import os

import httpx

from examples.experimental.video_gen_agent.generator import Clip, ClipRequest, GeneratorUnavailable
from miles.diffusion.api_client import DiffusionApiClient

URL_ENV = "AGENT_VIDEO_GEN_GENERATOR_URL"
CLIP_DIR_ENV = "AGENT_VIDEO_GEN_CLIP_DIR"
STEPS_ENV = "AGENT_VIDEO_GEN_NUM_INFERENCE_STEPS"
GUIDANCE_ENV = "AGENT_VIDEO_GEN_GUIDANCE_SCALE"
GUIDANCE_2_ENV = "AGENT_VIDEO_GEN_GUIDANCE_SCALE_2"


def generator_frames(request: ClipRequest) -> int:
    """The request's frame count rounded to the 4k+1 that Wan2.2's VAE takes."""
    return 4 * max(1, round((request.num_frames - 1) / 4)) + 1


class RouterClipGenerator:
    def __init__(self):
        self.client = DiffusionApiClient(server_url=os.environ[URL_ENV])
        self.clip_dir = os.environ[CLIP_DIR_ENV]
        self.num_inference_steps = int(os.environ[STEPS_ENV]) if os.environ.get(STEPS_ENV) else None
        self.guidance_scale = float(os.environ[GUIDANCE_ENV]) if os.environ.get(GUIDANCE_ENV) else None
        guidance_2 = os.environ.get(GUIDANCE_2_ENV)
        self.extra_sampling_params = {"guidance_scale_2": float(guidance_2)} if guidance_2 else None

    async def generate(self, request: ClipRequest) -> Clip:
        try:
            path = await self.client.generate_clip(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or None,
                seed=request.seed,
                num_frames=generator_frames(request),
                fps=request.fps,
                width=request.width,
                height=request.height,
                output_dir=self.clip_dir,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                image_path=request.image_path,
                extra_sampling_params=self.extra_sampling_params,
            )
        except httpx.HTTPError as error:
            raise GeneratorUnavailable(f"the generator's router failed: {error!r}") from error
        return Clip(path=path)
