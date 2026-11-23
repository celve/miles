from contextlib import contextmanager
from contextvars import ContextVar

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.diffusion_rollout.client import ImageClient
from miles.rollout.rm_hub import batched_async_rm
from miles.utils.retry_utils import NonRetryableError
from miles.utils.types import Sample

_image_client: ContextVar[ImageClient] = ContextVar("diffusion_image_client")


@contextmanager
def use_image_client(client: ImageClient):
    token = _image_client.set(client)
    try:
        yield
    finally:
        _image_client.reset(token)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    sample = input.sample
    try:
        client = _image_client.get()
        if input.evaluation:
            sample.diffusion = await client.generate_eval(sample, input.sampling_params)
        else:
            sample.diffusion, sample.reward = await client.generate(sample, input.sampling_params)
    except Exception as exc:
        # The client already retries HTTP failures, and refilling would advance diffusion's deterministic prompt schedule.
        raise NonRetryableError(f"diffusion generation failed for sample {sample.index}: {exc}") from exc
    sample.status = Sample.Status.COMPLETED
    if input.evaluation:
        await batched_async_rm(input.args, [sample], inplace_set_reward_field=True)
    return GenerateFnOutput(samples=sample)
