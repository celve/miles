from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from miles.utils.types import Sample

StepStrategy = Callable[[Namespace, Sample, int, int], tuple[list[int] | None, list[int] | None]]


@dataclass(frozen=True)
class ImageRequestPlan:
    group_key: tuple[int, int]
    position: int
    size: int
    seed: int
    sde_steps: list[int] | None
    returned_latents: list[int] | None


def window_group_for(args: Namespace, sample: Sample, index_offset: int) -> tuple[tuple[int, int], int]:
    n, m = args.n_samples_per_prompt, args.diffusion_microgroup_size
    window_group, position = divmod(sample.index - index_offset - sample.group_index * n, m)
    return (sample.group_index, window_group * m), position


def plan_for(
    args: Namespace, sample: Sample, num_steps: int, step_strategy: StepStrategy, index_offset: int = 0
) -> ImageRequestPlan:
    group_key, position = window_group_for(args, sample, index_offset)
    group_index, start = group_key
    n = args.n_samples_per_prompt
    seed = (args.rollout_seed + group_index * n) % (2**31) + start
    first = prompt_sample_at(sample, sample.index - position)
    steps, returned_latents = step_strategy(args, first, num_steps, seed)
    return ImageRequestPlan(
        group_key, position, min(args.diffusion_microgroup_size, n - start), seed, steps, returned_latents
    )


def prompt_sample_at(sample: Sample, index: int) -> Sample:
    return Sample(
        group_index=sample.group_index,
        index=index,
        prompt=sample.prompt,
        metadata=deepcopy(sample.metadata),
        label=deepcopy(sample.label),
    )
