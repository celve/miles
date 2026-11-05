"""PickScore batches as miles_diffusion does: samples per worker call, frames per CLIP forward, a video's mean."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.diffusion.rollout import pickscore
from miles.diffusion.rollout.pickscore import PickScoreBatchScorer, pickscore_rm


@pytest.mark.parametrize(
    "total,wanted,indices", [(5, None, [0, 1, 2, 3, 4]), (5, 3, [0, 2, 4]), (5, 1, [2]), (2, 3, [0, 1])]
)
def test_frames_are_evenly_spaced(total, wanted, indices):
    assert pickscore.sample_frame_indices(total, wanted) == indices


def test_frames_are_truncated_to_uint8():
    (frame,) = pickscore.video_to_uint8_frames(torch.full((3, 1, 2, 2), 0.5))

    assert frame.shape == (2, 2, 3) and frame.dtype.name == "uint8" and int(frame[0, 0, 0]) == 127


class _ValueScorer:
    """Scores a frame by its first pixel, recording each forward's size."""

    def __init__(self):
        self.forwards = []

    def __call__(self, prompts, images):
        self.forwards.append(len(images))
        return [float(image.getpixel((0, 0))[0]) for image in images]


def test_a_worker_forwards_fixed_frame_chunks_across_samples_and_averages_each_video():
    actor = object.__new__(PickScoreBatchScorer)
    actor.frames_per_forward, actor.scorer = 4, _ValueScorer()
    videos = [torch.arange(3.0).view(1, 3, 1, 1).expand(3, 3, 1, 1) * 10, torch.full((3, 2, 1, 1), 50.0)]

    scores = actor.score_batch(videos, ["a", "b"])

    assert actor.scorer.forwards == [4, 1]
    assert scores == [10.0, 50.0]


async def test_pickscore_rm_sends_only_the_scored_frames():
    sent = []

    class Pool:
        async def score(self, outputs, prompts):
            sent.extend(output.shape[1] for output in outputs)
            return [0.5] * len(outputs)

    samples = [SimpleNamespace(diffusion=SimpleNamespace(generated_output=torch.zeros(3, 5, 2, 2)), prompt="p")]

    assert await pickscore_rm(Namespace(diffusion_pickscore_num_frames=3), samples, pickscore_pool=Pool()) == [0.5]
    assert sent == [3]
