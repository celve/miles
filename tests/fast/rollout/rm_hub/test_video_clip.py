import numpy as np
import pytest

av = pytest.importorskip("av")

from miles.rollout.rm_hub.video import pickscore  # noqa: E402
from miles.rollout.rm_hub.video.clip import read_clip  # noqa: E402


def _write_clip(path, *, num_frames: int, fps: int = 8) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(num_frames):
            frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())


def test_a_clip_reads_as_every_frame_with_its_fps(tmp_path):
    _write_clip(tmp_path / "clip.mp4", num_frames=6)

    clip = read_clip(str(tmp_path / "clip.mp4"))

    assert clip.frames.shape == (6, 3, 48, 64) and clip.fps == 8.0
    assert clip.frames[0].float().mean() < clip.frames[-1].float().mean()


class _FailingPickScore:
    """Stands in for the CLIP model: raises, as an out-of-memory forward would, for the request `boom`."""

    def __init__(self, **_kwargs):
        pass

    def __call__(self, prompts, images):
        if "boom" in prompts:
            raise RuntimeError("CUDA out of memory")
        return [float(len(images))] * len(images)


def test_a_clip_whose_scoring_raises_scores_none_and_the_others_keep_their_scores(tmp_path, monkeypatch):
    _write_clip(tmp_path / "clip.mp4", num_frames=6)
    monkeypatch.setattr(pickscore, "PickScoreScorer", _FailingPickScore)
    scorer = pickscore.VideoPickScoreScorer(device=None, processor_path="", model_path="", num_frames=4)
    clip = str(tmp_path / "clip.mp4")

    scores = scorer.score_batch([clip, clip, str(tmp_path / "missing.mp4")], ["a dog", "boom", "a cat"])

    assert scores == [4.0, None, None]


@pytest.mark.parametrize("content", [None, b"not a video"])
def test_a_missing_or_unreadable_clip_reads_as_none(tmp_path, content):
    path = tmp_path / "clip.mp4"
    if content is not None:
        path.write_bytes(content)

    assert read_clip(str(path)) is None
