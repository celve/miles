"""The pairwise judge without its model: its anchor lookup, both orders, and a request without an anchor."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

av = pytest.importorskip("av")

from miles.rollout.rm_hub.video import pairwise  # noqa: E402


def _write_clip(path, *, num_frames: int, shade: int) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for _ in range(num_frames):
            frame = np.full((48, 64, 3), shade, dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())


class _Processor:
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: [{"1": 0, "2": 1}[text]])

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, message, tokenize, add_generation_prompt):
        return message[0]["content"][-1]["text"]

    def __call__(self, text, videos, video_metadata, **kwargs):
        self.calls.append((text[0], [len(v) for v in videos]))
        bright_first = videos[0].float().mean() > videos[1].float().mean()
        return SimpleNamespace(to=lambda device: {"bright_first": bright_first})


class _Model:
    """Prefers the brighter clip 3:1 and leans toward Video 1 on top, the bias both orders cancel."""

    def __call__(self, bright_first):
        logits = torch.zeros(1, 1, 2)
        logits[0, -1, 0 if bright_first else 1] = float(np.log(3.0))
        logits[0, -1, 0] += 0.5
        return SimpleNamespace(logits=logits)


def _scorer(anchor_dir):
    scorer = object.__new__(pairwise.VideoPairwiseScorer)
    scorer.device, scorer.anchor_dir, scorer.num_frames, scorer.max_pixels_per_frame = "cpu", str(anchor_dir), 4, 448**2
    scorer.processor, scorer.model, scorer.answer_ids = _Processor(), _Model(), [0, 1]
    return scorer


def test_a_clip_scores_its_mean_win_over_its_requests_anchor_in_both_orders(tmp_path):
    anchors = tmp_path / "anchors"
    anchors.mkdir()
    _write_clip(anchors / pairwise.anchor_name("a fox"), num_frames=6, shade=60)
    _write_clip(tmp_path / "bright.mp4", num_frames=9, shade=200)
    _write_clip(tmp_path / "dark.mp4", num_frames=9, shade=10)
    scorer = _scorer(anchors)

    bright, dark = scorer.score_batch([str(tmp_path / "bright.mp4"), str(tmp_path / "dark.mp4")], ["a fox"] * 2)

    # Video 1's logit bias of 0.5 lands on each side once, so the two scores add to 1.
    assert bright == pytest.approx(0.75, abs=0.02) and bright + dark == pytest.approx(1.0)
    assert [frames for _, frames in scorer.processor.calls] == [[4, 4]] * 4
    assert all('"a fox"' in text for text, _ in scorer.processor.calls)


def test_a_request_without_an_anchor_scores_none(tmp_path):
    _write_clip(tmp_path / "clip.mp4", num_frames=6, shade=100)

    assert _scorer(tmp_path / "anchors").score_batch([str(tmp_path / "clip.mp4")], ["a red car"]) == [None]
    assert pairwise.anchor_name("a red car") == pairwise.anchor_name("a red car") != pairwise.anchor_name("a fox")
