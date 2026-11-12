"""Phase 1's scorers without their models: OCR's unrequested-text rule, VQAScore's yes, UnifiedReward's answer
parsing, and HPSv3's checkpoint renaming."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

av = pytest.importorskip("av")

from miles.rollout.rm_hub.video import hpsv3, ocr, unified_reward, vqascore  # noqa: E402


def _write_clip(path, *, num_frames: int, fps: int = 8) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(num_frames):
            frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())


@pytest.mark.parametrize(
    "lines, request_text, found",
    [
        ([("A PANDA DRINKING", 0.99)], "A panda drinking coffee", {"apandadrinking"}),
        ([("OPEN", 0.99), ("open!", 0.95)], 'a shop sign that says "Open"', set()),
        ([("Café Open", 0.99)], "a sign reading “Café open 24 hours”", set()),
        ([("SALE", 0.79), ("x", 0.99), ("--", 0.99)], "a street", set()),
        ([("SALE", 0.9), ("Sale", 0.85), ("EXIT", 0.8)], 'a "door"', {"sale", "exit"}),
    ],
)
def test_text_counts_unless_the_request_quotes_it_or_ocr_is_unsure(lines, request_text, found):
    assert ocr.unrequested_lines(lines, request_text, min_confidence=0.8, min_chars=2) == found


class _Reader:
    """Stands in for PaddleOCR: reads `text` on every frame, as its 2.x `ocr()` returns it."""

    def __init__(self, text: str | None):
        self.text, self.frames = text, []

    def ocr(self, image, cls):
        self.frames.append(image.shape)
        return [[[[[0, 0]] * 4, (self.text, 0.97)]]] if self.text else [None]


@pytest.mark.parametrize("text, count", [("WATERMARK", 1.0), (None, 0.0)])
def test_the_ocr_scorer_counts_distinct_unrequested_lines_over_sampled_frames(tmp_path, text, count):
    _write_clip(tmp_path / "clip.mp4", num_frames=9)
    scorer = object.__new__(ocr.VideoOcrScorer)
    scorer.num_frames, scorer.min_confidence, scorer.min_chars, scorer._ocr = 4, 0.8, 2, _Reader(text)

    assert scorer.score_batch([str(tmp_path / "clip.mp4"), str(tmp_path / "missing.mp4")], ["a fox"] * 2) == [
        count,
        None,
    ]
    assert scorer._ocr.frames == [(48, 64, 3)] * 4


@pytest.mark.parametrize(
    "request_text",
    [
        "a chalkboard with the words of a poem",
        "a neon sign above the bar",
        "the company logo on a cloud with AWS icons",
        "a banner that says welcome home",
        "a t shirt printed in a modern font",
    ],
    ids=["text words", "signage", "logo or icon", "quoting verb", "print or font"],
)
def test_a_request_that_asks_for_text_is_not_read(tmp_path, request_text):
    _write_clip(tmp_path / "clip.mp4", num_frames=9)
    scorer = object.__new__(ocr.VideoOcrScorer)
    scorer.num_frames, scorer.min_confidence, scorer.min_chars, scorer._ocr = 4, 0.8, 2, _Reader("Data Pigeline")

    assert scorer.score_batch([str(tmp_path / "clip.mp4")], [request_text]) == [0.0]
    assert scorer._ocr.frames == []


@pytest.mark.parametrize("request_text", ["A panda drinking coffee", "a signal fire on a hill"])
def test_a_caption_of_the_request_still_counts_without_a_text_cue(tmp_path, request_text):
    _write_clip(tmp_path / "clip.mp4", num_frames=9)
    scorer = object.__new__(ocr.VideoOcrScorer)
    scorer.num_frames, scorer.min_confidence, scorer.min_chars, scorer._ocr = 4, 0.8, 2, _Reader("A PANDA DRINKING")

    assert scorer.score_batch([str(tmp_path / "clip.mp4")], [request_text]) == [1.0]


def test_unified_reward_reads_the_alignment_its_model_writes(tmp_path):
    _write_clip(tmp_path / "clip.mp4", num_frames=12)
    replies = iter(["Alignment Score (1-5): 3.25\nPhysics Score (1-5): 2\nStyle Score (1-5): 4", "I cannot judge."])
    seen = []

    class _Scorer(unified_reward.UnifiedRewardScorer):
        def __init__(self):
            self.axis = "Alignment"

        def reply(self, clip, request):
            seen.append(len(clip.frames))
            return next(replies)

    scorer = _Scorer()

    assert scorer.score_batch([str(tmp_path / "clip.mp4")] * 2, ["a fox"] * 2) == [3.25, None]
    assert seen == [12, 12]
    assert unified_reward.parse_score("Alignment Score (1-5): 2\nStyle Score (1-5): 4.5", "Style") == 4.5


def test_hpsv3_checkpoint_names_move_to_transformers_5_and_drop_the_lm_head():
    checkpoint = {
        "model.embed_tokens.weight": torch.zeros(2),
        "model.layers.0.mlp.up_proj.weight": torch.zeros(1),
        "visual.blocks.0.attn.qkv.weight": torch.zeros(1),
        "lm_head.weight": torch.zeros(1),
        "rm_head.0.weight": torch.zeros(1),
    }

    assert sorted(hpsv3._renamed_state_dict(checkpoint)) == [
        "language_model.embed_tokens.weight",
        "language_model.layers.0.mlp.up_proj.weight",
        "rm_head.0.weight",
        "visual.blocks.0.attn.qkv.weight",
    ]


def test_vqascore_counts_a_lower_case_yes(tmp_path):
    _write_clip(tmp_path / "clip.mp4", num_frames=12)

    class _Tokenizer:
        def encode(self, text, add_special_tokens):
            return [{"yes": 0, "Yes": 1, "YES": 2, " yes": 3, " Yes": 4}[text]]

    class _Processor:
        tokenizer = _Tokenizer()

        def apply_chat_template(self, message, tokenize, add_generation_prompt):
            return message[0]["content"][1]["text"]

        def __call__(self, text, videos, **kwargs):
            assert videos[0].shape[0] == 12 and "a fox" in text[0]
            return SimpleNamespace(to=lambda device: {})

    class _Model:
        def __call__(self):
            logits = torch.full((1, 1, 6), -1e9)
            logits[0, -1, 0], logits[0, -1, 5] = 0.0, 0.0
            return SimpleNamespace(logits=logits)

    scorer = object.__new__(vqascore.VideoVqaScorer)
    scorer.device, scorer.num_frames, scorer.max_pixels_per_frame = "cpu", 16, 448 * 448
    scorer.processor, scorer.model = _Processor(), _Model()
    scorer.yes_ids = [scorer.processor.tokenizer.encode(yes, add_special_tokens=False)[0] for yes in vqascore._YES]

    assert scorer.score_batch([str(tmp_path / "clip.mp4")], ["a fox"]) == [pytest.approx(0.5)]


def test_hpsv3_reduces_frames_to_the_mean_of_the_best_30_percent(tmp_path):
    _write_clip(tmp_path / "clip.mp4", num_frames=12)
    scorer = object.__new__(hpsv3.VideoHpsv3Scorer)
    scorer.num_frames, scorer.frames_per_forward, scorer.top_fraction = 8, 3, 0.3
    scorer.score_images = lambda images, request: [float(i) for i in range(len(images))]

    # Frames come in batches of 3, scored 0, 1, 2 / 0, 1, 2 / 0, 1: the best 3 of 8 are 2, 2 and 1.
    assert scorer.score_batch([str(tmp_path / "clip.mp4")], ["a fox"]) == [pytest.approx(5 / 3)]
    assert hpsv3.top_mean([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1.0) == 4.5
    assert hpsv3.top_mean([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 0.3) == 7.0
