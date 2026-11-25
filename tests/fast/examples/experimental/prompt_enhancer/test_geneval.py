"""`geneval.py` against a fake diffusion engine: GenEval's layout, seeds shared across arms, and resuming."""

import json
import sys

import pytest
from examples.experimental.prompt_enhancer import geneval
from examples.experimental.prompt_enhancer.geneval import final_prompt

from .conftest import png


@pytest.mark.parametrize(
    "response, expected",
    [
        ("x </think>\n<answer> a red cube </answer>", "a red cube"),
        ("x </think>\n<answer>two\nlines</answer><answer>second</answer>", "two\nlines"),
        ("no answer", "a photo"),
    ],
)
def test_the_final_prompt_is_reprompts_geneval_extraction(response, expected):
    assert final_prompt(response) == expected


def _run(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["geneval", *argv])
    geneval.main()


@pytest.fixture
def metadata(tmp_path):
    path = tmp_path / "evaluation_metadata.jsonl"
    rows = [{"tag": "single_object", "prompt": "a photo of a cat"}, {"tag": "counting", "prompt": "two dogs"}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_a_raw_arm_writes_genevals_layout_with_paired_seeds(monkeypatch, tmp_path, metadata, engine):
    out = tmp_path / "raw"

    _run(monkeypatch, "--raw", "--metadata", str(metadata), "--image-urls", engine.url, "--out", str(out),
         "--images-per-prompt", "3", "--seed-offset", "100000")  # fmt: skip

    for i, prompt in enumerate(["a photo of a cat", "two dogs"]):
        folder = out / "images" / f"{i:05}"
        assert json.loads((folder / "metadata.jsonl").read_text())["final_prompt"] == prompt
        assert [(folder / "samples" / f"{k:05}.png").read_bytes() for k in range(3)] == [png(512, 512)] * 3
    sent = sorted((r["prompt"], r["seed"]) for r in engine.requests)
    expected = [(p, i * 1000 + k + 100000) for i, p in enumerate(["a photo of a cat", "two dogs"]) for k in range(3)]
    assert sent == sorted(expected)
    assert {(r["num_inference_steps"], r["guidance_scale"]) for r in engine.requests} == {(9, 0.0)}


def test_a_rerun_renders_only_the_missing_images(monkeypatch, tmp_path, metadata, engine):
    out = tmp_path / "raw"
    argv = ["--raw", "--metadata", str(metadata), "--image-urls", engine.url, "--out", str(out)]
    _run(monkeypatch, *argv, "--images-per-prompt", "2")
    engine.requests.clear()

    _run(monkeypatch, *argv, "--images-per-prompt", "3")

    assert sorted(r["seed"] for r in engine.requests) == [2, 1002]


def test_a_policy_arm_renders_its_saved_rewrites(monkeypatch, tmp_path, metadata, engine):
    out = tmp_path / "trained"
    out.mkdir()
    rows = [{"response": "r", "final_prompt": "a fluffy cat"}, {"response": "r", "final_prompt": "two brown dogs"}]
    (out / "rewrites.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    _run(monkeypatch, "--policy", "unused", "--metadata", str(metadata), "--image-urls", engine.url, "--out", str(out),
         "--images-per-prompt", "1")  # fmt: skip

    assert sorted((r["prompt"], r["seed"]) for r in engine.requests) == [("a fluffy cat", 0), ("two brown dogs", 1000)]
