"""The anchor maker without engines: which requests it makes, its estimate, and its exit while any is missing."""

import json

import pytest
from examples.experimental.video_gen_agent.keyframe.eval import anchors, arms

from miles.rollout.rm_hub.video.pairwise import anchor_name


def _requests(path, requests):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"messages": [{"role": "user", "content": r}]}) + "\n" for r in requests))


def _args(tmp_path, **overrides):
    values = dict(run_id="test", reward="pairwise", video_mode="reference", data_dir=str(tmp_path), gpus="0,1,2")
    return anchors.AnchorArgs(**(values | overrides))


@pytest.fixture
def requests(tmp_path):
    _requests(tmp_path / "video_gen_requests/train.jsonl", ["a fox", "a red car"])
    _requests(tmp_path / "video_gen_requests/heldout.jsonl", ["a tv", "a fox"])


def test_only_requests_without_an_anchor_are_made_with_an_estimate(tmp_path, requests):
    args = _args(tmp_path)
    (tmp_path / "video_gen_anchors_reference").mkdir()
    (tmp_path / "video_gen_anchors_reference" / anchor_name("a red car")).write_bytes(b"clip")

    rows, summary = anchors.plan(args)

    assert rows == [arms.one_shot_row(r, "reference") for r in ("a fox", "a tv")]
    assert summary.startswith(f"2 of 3 requests have no anchor in {tmp_path}/video_gen_anchors_reference")
    assert "about 0.0 GPU-h" in summary and "on 2 video GPUs" in summary
    assert "no measured rate for preview mode" in anchors.plan(_args(tmp_path, video_mode="preview"))[1]
    assert anchors.plan(_args(tmp_path, video_mode="preview"))[0][0] == arms.one_shot_row("a fox", "t2v")


def test_main_makes_the_missing_anchors_then_exits_non_zero_while_any_is_missing(tmp_path, requests, monkeypatch):
    made = []

    async def make(args, rows):
        made.append([row["request"] for row in rows])
        for row in rows[:-1]:
            (tmp_path / "video_gen_anchors_reference" / anchor_name(row["request"])).write_bytes(b"clip")

    (tmp_path / "video_gen_anchors_reference").mkdir()
    monkeypatch.setattr(anchors, "_make", make)

    with pytest.raises(SystemExit, match="^1 of 3 requests have no anchor"):
        anchors.make_anchors(_args(tmp_path))
    assert made == [["a fox", "a red car", "a tv"]]


@pytest.mark.parametrize(
    "overrides, message", [(dict(reward="hpsv3+vqascore"), "anchors are for --reward pairwise"), (dict(gpus="0"), "--gpus")]
)
def test_the_anchor_maker_needs_the_pairwise_reward_and_a_video_gpu(tmp_path, overrides, message):
    with pytest.raises(AssertionError, match=message):
        _args(tmp_path, **overrides)
