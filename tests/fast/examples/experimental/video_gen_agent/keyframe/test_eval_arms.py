"""Phase 1's eval arms on CPU: prompts, clips from the fake generator, dumps, scoring with stub scorers, the paired
comparisons and the blind page."""

import asyncio
import json

import numpy as np
import pytest
import torch
from examples.experimental.video_gen_agent.generator import FakeClipGenerator
from examples.experimental.video_gen_agent.keyframe.eval import arms, blind, compare, score
from examples.experimental.video_gen_agent.keyframe.run_qwen35_9b_keyframe import ScriptArgs
from examples.experimental.video_gen_agent.phase0.reward import request_seed

av = pytest.importorskip("av")

REQUESTS = ["a fox", "a red car", "a tv"]


def _heldout(tmp_path):
    path = tmp_path / "heldout.jsonl"
    path.write_text("".join(json.dumps({"messages": [{"role": "user", "content": r}]}) + "\n" for r in REQUESTS))
    return path


def _dump(tmp_path, name: str, episodes: dict[str, dict]):
    samples = [{"metadata": {"video_gen": {"request": r} | record}} for r, record in episodes.items()]
    # A folded episode's later branch repeats the record; the first is read.
    samples.append({"metadata": {"video_gen": {"request": "a fox", "images": [{"prompt": "later"}]}}})
    path = tmp_path / name
    torch.save({"samples": samples}, path)
    return path


def test_the_generated_arms_prompts_and_seeds(tmp_path):
    final = arms.episodes(
        _dump(tmp_path, "eval_9.pt", {"a fox": {"images": [{"prompt": "fox, snow"}, {"prompt": "fox, close"}]}})
    )
    phase0 = {"a fox": "<think>hm</think> A fox runs.", "a red car": "A red car."}

    rows = arms.arm_prompts(arms.heldout_requests(_heldout(tmp_path)), phase0, final)

    by = {(row["arm"], row["request"]): row for row in rows}
    assert len(rows) == 9 and {row["seed"] for row in rows if row["request"] == "a tv"} == {request_seed("a tv")}
    assert by["B0", "a fox"]["prompt"] == "a fox" and by["B1", "a fox"]["prompt"] == "A fox runs."
    assert by["K-blind", "a fox"]["prompt"] == "fox, snow"
    assert by["B1", "a tv"]["prompt"] is None and by["K-blind", "a red car"]["prompt"] is None


def test_generate_asks_the_env_generator_for_the_runs_clip_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path / "clips"))
    rows = [
        {"arm": "B0", "request": "a fox", "prompt": "a fox", "seed": 7},
        {"arm": "B1", "request": "a fox", "prompt": None, "seed": 7},
    ]

    made, skipped = asyncio.run(arms.generate(rows, FakeClipGenerator()))

    asked = json.loads(open(made["path"]).read())
    assert (asked["width"], asked["height"], asked["fps"], asked["seconds"], asked["seed"]) == (832, 480, 16, 2.0, 7)
    assert skipped["path"] is None


def test_k_arms_replay_their_episodes_video_request_on_the_common_seed(tmp_path, monkeypatch):
    dump = _dump(
        tmp_path,
        "eval_0.pt",
        {
            "a fox": {
                "video": {"prompt": "fox, snow it runs", "seed": 0, "image_path": None},
                "submitted": {"path": "/kept/f.mp4"},
            },
            "a red car": {"video": {"prompt": "it drives", "seed": 5, "image_path": "/img/car.png"}, "submitted": {}},
            "a tv": {"video": None, "submitted": None},
        },
    )

    rows = arms.from_dump(dump, "K0", REQUESTS)

    assert rows[0] == {
        "arm": "K0",
        "request": "a fox",
        "prompt": "fox, snow it runs",
        "seed": request_seed("a fox"),
        "image_path": None,
        "episode_path": "/kept/f.mp4",
        "episode_seed": 0,
    }
    assert rows[2]["prompt"] is None
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path / "clips"))
    fox, car, tv = asyncio.run(arms.generate(rows, FakeClipGenerator()))
    asked = json.loads(open(car["path"]).read())
    assert (asked["seed"], asked["image_path"], asked["prompt"]) == (
        request_seed("a red car"),
        "/img/car.png",
        "it drives",
    )
    assert json.loads(open(fox["path"]).read())["seed"] == request_seed("a fox") and tv["path"] is None


def _write_clip(path, values):
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=16)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for value in values:
            frame = np.full((48, 64, 3), value, dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())


class _Table:
    def __init__(self, table):
        self.table = table

    def score_batch(self, paths, requests):
        return [self.table[path] for path in paths]


def test_scoring_gives_the_training_reward_its_gates_and_the_hack_monitors(tmp_path):
    still, moving = tmp_path / "still.mp4", tmp_path / "moving.mp4"
    _write_clip(still, [100] * 4)
    _write_clip(moving, [0, 80, 160, 240])
    rows = [
        {"arm": "K", "request": "a fox", "prompt": "a fox runs fast", "seed": 1, "path": str(moving)},
        {"arm": "B0", "request": "a fox", "prompt": "a fox", "seed": 1, "path": str(still)},
        {"arm": "K-blind", "request": "a fox", "prompt": None, "seed": 1, "path": None},
    ]
    config = score.reward_config(ScriptArgs(run_id="test"))
    low, high = config.video_reward_signal_ranges[:2]
    tables = {
        "hpsv3": {str(moving): high, str(still): high},
        "vqascore": {str(moving): 1.0, str(still): 1.0},
        "motion": {str(moving): 0.9, str(still): 0.1},
        "ocr": {str(moving): 1.0, str(still): 0.0},
        "judge": {str(moving): 0.8, str(still): 0.7},
        "pickscore": {str(moving): 0.9, str(still): 0.8},
    }

    k, b0, blind_row = score.score_rows(rows, {name: _Table(t) for name, t in tables.items()}, config)

    assert k["reward"] == pytest.approx(0.5) and k["fired"]["ocr"] and k["ocr_lines"] == 1.0
    assert b0["reward"] == 0.0 and b0["fired"]["motion"]
    assert k["frame_diff"] > 50 and b0["frame_diff"] < 1
    assert (k["prompt_words"], b0["prompt_words"]) == (4, 2)
    assert k["scores"]["judge"] == 0.8 and k["scores"]["pickscore"] == 0.9
    assert (blind_row["reward"], blind_row["frame_diff"], blind_row["prompt_words"]) == (0.0, None, None)


def test_scoring_uses_the_runs_pools_and_reward_settings_plus_the_outside_checks(monkeypatch):
    specs = score.scorer_specs(ScriptArgs(run_id="test", model_dir="/m"))
    gated = score.reward_config(ScriptArgs(run_id="test", adherence_gate=True))
    assert gated.video_reward_signal_pools == ["hpsv3", "vqascore"]
    assert (gated.video_reward_adherence_pool, gated.video_reward_adherence_threshold) == ("adherence", 0.016)
    assert specs["hpsv3"][1]["top_fraction"] == 0.3

    def stop(run):
        raise SystemExit(run.reward)

    monkeypatch.setattr(score, "scorer_specs", stop)
    with pytest.raises(SystemExit, match="^hpsv3\\+vqascore$"):
        score.main(["--out", "/dev/null", "/dev/null"])

    assert list(specs) == ["motion", "hpsv3", "vqascore", "ocr", "judge", "pickscore"]
    assert specs["judge"][1] == {"model_path": "/m/Qwen3-VL-8B-Instruct"}


def _rows(arm: str, values: dict[str, float]) -> list[dict]:
    return [
        {
            "arm": arm,
            "request": request,
            "reward": value,
            "scores": {"judge": value, "pickscore": value},
            "frame_diff": 10 * value,
            "ocr_lines": 0.0,
            "prompt_words": 5,
        }
        for request, value in values.items()
    ]


def test_arms_are_paired_by_request_with_a_two_sided_sign_test():
    k = {f"r{i}": 1.0 for i in range(10)}
    rows = (
        _rows("K", k)
        + _rows("B0", {f"r{i}": 0.0 for i in range(10)})
        + _rows("B1", {"r0": 2.0, "r1": 0.0, "r99": 0.0})
    )

    results = compare.compare(rows)

    assert results["B0"]["reward"] == {
        "requests": 10,
        "ours_mean": 1.0,
        "theirs_mean": 0.0,
        "wins": 10,
        "losses": 0,
        "ties": 0,
        "p": 2 / 2**10,
    }
    assert (results["B1"]["reward"]["requests"], results["B1"]["reward"]["wins"]) == (2, 1)
    assert results["B0"]["ocr_lines"]["ties"] == 10 and set(results) == {"B0", "B1"}
    bar = compare.win_bar(results, {"A": {"K": 15, "B0": 4, "tie": 1}, "B": {"K": 9, "B1": 11, "tie": 0}})
    assert bar == {
        "reward beats B0 and B1": False,
        "PickScore beats B0 and B1": False,
        "blind: K in >= 15 of 20 decided vs B0": True,
        "frame difference not below B0": True,
    }
    assert compare.win_bar(results, None)["blind: K in >= 15 of 20 decided vs B0"] is None
    missing = [row | {"frame_diff": None} for row in _rows("K", k)] + _rows("B0", {"r0": 0.0})
    assert compare.win_bar(compare.compare(missing), None)["frame difference not below B0"] is None


def test_the_blind_page_draws_outside_lin2741s_requests_and_keeps_the_key_apart(tmp_path):
    clips = {}
    for arm in ("K", "B0", "B1"):
        for i in range(30):
            path = tmp_path / f"{arm}_{i}.mp4"
            path.write_text(f"{arm} {i}")
            clips.setdefault(arm, {})[f"r{i}"] = str(path)
    excluded = {f"r{i}" for i in range(5)}

    pairs, key = blind.make(clips, excluded, tmp_path / "blind")

    requests = [pair["request"] for pair in pairs]
    assert len(pairs) == 40 and requests[:20] == requests[20:] and not set(requests) & excluded
    assert json.loads((tmp_path / "blind_key.json").read_text()) == key
    page = (tmp_path / "blind" / "index.html").read_text()
    assert f"const PAIRS = {json.dumps(pairs)};" in page and "__PAIRS__" not in page
    assert "_key.json" not in page and '"left"' not in json.dumps(pairs)
    assert 'src: p.pair + "_" + side + ".mp4"' in page and "picks.json" in page
    for pair in pairs:
        k_side = key[pair["pair"]]
        assert (tmp_path / "blind" / f"{pair['pair']}_{k_side}.mp4").read_text().startswith("K ")
    picks = {pair["pair"]: key[pair["pair"]] if pair["set"] == "A" else "tie" for pair in pairs}
    assert blind.unblind(pairs, key, picks) == {"A": {"K": 20, "B0": 0, "tie": 0}, "B": {"K": 0, "B1": 0, "tie": 20}}


def test_the_blind_draw_is_fixed():
    clips = {arm: {f"r{i}": f"{arm}{i}" for i in range(30)} for arm in ("K", "B0", "B1")}

    assert blind.draw(clips, set()) == blind.draw(clips, set())
    with pytest.raises(AssertionError, match="only 19 requests"):
        blind.draw(clips, {f"r{i}" for i in range(11)})
