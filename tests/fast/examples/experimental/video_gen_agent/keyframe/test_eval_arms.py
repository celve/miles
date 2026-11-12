"""Phase 1's eval arms on CPU: prompts, clips from the fake generator, dumps, scoring with stub scorers, the paired
comparisons and the blind page."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from examples.experimental.video_gen_agent.generator import FakeClipGenerator, FakeImageGenerator, GeneratorUnavailable
from examples.experimental.video_gen_agent.keyframe.eval import arms, blind, compare, score
from examples.experimental.video_gen_agent.keyframe.run_qwen35_9b_keyframe import ScriptArgs
from examples.experimental.video_gen_agent.phase0.reward import request_seed
from PIL import Image

from miles.rollout.rm_hub.video.pairwise import anchor_name

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


def test_the_i2v_one_shot_arm_conditions_the_video_on_an_image_of_the_raw_request(tmp_path, monkeypatch):
    rows = arms.arm_prompts(["a fox"], {}, {}, mode="i2v")
    (row,) = [row for row in rows if row["arm"] == "B0-i2v"]
    assert (row["prompt"], row["image_prompt"], row["seed"]) == ("a fox", "a fox", request_seed("a fox"))
    assert "B0-i2v" not in {row["arm"] for row in arms.arm_prompts(["a fox"], {}, {})}
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path / "made"))

    (made,) = asyncio.run(arms.generate([row], FakeClipGenerator(), FakeImageGenerator()))

    asked = json.loads(open(made["path"]).read())
    assert asked["image_path"] == made["image_path"] and Image.open(made["image_path"]).size == (832, 480)
    assert json.loads(Image.open(made["image_path"]).text["request"])["prompt"] == "a fox"
    with pytest.raises(ValueError, match="AGENT_VIDEO_GEN_IMAGE_GENERATOR"):
        asyncio.run(arms.generate([row], FakeClipGenerator()))


def test_in_reference_mode_both_image_arms_condition_the_video_on_a_reference(tmp_path, monkeypatch):
    final = {
        "a fox": {
            "images": [
                {"prompt": "a fox, isolated", "path": "/img/fox.png"},
                {"prompt": "snow", "path": "/img/snow.png"},
            ]
        }
    }
    by = {row["arm"]: row for row in arms.arm_prompts(["a fox"], {}, final, mode="reference")}
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path / "made"))

    one_shot, blind_k = asyncio.run(
        arms.generate([by["B0-ref"], by["K-blind"]], FakeClipGenerator(), FakeImageGenerator())
    )

    assert set(by) == {"B0", "B1", "K-blind", "B0-ref"}
    asked = json.loads(open(one_shot["path"]).read())
    assert asked["reference_paths"] == [one_shot["reference_paths"][0]] and asked["image_path"] is None
    assert json.loads(Image.open(one_shot["reference_paths"][0]).text["request"])["prompt"] == "a fox"
    asked = json.loads(open(blind_k["path"]).read())
    # Each image is one asset, so K-blind takes them all, as K's video does.
    assert (asked["reference_paths"], asked["image_path"], asked["seed"], asked["prompt"]) == (
        ["/img/fox.png", "/img/snow.png"],
        None,
        request_seed("a fox"),
        "a fox, isolated snow",
    )
    assert "reference_paths" not in by["B0"] and "reference_paths" not in by["B1"]


def test_in_i2v_mode_k_blind_animates_the_agents_recorded_first_image(tmp_path, monkeypatch):
    final = {"a fox": {"images": [{"prompt": "fox, snow", "path": "/img/first.png"}, {"prompt": "fox, close"}]}}
    rows = arms.arm_prompts(["a fox", "a tv"], {}, final, mode="i2v")
    by = {(row["arm"], row["request"]): row for row in rows}
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path / "made"))

    (made,) = asyncio.run(arms.generate([by["K-blind", "a fox"]], FakeClipGenerator()))

    asked = json.loads(open(made["path"]).read())
    assert (asked["image_path"], asked["prompt"], asked["seed"]) == (
        "/img/first.png",
        "fox, snow",
        request_seed("a fox"),
    )
    assert "image_path" not in by["B0", "a fox"] and "image_path" not in by["B1", "a fox"]
    assert by["K-blind", "a tv"]["prompt"] is None and "image_path" not in by["K-blind", "a tv"]
    t2v = next(r for r in arms.arm_prompts(["a fox"], {}, final) if r["arm"] == "K-blind")
    assert "image_path" not in t2v


class _Flaky:
    """Fails its first ``failures`` calls as an unavailable router does, then succeeds; records every request."""

    def __init__(self, failures: int):
        self.failures, self.requests = failures, []

    async def generate(self, request):
        self.requests.append(request)
        if len(self.requests) <= self.failures:
            raise GeneratorUnavailable("the generator's router failed: ReadError('')")
        return SimpleNamespace(path=f"/made/{len(self.requests)}")


def test_generate_retries_a_transient_failure_with_the_same_request():
    rows = [
        {
            "arm": "B0-ref",
            "request": "a fox",
            "prompt": "a fox",
            "image_prompt": "a fox",
            "seed": 7,
            "mode": "reference",
        }
    ]
    image, clip = _Flaky(failures=2), _Flaky(failures=1)

    (made,) = asyncio.run(arms.generate(rows, clip, image, retry_delay_s=0))

    assert made["path"] == "/made/2" and made["reference_paths"] == ["/made/3"]
    assert len(set(image.requests)) == 1 and len(image.requests) == 3
    assert clip.requests[0] == clip.requests[1] and clip.requests[0].seed == 7
    with pytest.raises(GeneratorUnavailable):
        asyncio.run(arms.generate(rows, _Flaky(failures=arms.TRIES), _Flaky(failures=0), retry_delay_s=0))


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


def test_an_h3_runs_arms_take_its_clip_length_and_frame_rate(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "AGENT_VIDEO_GEN_GENERATOR", "examples.experimental.video_gen_agent.generator.FakeClipGenerator"
    )
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path / "clips"))
    monkeypatch.delenv("AGENT_VIDEO_GEN_IMAGE_GENERATOR", raising=False)
    prompts = tmp_path / "prompts.jsonl"
    row = {"arm": "K-blind", "request": "a fox", "prompt": "a fox", "seed": 7, "reference_paths": ["/img/fox.png"]}
    prompts.write_text(json.dumps(row) + "\n")

    arms.main(["generate", str(prompts), "--out", str(tmp_path / "clips.jsonl"), "--seconds", "4", "--fps", "24"])

    (made,) = map(json.loads, (tmp_path / "clips.jsonl").read_text().splitlines())
    asked = json.loads(open(made["path"]).read())
    assert (asked["seconds"], asked["fps"], asked["width"], asked["height"]) == (4.0, 24, 832, 480)
    assert asked["reference_paths"] == ["/img/fox.png"]


def test_k_arms_replay_their_episodes_video_request_on_the_common_seed(tmp_path, monkeypatch):
    dump = _dump(
        tmp_path,
        "eval_0.pt",
        {
            "a fox": {
                "video": {"prompt": "fox, snow it runs", "seed": 0, "image_path": None},
                "submitted": {"path": "/kept/f.mp4"},
            },
            "a red car": {
                "video": {
                    "prompt": "it drives",
                    "negative_prompt": "blur, watermark",
                    "seed": 5,
                    "image_path": "/img/car.png",
                },
                "submitted": {},
            },
            "a tv": {"video": None, "submitted": None},
        },
    )

    rows = arms.from_dump(dump, "K0", REQUESTS)

    assert rows[0] == {
        "arm": "K0",
        "request": "a fox",
        "prompt": "fox, snow it runs",
        "negative_prompt": "",
        "seed": request_seed("a fox"),
        "image_path": None,
        "episode_path": "/kept/f.mp4",
        "episode_seed": 0,
    }
    assert rows[2]["prompt"] is None
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path / "clips"))
    fox, car, tv = asyncio.run(arms.generate(rows, FakeClipGenerator()))
    asked = json.loads(open(car["path"]).read())
    assert (asked["seed"], asked["image_path"], asked["prompt"], asked["negative_prompt"]) == (
        request_seed("a red car"),
        "/img/car.png",
        "it drives",
        "blur, watermark",
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


def test_scoring_the_pairwise_reward_needs_no_motion_pool_and_takes_the_anchor_dir(tmp_path, monkeypatch):
    run = ScriptArgs(run_id="test", model_dir="/m", reward="pairwise", anchor_dir="/anchors")
    specs = score.scorer_specs(run)
    assert list(specs) == ["pairwise", "ocr", "judge", "pickscore"]
    assert specs["pairwise"][1]["anchor_dir"] == "/anchors"
    config = score.reward_config(run)
    clip = tmp_path / "still.mp4"
    _write_clip(clip, [100] * 4)
    pools = {name: _Table({str(clip): value}) for name, value in (("pairwise", 0.8), ("ocr", 1.0))}

    (row,) = score.score_rows([{"path": str(clip), "request": "a fox", "prompt": "a fox"}], pools, config)

    assert config.video_reward_motion_pool is None
    assert row["reward"] == pytest.approx(0.4) and row["fired"]["ocr"] and not row["fired"]["motion"]

    def stop(run):
        raise SystemExit(run.anchor_dir)

    monkeypatch.setattr(score, "scorer_specs", stop)
    with pytest.raises(SystemExit, match="^/anchors$"):
        score.main(["--out", "/dev/null", "--reward", "pairwise", "--anchor-dir", "/anchors", "/dev/null"])


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


def test_in_i2v_mode_the_win_bar_holds_k_to_b0_i2v_and_still_reports_b0():
    k = {f"r{i}": 1.0 for i in range(10)}
    rows = (
        _rows("K", k)
        + _rows("B0", {f"r{i}": 0.0 for i in range(10)})
        + _rows("B0-i2v", {f"r{i}": 2.0 for i in range(10)})
        + _rows("B1", {f"r{i}": 0.0 for i in range(10)})
    )
    results = compare.compare(rows)

    bar = compare.win_bar(results, {"A": {"K": 15, "B0-i2v": 5, "tie": 0}}, mode="i2v")

    assert set(results) == {"B0", "B0-i2v", "B1"}
    assert bar["reward beats B0-i2v and B1"] is False and bar["blind: K in >= 15 of 20 decided vs B0-i2v"] is True
    assert compare.win_bar(results, None)["reward beats B0 and B1"] is True
    with pytest.raises(ValueError, match="not judged against B0-i2v"):
        compare.win_bar(results, {"A": {"K": 20, "B0": 0, "tie": 0}}, mode="i2v")
    with pytest.raises(ValueError, match="not judged against B0:"):
        compare.win_bar(results, {"A": {"K": 20, "B0-i2v": 0, "tie": 0}})


def test_in_reference_mode_the_win_bar_holds_k_to_b0_ref_and_refuses_other_picks():
    k = {f"r{i}": 1.0 for i in range(10)}
    rows = _rows("K", k) + _rows("B0", {f"r{i}": 2.0 for i in range(10)})
    rows += _rows("B0-ref", {f"r{i}": 0.0 for i in range(10)}) + _rows("B1", {f"r{i}": 0.0 for i in range(10)})
    results = compare.compare(rows)

    bar = compare.win_bar(results, {"A": {"K": 16, "B0-ref": 4, "tie": 0}}, mode="reference")

    assert set(results) == {"B0", "B0-ref", "B1"}
    assert bar["reward beats B0-ref and B1"] is True and bar["blind: K in >= 15 of 20 decided vs B0-ref"] is True
    assert compare.win_bar(results, None)["reward beats B0 and B1"] is False
    for picks in ({"K": 20, "B0": 0, "tie": 0}, {"K": 20, "B0-i2v": 0, "tie": 0}):
        with pytest.raises(ValueError, match="not judged against B0-ref"):
            compare.win_bar(results, {"A": picks}, mode="reference")


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
    assert "by the same frozen Wan2.2-T2V-A14B" in page
    assert "_key.json" not in page and '"left"' not in json.dumps(pairs)
    assert 'src: p.pair + "_" + side + ".mp4"' in page and "picks.json" in page
    # The export button is styled as a tab but must not switch sets, which would empty the list.
    assert 'querySelectorAll(".tab[data-set]")' in page and 'querySelectorAll(".tab")' not in page
    for pair in pairs:
        k_side = key[pair["pair"]]
        assert (tmp_path / "blind" / f"{pair['pair']}_{k_side}.mp4").read_text().startswith("K ")
    picks = {pair["pair"]: key[pair["pair"]] if pair["set"] == "A" else "tie" for pair in pairs}
    assert blind.unblind(pairs, key, picks) == {"A": {"K": 20, "B0": 0, "tie": 0}, "B": {"K": 0, "B1": 0, "tie": 20}}


def test_in_i2v_mode_blind_set_a_is_k_against_b0_i2v(tmp_path):
    clips = {}
    for arm in ("K", "B0-i2v", "B1"):
        for i in range(25):
            path = tmp_path / f"{arm}_{i}.mp4"
            path.write_text(arm)
            clips.setdefault(arm, {})[f"r{i}"] = str(path)

    pairs, key = blind.make(clips, set(), tmp_path / "blind", mode="i2v")

    page = (tmp_path / "blind" / "index.html").read_text()
    assert "in set A both clips are image-to-video" in page and "__MODELS__" not in page
    a_pair = next(pair for pair in pairs if pair["set"] == "A")
    other = "right" if key[a_pair["pair"]] == "left" else "left"
    assert (tmp_path / "blind" / f"{a_pair['pair']}_{other}.mp4").read_text() == "B0-i2v"
    against_k = {"left": "right", "right": "left"}
    picks = {pair["pair"]: against_k[key[pair["pair"]]] if pair["set"] == "A" else "tie" for pair in pairs}
    assert blind.unblind(pairs, key, picks) == {
        "A": {"K": 0, "B0-i2v": 20, "tie": 0},
        "B": {"K": 0, "B1": 0, "tie": 20},
    }
    with pytest.raises(KeyError):
        blind.draw(clips, set())


def test_in_reference_mode_blind_set_a_is_k_against_b0_ref(tmp_path):
    clips = {arm: {f"r{i}": arm for i in range(25)} for arm in ("K", "B0-ref", "B1")}
    for arm, by_request in clips.items():
        for request in by_request:
            path = tmp_path / f"{arm}_{request}.mp4"
            path.write_text(arm)
            by_request[request] = str(path)

    pairs, key = blind.make(clips, set(), tmp_path / "blind", mode="reference")

    assert {pair["baseline"] for pair in pairs if pair["set"] == "A"} == {"B0-ref"}
    assert "made by Wan2.1-VACE-14B" in (tmp_path / "blind" / "index.html").read_text()


def test_the_blind_draw_is_fixed():
    clips = {arm: {f"r{i}": f"{arm}{i}" for i in range(30)} for arm in ("K", "B0", "B1")}

    assert blind.draw(clips, set()) == blind.draw(clips, set())
    with pytest.raises(AssertionError, match="only 19 requests"):
        blind.draw(clips, {f"r{i}" for i in range(11)})


def test_k_arms_replay_a_reference_mode_episodes_images(tmp_path, monkeypatch):
    video = {
        "prompt": "a corgi a park it runs",
        "seed": 0,
        "image_path": None,
        "reference_paths": ["/i/a.png", "/i/b.png"],
    }
    dump = _dump(tmp_path, "eval_0.pt", {"a fox": {"video": video, "submitted": {"path": "/kept/f.mp4"}}})

    (row, *_) = arms.from_dump(dump, "K", REQUESTS)
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path / "clips"))
    (made,) = asyncio.run(arms.generate([row], FakeClipGenerator()))

    asked = json.loads(open(made["path"]).read())
    assert row["reference_paths"] == ["/i/a.png", "/i/b.png"] and asked["reference_paths"] == ["/i/a.png", "/i/b.png"]
    assert (asked["prompt"], asked["image_path"], asked["seed"]) == (
        "a corgi a park it runs",
        None,
        request_seed("a fox"),
    )


def test_b1_rewrite_replaces_b1_where_phase0_has_no_answers_and_rows_carry_categories(tmp_path):
    heldout = tmp_path / "heldout.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": r}], "metadata": {"category": c}}
        for r, c in (("a fox", "x"), ("a tv", "y"))
    ]
    heldout.write_text("".join(json.dumps(row) + "\n" for row in rows))
    rewrites = tmp_path / "rewrites.jsonl"
    rewrites.write_text(json.dumps({"request": "a fox", "answer": "A red fox trots through snow."}) + "\n")
    dump = _dump(tmp_path, "eval_9.pt", {"a fox": {"images": [{"prompt": "fox, snow"}]}})
    out = tmp_path / "prompts.jsonl"

    arms.main(
        [
            "prompts",
            "--heldout",
            str(heldout),
            "--b1-rewrite",
            str(rewrites),
            "--final-dump",
            str(dump),
            "--out",
            str(out),
        ]
    )

    made = [json.loads(line) for line in out.read_text().splitlines()]
    by = {(row["arm"], row["request"]): row for row in made}
    assert {row["arm"] for row in made} == {"B0", "B1-rewrite", "K-blind"}
    assert (
        by["B1-rewrite", "a fox"]["prompt"] == "A red fox trots through snow."
        and by["B1-rewrite", "a tv"]["prompt"] is None
    )
    assert {(row["request"], row["category"]) for row in made} == {("a fox", "x"), ("a tv", "y")}


def test_rewrite_asks_the_served_planner_with_phase0s_prompt_greedy_and_without_thinking():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from examples.experimental.video_gen_agent.phase0.make_planner_data import SYSTEM_PROMPT

    asked = []

    class Planner(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            asked.append((self.path, payload))
            answer = f"A clip of {payload['messages'][-1]['content']}."
            body = json.dumps({"choices": [{"message": {"content": answer}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Planner)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        rows = asyncio.run(arms.rewrite(["a fox", "a tv"], f"http://127.0.0.1:{server.server_port}"))
    finally:
        server.shutdown()

    assert rows == [
        {"request": "a fox", "answer": "A clip of a fox."},
        {"request": "a tv", "answer": "A clip of a tv."},
    ]
    path, payload = asked[0]
    assert path == "/v1/chat/completions" and payload["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert (payload["temperature"], payload["chat_template_kwargs"]) == (0, {"enable_thinking": False})


def test_the_win_bar_can_hold_k_to_b1_rewrite_and_compare_splits_by_category(tmp_path, capsys):
    k = {f"r{i}": 1.0 for i in range(10)}
    rows = _rows("K", k) + _rows("B0-ref", {f"r{i}": 0.0 for i in range(10)})
    rows += _rows("B1-rewrite", {f"r{i}": 0.0 for i in range(10)})
    rows = [row | {"category": "spatial" if row["request"] < "r5" else "numeracy"} for row in rows]
    results = compare.compare(rows)

    bar = compare.win_bar(results, None, mode="reference", b1="B1-rewrite")

    assert set(results) == {"B0-ref", "B1-rewrite"}
    assert bar["reward beats B0-ref and B1-rewrite"] is True
    assert compare.win_bar(results, None, mode="reference")["reward beats B0-ref and B1"] is None
    scored = tmp_path / "scored.jsonl"
    scored.write_text("".join(json.dumps(row) + "\n" for row in rows))
    compare.main([str(scored), "--mode", "reference", "--b1", "B1-rewrite", "--by-category"])
    printed = capsys.readouterr().out
    assert "- reward beats B0-ref and B1-rewrite: pass" in printed
    assert "## numeracy" in printed and "| B1-rewrite | reward | 5 | 1.0000 | 0.0000 | 5 | 0 | 0 |" in printed


def test_anchors_are_each_requests_one_shot_clip_placed_by_request_and_checked(tmp_path, monkeypatch, capsys):
    heldout = _heldout(tmp_path)
    arms.main(
        ["anchor-prompts", str(heldout), str(heldout), "--mode", "reference", "--out", str(tmp_path / "a.jsonl")]
    )
    rows = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert rows == [arms.one_shot_row(request, "reference") for request in REQUESTS]
    assert rows[0] == arms.arm_prompts(["a fox"], {}, {}, mode="reference")[-1]
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path / "made"))
    made = asyncio.run(arms.generate(rows[:2], FakeClipGenerator(), FakeImageGenerator())) + [rows[2] | {"path": None}]
    (tmp_path / "made.jsonl").write_text("".join(json.dumps(row) + "\n" for row in made))
    anchors = tmp_path / "anchors"

    arms.main(["place-anchors", str(tmp_path / "made.jsonl"), "--anchor-dir", str(anchors)])

    assert (anchors / anchor_name("a red car")).read_bytes() == open(made[1]["path"], "rb").read()
    with pytest.raises(SystemExit, match="1 requests have no anchor clip .* 'a tv'"):
        arms.main(["check-anchors", str(heldout), "--anchor-dir", str(anchors)])
    (anchors / anchor_name("a tv")).write_bytes(b"clip")
    arms.main(["check-anchors", str(heldout), "--anchor-dir", str(anchors)])


class _Planner:
    """A served planner's ``/v1/chat/completions`` answering each request with ``answer(payload)``."""

    def __init__(self, answer):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        self.asked = []
        asked = self.asked

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                asked.append(payload)
                body = json.dumps({"choices": [{"message": {"content": answer(payload)}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.server.shutdown()


def _made(path: str) -> dict:
    return json.loads(Image.open(path).text["request"])


def test_bo3_lets_the_planner_pick_among_three_keyframes_the_first_of_which_is_b0_refs(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path / "made"))
    with _Planner(lambda payload: "Image 2 shows it best.") as planner:
        (row,) = asyncio.run(arms.keyframes("bo3", ["a fox"], planner.url, FakeImageGenerator()))
        (b0_ref,) = asyncio.run(
            arms.generate([arms.one_shot_row("a fox", "reference")], FakeClipGenerator(), FakeImageGenerator())
        )

    seed = request_seed("a fox")
    assert [_made(path)["seed"] for path in row["candidates"]] == [seed, seed + 1, seed + 2]
    assert row["candidates"][0] == b0_ref["reference_paths"][0]
    assert (row["pick"], row["reference_paths"], row["prompt"], row["seed"]) == (
        1,
        [row["candidates"][1]],
        "a fox",
        seed,
    )
    (payload,) = planner.asked
    shown = payload["messages"][-1]["content"]
    assert [part["image_url"]["url"] for part in shown if part["type"] == "image_url"] == [
        f"file://{path}" for path in row["candidates"]
    ]
    assert (payload["temperature"], payload["chat_template_kwargs"]) == (0, {"enable_thinking": False})
    (made,) = asyncio.run(arms.generate([row], FakeClipGenerator()))
    asked = json.loads(open(made["path"]).read())
    assert (asked["reference_paths"], asked["prompt"], asked["seed"]) == ([row["candidates"][1]], "a fox", seed)


def test_comp_key_composes_k0s_first_three_assets_into_the_clips_first_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path / "made"))
    assets = ["/i/fox.png", "/i/ball.png", "/i/tree.png", "/i/sun.png"]
    video = {"prompt": "the fox chases the ball", "seed": 3, "image_path": None, "reference_paths": assets}
    dump = _dump(tmp_path, "eval_0.pt", {"a fox": {"video": video}})
    k0 = {row["request"]: row for row in arms.from_dump(dump, "K0", ["a fox", "a tv"])}
    with _Planner(
        lambda payload: " The fox in image 1 leaps at the ball in image 2 by the tree in image 3. "
    ) as planner:
        fox, tv = asyncio.run(
            arms.keyframes(
                "comp-key", ["a fox", "a tv"], planner.url, None, k0=k0, edit_generator=FakeImageGenerator()
            )
        )

    composed = _made(fox["image_path"])
    assert composed["image_paths"] == assets[:3] and composed["seed"] == request_seed("a fox")
    assert composed["prompt"] == "The fox in image 1 leaps at the ball in image 2 by the tree in image 3."
    assert (fox["prompt"], fox["reference_paths"]) == ("the fox chases the ball", assets)
    (payload,) = planner.asked
    assert [p["image_url"]["url"] for p in payload["messages"][-1]["content"] if p["type"] == "image_url"] == [
        f"file://{path}" for path in assets[:3]
    ]
    assert tv["prompt"] is None
    (made, missing) = asyncio.run(arms.generate([fox, tv], FakeClipGenerator()))
    asked = json.loads(open(made["path"]).read())
    assert (asked["image_path"], asked["reference_paths"]) == (fox["image_path"], assets)
    assert missing["path"] is None


def test_end_key_renders_the_planners_first_and_last_frames_for_dynamic_and_motion_requests_only(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path / "made"))
    plan = {"first": "a green leaf", "last": "a red leaf", "video": "the leaf turns red"}
    answers = {"a leaf turns red": "```json\n" + json.dumps(plan) + "\n```", "a car drives left": "no idea"}
    categories = {
        "a leaf turns red": "dynamic_attribute_binding",
        "a car drives left": "motion_binding",
        "a red car": "consistent_attribute_binding",
    }
    with _Planner(
        lambda payload: answers[payload["messages"][-1]["content"].split("\n")[0].split(": ", 1)[1]]
    ) as planner:
        leaf, car = asyncio.run(
            arms.keyframes("end-key", list(categories), planner.url, FakeImageGenerator(), categories=categories)
        )

    seed = request_seed("a leaf turns red")
    assert [(_made(leaf[key])["prompt"], _made(leaf[key])["seed"]) for key in ("image_path", "last_image_path")] == [
        ("a green leaf", seed),
        ("a red leaf", seed),
    ]
    assert (leaf["prompt"], leaf["reference_paths"]) == ("the leaf turns red", [leaf["image_path"]])
    assert (car["request"], car["prompt"], car["planner_answer"]) == ("a car drives left", None, "no idea")
    (made, _) = asyncio.run(arms.generate([leaf, car], FakeClipGenerator()))
    asked = json.loads(open(made["path"]).read())
    assert (asked["image_path"], asked["last_image_path"], asked["reference_paths"]) == (
        leaf["image_path"],
        leaf["last_image_path"],
        [leaf["image_path"]],
    )


def test_a_pick_or_an_end_key_plan_the_planner_garbles_is_recorded_as_none():
    assert [arms.parse_pick(answer, 3) for answer in ("3", "Image 1.", "the 4th", "none", "12")] == [
        2,
        0,
        None,
        None,
        None,
    ]
    assert arms.parse_end_key('{"first": "a", "last": "b"}') is None
    assert arms.parse_end_key('{"first": "a", "last": "b", "video": " "}') is None
    assert arms.parse_end_key('Sure: {"first": "a", "last": "b", "video": "c"}') == {
        "first": "a",
        "last": "b",
        "video": "c",
    }


def test_compare_holds_any_arm_against_the_baselines_and_keeps_the_win_bar_for_k(tmp_path, capsys):
    rows = [
        {
            "arm": arm,
            "request": f"r{i}",
            "reward": reward + 0.01 * i,
            "scores": {},
            "frame_diff": 1.0,
            "ocr_lines": 0,
            "prompt_words": 3,
        }
        for i in range(6)
        for arm, reward in (("bo3", 0.9), ("B0-ref", 0.5), ("B1-rewrite", 0.95))
    ]
    path = tmp_path / "scored.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    results = compare.compare(rows, "bo3")
    compare.main([str(path), "--ours", "bo3", "--mode", "reference"])

    assert (results["B0-ref"]["reward"]["wins"], results["B1-rewrite"]["reward"]["losses"]) == (6, 6)
    out = capsys.readouterr().out
    assert "| bo3 vs | metric |" in out and "| B0-ref | reward | 6 |" in out and "win" not in out.split("\n")[-2]
    assert "beats" not in out


def test_b1_ref_renders_the_rewrite_as_b0_ref_renders_the_raw_request(tmp_path, monkeypatch):
    rewrites = {"a fox": "<think>x</think> A red fox trots through snow."}
    rows = arms.arm_prompts(["a fox", "a tv"], None, {}, mode="reference", rewrites=rewrites)
    by = {(row["arm"], row["request"]): row for row in rows}
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path / "made"))

    b0_ref, b1_ref = asyncio.run(
        arms.generate([by["B0-ref", "a fox"], by["B1-ref", "a fox"]], FakeClipGenerator(), FakeImageGenerator())
    )

    assert ("B1-ref", "a tv") not in by and "B1-ref" not in {
        r["arm"] for r in arms.arm_prompts(["a fox"], None, {}, rewrites=rewrites)
    }
    image = _made(b1_ref["reference_paths"][0])
    assert (image["prompt"], image["seed"]) == (
        "A red fox trots through snow.",
        _made(b0_ref["reference_paths"][0])["seed"],
    )
    asked = json.loads(open(b1_ref["path"]).read())
    assert (asked["prompt"], asked["reference_paths"], asked["seed"]) == (
        "A red fox trots through snow.",
        b1_ref["reference_paths"],
        request_seed("a fox"),
    )
