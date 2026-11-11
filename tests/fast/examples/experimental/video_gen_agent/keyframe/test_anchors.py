"""The anchor maker without engines: which requests it makes, its estimate, and its exit while any is missing."""

import json

import pytest
from examples.experimental.video_gen_agent.keyframe.eval import anchors, arms
from examples.experimental.video_gen_agent.keyframe.run_qwen35_9b_keyframe import _agent_env

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
    "overrides, message",
    [(dict(reward="hpsv3+vqascore"), "anchors are for --reward pairwise"), (dict(gpus="0"), "--gpus")],
)
def test_the_anchor_maker_needs_the_pairwise_reward_and_a_video_gpu(tmp_path, overrides, message):
    with pytest.raises(AssertionError, match=message):
        _args(tmp_path, **overrides)


class _Engine:
    """A launched engine's process handle, standing in for the subprocess."""

    def terminate(self):
        pass

    def wait(self, timeout):
        pass


def test_h3_anchors_are_made_by_the_runs_generator_at_its_clip_length_on_its_engines(tmp_path, requests, monkeypatch):
    from tests.fast.diffusion.test_frozen_generator import _fake_image_engine
    from tests.fast.examples.experimental.video_gen_agent.keyframe.test_h3_generator import _fake_h3_engine

    from miles.utils.http_utils import find_available_port
    from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

    seen, launched, servers = [], [], []

    def launch(model_path, gpus, overrides, ports):
        next(ports)
        image = model_path.endswith("Z-Image-Turbo")
        app = _fake_image_engine(channels=3, height=480, width=832, seen=[]) if image else _fake_h3_engine("h3", seen)
        app.get("/health_generate")(lambda: {})
        server = UvicornThreadServer(app, host="127.0.0.1", port=find_available_port(38500 + 50 * len(servers)))
        server.start()
        servers.append(server)
        launched.append((model_path.rsplit("/", 1)[1], gpus, overrides))
        return _Engine(), server.url

    monkeypatch.setattr(anchors, "_launch", launch)
    monkeypatch.setattr("examples.experimental.video_gen_agent.keyframe.h3_generator.POLL_S", 0.01)
    args = _args(tmp_path, video_mode="h3-ref", gpus="0,1,2", port_base=find_available_port(39000))
    # The maker exports the run's env as the launcher would; registered here so teardown takes it back out.
    for name in _agent_env(args) | dict.fromkeys(("GENERATOR_URL", "IMAGE_GENERATOR_URL", "IMAGE_DIR", "CLIP_DIR")):
        monkeypatch.setenv(name if name.startswith("AGENT_") else f"AGENT_VIDEO_GEN_{name}", "")
    try:
        anchors.make_anchors(args)
    finally:
        for server in servers:
            server.stop()

    assert launched == [
        ("Z-Image-Turbo", [0], {}),
        ("FastVideo-FastH3-Omni-8-Step-V1", [1, 2], {"ulysses_degree": 2}),
    ]
    bodies = [entry[1] for entry in seen if isinstance(entry[1], dict)]
    assert len(bodies) == 3 and {(b["seconds"], b["target"]["duration_seconds"], b["task"]) for b in bodies} == {
        (4, 4.0, "ref2va")
    }
    anchor_dir = tmp_path / "video_gen_anchors_h3-ref"
    assert sorted(p.name for p in anchor_dir.iterdir() if p.is_file()) == sorted(
        anchor_name(r) for r in ("a fox", "a red car", "a tv")
    )


def test_h3_anchors_need_whole_engines_and_have_a_measured_rate(tmp_path, requests):
    with pytest.raises(AssertionError, match="do not split into engines of 2"):
        _args(tmp_path, video_mode="h3-ref", gpus="0,1")
    assert "about 0.0 GPU-h" in anchors.plan(_args(tmp_path, video_mode="h3-ref", gpus="0,1,2"))[1]
