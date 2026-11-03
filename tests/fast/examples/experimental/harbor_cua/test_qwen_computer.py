"""Qwen3-VL's grid coordinates reach computer-1's 1024x900 desktop as the pixels it pointed at."""

import base64
import json

import pytest

pytest.importorskip("harbor", reason="pip install harbor -- the recipe's one extra dependency")

from examples.experimental.harbor_cua.qwen_computer import Qwen3VLComputer1  # noqa: E402
from harbor.agents.computer_1.runtime import DisplayGeometry, normalize_completion_action  # noqa: E402
from harbor.llms.base import LLMResponse  # noqa: E402

from miles.utils.eval_record import load_episode, start_episode  # noqa: E402

DESKTOP = DisplayGeometry(desktop_width=1024, desktop_height=900, window_width=1024, window_height=900)


@pytest.fixture
def provider(tmp_path):
    agent = Qwen3VLComputer1(logs_dir=tmp_path, model_name="openai/model", provider="litellm", enable_images=True)
    return agent._build_provider()


def _executed(provider, action: dict, desktop: DisplayGeometry = DESKTOP):
    step = provider.parse(LLMResponse(content=json.dumps({"analysis": "", "plan": "", "action": action})))
    return normalize_completion_action(step.action, desktop)


def test_the_prompt_names_the_screen_qwen3_vl_answers_on(provider):
    text = provider.record_text("Click Start.")

    assert "1000x1000 pixels" in text
    assert "1024x900" not in text


def test_lin_2045s_base_clicks_land_on_the_buttons_they_aimed_at(provider):
    start = _executed(provider, {"type": "click", "x": 499, "y": 284})
    finish = _executed(provider, {"type": "click", "x": 499, "y": 431})

    assert (start.x, start.y) == (511, 256)
    assert (finish.x, finish.y) == (511, 388)
    assert (finish.model_x, finish.model_y) == (499, 431)


def test_a_drag_scales_both_ends(provider):
    drag = _executed(provider, {"type": "drag", "x": 0, "y": 0, "end_x": 999, "end_y": 999})

    assert (drag.x, drag.y, drag.end_x, drag.end_y) == (0, 0, 1023, 899)


def test_the_grid_scales_to_whatever_desktop_the_agent_runs(tmp_path):
    agent = Qwen3VLComputer1(
        logs_dir=tmp_path,
        model_name="openai/model",
        provider="litellm",
        enable_images=True,
        desktop_width=1280,
        desktop_height=720,
        window_width=1280,
        window_height=720,
    )
    provider = agent._build_provider()
    corner = _executed(provider, {"type": "click", "x": 999, "y": 999}, agent._desktop_geometry)

    assert "1000x1000 pixels" in provider.record_text("Click Start.")
    assert (corner.x, corner.y) == (1279, 719)


def test_a_reply_without_an_action_is_still_retried(provider):
    step = provider.parse(LLMResponse(content="no json here"))

    assert step.needs_retry
    assert step.action is None


def _data_url(payload: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode()


def _reply(action: dict) -> LLMResponse:
    return LLMResponse(content=json.dumps({"analysis": "a", "plan": "p", "action": action}))


def test_a_recording_agent_keeps_each_turns_screen_reply_and_action(tmp_path):
    directory = start_episode(tmp_path / "episode", stack="harbor", task_id="t", sample_index=0)
    agent = Qwen3VLComputer1(
        logs_dir=tmp_path, model_name="openai/model", provider="litellm", enable_images=True, record_dir=str(directory)
    )
    provider = agent._build_provider()

    provider.initial_messages("Click Start.", _data_url(b"screen-0"))
    provider.parse(_reply({"type": "click", "x": 499, "y": 284}))
    provider.parse(LLMResponse(content="not json"))
    provider.follow_up_messages(None, "clicked", _data_url(b"screen-1"))
    provider.parse(_reply({"type": "done", "result": "ok"}))

    episode = load_episode(directory)
    assert episode.instruction == "Click Start."
    assert [step.screenshot.read_bytes() for step in episode.steps] == [b"screen-0", b"screen-0", b"screen-1"]
    assert (episode.steps[0].action["type"], episode.steps[0].action["x"]) == ("click", 499)
    assert (episode.steps[1].action, episode.steps[1].output) == (None, "not json")
    assert episode.steps[1].error.startswith("ERROR")
    assert episode.steps[2].action["type"] == "done"


def test_an_agent_without_a_record_dir_records_nothing(provider):
    assert provider.record is None
