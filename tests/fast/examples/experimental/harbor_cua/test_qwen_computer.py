"""Qwen3-VL's grid coordinates reach computer-1's 1024x900 desktop as the pixels it pointed at."""

import json

import pytest

pytest.importorskip("harbor", reason="pip install harbor -- the recipe's one extra dependency")

from examples.experimental.harbor_cua.qwen_computer import Qwen3VLComputer1  # noqa: E402
from harbor.agents.computer_1.runtime import DisplayGeometry, normalize_completion_action  # noqa: E402
from harbor.llms.base import LLMResponse  # noqa: E402

DESKTOP = DisplayGeometry(desktop_width=1024, desktop_height=900, window_width=1024, window_height=900)


@pytest.fixture
def provider(tmp_path):
    agent = Qwen3VLComputer1(logs_dir=tmp_path, model_name="openai/model", provider="litellm", enable_images=True)
    return agent._build_provider()


def _executed(provider, action: dict):
    step = provider.parse(LLMResponse(content=json.dumps({"analysis": "", "plan": "", "action": action})))
    return normalize_completion_action(step.action, DESKTOP)


def test_the_prompt_names_the_screen_qwen3_vl_answers_on(provider):
    text = provider.record_text("Click Start.")

    assert "1000x1000 pixels" in text
    assert "1024x900" not in text


def test_lin_2045s_base_clicks_land_on_the_buttons_they_aimed_at(provider):
    """LIN-2045's base model clicked (499, 284) and (499, 431) for buttons centred at (512, 255) and (512, 391)."""
    start = _executed(provider, {"type": "click", "x": 499, "y": 284})
    finish = _executed(provider, {"type": "click", "x": 499, "y": 431})

    assert (start.x, start.y) == (511, 256)
    assert (finish.x, finish.y) == (511, 388)
    assert (finish.model_x, finish.model_y) == (499, 431)


def test_a_drag_scales_both_ends(provider):
    drag = _executed(provider, {"type": "drag", "x": 0, "y": 0, "end_x": 999, "end_y": 999})

    assert (drag.x, drag.y, drag.end_x, drag.end_y) == (0, 0, 1023, 899)


def test_a_reply_without_an_action_is_still_retried(provider):
    step = provider.parse(LLMResponse(content="no json here"))

    assert step.needs_retry
    assert step.action is None
