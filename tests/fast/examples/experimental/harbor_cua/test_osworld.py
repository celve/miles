from __future__ import annotations

import json
from io import BytesIO

import httpx
import pytest
from examples.experimental.harbor_cua import trial_driver
from examples.experimental.harbor_cua.options import HarborCuaOptions
from examples.experimental.harbor_cua.osworld_qwen.qwen3vl import (
    Qwen3VLPolicy,
    parse_response,
    process_image,
    smart_resize,
)
from examples.experimental.harbor_cua.recorder import CallRecorder
from PIL import Image


def screenshot(width: int = 1920, height: int = 1080) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), (40, 90, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


def tool_call(action: str, **arguments) -> str:
    return f"Action: do it\n<tool_call>\n{json.dumps({'name': 'computer_use', 'arguments': {'action': action, **arguments}})}\n</tool_call>"


def test_a_1080p_screenshot_pads_to_multiples_of_32():
    assert smart_resize(1080, 1920, factor=32, max_pixels=16 * 16 * 4 * 12800) == (1088, 1920)
    _, width, height = process_image(screenshot())
    assert (width, height) == (1920, 1088)
    assert (width // 16) * (height // 16) // 4 == 2040


def test_the_first_step_carries_the_instruction_and_no_previous_actions():
    messages, screen = Qwen3VLPolicy().build_messages("Open the file.", screenshot())

    assert screen == (1920, 1080)
    assert [m["role"] for m in messages] == ["system", "user"]
    image, text = messages[1]["content"]
    assert image["image_url"]["url"].startswith("data:image/png;base64,")
    assert "Instruction: Open the file." in text["text"]
    assert "Previous actions:\nNone" in text["text"]
    assert "The screen's resolution is 1000x1000." in messages[0]["content"][0]["text"]


def test_history_keeps_the_last_four_steps_and_lists_older_actions_as_text():
    policy = Qwen3VLPolicy(history_n=4)
    for step in range(6):
        policy.build_messages("Do the task.", screenshot())
        policy.record(
            f"Action: step {step}\n" + tool_call("left_click", coordinate=[10, 10]).split("\n", 1)[1], (1920, 1080)
        )

    messages, _ = policy.build_messages("Do the task.", screenshot())

    assert [m["role"] for m in messages] == ["system"] + ["user", "assistant"] * 4 + ["user"]
    first_history = messages[1]["content"]
    assert len(first_history) == 2 and "Step 1: step 0\nStep 2: step 1" in first_history[1]["text"]
    assert all(len(m["content"]) == 1 for m in messages[3:9:2])
    assert messages[2]["content"][0]["text"].startswith("Action: step 2")


@pytest.mark.parametrize(
    "response, code",
    [
        (tool_call("left_click", coordinate=[500, 500]), ["pyautogui.click(960, 540)"]),
        (tool_call("double_click", coordinate=[999, 0]), ["pyautogui.doubleClick(1920, 0)"]),
        (tool_call("right_click"), ["pyautogui.rightClick()"]),
        (tool_call("key", keys=["ctrl", "s"]), ["pyautogui.hotkey('ctrl', 's')"]),
        (tool_call("key", keys=["enter"]), ["pyautogui.press('enter')"]),
        (
            tool_call("type", text="a\nb"),
            [
                "pyautogui.typewrite('a', interval=0.03)",
                "pyautogui.press('enter')",
                "pyautogui.typewrite('b', interval=0.03)",
            ],
        ),
        (tool_call("scroll", pixels=-5), ["pyautogui.scroll(-5)"]),
        (tool_call("left_click_drag", coordinate=[100, 200]), ["pyautogui.dragTo(192, 216, duration=0.5)"]),
        (tool_call("wait", time=2), ["WAIT"]),
        (tool_call("terminate", status="success"), ["DONE"]),
    ],
)
def test_tool_calls_become_pyautogui_on_the_real_screen(response, code):
    instruction, actions = parse_response(response, 1920, 1080)
    assert actions == code
    assert instruction == "do it"


def test_a_malformed_coordinate_is_skipped_rather_than_raised():
    assert parse_response(tool_call("left_click", coordinate=[1, 2, 3]), 1920, 1080)[1] == []


def test_a_terminate_without_an_action_line_names_itself():
    response = '<tool_call>\n{"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}\n</tool_call>'
    assert parse_response(response, 1920, 1080) == ("Performing DONE action", ["DONE"])


def test_a_remote_runner_needs_a_recorder_its_machine_can_reach():
    with pytest.raises(ValueError, match="recorder"):
        HarborCuaOptions(runner_url="http://kvm:8765")
    HarborCuaOptions(runner_url="http://kvm:8765", recorder_host="0.0.0.0", recorder_advertise_host="100.1.2.3")


async def test_the_recorder_advertises_the_host_the_runner_dials():
    recorder = CallRecorder(upstream="http://engine", host="127.0.0.1", advertise_host="gpu.example.ts.net")
    await recorder.start()
    try:
        assert recorder.base_url("tok") == f"http://gpu.example.ts.net:{recorder.port}/r/tok/v1"
    finally:
        await recorder.stop()


@pytest.fixture
def runner(monkeypatch):
    """Answer the runner's POST with ``reply``, a payload or an exception."""
    seen = {}

    def install(reply):
        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            if isinstance(reply, Exception):
                raise reply
            return httpx.Response(200, json=reply)

        real = httpx.AsyncClient
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
        return seen

    return install


async def remote(**overrides):
    kwargs = {
        "runner_url": "http://kvm:8765",
        "instance_id": "gimp__7a4deb26",
        "api_base": "http://gpu:1/r/tok/v1",
        "sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 1024, "sampling_seed": 7},
        "max_steps": 15,
        "history_n": 4,
        "timeout_s": 60.0,
    }
    return await trial_driver.run_remote_trial(**{**kwargs, **overrides})


async def test_the_runner_receives_miles_sampling_rather_than_upstream_greedy(runner):
    seen = runner({"exit_status": "AgentError", "error": "boom"})
    await remote()
    assert {k: seen["body"][k] for k in ("temperature", "top_p", "top_k", "max_tokens", "seed", "history_n")} == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 1024,
        "seed": 7,
        "history_n": 4,
    }


async def test_a_runner_failure_is_an_infrastructure_failure(runner):
    runner({"exit_status": "AgentError", "error": "RuntimeError: vm never became ready"})
    outcome = await remote()
    assert outcome.infrastructure_failure
    assert "vm never became ready" in outcome.metadata["runner_error"]


async def test_an_unreachable_runner_is_an_infrastructure_failure(runner):
    runner(httpx.ConnectError("refused"))
    assert (await remote()).infrastructure_failure


async def test_a_trial_timeout_is_scored_like_a_local_one(runner):
    runner({"exit_status": "TimeLimitExceeded", "error": "timeout"})
    outcome = await remote()
    assert not outcome.infrastructure_failure and outcome.reward == 0.0
