"""The computer MCP server: Qwen3-VL's computer_use tool over stdio, driving xdotool on the 0..999 grid."""

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest
from examples.experimental.harbor_cua import computer_mcp as mcp
from examples.experimental.harbor_cua.osworld_qwen.qwen3vl import _system_prompt

from tests.fast.examples.experimental.harbor_cua.conftest import png_bytes

GEOMETRY = (1920, 1080)


def _qwen_tool() -> dict:
    tools = _system_prompt().split("<tools>\n", 1)[1].split("\n</tools>", 1)[0]
    return json.loads(tools)["function"]


def test_the_tool_is_qwen3_vls_computer_use():
    qwen = _qwen_tool()
    assert qwen["name"] == "computer_use"
    assert mcp.TOOL["description"] == qwen["description"]
    assert set(qwen["parameters"]["properties"]["action"]["enum"]) <= set(mcp.ACTIONS)
    assert mcp.TOOL["inputSchema"]["properties"].keys() == qwen["parameters"]["properties"].keys()


@pytest.mark.parametrize(
    "args, argv",
    [
        ({"action": "left_click", "coordinate": [500, 300]}, ["mousemove", "960", "324", "click", "1"]),
        ({"action": "right_click", "coordinate": [0, 999]}, ["mousemove", "0", "1079", "click", "3"]),
        (
            {"action": "double_click", "coordinate": [999, 0]},
            ["mousemove", "1919", "0", "click", "--repeat", "2", "1"],
        ),
        ({"action": "triple_click", "coordinate": [10, 10]}, ["mousemove", "19", "11", "click", "--repeat", "3", "1"]),
        (
            {"action": "left_click_drag", "coordinate": [500, 500]},
            ["mousedown", "1", "mousemove", "960", "540", "mouseup", "1"],
        ),
        ({"action": "type", "text": "hi\nthere"}, ["type", "--clearmodifiers", "--", "hi\nthere"]),
        ({"action": "key", "keys": ["ctrl", "c"]}, ["key", "--clearmodifiers", "ctrl+c"]),
        ({"action": "key", "keys": ["Enter"]}, ["key", "--clearmodifiers", "Return"]),
        ({"action": "scroll", "pixels": -5}, ["click", "--repeat", "5", "5"]),
        (
            {"action": "scroll", "pixels": 3, "coordinate": [500, 500]},
            ["mousemove", "960", "540", "click", "--repeat", "3", "4"],
        ),
        ({"action": "wait", "time": 1}, None),
        ({"action": "terminate", "status": "success"}, None),
    ],
)
def test_xdotool_argv(args, argv):
    assert mcp.xdotool_argv(args, GEOMETRY) == argv


@pytest.mark.parametrize(
    "args",
    [
        {"action": "fly"},
        {"action": "left_click"},
        {"action": "left_click", "coordinate": ["a", 1]},
        {"action": "key", "keys": []},
        {"action": "type"},
        {"action": "scroll", "pixels": "up"},
    ],
)
def test_malformed_actions_are_the_policys_errors(args):
    with pytest.raises(mcp.ActionError):
        mcp.xdotool_argv(args, GEOMETRY)


def test_clicks_match_computer_1(monkeypatch):
    runtime = pytest.importorskip("harbor.agents.computer_1.runtime")
    geometry = runtime.DisplayGeometry(desktop_width=GEOMETRY[0], desktop_height=GEOMETRY[1])
    for qwen, harbor_type in (
        ("left_click", "click"),
        ("double_click", "double_click"),
        ("right_click", "right_click"),
    ):
        for x, y in ((0, 0), (500, 300), (999, 999), (123, 877)):
            hx, hy = runtime.scale_normalized_coordinate(x, y, geometry)
            action = runtime.ComputerAction(type=harbor_type, x=hx, y=hy)
            assert [mcp.xdotool_argv({"action": qwen, "coordinate": [x, y]}, GEOMETRY)] == runtime.build_xdotool_argv(
                action, geometry
            )


@pytest.fixture
def fake_desktop(tmp_path):
    """A PATH whose xdotool logs its argv, and a capture command that copies a fixed PNG."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "xdotool").write_text(
        '#!/bin/bash\necho "$*" >> "$FAKE_XDOTOOL_LOG"\n[ "$1" = getdisplaygeometry ] && echo "1920 1080"\nexit 0\n'
    )
    (bin_dir / "xdotool").chmod(0o755)
    screen = tmp_path / "screen.png"
    screen.write_bytes(png_bytes((1, 2, 3)))
    return {"bin": bin_dir, "screen": screen, "xdotool_log": tmp_path / "xdotool.log", "log": tmp_path / "mcp.jsonl"}


def _serve(fake_desktop, messages: list[dict], capture_cmd: str) -> list[dict]:
    env = {"PATH": f"{fake_desktop['bin']}:/usr/bin:/bin", "FAKE_XDOTOOL_LOG": str(fake_desktop["xdotool_log"])}
    argv = [sys.executable, mcp.__file__, "--settle-sec", "0", "--capture-cmd", capture_cmd]
    argv += ["--log", str(fake_desktop["log"])]
    stdin = "".join(json.dumps(m) + "\n" for m in messages)
    out = subprocess.run(argv, input=stdin, capture_output=True, text=True, env=env, timeout=60, check=True).stdout
    return [json.loads(line) for line in out.splitlines()]


def _call(i: int, arguments: dict) -> dict:
    return {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": "use", "arguments": arguments}}


def test_an_mcp_session_over_stdio(fake_desktop):
    responses = _serve(
        fake_desktop,
        [
            {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            _call(2, {"action": "left_click", "coordinate": [500, 300]}),
            _call(3, {"action": "left_click"}),
            _call(4, {"action": "terminate", "status": "success"}),
        ],
        f"cp {fake_desktop['screen']} {{out}}",
    )
    assert [r["id"] for r in responses] == [0, 1, 2, 3, 4]
    assert responses[0]["result"]["protocolVersion"] == "2025-03-26"
    assert responses[1]["result"]["tools"] == [mcp.TOOL]
    text, image = responses[2]["result"]["content"]
    assert image["mimeType"] == "image/png"
    assert base64.b64decode(image["data"]) == fake_desktop["screen"].read_bytes()
    assert responses[3]["result"]["isError"] is True
    assert [part["type"] for part in responses[4]["result"]["content"]] == ["text"]
    assert fake_desktop["xdotool_log"].read_text().splitlines() == ["getdisplaygeometry", "mousemove 960 324 click 1"]
    rows = [json.loads(line) for line in fake_desktop["log"].read_text().splitlines()]
    assert [("error" in r, r.get("screenshot")) for r in rows] == [(False, True), (True, None), (False, False)]


def test_a_failed_capture_is_logged_as_the_desktops_failure(fake_desktop):
    responses = _serve(fake_desktop, [_call(1, {"action": "wait", "time": 0})], "exit 3")
    assert responses[0]["result"]["isError"] is True
    (row,) = [json.loads(line) for line in Path(fake_desktop["log"]).read_text().splitlines()]
    assert "infra_error" in row
