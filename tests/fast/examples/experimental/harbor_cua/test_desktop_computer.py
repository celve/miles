"""computer-1 on a desktop its image already runs: no browser of its own, a settle after each action."""

import asyncio
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor", reason="pip install harbor -- the recipe's one extra dependency")

from examples.experimental.harbor_cua import desktop_computer as dc  # noqa: E402
from harbor.agents.computer_1.computer_1 import Computer1  # noqa: E402
from harbor.agents.computer_1.providers import generic  # noqa: E402
from harbor.agents.computer_1.runtime import ComputerAction, RuntimeRequestError  # noqa: E402


class FakeEnvironment:
    default_user = "user"

    def __init__(self, fail_on: str = "", display: str = "1920 1080", block_output: str = "blocked: /usr/bin/xterm"):
        self.commands: list[str] = []
        self.users: list[str | None] = []
        self._fail_on = fail_on
        self._display = display
        self._block_output = block_output

    async def exec(self, command, timeout_sec=None, user=None, **_):
        self.commands.append(command)
        self.users.append(user)
        if self._fail_on and self._fail_on in command:
            return SimpleNamespace(return_code=1, stdout="", stderr="boom")
        if "getdisplaygeometry" in command:
            return SimpleNamespace(return_code=0, stdout=self._display + "\n", stderr="")
        if "gsettings" in command:
            return SimpleNamespace(return_code=0, stdout=self._block_output, stderr="")
        return SimpleNamespace(return_code=0, stdout="ok" if "X11-unix" in command else "", stderr="")


def _agent(tmp_path, **kwargs):
    return dc.DesktopQwen3VLComputer1(
        logs_dir=tmp_path,
        model_name="openai/model",
        provider="litellm",
        enable_images=True,
        desktop_width=1920,
        desktop_height=1080,
        **kwargs,
    )


def _session(environment, settle_sec=2.0, screenshot_size=(1280, 720)):
    return dc.DesktopSession(
        environment=environment,
        agent_dir="/logs/agent",
        desktop_width=1920,
        desktop_height=1080,
        window_width=1920,
        window_height=1080,
        settle_sec=settle_sec,
        screenshot_size=screenshot_size,
    )


@pytest.fixture
def sleeps(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(dc.asyncio, "sleep", fake_sleep)
    return slept


def test_the_prompt_offers_no_browser_or_zoom_actions(tmp_path):
    provider = _agent(tmp_path)._build_provider()
    text = provider.record_text("Bold the title.")

    assert '"zoom"' not in text and "zoom_region" not in text and "navigate" not in text
    assert "1000x1000 pixels" in text
    assert provider.screenshot_format == "png"


def test_a_changed_harbor_template_fails_loudly():
    template = (generic._TEMPLATES_DIR / "computer-1-json.txt").read_text()

    with pytest.raises(ValueError, match="navigate"):
        dc.desktop_prompt_template(template.replace('\n- For "navigate": provide the destination URL in "url".', ""))


def test_start_uses_the_running_display_and_launches_nothing(sleeps):
    env = FakeEnvironment()
    asyncio.run(_session(env).start())

    joined = "\n".join(env.commands)
    assert "X11-unix/X1" in joined
    for launched in ("Xvfb", "startxfce4", "x11vnc", "chromium"):
        assert launched not in joined


def test_clicks_follow_the_display_not_the_configured_size(sleeps):
    env = FakeEnvironment(display="1920 1080")
    session = dc.DesktopSession(
        environment=env,
        agent_dir="/logs/agent",
        desktop_width=1024,
        desktop_height=900,
        settle_sec=0,
        screenshot_size=None,
    )
    asyncio.run(session.start())
    asyncio.run(session.execute(ComputerAction(type="click", x=999, y=999, source="normalized_completion")))

    assert (session.geometry.desktop_width, session.geometry.desktop_height) == (1920, 1080)
    assert "mousemove 1919 1079 click 1" in env.commands[-1]


def test_the_session_lives_while_the_display_does():
    assert asyncio.run(_session(FakeEnvironment()).is_session_alive())
    assert not asyncio.run(_session(FakeEnvironment(fail_on="X11-unix")).is_session_alive())


@pytest.mark.parametrize("action_type", sorted(dc.UNAVAILABLE_ACTIONS))
def test_browser_and_zoom_actions_are_refused(sleeps, action_type):
    env = FakeEnvironment()

    with pytest.raises(RuntimeRequestError, match="not available"):
        asyncio.run(_session(env).execute(ComputerAction(type=action_type, url="https://x", zoom_region=[0, 0, 9, 9])))
    assert env.commands == []


def test_a_screen_changing_action_settles_before_the_screenshot(sleeps):
    env = FakeEnvironment()
    asyncio.run(_session(env, settle_sec=2.0).execute(ComputerAction(type="click", x=10, y=20)))

    assert any("xdotool" in c for c in env.commands)
    assert sleeps[-1] == 2.0


def test_wait_and_done_add_no_settle(sleeps):
    session = _session(FakeEnvironment(), settle_sec=2.0)
    asyncio.run(session.execute(ComputerAction(type="wait")))
    asyncio.run(session.execute(ComputerAction(type="done", text="ok")))

    assert sleeps == [1.0]


def test_a_failed_action_relaunches_no_browser(sleeps):
    env = FakeEnvironment(fail_on="xdotool")

    with pytest.raises(RuntimeRequestError):
        asyncio.run(_session(env).execute(ComputerAction(type="click", x=10, y=20)))
    assert not any("chromium" in c for c in env.commands)


def test_screenshots_are_shrunk_in_the_sandbox():
    env = FakeEnvironment()
    asyncio.run(_session(env).fetch_screenshot("/logs/agent/screenshot_ep0.png"))
    asyncio.run(_session(env, screenshot_size=None).fetch_screenshot("/logs/agent/screenshot_ep1.png"))

    assert "-resize 1280x720!" in env.commands[0]
    assert "-resize" not in env.commands[1]
    with pytest.raises(ValueError, match="PNG"):
        asyncio.run(_session(env).fetch_screenshot("/logs/agent/screenshot_ep2.webp"))


def test_an_image_without_imagemagick_can_bring_its_own_capture():
    class OwnCapture(dc.DesktopSession):
        def _capture_script(self, env_png):
            return f"grab-screen {env_png}"

    env = FakeEnvironment()
    asyncio.run(
        OwnCapture(environment=env, agent_dir="/logs/agent", settle_sec=0, screenshot_size=None).fetch_screenshot(
            "/x/s.png"
        )
    )

    assert "grab-screen /tmp/computer_1-screenshots/latest.png" in env.commands[0]
    assert "import -window" not in env.commands[0]


def test_setup_builds_a_desktop_session_from_the_agents_settings(tmp_path, sleeps):
    agent = _agent(tmp_path, settle_sec=1.5, screenshot_width=1280, screenshot_height=720)
    asyncio.run(agent.setup(FakeEnvironment()))

    session = agent._session
    assert type(session) is agent.session_class is dc.DesktopSession
    assert (session._settle_sec, session._screenshot_size) == (1.5, (1280, 720))
    assert (session.geometry.window_width, session.geometry.window_height) == (1920, 1080)


def _image_turn(text):
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": "x"}}],
    }


def _images(messages):
    return sum(
        part.get("type") == "image_url"
        for m in messages
        for part in (m["content"] if isinstance(m.get("content"), list) else [])
    )


_PARSE_RETRY = "Previous response had parsing errors"


@pytest.mark.parametrize(
    ("keep", "prompt", "carried"),
    [
        (1, [_image_turn("now")], 1),
        (1, _PARSE_RETRY, 1),
        (2, [_image_turn("now")], 2),
        (2, _PARSE_RETRY, 2),
        (None, [_image_turn("now")], 4),
    ],
)
def test_a_request_carries_at_most_keep_screenshots(tmp_path, monkeypatch, keep, prompt, carried):
    seen = {}

    async def fake_query(self, chat, prompt, *args, **kwargs):
        seen["images"] = _images(chat.messages) + (_images(prompt) if isinstance(prompt, list) else 0)

    monkeypatch.setattr(Computer1, "_query_litellm", fake_query)
    chat = SimpleNamespace(messages=[_image_turn("t0"), _image_turn("t1"), _image_turn("t2")])
    asyncio.run(_agent(tmp_path, keep_screenshots=keep)._query_litellm(chat, prompt))

    assert seen["images"] == carried


@pytest.mark.parametrize(("latest", "history_images"), [("/logs/agent/screenshot_ep1.png", 0), (None, 1)])
def test_the_turn_cap_fallback_carries_one_screen(tmp_path, latest, history_images):
    agent = _agent(tmp_path, keep_screenshots=1)
    agent._chat = SimpleNamespace(
        messages=[_image_turn("t0"), {"role": "assistant", "content": "a"}, _image_turn("t1")]
    )
    agent._latest_screenshot_path = latest

    assert _images(agent._fallback_message_history()) == history_images
    assert _images(agent._chat.messages) == 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"screenshot_width": 1280}, "together"),
        ({"keep_screenshots": 0}, "at least 1"),
        ({"keep_turns": 0}, "at least 1"),
        ({"loop_limit": 1}, "at least 2"),
    ],
)
def test_inconsistent_settings_are_refused(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _agent(tmp_path, **kwargs)


def _reply(action):
    return {"role": "assistant", "content": f'{{"analysis": "a", "plan": "p", "action": {action}}}'}


def _turns(n):
    messages = [_image_turn("instruction")]
    for i in range(n):
        messages += [_reply(f'{{"type": "click", "x": {i}, "y": 0}}'), _image_turn(f"t{i + 1}")]
    return messages[:-1]


def _run_queries(tmp_path, monkeypatch, agent, histories):
    seen = []

    async def fake_query(self, chat, prompt, *args, **kwargs):
        seen.append([dict(m) for m in chat.messages])

    monkeypatch.setattr(Computer1, "_query_litellm", fake_query)
    chat = SimpleNamespace(messages=[])
    for extra in histories:
        chat.messages += extra
        asyncio.run(agent._query_litellm(chat, [_image_turn("now")]))
    return seen


def test_turns_beyond_keep_turns_collapse_to_one_line_each(tmp_path, monkeypatch):
    agent = _agent(tmp_path, keep_turns=2)
    seen = _run_queries(tmp_path, monkeypatch, agent, [_turns(4)])

    head, summary, *kept = seen[0]
    assert head["content"][0]["text"] == "instruction"
    assert summary["content"].splitlines()[1:] == [
        '1. {"type":"click","x":0,"y":0}',
        '2. {"type":"click","x":1,"y":0}',
    ]
    assert [m["role"] for m in kept] == ["user", "assistant", "user", "assistant"]
    assert '"x": 3' in kept[-1]["content"]


def test_the_summary_grows_across_calls_without_renumbering(tmp_path, monkeypatch):
    agent = _agent(tmp_path, keep_turns=1)
    later = [_image_turn("t5"), _reply('{"type": "type", "text": "hi"}')]
    seen = _run_queries(tmp_path, monkeypatch, agent, [_turns(3), later])

    lines = seen[1][1]["content"].splitlines()[1:]
    assert lines == [f'{i + 1}. {{"type":"click","x":{i},"y":0}}' for i in range(3)]
    assert len(seen[1]) == 4


def test_a_short_history_is_left_alone(tmp_path, monkeypatch):
    seen = _run_queries(tmp_path, monkeypatch, _agent(tmp_path, keep_turns=3), [_turns(3)])
    assert seen[0] == _turns(3)


def _record_executions(monkeypatch):
    executed = []

    async def fake_execute(self, action, episode):
        executed.append(action)

    monkeypatch.setattr(Computer1, "_execute_action", fake_execute)
    return executed


@pytest.mark.parametrize(
    ("actions", "executed_before_stop"),
    [
        ([ComputerAction(type="click", x=5, y=5)] * 5, 4),
        ([ComputerAction(type="click", x=5, y=5), ComputerAction(type="key", keys=["Tab"])] * 5, 9),
    ],
)
def test_the_action_that_closes_a_loop_is_never_executed(tmp_path, monkeypatch, actions, executed_before_stop):
    executed = _record_executions(monkeypatch)
    agent = _agent(tmp_path, loop_limit=5)
    with pytest.raises(dc.LoopDetected):
        for action in actions:
            asyncio.run(agent._execute_action(action, 0))
    assert len(executed) == executed_before_stop
    assert agent._early_termination_reason == "loop_detected"


@pytest.mark.parametrize(
    "actions",
    [[ComputerAction(type="click", x=5, y=5)] * 4, [ComputerAction(type="click", x=i, y=5) for i in range(5)]],
)
def test_actions_short_of_a_loop_all_execute(tmp_path, monkeypatch, actions):
    executed = _record_executions(monkeypatch)
    agent = _agent(tmp_path, loop_limit=5)
    for action in actions:
        asyncio.run(agent._execute_action(action, 0))
    assert len(executed) == len(actions)


def test_a_detected_loop_ends_the_run_without_an_error(tmp_path, monkeypatch):
    async def looping_run(self, *args, **kwargs):
        raise dc.LoopDetected("click")

    monkeypatch.setattr(dc.Qwen3VLComputer1, "run", looping_run)
    asyncio.run(_agent(tmp_path, loop_limit=5).run("task", FakeEnvironment(), SimpleNamespace()))


def test_harbors_own_llm_calls_can_be_kept_off_the_policy(tmp_path, monkeypatch):
    calls = []

    async def fallback(self, instruction):
        calls.append(instruction)

    monkeypatch.setattr(Computer1, "_maybe_write_final_answer_fallback", fallback)
    off = _agent(tmp_path, compaction=False, final_answer_fallback=False)
    asyncio.run(off._maybe_write_final_answer_fallback("task"))
    assert calls == []
    assert asyncio.run(off._compactor.maybe_proactively_compact(SimpleNamespace(messages=[]), "p", "task")) is None
    assert asyncio.run(off._compactor.reactive_compaction(SimpleNamespace(messages=[]), "p", "task")) is None

    asyncio.run(_agent(tmp_path)._maybe_write_final_answer_fallback("task"))
    assert calls == ["task"]


def test_the_terminal_block_runs_as_root_before_the_first_screen(tmp_path, monkeypatch):
    seen = []

    async def first_screen(self, instruction, environment, context):
        seen.append(len(environment.commands))

    monkeypatch.setattr(dc.Qwen3VLComputer1, "run", first_screen)
    env = FakeEnvironment()
    asyncio.run(_agent(tmp_path, block_terminal="ga").run("task", env, SimpleNamespace(metadata=None)))

    assert seen == [1] and env.users == ["root"]
    assert "gnome-terminal" in env.commands[0] and env.commands[0].endswith(" block ga")
    assert (tmp_path / "terminal_block.txt").read_text() == "blocked: /usr/bin/xterm"


def test_a_block_that_leaves_a_terminal_loses_the_sandbox(tmp_path):
    env = FakeEnvironment(block_output="STILL-RUNNABLE gnome-terminal")

    with pytest.raises(RuntimeRequestError, match="STILL-RUNNABLE"):
        asyncio.run(_agent(tmp_path, block_terminal="ga").run("task", env, SimpleNamespace(metadata=None)))


_VRML = (
    '#VRML_SIM R2023b utf8\nWorldInfo {\n  gravity 0 0 0\n}\nViewpoint {\n  position 0 0 5\n}\nRobot {\n  name "cubesat"\n}\n'
    * 2
)
_SUMO = '<additional>\n  <parkingArea id="pa_1" lane="E_0_0" startPos="10" endPos="40"/>\n' * 4 + "</additional>\n"
_REPORT = (
    "Freeform Fin Integration Report\n=====\nFin Type: Freeform (3 fins)\nThickness: 3.0 mm\nApogee: 233 m\nStability: 3.97 cal\n"
    * 2
)
_GMAT = "%General Mission Analysis Tool(GMAT) Script\nCreate Spacecraft Sat;\nSat.SMA = 7000;\nCreate Propagator Prop;\nBeginMissionSequence;\nPropagate Prop(Sat) {Sat.ElapsedDays = 1};\n"
_COURSE_FORM = (
    "Create a new course\nCourse name: Introduction to Cell Biology\nShort name: BIO101\n"
    "Summary: This course covers cellular structures and their functions in living organisms.\nVisibility: Show"
)
_BEGIN_REPORT = (
    "Begin Analysis\nStudy: Myoglobin solvent-accessible surface area\nMethod: Values were measured in the application.\n"
    "Result: The selected region has an accessible surface area of 120 square angstroms.\n"
    "Conclusion: The measured area agrees with the displayed structure."
)
_BEGIN_BLOCKS = "Begin Tank\n   Name: PropTank\n   InitialMass: 400.0\n   Isp: 285.0\nEnd Tank\n" * 3
_JSON = (
    '{\n  "sma_angle_degrees": 28.4,\n  "aortomesenteric_distance_mm": 11.2,\n  "notes": "measured in the axial view"\n}\n'
    * 2
)


@pytest.mark.parametrize(
    ("typed", "gated"),
    [
        ("cat > /tmp/out.csv << EOF", True),
        ("import bpy", True),
        ("cd /home/ga && ls -l", True),
        ("  echo indented", True),
        ("\n  ls -l /home/ga", True),
        (_GMAT, True),
        (_BEGIN_BLOCKS, True),
        (".shell python3 /tmp/fk.py", True),
        ("!mkdir -p /home/ga/exports", True),
        ("Sub Run()\n  Shell \"bash -c 'ls'\"\nEnd Sub", True),
        (_VRML, True),
        (_SUMO, True),
        ('Fog {\n  fogType "EXPONENTIAL"\n  visibilityRange 50.0\n}', True),
        ("  ContactProperties { coulombFriction 0.4 }", True),
        ("Quarterly report", False),
        ("=SUM(A1:A9)", False),
        ("Introduction to Cells - an overview of cell biology", False),
        ("An echo of the old design", False),
        (_REPORT, False),
        (_JSON, False),
        (_COURSE_FORM, False),
        (_BEGIN_REPORT, False),
        ("Results {see below}", False),
        ("Total {count} items", False),
        ("Results {see [appendix]}", False),
        ("Results {see 2nd paragraph}", False),
        ("Report {status TRUE positive}", False),
        ('{"notes": "Current Results {count 2} were measured"}', False),
        ("SELECT 'Current Results {count 2}' AS notes;", False),
    ],
)
def test_typed_shell_or_code_gates_the_reward(tmp_path, monkeypatch, typed, gated):
    _record_executions(monkeypatch)

    async def one_action(self, instruction, environment, context):
        await self._execute_action(ComputerAction(type="type", text=typed), 0)

    monkeypatch.setattr(dc.Qwen3VLComputer1, "run", one_action)
    context = SimpleNamespace(metadata={"n_episodes": 1})
    asyncio.run(_agent(tmp_path, gate_typed_code=True).run("task", FakeEnvironment(), context))

    assert (dc.REWARD_GATE in context.metadata) is gated
    assert context.metadata["n_episodes"] == 1


def test_a_malformed_typed_node_is_rejected_without_backtracking(tmp_path, monkeypatch):
    _record_executions(monkeypatch)

    async def one_action(self, instruction, environment, context):
        await self._execute_action(ComputerAction(type="type", text="WorldInfo { gravity " + "111 " * 14 + "oops }"), 0)

    monkeypatch.setattr(dc.Qwen3VLComputer1, "run", one_action)
    context = SimpleNamespace(metadata={"n_episodes": 1})
    start = time.perf_counter()
    asyncio.run(_agent(tmp_path, gate_typed_code=True).run("task", FakeEnvironment(), context))

    assert time.perf_counter() - start < 1.0
    assert dc.REWARD_GATE not in context.metadata


def test_without_the_gate_typed_code_is_not_marked(tmp_path, monkeypatch):
    _record_executions(monkeypatch)

    async def one_action(self, instruction, environment, context):
        await self._execute_action(ComputerAction(type="type", text="cat > /tmp/out.csv"), 0)

    monkeypatch.setattr(dc.Qwen3VLComputer1, "run", one_action)
    context = SimpleNamespace(metadata=None)
    asyncio.run(_agent(tmp_path).run("task", FakeEnvironment(), context))

    assert context.metadata is None
