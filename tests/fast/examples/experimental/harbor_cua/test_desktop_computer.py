"""computer-1 on a desktop its image already runs: no browser of its own, a settle after each action."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor", reason="pip install harbor -- the recipe's one extra dependency")

from examples.experimental.harbor_cua import desktop_computer as dc  # noqa: E402
from harbor.agents.computer_1.computer_1 import Computer1  # noqa: E402
from harbor.agents.computer_1.providers import generic  # noqa: E402
from harbor.agents.computer_1.runtime import ComputerAction, RuntimeRequestError  # noqa: E402


class FakeEnvironment:
    default_user = "user"

    def __init__(self, fail_on: str = ""):
        self.commands: list[str] = []
        self._fail_on = fail_on

    async def exec(self, command, timeout_sec=None, user=None, **_):
        self.commands.append(command)
        if self._fail_on and self._fail_on in command:
            return SimpleNamespace(return_code=1, stdout="", stderr="boom")
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
    [({"screenshot_width": 1280}, "together"), ({"keep_screenshots": 0}, "at least 1")],
)
def test_inconsistent_settings_are_refused(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _agent(tmp_path, **kwargs)
