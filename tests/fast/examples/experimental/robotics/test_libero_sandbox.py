"""``libero_sandbox.Sandbox`` against a fake e2b SDK and a local server standing in for the simulator's /health."""

import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import libero_sandbox
import pytest

TOKEN = "traffic-token"


@pytest.fixture
def health():
    """A /health server that answers 401 unless the request carries ``required`` as the traffic token."""
    state = {"required": None, "seen": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            token = self.headers.get(libero_sandbox.TRAFFIC_TOKEN_HEADER)
            state["seen"].append(token)
            self.send_response(200 if token == state["required"] else 401)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["host"] = f"127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()


def start(monkeypatch, host: str, token: str | None) -> libero_sandbox.Sandbox:
    class Handle:
        sandbox_id = "sbx-test"
        traffic_access_token = token
        commands = types.SimpleNamespace(run=lambda *args, **kwargs: None)

        def get_host(self, port):
            return host

        def kill(self, **kwargs):
            pass

        def set_timeout(self, *args, **kwargs):
            pass

    e2b = types.ModuleType("e2b")
    e2b.Sandbox = types.SimpleNamespace(create=lambda **kwargs: Handle())
    monkeypatch.setitem(sys.modules, "e2b", e2b)
    monkeypatch.setenv("E2B_API_KEY", "e2b_" + "0" * 40)
    monkeypatch.setenv("LIBERO_READY_TIMEOUT_S", "3")
    monkeypatch.delenv("LIBERO_E2B_PROXY", raising=False)
    monkeypatch.delenv("LIBERO_E2B_URL_SCHEME", raising=False)
    return libero_sandbox.Sandbox(template="libero-sim-test", max_lifetime_s=600, labels={})


def test_a_private_sandbox_is_reached_with_its_traffic_token(monkeypatch, health):
    health["required"] = TOKEN

    sandbox = start(monkeypatch, health["host"], TOKEN)

    assert sandbox.headers == {"e2b-traffic-access-token": TOKEN}
    assert health["seen"][-1] == TOKEN
    sandbox.kill()


def test_a_public_sandbox_sends_no_token(monkeypatch, health):
    sandbox = start(monkeypatch, health["host"], None)

    assert sandbox.headers == {}
    assert health["seen"] == [None]
    sandbox.kill()
