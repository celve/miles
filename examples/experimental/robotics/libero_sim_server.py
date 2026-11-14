"""One LIBERO episode at a time over HTTP, on PEAR's own simulator code.

Runs in the LIBERO environment PEAR's installer builds (``scripts/install.sh libero``), inside a
``libero-sim`` sandbox or on a devbox, with a PEAR checkout at ``--pear-root``. Scene construction,
seed, initial state, stabilization, the closed-loop pose controller and success all come from
``pear.envs.libero``; this file only keeps the episode's scene open between actions where PEAR
replays the native actions on every call. ``/replay`` runs PEAR's replay, for the probe that checks
the two agree; it builds and closes a scene of its own, which breaks an open episode's renderer, so it
closes the open episode first.

  POST /reset  {"suite", "task_id", "trial_id"}  -> observation
  POST /act    {"action": {"arm": {...}, "gripper": {"width": ...}}}  -> observation
  POST /replay {"suite", "task_id", "trial_id", "actions"}  -> PEAR's observation after those native actions
  GET  /health

An observation is ``{"state", "images": {"agentview", "wrist"} | null, "done", "success", "outcome",
"executed", "actions", "steps", "max_steps", "instruction", "seconds"}``: frames as base64 PNG, and the
native actions the call executed.

Start it through ``libero_sandbox.START_SCRIPT``, which sets the Mesa variables PEAR's
``LiberoEnv.setup_env`` gives its simulator processes before Python starts.
"""

import argparse
import base64
import io
import json
import logging
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logger = logging.getLogger("libero_sim_server")


class InvalidRequest(ValueError):
    pass


class Episode:
    """One episode's open scene, as PEAR's ``LiberoEnv.rollout`` would leave it after each call."""

    def __init__(self, suite: str, task_id: int, trial_id: int):
        from pear.envs.libero.env import LiberoEnv

        self.env = LiberoEnv(suite, task_id, trial_id)
        self.max_steps = LiberoEnv.MAX_STEP_N_BY_SUITE[suite]
        self.actions: list[list[float]] = []
        self.done = False
        self.observation = self.env._open_scene()

    def first(self) -> dict[str, Any]:
        return self._result([], [], None, rendered=True)

    def act(self, action: dict[str, Any]) -> dict[str, Any]:
        from pear.envs.libero.controller import MotionController

        if self.done:
            raise InvalidRequest("the episode has ended")
        with self.env._no_render():
            self.observation, actions, rewards, self.done, outcome = MotionController(
                self.env.scene, self.observation, action
            ).move(self.max_steps - len(self.actions))
        self.actions.extend(actions)
        return self._result(actions, rewards, outcome, rendered=False)

    def close(self) -> None:
        self.env._close_scene()

    def _result(self, actions, rewards, outcome, *, rendered: bool) -> dict[str, Any]:
        """The fields of ``LiberoEnv.rollout``'s result, in its order and with its rules."""
        from pear.envs.libero.controller import MotionController

        result = {
            "images": None,
            "state": MotionController.state(self.env.scene, self.observation),
            "actions": actions,
            "rewards": rewards,
            "done": self.done,
            "executed": len(actions),
            "outcome": outcome,
            "instruction": _instruction(self.env),
        }
        if not self.done:
            observation = self.observation
            if not rendered:
                observation = self._render()
            result["images"] = [_frame(self.env, observation, camera) for camera in self.env.CAMERAS]
        return _encode(result, len(self.actions), self.max_steps)

    def _render(self) -> dict[str, Any]:
        """PEAR's frame path (a forced observation update), with the observables' sampling state put back.

        A forced update marks each observable sampled mid-period, so the scene's next steps would skip a sample
        and its observations lag the replay's; PEAR closes the scene after its forced update and never sees it."""
        robosuite_env = self.env.scene.env
        saved = [(observable, dict(vars(observable))) for observable in robosuite_env._observables.values()]
        try:
            return robosuite_env._get_observations(force_update=True)
        finally:
            for observable, attributes in saved:
                vars(observable).update(attributes)


def _instruction(env) -> str:
    import re

    language = str(env._benchmark().get_task(env.task_id).language)
    return re.sub(r"\b(left|right)\b", lambda m: env.MIRRORED_SIDE_BY_SIDE[m.group(1)], language)


def _frame(env, observation, camera):
    import numpy as np
    from PIL import Image

    pixels = observation[env.OBSERVATION_KEY_BY_CAMERA[camera]][::-1, ::-1]
    return Image.fromarray(np.ascontiguousarray(pixels, dtype=np.uint8))


def _encode(result: dict[str, Any], steps: int, max_steps: int) -> dict[str, Any]:
    from pear.agent_loop.episode import Episode as PearEpisode
    from pear.envs.libero.env import LiberoEnv

    images = None
    if result["images"] is not None:
        images = {}
        for camera, image in zip(LiberoEnv.CAMERAS, result["images"], strict=True):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            images[camera] = base64.b64encode(buffer.getvalue()).decode()
    return {
        "state": result["state"],
        "images": images,
        "done": result["done"],
        "success": PearEpisode.is_success(result),
        "outcome": result["outcome"],
        "executed": result["executed"],
        "actions": result["actions"],
        "steps": steps,
        "max_steps": max_steps,
        "instruction": result["instruction"],
    }


class Simulator:
    """The server's one episode. Every scene call runs on one thread: an EGL context is current only on the
    thread that made it, and a render from another thread returns another camera's frame or a black one."""

    def __init__(self):
        self.thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="simulator")
        self.episode: Episode | None = None

    def reset(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.thread.submit(self._reset, *_episode_args(body)).result()

    def act(self, body: dict[str, Any]) -> dict[str, Any]:
        from pear.agent_loop.protocol import InvalidAction, JsonBlockProtocol
        from pear.envs.libero.env import LiberoEnv

        action = body.get("action")
        try:
            JsonBlockProtocol.check_action(action, LiberoEnv.SCHEMA)
        except InvalidAction as exc:
            raise InvalidRequest(f"invalid action: {exc}") from None
        return self.thread.submit(lambda: self._episode().act(action)).result()

    def replay(self, body: dict[str, Any]) -> dict[str, Any]:
        args, actions = _episode_args(body), body.get("actions", [])
        return self.thread.submit(self._replay, *args, actions).result()

    def _reset(self, suite: str, task_id: int, trial_id: int) -> dict[str, Any]:
        self._close()
        try:
            self.episode = Episode(suite, task_id, trial_id)
        except (ValueError, AssertionError) as exc:
            raise InvalidRequest(str(exc)) from None
        return self.episode.first()

    def _replay(self, suite: str, task_id: int, trial_id: int, actions: list[list[float]]) -> dict[str, Any]:
        from pear.envs.libero.env import LiberoEnv

        self._close()
        try:
            env = LiberoEnv(suite, task_id, trial_id)
        except (ValueError, AssertionError) as exc:
            raise InvalidRequest(str(exc)) from None
        return _encode(env.rollout(actions, None), len(actions), LiberoEnv.MAX_STEP_N_BY_SUITE[suite])

    def _close(self) -> None:
        if self.episode is not None:
            self.episode.close()
            self.episode = None

    def _episode(self) -> Episode:
        if self.episode is None:
            raise InvalidRequest("no episode: POST /reset first")
        return self.episode


def _episode_args(body: dict[str, Any]) -> tuple[str, int, int]:
    try:
        return str(body["suite"]), int(body["task_id"]), int(body["trial_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidRequest(f"an episode needs suite, task_id and trial_id: {exc!r}") from None


def make_handler(simulator: Simulator):
    routes = {"/reset": simulator.reset, "/act": simulator.act, "/replay": simulator.replay}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            if self.path == "/health":
                self._reply(200, {"ok": True})
            else:
                self._reply(404, {"error": f"no route {self.path}"})

        def do_POST(self):
            route = routes.get(self.path)
            if route is None:
                self._reply(404, {"error": f"no route {self.path}"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except json.JSONDecodeError as exc:
                self._reply(400, {"error": f"body is not JSON: {exc}"})
                return
            start = time.monotonic()
            try:
                result = route(body)
            except InvalidRequest as exc:
                self._reply(400, {"error": str(exc)})
                return
            except Exception:
                logger.exception("%s failed", self.path)
                self._reply(500, {"error": traceback.format_exc()})
                return
            result["seconds"] = round(time.monotonic() - start, 3)
            self._reply(200, result)

        def _reply(self, status: int, payload: dict[str, Any]):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            logger.info("%s %s", self.address_string(), format % args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--pear-root", required=True, help="a PEAR checkout (mit-han-lab/pear)")
    parser.add_argument(
        "--warm",
        default=None,
        help="an episode to build before serving, as JSON reset args: imports and the first scene build happen here",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    sys.path.insert(0, args.pear_root)

    simulator = Simulator()
    if args.warm:
        start = time.monotonic()
        simulator.reset(json.loads(args.warm))
        logger.info("warm episode built in %.1fs", time.monotonic() - start)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(simulator))
    logger.info("serving on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
