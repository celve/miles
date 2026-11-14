"""Check a ``libero_sim_server`` against PEAR's semantics, and time it.

  1. reset is reproducible: the same episode reset on every server (and twice on the first) gives the same
     state and frames;
  2. the stateful scene matches PEAR's stateless replay of its native actions after each scripted action
     (``/replay``, on the first server: a replay closes that server's open episode);
  3. reset and action wall times.

Standard library only, so it runs anywhere the server is reachable; numpy and PIL, when present, add a
per-frame pixel difference to a mismatch. Exits 1 on any mismatch.

    python libero_probe.py --url http://127.0.0.1:8000 [--url http://<second server>] --out /tmp/probe
"""

import argparse
import base64
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

# pose changes that move the arm in each axis, turn it, and close and open the gripper
SCRIPT = [
    {"arm": {"x": 0, "y": 0, "z": -5, "roll": 0, "pitch": 0, "yaw": 0}, "gripper": {"width": 0}},
    {"arm": {"x": 4, "y": -3, "z": 0, "roll": 0, "pitch": 0, "yaw": 0.3}, "gripper": {"width": 0}},
    {"arm": {"x": 0, "y": 0, "z": -3, "roll": 0, "pitch": 0, "yaw": 0}, "gripper": {"width": -8}},
    {"arm": {"x": 0, "y": 0, "z": 6, "roll": 0, "pitch": 0, "yaw": 0}, "gripper": {"width": 0}},
    {"arm": {"x": -2, "y": 2, "z": 0, "roll": 0.1, "pitch": -0.1, "yaw": 0}, "gripper": {"width": 8}},
]


def post(url: str, path: str, body: dict, timeout_s: float = 600) -> dict:
    request = urllib.request.Request(
        url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        result = json.loads(response.read())
    result["wall"] = round(time.monotonic() - started, 3)
    return result


def pixel_diff(a: str, b: str) -> str:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return "differs"
    pa, pb = (np.asarray(Image.open(io.BytesIO(base64.b64decode(x))), dtype=np.int16) for x in (a, b))
    diff = np.abs(pa - pb)
    return f"max {diff.max()}, mean {diff.mean():.4f}, pixels {(diff.max(axis=-1) > 0).mean():.4%}"


def compare(label: str, a: dict, b: dict) -> list[str]:
    problems = []
    if a["state"] != b["state"]:
        problems.append(f"{label}: state {a['state']} != {b['state']}")
    for camera in a["images"] or {}:
        if a["images"][camera] != b["images"][camera]:
            problems.append(f"{label}: {camera} frame {pixel_diff(a['images'][camera], b['images'][camera])}")
    return problems


def save_frames(out: Path | None, name: str, observation: dict) -> None:
    if out is None or not observation.get("images"):
        return
    out.mkdir(parents=True, exist_ok=True)
    for camera, data in observation["images"].items():
        (out / f"{name}_{camera}.png").write_bytes(base64.b64decode(data))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url", action="append", required=True, help="a simulator server; give two, the last plays the episode"
    )
    parser.add_argument("--episode", default='{"suite": "libero_goal", "task_id": 0, "trial_id": 0}')
    parser.add_argument("--rounds", type=int, default=1, help="play the script this many times (or until done)")
    parser.add_argument("--out", type=Path, default=None, help="where to save the frames")
    args = parser.parse_args()
    episode = json.loads(args.episode)
    problems: list[str] = []
    timings: dict[str, list[float]] = {"reset": [], "act": [], "replay": []}

    first = post(args.url[0], "/reset", episode)
    timings["reset"].append(first["wall"])
    save_frames(args.out, "reset", first)
    print(f"instruction: {first['instruction']!r}; state {first['state']}")
    for i, url in enumerate(args.url):
        again = post(url, "/reset", episode)
        timings["reset"].append(again["wall"])
        problems += compare(f"reset on server {i} vs the first reset", first, again)

    observation, actions = again, []
    for step, action in enumerate(SCRIPT * args.rounds):
        observation = post(args.url[-1], "/act", {"action": action})
        timings["act"].append(observation["wall"])
        print(
            f"act {step}: outcome {observation['outcome']}, executed {observation['executed']}, "
            f"steps {observation['steps']}, done {observation['done']}, {observation['wall']:.2f}s"
        )
        save_frames(args.out, f"act{step}", observation)
        actions += observation["actions"]
        replay = post(args.url[0], "/replay", {**episode, "actions": actions})
        timings["replay"].append(replay["wall"])
        problems += compare(f"act {step} stateful vs PEAR replay", observation, replay)
        if observation["done"]:
            break

    print("timings (s):", json.dumps({k: [round(v, 2) for v in vs] for k, vs in timings.items()}))
    for problem in problems:
        print("MISMATCH", problem)
    print("PASS" if not problems else f"FAIL ({len(problems)} mismatches)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
