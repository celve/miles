"""Per-episode records of evaluated agent trajectories, written by ``--eval-record-dir``.

An episode's directory holds ``episode.json`` (stack, task, sample index, instruction),
``steps.jsonl`` (one line per model call: the screenshot it saw, its raw output, the
parsed action), the screenshots themselves, and ``outcome.json`` (reward and status).
Steps are appended as the episode runs, so one that dies early still shows how far it got.
The producers are the Harbor computer-1 binding and ``examples/experimental/cua_lite/record.py``;
``scripts/tools/render_eval_records.py`` reads the records back.
"""

import base64
import binascii
import json
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from miles.utils.file_utils import atomic_write_text

EPISODE_FILE = "episode.json"
STEPS_FILE = "steps.jsonl"
OUTCOME_FILE = "outcome.json"

_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def eval_point_dir(root: str, rollout_id: int) -> str:
    return f"{root}/eval_{rollout_id}"


def episode_dir(parent: str | Path, task_id: str, sample_index: int) -> Path:
    """``<parent>/<task>/<sample_index>``, with the task id made safe as one path component."""
    safe_task = _UNSAFE.sub("_", str(task_id)).strip("._") or "task"
    return Path(parent) / safe_task / str(sample_index)


def decode_data_url(url: str) -> tuple[bytes, str] | None:
    """The bytes and media type of a base64 ``data:`` URL, or None for anything else."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    header, _, payload = url[len("data:") :].partition(",")
    if "base64" not in header:
        return None
    try:
        return base64.b64decode(unquote(payload)), header.split(";")[0] or "image/png"
    except (binascii.Error, ValueError):
        return None


def start_episode(
    directory: str | Path, *, stack: str, task_id: str, sample_index: int, dataset: str | None = None
) -> Path:
    """Empty ``directory`` and describe the episode about to run in it; a retried episode replaces the last try."""
    directory = Path(directory)
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    _write_json(
        directory / EPISODE_FILE,
        {"stack": stack, "task_id": str(task_id), "sample_index": sample_index, "dataset": dataset},
    )
    return directory


class EpisodeWriter:
    """Appends the steps of an episode ``start_episode`` opened; one writer per episode."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        if not (self.directory / EPISODE_FILE).is_file():
            raise FileNotFoundError(f"{self.directory} has no {EPISODE_FILE}; open it with start_episode first")
        self._next_index = 0

    def set_instruction(self, instruction: str) -> None:
        path = self.directory / EPISODE_FILE
        _write_json(path, {**json.loads(path.read_text()), "instruction": instruction})

    def add_step(
        self,
        *,
        output: str,
        screenshot: bytes | None = None,
        media_type: str = "image/png",
        action: Any = None,
        error: str | None = None,
    ) -> None:
        index = self._next_index
        self._next_index += 1
        screenshot_name = None
        if screenshot is not None:
            screenshot_name = f"step_{index:03d}.{_EXTENSIONS.get(media_type, 'png')}"
            (self.directory / screenshot_name).write_bytes(screenshot)
        line = {"index": index, "screenshot": screenshot_name, "output": output, "action": action, "error": error}
        with open(self.directory / STEPS_FILE, "a") as f:
            f.write(json.dumps(line, default=str) + "\n")


def finish_episode(directory: str | Path, *, reward: float | None, status: str, **extra: Any) -> None:
    _write_json(Path(directory) / OUTCOME_FILE, {"reward": reward, "status": status, **extra})


@dataclass(frozen=True)
class RecordedStep:
    index: int
    screenshot: Path | None
    output: str
    action: Any
    error: str | None


@dataclass(frozen=True)
class RecordedEpisode:
    directory: Path
    stack: str
    task_id: str
    sample_index: int
    dataset: str | None
    instruction: str | None
    steps: list[RecordedStep]
    reward: float | None
    status: str
    outcome: dict[str, Any] = field(default_factory=dict)


def load_episode(directory: str | Path) -> RecordedEpisode:
    directory = Path(directory)
    meta = json.loads((directory / EPISODE_FILE).read_text())
    steps = []
    if (steps_path := directory / STEPS_FILE).is_file():
        for line in steps_path.read_text().splitlines():
            if line.strip():
                raw = json.loads(line)
                steps.append(
                    RecordedStep(
                        index=raw["index"],
                        screenshot=directory / raw["screenshot"] if raw.get("screenshot") else None,
                        output=raw.get("output") or "",
                        action=raw.get("action"),
                        error=raw.get("error"),
                    )
                )
    outcome_path = directory / OUTCOME_FILE
    outcome = json.loads(outcome_path.read_text()) if outcome_path.is_file() else {}
    return RecordedEpisode(
        directory=directory,
        stack=meta["stack"],
        task_id=meta["task_id"],
        sample_index=meta["sample_index"],
        dataset=meta.get("dataset"),
        instruction=meta.get("instruction"),
        steps=steps,
        reward=outcome.get("reward"),
        # An episode with no outcome never finished: its process died or is still running.
        status=outcome.get("status", "unfinished"),
        outcome=outcome,
    )


def iter_episodes(root: str | Path) -> Iterator[RecordedEpisode]:
    for path in sorted(Path(root).rglob(EPISODE_FILE)):
        yield load_episode(path.parent)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, indent=1, default=str))
