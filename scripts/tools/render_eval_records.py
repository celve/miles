"""Render two ``--eval-record-dir`` recordings of the same tasks, base and checkpoint, side by side.

Each root is a recording of one eval point (``<eval_record_dir>/eval_<rollout_id>``, or a directory
above it holding only one); episodes pair by task and by their rank among that task's samples. An
episode without a reward (discarded, or never finished) is shown as unscored and left out of pass
rates and of ``--select contrast``. The page is one
self-contained HTML file: per task, the instruction, both sides' pass rates, and every step's
screenshot, model output and action in two aligned columns.

    python scripts/tools/render_eval_records.py --base runs/base/eval_0 --checkpoint runs/ckpt/eval_0 \\
        --out compare.html --select contrast --pair contrast --gif-dir gifs
"""

import base64
import html
import io
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image, ImageDraw

from miles.utils.eval_record import RecordedEpisode, RecordedStep, iter_episodes

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum


class Select(StrEnum):
    all = "all"
    contrast = "contrast"


class Pair(StrEnum):
    same = "same"
    contrast = "contrast"


@dataclass(frozen=True)
class TaskPair:
    task_id: str
    base: list[RecordedEpisode]
    checkpoint: list[RecordedEpisode]


_EVAL_POINT = re.compile(r"eval_\d+")


def _eval_point(episode: RecordedEpisode) -> Path | None:
    return next((p for p in [episode.directory, *episode.directory.parents] if _EVAL_POINT.fullmatch(p.name)), None)


def load_by_task(root: Path) -> dict[str, list[RecordedEpisode]]:
    """Episodes by task; a root holding several eval points is refused, since each point is different weights."""
    episodes = list(iter_episodes(root))
    if len(points := {_eval_point(episode) for episode in episodes}) > 1:
        names = ", ".join(sorted(str(point) for point in points))
        raise typer.BadParameter(f"{root} holds several eval points ({names}); pass one of them")
    by_task: dict[str, list[RecordedEpisode]] = defaultdict(list)
    for episode in episodes:
        by_task[episode.task_id].append(episode)
    for task_episodes in by_task.values():
        task_episodes.sort(key=lambda episode: episode.sample_index)
    return dict(by_task)


def scored(episodes: list[RecordedEpisode]) -> list[RecordedEpisode]:
    return [episode for episode in episodes if episode.reward is not None]


def passed(episode: RecordedEpisode, threshold: float) -> bool:
    return episode.reward is not None and episode.reward >= threshold


def pass_rate(episodes: list[RecordedEpisode], threshold: float) -> float | None:
    """Over the scored episodes only; None when none was scored."""
    if not (counted := scored(episodes)):
        return None
    return sum(passed(episode, threshold) for episode in counted) / len(counted)


def pass_count(episodes: list[RecordedEpisode], threshold: float) -> str:
    counted = scored(episodes)
    unscored = len(episodes) - len(counted)
    return f"{sum(passed(e, threshold) for e in counted)}/{len(counted)}" + (
        f" ({unscored} unscored)" if unscored else ""
    )


def select_tasks(
    base: dict[str, list[RecordedEpisode]],
    checkpoint: dict[str, list[RecordedEpisode]],
    *,
    select: Select,
    tasks: list[str],
    threshold: float,
    max_base_pass: float,
    min_checkpoint_pass: float,
) -> list[TaskPair]:
    """Tasks both sides recorded; ``contrast`` keeps those base mostly failed and the checkpoint mostly passed."""
    shared = sorted(base.keys() & checkpoint.keys())
    if tasks:
        shared = [task for task in shared if task in set(tasks)]
    pairs = [TaskPair(task, base[task], checkpoint[task]) for task in shared]
    if select is Select.contrast:
        rates = {
            pair.task_id: (pass_rate(pair.base, threshold), pass_rate(pair.checkpoint, threshold)) for pair in pairs
        }
        pairs = [
            pair
            for pair in pairs
            if None not in rates[pair.task_id]
            and rates[pair.task_id][0] <= max_base_pass
            and rates[pair.task_id][1] >= min_checkpoint_pass
        ]
        pairs.sort(key=lambda pair: rates[pair.task_id][0] - rates[pair.task_id][1])
    return pairs


def pick_episodes(
    pair: TaskPair, *, mode: Pair, rank: int, threshold: float
) -> tuple[RecordedEpisode, RecordedEpisode]:
    """``same`` shows the rank-th sample of each side; ``contrast`` a scored failed base and a passed checkpoint."""
    if mode is Pair.contrast:
        base_scored, checkpoint_scored = scored(pair.base) or pair.base, scored(pair.checkpoint) or pair.checkpoint
        base = next((episode for episode in base_scored if not passed(episode, threshold)), base_scored[0])
        checkpoint = next(
            (episode for episode in checkpoint_scored if passed(episode, threshold)), checkpoint_scored[0]
        )
        return base, checkpoint
    return pair.base[min(rank, len(pair.base) - 1)], pair.checkpoint[min(rank, len(pair.checkpoint) - 1)]


def _thumbnail(path: Path | None, width: int) -> Image.Image | None:
    if path is None or not path.is_file():
        return None
    image = Image.open(path).convert("RGB")
    if image.width > width:
        image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    return image


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _step_cell(step: RecordedStep | None, width: int) -> str:
    if step is None:
        return '<td class="empty"></td>'
    image = _thumbnail(step.screenshot, width)
    shot = f'<img src="{_data_url(image)}" alt="step {step.index} screenshot">' if image is not None else ""
    action = json.dumps(step.action, ensure_ascii=False) if step.action is not None else "(no action)"
    error = f'<div class="error">{html.escape(step.error)}</div>' if step.error else ""
    return (
        f"<td>{shot}<div class='action'>{html.escape(action)}</div>{error}"
        f"<details><summary>model output</summary><pre>{html.escape(step.output)}</pre></details></td>"
    )


def _side_header(label: str, episode: RecordedEpisode, episodes: list[RecordedEpisode], threshold: float) -> str:
    verdict = "unscored" if episode.reward is None else "pass" if passed(episode, threshold) else "fail"
    return (
        f"<th><span class='label'>{html.escape(label)}</span> "
        f"<span class='badge {verdict}'>{verdict} · reward {episode.reward}</span>"
        f"<div class='sub'>sample {episode.sample_index} · {len(episode.steps)} steps · {html.escape(episode.status)}"
        f" · task pass {pass_count(episodes, threshold)}</div></th>"
    )


def render_task(
    pair: TaskPair,
    base: RecordedEpisode,
    checkpoint: RecordedEpisode,
    *,
    labels: tuple[str, str],
    threshold: float,
    width: int,
) -> str:
    instruction = base.instruction or checkpoint.instruction or "(no instruction recorded)"
    rows = []
    for index in range(max(len(base.steps), len(checkpoint.steps))):
        left = base.steps[index] if index < len(base.steps) else None
        right = checkpoint.steps[index] if index < len(checkpoint.steps) else None
        rows.append(f"<tr><td class='n'>{index + 1}</td>{_step_cell(left, width)}{_step_cell(right, width)}</tr>")
    return (
        f"<section id='{html.escape(pair.task_id)}'><h2>{html.escape(pair.task_id)}</h2>"
        f"<p class='instruction'>{html.escape(instruction)}</p>"
        f"<table class='steps'><thead><tr><th></th>"
        f"{_side_header(labels[0], base, pair.base, threshold)}"
        f"{_side_header(labels[1], checkpoint, pair.checkpoint, threshold)}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def render_page(sections: list[str], summary: list[tuple[str, str, str]], labels: tuple[str, str]) -> str:
    summary_rows = "".join(
        f"<tr><td><a href='#{html.escape(task)}'>{html.escape(task)}</a></td><td>{b}</td><td>{c}</td></tr>"
        for task, b, c in summary
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CUA trajectories: {html.escape(labels[0])} vs {html.escape(labels[1])}</title>
<style>
:root {{ --bg: #ffffff; --fg: #1d1d1f; --muted: #6e6e73; --line: #d2d2d7; --pass: #1a7f37; --fail: #b42318; --panel: #f5f5f7; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #161618; --fg: #f2f2f4; --muted: #a1a1a6; --line: #3a3a3c; --pass: #3fb950; --fail: #f85149; --panel: #232326; }}
}}
body {{ background: var(--bg); color: var(--fg); font: 14px/1.45 system-ui, sans-serif; margin: 0 auto; padding: 16px; max-width: 1800px; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 32px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-top: 1px solid var(--line); padding: 8px; vertical-align: top; text-align: left; }}
table.steps {{ table-layout: fixed; }} table.steps td.n {{ width: 32px; color: var(--muted); }}
table.steps img {{ width: 100%; height: auto; border: 1px solid var(--line); border-radius: 4px; }}
td.empty {{ background: var(--panel); }}
.instruction {{ background: var(--panel); padding: 8px 12px; border-radius: 6px; white-space: pre-wrap; }}
.action {{ font-family: ui-monospace, monospace; font-size: 12px; margin-top: 6px; word-break: break-word; }}
.error {{ color: var(--fail); font-size: 12px; }}
.label {{ font-weight: 600; }} .sub {{ color: var(--muted); font-size: 12px; font-weight: normal; }}
.badge {{ font-size: 12px; padding: 1px 6px; border-radius: 4px; color: #fff; }}
.badge.pass {{ background: var(--pass); }} .badge.fail {{ background: var(--fail); }} .badge.unscored {{ background: var(--muted); }}
pre {{ white-space: pre-wrap; word-break: break-word; font-size: 12px; }}
a {{ color: inherit; }}
</style></head><body>
<h1>{html.escape(labels[0])} vs {html.escape(labels[1])}</h1>
<table><thead><tr><th>task</th><th>{html.escape(labels[0])} passed</th><th>{html.escape(labels[1])} passed</th></tr></thead>
<tbody>{summary_rows}</tbody></table>
{''.join(sections)}
</body></html>
"""


def write_gif(
    base: RecordedEpisode, checkpoint: RecordedEpisode, path: Path, *, width: int, labels: tuple[str, str]
) -> None:
    """Both episodes stepping together, base on the left; the shorter one holds its last frame."""
    sides = [[_thumbnail(step.screenshot, width) for step in episode.steps] for episode in (base, checkpoint)]
    shots = [image for side in sides for image in side if image is not None]
    if not shots:
        return
    column, height = max(image.width for image in shots), max(image.height for image in shots)
    frames, last = [], [None, None]
    for index in range(max(len(side) for side in sides)):
        frame = Image.new("RGB", (2 * column, height + 24), "white")
        draw = ImageDraw.Draw(frame)
        for side, images in enumerate(sides):
            if index < len(images) and images[index] is not None:
                last[side] = images[index]
            if last[side] is not None:
                frame.paste(last[side], (side * column, 24))
            draw.text((side * column + 6, 5), f"{labels[side]} - step {min(index, len(images) - 1) + 1}", fill="black")
        frames.append(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1200, loop=0)


def main(
    base: Annotated[Path, typer.Option(help="Recording of the base model: an eval point's directory or above.")],
    checkpoint: Annotated[Path, typer.Option(help="Recording of the trained checkpoint on the same tasks.")],
    out: Annotated[Path, typer.Option(help="The HTML page to write.")],
    base_label: str = "base",
    checkpoint_label: str = "checkpoint",
    select: Annotated[
        Select, typer.Option(help="contrast: tasks base mostly failed and the checkpoint mostly passed.")
    ] = Select.all,
    task: Annotated[list[str] | None, typer.Option(help="Only these task ids (repeatable).")] = None,
    pass_threshold: Annotated[float, typer.Option(help="A reward at or above this counts as a pass.")] = 1.0,
    max_base_pass: Annotated[
        float, typer.Option(help="contrast: the most of its samples base may have passed.")
    ] = 0.0,
    min_checkpoint_pass: Annotated[
        float, typer.Option(help="contrast: the fewest the checkpoint must have passed.")
    ] = 0.5,
    pair: Annotated[
        Pair, typer.Option(help="same: each side's --rank-th sample; contrast: a failed base, a passed checkpoint.")
    ] = Pair.same,
    rank: int = 0,
    max_tasks: int = 20,
    image_width: Annotated[int, typer.Option(help="Screenshots are scaled down to this width, as JPEG.")] = 800,
    gif_dir: Annotated[Path | None, typer.Option(help="Also write one side-by-side GIF per task here.")] = None,
) -> None:
    labels = (base_label, checkpoint_label)
    pairs = select_tasks(
        load_by_task(base),
        load_by_task(checkpoint),
        select=select,
        tasks=task or [],
        threshold=pass_threshold,
        max_base_pass=max_base_pass,
        min_checkpoint_pass=min_checkpoint_pass,
    )[:max_tasks]
    if not pairs:
        raise typer.BadParameter("no task was recorded on both sides and passed the selection")
    sections, summary = [], []
    for task_pair in pairs:
        left, right = pick_episodes(task_pair, mode=pair, rank=rank, threshold=pass_threshold)
        sections.append(
            render_task(task_pair, left, right, labels=labels, threshold=pass_threshold, width=image_width)
        )
        summary.append(
            (
                task_pair.task_id,
                pass_count(task_pair.base, pass_threshold),
                pass_count(task_pair.checkpoint, pass_threshold),
            )
        )
        if gif_dir is not None:
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in task_pair.task_id)
            write_gif(left, right, gif_dir / f"{safe}.gif", width=min(image_width, 640), labels=labels)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_page(sections, summary, labels))
    print(f"wrote {out}: {len(pairs)} tasks")


if __name__ == "__main__":
    typer.run(main)
