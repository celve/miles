"""Turn a NeMo Gym OSWorld task file into Miles prompt rows and a held-out split.

Miles reads ``prompt`` / ``label`` / ``metadata`` by the launcher's key flags.
The instruction itself never goes in ``prompt``: the policy only ever sees it
inside the environment, so ``prompt`` carries an identifier and nothing the
model reads. ``metadata.osworld_task`` is what the generate function POSTs back
to Gym.

The split is stratified by OSWorld domain, so the held-out side carries the same
domain mix rather than whichever domains happened to sort last, and is written
with the seed and the source file's digest. Without that record a before/after
evaluation cannot be shown to have been measured on tasks the policy never
trained on -- which is the one claim such a number is for.

Usage:
    python3 scripts/tools/make_osworld_split.py \\
        --source <gym>/benchmarks/osworld/data/test_small.jsonl \\
        --out-dir /root/datasets/osworld --held-out-fraction 0.3 --seed 20261016
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import typer


@dataclass(frozen=True)
class Task:
    task_id: str
    domain: str
    row: dict


def _read_tasks(source: Path) -> list[Task]:
    tasks: list[Task] = []
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        metadata = raw.get("verifier_metadata") or {}
        osworld = metadata.get("osworld_task") or {}
        domain = osworld.get("domain") or metadata.get("domain") or "unknown"
        task_id = metadata.get("task_id") or osworld.get("id") or f"task-{len(tasks)}"
        tasks.append(
            Task(
                task_id=str(task_id),
                domain=str(domain),
                row={
                    "prompt": f"{domain}/{task_id}",
                    "label": domain,
                    "metadata": {
                        "osworld_task": {
                            "responses_create_params": raw.get("responses_create_params") or {},
                            "verifier_metadata": metadata,
                        }
                    },
                },
            )
        )
    return tasks


def _stratified_split(tasks: list[Task], held_out_fraction: float, seed: int) -> tuple[list[Task], list[Task]]:
    """Hold out the same fraction of every domain, deterministically.

    A domain with a single task goes to training: a held-out side of one task
    measures nothing, and losing the domain from training costs more.
    """
    by_domain: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        by_domain[task.domain].append(task)

    train: list[Task] = []
    held_out: list[Task] = []
    for domain in sorted(by_domain):
        domain_tasks = sorted(by_domain[domain], key=lambda t: t.task_id)
        random.Random(f"{seed}:{domain}").shuffle(domain_tasks)
        count = int(len(domain_tasks) * held_out_fraction) if len(domain_tasks) > 1 else 0
        held_out.extend(domain_tasks[:count])
        train.extend(domain_tasks[count:])
    return train, held_out


def _write(path: Path, tasks: list[Task]) -> None:
    path.write_text("".join(f"{json.dumps(task.row)}\n" for task in tasks))


def _domain_counts(tasks: list[Task]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for task in tasks:
        counts[task.domain] += 1
    return dict(sorted(counts.items()))


def main(
    source: Path = typer.Option(..., help="Gym OSWorld task jsonl"),
    out_dir: Path = typer.Option(..., help="where train.jsonl, held_out.jsonl and provenance.json go"),
    held_out_fraction: float = typer.Option(0.3),
    seed: int = typer.Option(20261016),
    only_domains: str = typer.Option("", help="comma-separated OSWorld domains to keep"),
) -> None:
    tasks = _read_tasks(source)
    keep = {domain.strip() for domain in only_domains.split(",")} if only_domains else None
    if keep is not None:
        tasks = [task for task in tasks if task.domain in keep]
    if not tasks:
        raise typer.BadParameter(f"no tasks read from {source}")

    train, held_out = _stratified_split(tasks, held_out_fraction, seed)
    overlap = {task.task_id for task in train} & {task.task_id for task in held_out}
    assert not overlap, f"a task is on both sides of the split: {sorted(overlap)}"

    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "train.jsonl", train)
    _write(out_dir / "held_out.jsonl", held_out)

    provenance = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": seed,
        "held_out_fraction": held_out_fraction,
        "only_domains": sorted(keep) if keep is not None else None,
        "counts": {"total": len(tasks), "train": len(train), "held_out": len(held_out)},
        "domains": {"train": _domain_counts(train), "held_out": _domain_counts(held_out)},
        "train_task_ids": sorted(task.task_id for task in train),
        "held_out_task_ids": sorted(task.task_id for task in held_out),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({key: provenance[key] for key in ("counts", "domains", "source_sha256", "seed")}, indent=2))


if __name__ == "__main__":
    typer.run(main)
