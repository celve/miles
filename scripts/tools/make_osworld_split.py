"""Turn a NeMo Gym OSWorld task file into Miles prompt rows and a held-out split."""

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


def read_task_outcomes(rollouts: Path) -> dict[str, list[float]]:
    """Every graded reward a probe recorded, keyed by task."""
    outcomes: dict[str, list[float]] = defaultdict(list)
    for line in rollouts.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("runtime_eligible") is not True or row.get("mask_sample") is not False:
            continue
        reward = row.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            continue
        metadata = row.get("verifier_metadata") or {}
        task_id = metadata.get("task_id") or (metadata.get("osworld_task") or {}).get("id")
        if task_id:
            outcomes[str(task_id)].append(float(reward))
    return dict(outcomes)


def select_by_variance(tasks: list[Task], outcomes: dict[str, list[float]]) -> tuple[list[Task], dict]:
    """Keep the tasks whose own rewards differ, and say why each was dropped."""
    kept: list[Task] = []
    report = {"with_spread": [], "no_spread": [], "not_probed": []}
    for task in tasks:
        rewards = outcomes.get(task.task_id)
        if not rewards or len(rewards) < 2:
            report["not_probed"].append(task.task_id)
            continue
        record = {
            "task_id": task.task_id,
            "samples": len(rewards),
            "rewards": sorted(rewards),
            "spread": max(rewards) - min(rewards),
            "successes": sum(1 for reward in rewards if reward > 0),
        }
        if max(rewards) == min(rewards):
            report["no_spread"].append(record)
        else:
            report["with_spread"].append(record)
            kept.append(task)
    return kept, report


def _stratified_split(tasks: list[Task], held_out_fraction: float, seed: int) -> tuple[list[Task], list[Task]]:
    """Hold out the same fraction of every domain, deterministically."""
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
    variance_from: Path = typer.Option(
        None,
        help="a probe's rollouts.jsonl; keeps only tasks whose own samples disagree, which is what GRPO needs",
    ),
) -> None:
    tasks = _read_tasks(source)
    keep = {domain.strip() for domain in only_domains.split(",")} if only_domains else None
    if keep is not None:
        tasks = [task for task in tasks if task.domain in keep]
    if not tasks:
        raise typer.BadParameter(f"no tasks read from {source}")

    train_candidates, held_out = _stratified_split(tasks, held_out_fraction, seed)
    if held_out_fraction > 0 and not held_out:
        raise typer.BadParameter(
            f"a held-out fraction of {held_out_fraction} yielded no evaluation tasks from {len(tasks)} "
            "candidates. The launcher still schedules held-out evaluation, so this would run and report "
            "nothing; lower the fraction, add tasks, or pass 0 deliberately."
        )

    variance_report = None
    if variance_from is not None:
        outcomes = read_task_outcomes(variance_from)
        train_candidates, variance_report = select_by_variance(train_candidates, outcomes)
        if not train_candidates:
            raise typer.BadParameter(
                f"no training task in {source} showed differing rewards in {variance_from}. Every GRPO group "
                "drawn from this probe would have zero advantage, so there is nothing to train on; probe more "
                "samples or other tasks."
            )

    train = train_candidates
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
        "selection": "reward spread on the training side only" if variance_report else "domain stratification only",
        "variance_probe": str(variance_from) if variance_from else None,
        "variance_report": variance_report,
        "counts": {"candidates": len(tasks), "train": len(train), "held_out": len(held_out)},
        "domains": {"train": _domain_counts(train), "held_out": _domain_counts(held_out)},
        "train_task_ids": sorted(task.task_id for task in train),
        "held_out_task_ids": sorted(task.task_id for task in held_out),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({key: provenance[key] for key in ("counts", "domains", "selection", "seed")}, indent=2))


if __name__ == "__main__":
    typer.run(main)
