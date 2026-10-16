"""Turn a NeMo Gym OSWorld task file into Miles prompt rows and a held-out split.

Miles reads ``prompt`` / ``label`` / ``metadata`` by the launcher's key flags.
The instruction itself never goes in ``prompt``: the policy only ever sees it
inside the environment, so ``prompt`` carries an identifier and nothing the
model reads. ``metadata.osworld_task`` is what the generate function POSTs back
to Gym.

**Selection is by measured within-task variance, not by aggregate pass rate.**
A GRPO group is the N samples of *one* prompt, so the advantage is zero unless
that task's own samples disagree -- a task every sample solves and a task no
sample solves both contribute exactly nothing, however interesting the corpus
looks in aggregate. LIN-1996 measured this directly: across four optimizer
steps its gradient norm was non-zero on exactly the one step where a single
task scored 3 of 4, and zero on steps whose tasks were internally unanimous
even at a mean reward of 0.5. So ``--variance-from`` takes a probe's
``rollouts.jsonl`` -- the same task file run with ``num_repeats`` > 1 -- and
keeps only the tasks whose successes fall strictly between none and all.

Without ``--variance-from`` the split is stratified by domain alone, which
reproduces the earlier behaviour and is the right thing for an evaluation-only
split, but will waste most optimizer steps if used for training.

The split is written with the seed, the source file's digest and both task-id
lists, because a before/after number is only worth reporting if it can be shown
to have been measured on tasks the policy never trained on.

Usage:
    python3 scripts/tools/make_osworld_split.py \\
        --source <gym>/benchmarks/osworld/data/test_small.jsonl \\
        --out-dir /root/datasets/osworld --held-out-fraction 0.3 --seed 20261016 \\
        --variance-from <run>/rollouts.jsonl
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


def read_task_outcomes(rollouts: Path) -> dict[str, list[float]]:
    """Every graded reward a probe recorded, keyed by task.

    Only rows Gym admitted count. An episode it refused never measured the
    policy, so letting it stand in for a failure would make a task look
    unanimous when nothing was learned about it.
    """
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
    """Keep the tasks whose own samples disagree, and say why each was dropped."""
    kept: list[Task] = []
    report = {"with_variance": [], "unanimous_success": [], "unanimous_failure": [], "not_probed": []}
    for task in tasks:
        rewards = outcomes.get(task.task_id)
        if not rewards or len(rewards) < 2:
            report["not_probed"].append(task.task_id)
            continue
        successes = sum(1 for reward in rewards if reward > 0)
        if successes == 0:
            report["unanimous_failure"].append(task.task_id)
        elif successes == len(rewards):
            report["unanimous_success"].append(task.task_id)
        else:
            report["with_variance"].append({"task_id": task.task_id, "successes": successes, "samples": len(rewards)})
            kept.append(task)
    return kept, report


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

    variance_report = None
    if variance_from is not None:
        outcomes = read_task_outcomes(variance_from)
        tasks, variance_report = select_by_variance(tasks, outcomes)
        if not tasks:
            raise typer.BadParameter(
                f"no task in {source} showed within-task variance in {variance_from}. Every GRPO group would "
                "have zero advantage, so there is nothing to train on; probe more samples or other tasks."
            )

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
        "selection": "within-task variance" if variance_report else "domain stratification only",
        "variance_probe": str(variance_from) if variance_from else None,
        "variance_report": variance_report,
        "counts": {"selected": len(tasks), "train": len(train), "held_out": len(held_out)},
        "domains": {"train": _domain_counts(train), "held_out": _domain_counts(held_out)},
        "train_task_ids": sorted(task.task_id for task in train),
        "held_out_task_ids": sorted(task.task_id for task in held_out),
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({key: provenance[key] for key in ("counts", "domains", "selection", "seed")}, indent=2))


if __name__ == "__main__":
    typer.run(main)
