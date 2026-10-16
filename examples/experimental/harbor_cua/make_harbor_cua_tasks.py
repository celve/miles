"""Write the harbor_cua task directories and the train/evaluation JSONL.

Task directories are generated rather than checked in: the families table is the
one source of truth, and a Harbor task dir is four near-identical files around
one page. Rerunning this writes the same bytes, and every row records which
family and variant it came from and which split it is in.

    python -m examples.experimental.harbor_cua.make_harbor_cua_tasks \
        --tasks-dir /root/harbor_cua_tasks \
        --data-dir /root/datasets/harbor_cua
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from examples.experimental.harbor_cua.tasks.families import (
    FAMILIES,
    PAGE_PLACEHOLDER,
    Family,
    Variant,
    instance_id,
    split_variants,
)

_HERE = Path(__file__).resolve().parent
_TEMPLATES = _HERE / "tasks"
AGENT_NAME = "computer-1"

_TASK_TOML = """schema_version = "1.1"

[task]
name = "{group}/{family}-{variant}"
description = "{description}"
authors = []
keywords = ["computer-use", "computer-1", "harbor-cua"]

[metadata]
category = "computer-use"
tags = ["harbor-cua", "{family}"]
family = "{family}"
variant = "{variant}"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = {agent_timeout}

[environment]
build_timeout_sec = 1800.0
cpus = 2
memory_mb = 4096
storage_mb = 4096
gpus = 0
network_mode = "no-network"
mcp_servers = []

[verifier.env]
STAGES_TOTAL = "{stages_total}"
EXPECTED_ANSWER = "{expected_answer}"

[environment.env]

[solution.env]
"""


def substitute(text: str, substitutions: dict[str, str]) -> str:
    for name, value in substitutions.items():
        text = text.replace(f"__{name}__", value)
    return text


def write_task(tasks_dir: Path, family: Family, variant: Variant) -> Path:
    task_dir = tasks_dir / instance_id(family, variant)
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)

    page = (_TEMPLATES / "pages" / family.page).read_text()
    (task_dir / "environment" / PAGE_PLACEHOLDER).write_text(substitute(page, variant.substitutions))
    shutil.copyfile(_TEMPLATES / "base" / "Dockerfile", task_dir / "environment" / "Dockerfile")

    verifier = task_dir / "tests" / "test.sh"
    shutil.copyfile(_TEMPLATES / "base" / "verify.sh", verifier)
    verifier.chmod(0o755)

    instruction = substitute(family.instruction, variant.substitutions)
    (task_dir / "instruction.md").write_text(instruction + "\n")
    (task_dir / "task.toml").write_text(
        _TASK_TOML.format(
            group=instance_id(family, variant).split("--")[0],
            family=family.name,
            variant=variant.key,
            description=instruction.splitlines()[1].strip(),
            agent_timeout=float(family.max_turns * 60),
            stages_total=variant.stages_total,
            expected_answer=variant.expected_answer or "",
        )
    )
    return task_dir


def data_row(family: Family, variant: Variant, split: str) -> dict:
    name = instance_id(family, variant)
    return {
        "prompt": [{"role": "user", "content": name}],
        "label": name,
        "metadata": {
            "instance_id": name,
            "agent_name": AGENT_NAME,
            "family": family.name,
            "variant": variant.key,
            "split": split,
            "stages_total": variant.stages_total,
            "has_expected_answer": variant.expected_answer is not None,
            "max_turns": family.max_turns,
            "provenance": "examples/experimental/harbor_cua/tasks/families.py",
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def generate(*, tasks_dir: Path, data_dir: Path, train_per_family: int, eval_per_family: int) -> dict[str, int]:
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    for family in FAMILIES:
        train, evaluation = split_variants(family, train=train_per_family, evaluation=eval_per_family)
        for variant in train:
            write_task(tasks_dir, family, variant)
            train_rows.append(data_row(family, variant, "train"))
        for variant in evaluation:
            write_task(tasks_dir, family, variant)
            eval_rows.append(data_row(family, variant, "eval"))

    write_jsonl(data_dir / "train.jsonl", train_rows)
    write_jsonl(data_dir / "eval.jsonl", eval_rows)
    overlap = {row["label"] for row in train_rows} & {row["label"] for row in eval_rows}
    if overlap:
        raise ValueError(f"the split shares instances: {sorted(overlap)}")
    return {"train": len(train_rows), "eval": len(eval_rows), "families": len(FAMILIES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True, help="where Harbor task directories are written")
    parser.add_argument("--data-dir", type=Path, required=True, help="where train.jsonl and eval.jsonl are written")
    parser.add_argument("--train-per-family", type=int, default=4)
    parser.add_argument("--eval-per-family", type=int, default=2)
    args = parser.parse_args()

    counts = generate(
        tasks_dir=args.tasks_dir,
        data_dir=args.data_dir,
        train_per_family=args.train_per_family,
        eval_per_family=args.eval_per_family,
    )
    print(
        f"{counts['families']} families -> {counts['train']} training and {counts['eval']} evaluation instances\n"
        f"tasks: {args.tasks_dir}\ndata:  {args.data_dir}"
    )


if __name__ == "__main__":
    main()
