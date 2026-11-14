"""Write LIBERO episode rows for ``libero_agent_function``: one row per (suite, task, initial state).

A row's prompt is a label only; the simulator gives the task at reset, as in PEAR. A GRPO group is
the ``--n-samples-per-prompt`` episodes of one row, so it shares task and initial state.

LIBERO has 50 initial states (trials) per task. PEAR trains on all 50 and validates on LIBERO-Goal's
trials 40-49 (pear/envs/libero/configs/libero_gso_all.json, libero_goal_val.json); held-out here means
trials no training row uses, so the default split trains on 0-39 and holds out 40-49.

    python make_libero_data.py --out-dir data --suites libero_goal libero_spatial libero_object
"""

import argparse
import json
from pathlib import Path

SUITES = ("libero_goal", "libero_spatial", "libero_object", "libero_10")
TASKS_PER_SUITE = 10
TRIALS_PER_TASK = 50


def rows(suites: list[str], tasks: list[int], trials: range, split: str) -> list[dict]:
    return [
        {
            "prompt": [{"role": "user", "content": f"{suite}/task_{task} trial {trial}"}],
            "label": f"{suite}/task_{task}",
            "metadata": {"suite": suite, "task_id": task, "trial_id": trial, "split": split},
        }
        for suite in suites
        for task in tasks
        for trial in trials
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--suites", nargs="+", choices=SUITES, default=["libero_goal", "libero_spatial", "libero_object"]
    )
    parser.add_argument("--tasks", type=int, nargs="+", default=list(range(TASKS_PER_SUITE)))
    parser.add_argument("--heldout-from", type=int, default=40, help="trials from this one on are held out")
    args = parser.parse_args()
    if not 0 < args.heldout_from < TRIALS_PER_TASK:
        parser.error(f"--heldout-from must be in 1..{TRIALS_PER_TASK - 1}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": range(args.heldout_from), "heldout": range(args.heldout_from, TRIALS_PER_TASK)}
    for split, trials in splits.items():
        path = args.out_dir / f"libero.{split}.jsonl"
        split_rows = rows(args.suites, args.tasks, trials, split)
        path.write_text("".join(json.dumps(row) + "\n" for row in split_rows))
        print(f"{path}: {len(split_rows)} rows")


if __name__ == "__main__":
    main()
