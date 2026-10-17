"""Build each task's sandbox template once, serially, before any rollout fans out.

Harbor asks the provider whether a template exists and builds it when it does
not, with no retry around either call (``environments/e2b.py``). So N rollouts
starting on one unbuilt template race, one build wins, and the rest die with
``409: template <alias> is being built by another request`` -- which a trial
reports as an agent error, and a GRPO group then loses wholesale.

Warming is therefore a prerequisite of a reproducible run, not a nicety. Harbor's
``nop`` agent does it for the price of one sandbox start per task.

    python -m examples.experimental.harbor_cua.warm_templates \\
        --tasks-dir /root/harbor_cua_tasks --data /root/datasets/harbor_cua/train.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def instance_ids(data_files: list[Path]) -> list[str]:
    seen: dict[str, None] = {}
    for path in data_files:
        for line in path.read_text().splitlines():
            if line.strip():
                seen.setdefault(json.loads(line)["metadata"]["instance_id"], None)
    return list(seen)


async def warm_one(instance_id: str, *, tasks_dir: Path, trials_dir: Path, timeout_s: float) -> str:
    from examples.experimental.harbor.harbor_agent_function import _environment_config, _task_path
    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
    from harbor.trial.trial import Trial

    config = TrialConfig(
        task=TaskConfig(path=_task_path(tasks_dir, instance_id)),
        agent=AgentConfig(name="nop", model_name="nop", env={}, kwargs={}),
        environment=_environment_config(),
        trials_dir=trials_dir,
    )
    trial = await Trial.create(config)
    await asyncio.wait_for(trial.run(), timeout=timeout_s)
    return instance_id


async def main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from examples.experimental.harbor.harbor_agent_function import _ensure_provider_key

    _ensure_provider_key()
    ids = args.instances or instance_ids([Path(path) for path in args.data])
    failures = []
    for position, instance_id in enumerate(ids, start=1):
        try:
            await warm_one(
                instance_id,
                tasks_dir=Path(args.tasks_dir),
                trials_dir=Path(args.trials_dir),
                timeout_s=args.timeout,
            )
            logger.info("warmed %d/%d %s", position, len(ids), instance_id)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{instance_id}: {type(exc).__name__}: {exc}")
            logger.error("failed %d/%d %s: %s", position, len(ids), instance_id, exc)
    if failures:
        print("templates that did not warm:\n  " + "\n  ".join(failures))
        return 1
    print(f"warmed {len(ids)} templates")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", required=True)
    parser.add_argument("--data", nargs="*", default=[], help="jsonl files whose rows name the instances")
    parser.add_argument("--instances", nargs="*", default=[], help="instance ids, in place of --data")
    parser.add_argument("--trials-dir", default=os.getenv("HARBOR_TRIALS_DIR", "/tmp/harbor_cua_warm"))
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if not args.data and not args.instances:
        parser.error("pass --data or --instances")
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
