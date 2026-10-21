"""Run Harbor OSWorld trials on request, on the machine that hosts the OSWorld VMs.

Harbor's Docker environment bind-mounts each trial's log directories from the host it
runs on, and OSWorld's VMs need ``/dev/kvm``, so the trials run here and Miles's
generate function asks for them over HTTP. The reply is Harbor's own ``TrialResult``,
which the caller turns into a reward with the same function as an in-process trial.
Imports Harbor and its OSWorld adapter, never Miles.

    python -m examples.experimental.harbor_cua.osworld_service \\
        --tasks-dir <osworld tasks> --trials-dir <dir> --host <tailnet address> --port 8765 --max-trials 16
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

AGENT_IMPORT_PATH = "examples.experimental.harbor_cua.osworld_agent:QwenOSWorldAgent"


class TrialRequest(BaseModel):
    instance_id: str
    api_base: str
    model_name: str
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    max_tokens: int = 2048
    seed: int | None = None
    max_steps: int = 15
    history_n: int = 4
    timeout_s: float = 3600.0


def build_trial_config(request: TrialRequest, *, tasks_dir: Path, trials_dir: Path):
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig

    return TrialConfig(
        task=TaskConfig(path=tasks_dir / request.instance_id),
        agent=AgentConfig(
            import_path=AGENT_IMPORT_PATH,
            model_name=request.model_name,
            kwargs={
                "api_base": request.api_base,
                "history_n": request.history_n,
                "top_k": request.top_k,
                "seed": request.seed,
                "max_steps": request.max_steps,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
        ),
        environment=EnvironmentConfig(type="docker", delete=True),
        trials_dir=trials_dir,
    )


async def run_trial(request: TrialRequest, *, tasks_dir: Path, trials_dir: Path) -> dict[str, Any]:
    from harbor.trial.trial import Trial

    trial = None
    try:
        trial = await Trial.create(build_trial_config(request, tasks_dir=tasks_dir, trials_dir=trials_dir))
        result = await asyncio.wait_for(trial.run(), timeout=request.timeout_s)
    except TimeoutError:
        logger.error("osworld trial %s exceeded %ss", request.instance_id, request.timeout_s)
        return {"exit_status": "TimeLimitExceeded", "error": "timeout", "trial_dir": _trial_dir(trial)}
    except Exception as exc:  # noqa: BLE001
        logger.error("osworld trial %s failed: %s", request.instance_id, exc, exc_info=True)
        return {"exit_status": "AgentError", "error": f"{type(exc).__name__}: {exc}", "trial_dir": _trial_dir(trial)}
    return {"trial_result": result.model_dump(mode="json"), "trial_dir": _trial_dir(trial)}


def _trial_dir(trial) -> str | None:
    paths = getattr(trial, "paths", None)
    return str(paths.trial_dir) if paths is not None else None


def build_app(*, tasks_dir: Path, trials_dir: Path, max_trials: int) -> FastAPI:
    app = FastAPI()
    slots = asyncio.Semaphore(max_trials)
    running = {"trials": 0}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "running": running["trials"], "max_trials": max_trials}

    @app.post("/trials")
    async def trials(request: TrialRequest) -> dict[str, Any]:
        async with slots:
            running["trials"] += 1
            try:
                return await run_trial(request, tasks_dir=tasks_dir, trials_dir=trials_dir)
            finally:
                running["trials"] -= 1

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--trials-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-trials", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    app = build_app(tasks_dir=args.tasks_dir, trials_dir=args.trials_dir, max_trials=args.max_trials)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
