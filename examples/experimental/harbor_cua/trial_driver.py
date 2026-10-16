"""Run one Harbor ``computer-1`` trial against the recorder's endpoint.

The environment, verifier and task-path handling are Miles's existing Harbor
example's, imported rather than copied; only the agent block differs, because
``computer-1`` is the one Harbor agent that runs in the worker process instead
of in the sandbox and so needs its endpoint on the agent kwargs rather than in
``AgentConfig.env``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.experimental.harbor.harbor_agent_function import (
    _ensure_provider_key,
    _env_int,
    _environment_config,
    _task_path,
    _trial_timeout_s,
    trial_result_to_metadata,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "computer-1"
PROVIDER_NAME = "litellm"


@dataclass(frozen=True)
class TrialOutcome:
    reward: float
    exit_status: str
    metadata: dict[str, Any]
    trial_dir: str | None

    @property
    def infrastructure_failure(self) -> bool:
        """Whether the trial failed on us rather than on the policy.

        ``Submitted`` means the verifier ran and its number is the task's. A
        timeout is the policy stalling as often as it is the sandbox, so it is
        left as a scored task failure, which is what Miles's existing Harbor
        path does.
        """
        return self.exit_status == "AgentError"


def build_trial_config(
    *,
    instance_id: str,
    api_base: str,
    session_id: str,
    sampling_params: dict[str, Any],
    max_turns: int,
):
    """A ``computer-1`` trial pointed at the recorder.

    Three kwargs are load-bearing and each would fail quietly if left out:
    ``max_seq_len`` is deliberately absent because ``Computer1.__init__`` has no
    such parameter and the ``TypeError`` would surface only as a zero reward;
    ``enable_images=True`` is explicit because the fallback asks litellm whether
    the model supports vision and a ``False`` there blinds the agent silently;
    and ``session_id`` is what makes the compactor's own calls identifiable,
    since only ``LiteLLM.call`` sends it and the agent loop does not.
    """
    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig, VerifierConfig

    tasks_dir = Path(os.environ["HARBOR_TASKS_DIR"])
    model = f"openai/{os.getenv('AGENT_MODEL_NAME', 'model')}"
    agent_timeout = _env_int("AGENT_TIMEOUT")
    verifier_timeout = _env_int("HARBOR_VERIFIER_TIMEOUT_SEC")

    extra: dict[str, Any] = {}
    if verifier_timeout is not None:
        extra["verifier"] = VerifierConfig(override_timeout_sec=float(verifier_timeout))

    return TrialConfig(
        task=TaskConfig(path=_task_path(tasks_dir, instance_id)),
        agent=AgentConfig(
            name=AGENT_NAME,
            model_name=model,
            override_timeout_sec=float(agent_timeout) if agent_timeout else None,
            env={},
            kwargs=_agent_kwargs(
                api_base=api_base,
                session_id=session_id,
                sampling_params=sampling_params,
                max_turns=max_turns,
            ),
        ),
        environment=_environment_config(),
        trials_dir=Path(os.getenv("HARBOR_TRIALS_DIR", "/tmp/harbor_trials")),
        timeout_multiplier=float(os.getenv("HARBOR_TIMEOUT_MULTIPLIER", "2.0")),
        **extra,
    )


def _agent_kwargs(
    *, api_base: str, session_id: str, sampling_params: dict[str, Any], max_turns: int
) -> dict[str, Any]:
    call_kwargs = {
        key: value
        for key, value in sampling_params.items()
        if key in {"top_p", "top_k", "frequency_penalty", "presence_penalty", "stop"}
    }
    if (max_tokens := sampling_params.get("max_new_tokens") or sampling_params.get("max_tokens")) is not None:
        call_kwargs["max_tokens"] = int(max_tokens)
    return {
        "api_base": api_base,
        "provider": PROVIDER_NAME,
        "enable_images": True,
        "session_id": session_id,
        "collect_rollout_details": True,
        "temperature": float(sampling_params.get("temperature", 1.0)),
        "max_turns": max_turns,
        "desktop_width": _env_int("HARBOR_CUA_DESKTOP_WIDTH") or 1024,
        "desktop_height": _env_int("HARBOR_CUA_DESKTOP_HEIGHT") or 900,
        "llm_kwargs": {"api_key": "dummy"},
        "llm_call_kwargs": call_kwargs,
        "model_info": {
            "max_input_tokens": int(os.getenv("AGENT_MAX_INPUT_TOKENS", "32768")),
            "max_output_tokens": int(os.getenv("AGENT_MAX_OUTPUT_TOKENS", "2048")),
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "supports_vision": True,
        },
    }


async def run_trial(
    *,
    instance_id: str,
    api_base: str,
    session_id: str,
    sampling_params: dict[str, Any],
    max_turns: int,
) -> TrialOutcome:
    from harbor.trial.trial import Trial

    logger.info("harbor cua trial %s posting to %s", instance_id, api_base)
    _ensure_provider_key()
    config = build_trial_config(
        instance_id=instance_id,
        api_base=api_base,
        session_id=session_id,
        sampling_params=sampling_params,
        max_turns=max_turns,
    )
    timeout_s = _trial_timeout_s()
    trial = None
    try:
        trial = await Trial.create(config)
        result = await asyncio.wait_for(trial.run(), timeout=timeout_s)
    except TimeoutError:
        logger.error("harbor cua trial for %s exceeded %ss", instance_id, timeout_s)
        return TrialOutcome(reward=0.0, exit_status="TimeLimitExceeded", metadata={}, trial_dir=_trial_dir(trial))
    except Exception as exc:  # noqa: BLE001
        logger.error("harbor cua trial for %s failed: %s", instance_id, exc, exc_info=True)
        return TrialOutcome(reward=0.0, exit_status="AgentError", metadata={}, trial_dir=_trial_dir(trial))

    verdict = trial_result_to_metadata(result)
    return TrialOutcome(
        reward=float(verdict["reward"]),
        exit_status=str(verdict["exit_status"]),
        metadata=verdict,
        trial_dir=_trial_dir(trial),
    )


def _trial_dir(trial) -> str | None:
    paths = getattr(trial, "paths", None)
    return str(paths.trial_dir) if paths is not None else None
