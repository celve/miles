"""Harbor's OSWorld agent running the Qwen3-VL policy in ``osworld_qwen`` inside the task container.

Imported by Harbor on the machine that runs the OSWorld trials, which needs Harbor's
OSWorld adapter installed and no Miles. Built against harbor-miles-v0.20.0 at
8f88560ce9: it overrides that ``OSWorldAgent``'s container-runner hooks, so a Harbor
upgrade that renames them breaks it loudly.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from osworld.custom_agent.agent import (
    CONTAINER_REQUEST_PATH,
    CONTAINER_RESPONSE_PATH,
    CONTAINER_RUNNER_ROOT,
    OSWorldAgent,
)
from osworld.custom_agent.runner import OSWorldContainerCommandError

POLICY_PACKAGE = "osworld_qwen"


class QwenOSWorldAgent(OSWorldAgent):
    def __init__(
        self,
        *args: Any,
        api_base: str,
        api_key: str = "dummy",
        history_n: int = 4,
        top_k: int | None = None,
        seed: int | None = None,
        num_retries: int = 2,
        **kwargs: Any,
    ) -> None:
        kwargs["observation_type"] = "screenshot"
        # OSWorldRunnerConfig demands a template for screenshot observations; this policy never renders one.
        kwargs.setdefault("prompt_template", "unused: the Qwen3-VL policy builds its own prompt")
        super().__init__(*args, **kwargs)
        self._policy_request = {
            "api_base": api_base,
            "api_key": api_key,
            "history_n": history_n,
            "top_k": top_k,
            "seed": seed,
            "num_retries": num_retries,
        }

    @staticmethod
    def name() -> str:
        return "harbor-cua-osworld-qwen3vl"

    async def _prepare_container_runner(self, environment: BaseEnvironment) -> None:
        await super()._prepare_container_runner(environment)
        await environment.upload_dir(
            source_dir=Path(__file__).resolve().parent / POLICY_PACKAGE,
            target_dir=f"{CONTAINER_RUNNER_ROOT}/{POLICY_PACKAGE}",
        )

    def _container_request_payload(self, instruction: str) -> dict[str, Any]:
        return {**super()._container_request_payload(instruction), **self._policy_request}

    async def _run_container_runner(
        self,
        environment: BaseEnvironment,
        *,
        mode: str,
        instruction: str = "",
        context: AgentContext | None = None,
    ) -> None:
        # The parent hardcodes ``-m osworld.custom_agent.runner``; this is its body with the module swapped.
        request_path = self.logs_dir / Path(CONTAINER_REQUEST_PATH).name
        response_path = self.logs_dir / Path(CONTAINER_RESPONSE_PATH).name
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text(
            json.dumps(self._container_request_payload(instruction), indent=2) + "\n", encoding="utf-8"
        )
        if response_path.exists():
            response_path.unlink()
        command = (
            f"PYTHONPATH={shlex.quote(CONTAINER_RUNNER_ROOT)} "
            "${OSWORLD_VERIFIER_PYTHON:-/opt/osworld-verifier/bin/python} "
            f"-m {POLICY_PACKAGE}.runner "
            f"--mode {shlex.quote(mode)} "
            f"--request {shlex.quote(CONTAINER_REQUEST_PATH)}"
        )
        try:
            await self._host_runner.exec_as_agent(environment, command=command)
        except OSWorldContainerCommandError as exc:
            raise RuntimeError(f"OSWorld container runner failed (mode={mode}):\n{exc}") from exc
        if context is not None:
            if not response_path.exists():
                raise RuntimeError(f"OSWorld container runner did not write {response_path}")
            self._populate_context_from_payload(context, json.loads(response_path.read_text(encoding="utf-8")))
