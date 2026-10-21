"""Harbor's OSWorld container runner with the Qwen3-VL policy in place of PromptAgent.

Runs inside the task container, beside the VM, where Harbor's own runner runs: it
replaces only ``_predict``, so the VM session, the action execution and the
trajectory recording are Harbor's. Each episode's request names the URL its model
calls go to, so one recorder can tell concurrent episodes apart.

    python -m osworld_qwen.runner --mode run --request <request.json>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import litellm
from osworld.custom_agent.core import OSWorldPromptRunner, OSWorldRunnerConfig

from .qwen3vl import Qwen3VLPolicy

logger = logging.getLogger(__name__)


class QwenPromptRunner(OSWorldPromptRunner):
    def __init__(self, *, logs_dir: Path, config: OSWorldRunnerConfig, request: dict[str, Any]) -> None:
        super().__init__(logs_dir=logs_dir, config=config)
        self._api_base = request["api_base"]
        self._api_key = request.get("api_key") or "dummy"
        self._num_retries = int(request.get("num_retries", 2))
        extra_body = {"top_k": int(request["top_k"])} if request.get("top_k") is not None else {}
        self._call_extras: dict[str, Any] = {"extra_body": extra_body} if extra_body else {}
        if request.get("seed") is not None:
            self._call_extras["seed"] = int(request["seed"])
        self._policy = Qwen3VLPolicy(history_n=int(request.get("history_n", 4)))

    async def _predict(self, model: str, instruction: str, obs: dict[str, Any]) -> tuple[str, list[str], int, int]:
        messages, screen = self._policy.build_messages(instruction, obs["screenshot"])
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                api_base=self._api_base,
                api_key=self._api_key,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                num_retries=self._num_retries,
                **self._call_extras,
            )
            text = str(response.choices[0].message.content or "")
            usage = getattr(response, "usage", None)
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qwen3-VL call failed: %s: %s", type(exc).__name__, exc)
            text, in_tok, out_tok = "", 0, 0
        actions = self._policy.record(text, screen)
        self._actions.append(actions)
        self._thoughts.append(text)
        return text, actions, in_tok, out_tok


async def _run(mode: str, request_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    response_path = Path(request.get("response_path") or "/logs/agent/osworld_agent_response.json")
    task_dir = Path(request.get("task_dir") or "/osworld-task")
    runner = QwenPromptRunner(
        logs_dir=Path(request.get("logs_dir") or "/logs/agent"),
        config=OSWorldRunnerConfig.from_request(request),
        request=request,
    )
    try:
        await runner.setup_local_vm(task_dir, run_initial_setup=mode != "run")
        if mode == "setup":
            payload: dict[str, Any] = {"ok": True, "mode": mode}
        else:
            payload = await runner.run(str(request.get("instruction") or ""))
            payload.update(ok=True, mode=mode)
    finally:
        await runner.close()
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--mode", choices=["setup", "run", "full"], required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    return asyncio.run(_run(args.mode, args.request))


if __name__ == "__main__":
    raise SystemExit(main())
