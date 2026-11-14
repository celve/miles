"""A LIBERO episode for Miles, in PEAR's protocol: ``--custom-agent-function-path libero_agent_function.run``.

Each episode gets a fresh ``libero-sim`` sandbox (``libero_sandbox``), resets it to the row's
``metadata`` ``{suite, task_id, trial_id}``, and runs PEAR's conversation
(https://github.com/mit-han-lab/pear at f32aa1f, Apache-2.0): the system prompt states the
protocol and the task; every user turn is the robot's state as JSON, a reminder, and the
``[agentview]`` and ``[wrist]`` frames; every answer ends with one fenced json block, a pose
change the sandbox's closed-loop controller carries out. The policy is reached through the
session server, so tokens, log-probs and media are recorded there. The reward is PEAR's: 1.0
when the simulator ends the episode with the task done, else 0.0; ``reward_func`` reads it.

The episode ends when the simulator ends it (``env``), an answer has no valid action
(``parse``, PEAR's rule: reward 0), an answer hits its token cap (``length``), the next turn
would not fit the context budget (``budget``), the turn cap is reached (``max_turns``), or the
wall clock runs out (``timeout``). Only a sandbox that cannot be started or a simulator that is
lost or fails raises ``InfraAbort``, which discards the sample.

Env vars:
  LIBERO_MAX_TURNS           the most actions an episode takes (default 64)
  LIBERO_MAX_EPISODE_S       the wall clock of an episode (default 3600)
  LIBERO_CONTEXT_TOKENS      the context budget (default: metadata["max_seq_len"], else 32768)
  LIBERO_TURN_RESERVE_TOKENS the room left for the next observation (default 1024)
  LIBERO_SIM_TIMEOUT_S       one reset or action (default 300)
  AGENT_MODEL_NAME           the model name sent to the policy (default "model")
and libero_sandbox's.
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import libero_sandbox
from openai import AsyncOpenAI

from miles.rollout.agentic.agent_function import InfraAbort
from miles.rollout.agentic.rollout_limits import rollout_semaphore
from miles.rollout.agentic.session import openai_session_url
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

WORKER_ENV = ("LIBERO_*", "AGENT_*")

# PEAR's pear/envs/system_prompt.md, pear/envs/libero/environment_prompt.md and EnvironmentTool.REMINDER, verbatim.
SYSTEM_PROMPT_TEMPLATE = """You are controlling a robot. Each turn you see the robot's current state and its camera images, and you respond with the next action.

## OBSERVATION

The robot is made of named parts, and each part has named quantities you can set. Every turn begins with the current value of every such quantity, grouped by part, followed by the camera images:

{"<part>": {"<quantity>": <value>, ...}, ...}

Lengths are in centimeters and angles in radians; any other quantity is a plain number. Nothing beyond these names is given about what each quantity corresponds to.

## ACTION

The action changes the listed quantities by these amounts:

{"<part>": {"<quantity>": <change>, ...}, ...}

Write the change of every quantity; 0 leaves a quantity as it is.

## EXECUTION

The action ends once every target is reached and the robot is at rest. If something keeps a target from being reached, the robot stops there and the action ends. Either way, you then receive the next observation.

## RESPONSE FORMAT

End your response with exactly one fenced json block containing the action, and nothing after it:

```json
<action>
```

{environment}## TASK

{instruction}
"""
ENVIRONMENT_PROMPT = """- agentview: a fixed camera; the robot's base is at the top of its image, +x points toward the bottom of the image and +y toward its left.
- wrist: a camera on the gripper; with the gripper pointing straight down at yaw 0, +x points toward the top of its image and +y toward its right.
- the episode ends by itself as soon as the task is done: while observations keep coming, the task is not done yet."""
REMINDER = "The current camera frames follow. End your response with one json block containing the next action."

SCHEMA = {"arm": ("x", "y", "z", "roll", "pitch", "yaw"), "gripper": ("width",)}
CAMERAS = ("agentview", "wrist")
_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_THINK_END = "</think>"


class SimulatorFailure(Exception):
    """The simulator server could not be reached or failed a request it accepted."""


@dataclass
class EpisodeLog:
    turns: int = 0
    end_reason: str = "max_turns"
    success: bool = False
    native_steps: int = 0
    outcomes: list[str | None] = field(default_factory=list)
    gen_times: list[float] = field(default_factory=list)
    sim_times: list[float] = field(default_factory=list)


def system_prompt(instruction: str) -> str:
    section = f"## ENVIRONMENT\n\n{ENVIRONMENT_PROMPT}\n\n"
    return SYSTEM_PROMPT_TEMPLATE.replace("{environment}", section).replace("{instruction}", instruction)


def observation_message(observation: dict[str, Any]) -> dict[str, Any]:
    content = [{"type": "text", "text": json.dumps(observation["state"]) + "\n\n" + REMINDER}]
    for camera in CAMERAS:
        content.append({"type": "text", "text": f"[{camera}]"})
        url = "data:image/png;base64," + observation["images"][camera]
        content.append({"type": "image_url", "image_url": {"url": url}})
    return {"role": "user", "content": content}


def parse_action(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """The action of an answer's last json block after its thinking, else None and the reason (PEAR's checks)."""
    answer = text.rpartition(_THINK_END)[2]
    blocks = _JSON_BLOCK.findall(answer)
    if not blocks:
        return None, "no ```json block found"
    try:
        action = json.loads(blocks[-1])
    except json.JSONDecodeError as exc:
        return None, f"the ```json block is not valid JSON ({exc.msg})"
    if not isinstance(action, dict) or set(action) != set(SCHEMA):
        return None, f"the json block must map exactly the parts {', '.join(SCHEMA)} to their quantities"
    for part, quantities in action.items():
        if not isinstance(quantities, dict) or set(quantities) != set(SCHEMA[part]):
            return None, f"{part!r} must give exactly its quantities {', '.join(SCHEMA[part])}"
        for name, value in quantities.items():
            # NaN, inf and an int past every float fail the bound; a bool is no number here
            if type(value) not in (int, float) or not abs(value) <= sys.float_info.max:
                return None, f"{part}.{name} must be a finite number"
    return action, None


class SimulatorClient:
    def __init__(self, url: str, timeout_s: float):
        self._client = httpx.AsyncClient(base_url=url, timeout=timeout_s)

    async def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(path, json=body)
        except httpx.HTTPError as exc:
            raise SimulatorFailure(f"{path}: {exc!r}") from exc
        if response.status_code != 200:
            raise SimulatorFailure(f"{path} answered {response.status_code}: {response.text[-2000:]}")
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()


def _answer_text(message: Any) -> str:
    """The answer the action is parsed from: the content, with any thinking a reasoning parser left in it."""
    return message.content or ""


def _context_budget(metadata: dict[str, Any]) -> int:
    return int(os.environ.get("LIBERO_CONTEXT_TOKENS") or metadata.get("max_seq_len") or 32768)


async def play(
    sim: SimulatorClient,
    policy: AsyncOpenAI,
    *,
    env_args: dict[str, Any],
    request_kwargs: dict[str, Any],
    context_budget: int,
    log: EpisodeLog,
) -> None:
    """Reset, then policy -> action -> observation until the episode ends; ``log`` records it as it goes."""
    model = os.environ.get("AGENT_MODEL_NAME", "model")
    max_turns = int(os.environ.get("LIBERO_MAX_TURNS", "64"))
    reserve = int(os.environ.get("LIBERO_TURN_RESERVE_TOKENS", "1024"))
    turn_cap = request_kwargs.get("max_tokens")

    observation = await _timed(log, sim.post("/reset", env_args))
    messages = [{"role": "system", "content": system_prompt(observation["instruction"])}]
    messages.append(observation_message(observation))
    context = 0
    while log.turns < max_turns:
        room = context_budget - context - reserve
        if room <= 0:
            log.end_reason = "budget"
            return
        start = time.monotonic()
        completion = await policy.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={**request_kwargs, "max_tokens": min(room, turn_cap) if turn_cap else room},
        )
        log.gen_times.append(time.monotonic() - start)
        choice = completion.choices[0]
        # the whole message, reasoning included: the session server matches later requests against what it stored
        messages.append(choice.message.model_dump(exclude_none=True))
        context = completion.usage.prompt_tokens + completion.usage.completion_tokens
        if choice.finish_reason == "length":
            log.end_reason = "length"
            return
        action, reason = parse_action(_answer_text(choice.message))
        if action is None:
            logger.info("episode %s: no action (%s)", env_args, reason)
            log.end_reason = "parse"
            return
        observation = await _timed(log, sim.post("/act", {"action": action}))
        log.turns += 1
        log.native_steps = observation["steps"]
        log.outcomes.append(observation["outcome"])
        if observation["done"]:
            log.end_reason, log.success = "env", bool(observation["success"])
            return
        messages.append(observation_message(observation))


async def _timed(log: EpisodeLog, call):
    start = time.monotonic()
    try:
        return await call
    finally:
        log.sim_times.append(time.monotonic() - start)


def _env_args(metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        return {
            "suite": str(metadata["suite"]),
            "task_id": int(metadata["task_id"]),
            "trial_id": int(metadata["trial_id"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"a LIBERO row's metadata needs suite, task_id and trial_id: {exc!r}") from None


async def _start_sandbox(env_args: dict[str, Any], max_episode_s: float) -> libero_sandbox.Sandbox:
    labels = {key: str(value) for key, value in env_args.items()}
    limit = int(os.environ.get("LIBERO_CREATE_CONCURRENCY", "8"))
    try:
        async with rollout_semaphore("libero-sandbox-create", limit):
            return await asyncio.to_thread(
                libero_sandbox.Sandbox,
                template=libero_sandbox.selected_template(),
                max_lifetime_s=int(max_episode_s) + 30 * 60,
                labels=labels,
            )
    except Exception as exc:
        raise InfraAbort("SandboxStartFailed", f"the libero-sim sandbox did not start: {exc!r}") from exc


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """One LIBERO episode by the trained policy; the row's prompt is unused, as in PEAR (the simulator gives the task)."""
    metadata = metadata or {}
    env_args = _env_args(metadata)
    max_episode_s = float(os.environ.get("LIBERO_MAX_EPISODE_S", "3600"))
    started = time.monotonic()
    sandbox = await _start_sandbox(env_args, max_episode_s)
    sandbox_start_s = time.monotonic() - started
    sim = SimulatorClient(sandbox.url, float(os.environ.get("LIBERO_SIM_TIMEOUT_S", "300")))
    policy = AsyncOpenAI(base_url=openai_session_url(base_url), api_key="EMPTY")
    log = EpisodeLog()
    try:
        await asyncio.wait_for(
            play(
                sim,
                policy,
                env_args=env_args,
                request_kwargs=dict(request_kwargs or {}),
                context_budget=_context_budget(metadata),
                log=log,
            ),
            timeout=max_episode_s,
        )
    except TimeoutError:
        log.end_reason = "timeout"
    except SimulatorFailure as exc:
        tail = await asyncio.to_thread(sandbox.log_tail)
        raise InfraAbort("SimulatorLost", f"{exc}\n--- simulator log ---\n{tail}") from exc
    finally:
        await sim.aclose()
        await policy.close()
        await asyncio.to_thread(sandbox.kill)
    return _result(log, env_args, sandbox_start_s)


def _result(log: EpisodeLog, env_args: dict[str, Any], sandbox_start_s: float) -> dict[str, Any]:
    reward = 1.0 if log.success else 0.0
    return {
        "reward": reward,
        "exit_status": "completed",
        "libero": {**env_args, "success": log.success, "end_reason": log.end_reason, "outcomes": log.outcomes},
        "agent_metrics": {
            "success": reward,
            "turns": log.turns,
            "native_steps": log.native_steps,
            "end_reason": log.end_reason,
            "sandbox_start_time": sandbox_start_s,
            "reset_time": log.sim_times[0] if log.sim_times else 0.0,
            "gen_times": log.gen_times,
            "sim_times": log.sim_times,
            "total_gen_time": sum(log.gen_times),
            # Sample.non_generation_time: everything outside policy generation
            "total_tool_time": sandbox_start_s + sum(log.sim_times),
        },
    }


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """``--custom-rm-path libero_agent_function.reward_func``: the reward ``run`` left in the metadata."""
    if isinstance(samples, list):
        return [sample.metadata.get("reward", 0.0) for sample in samples]
    return samples.metadata.get("reward", 0.0)
