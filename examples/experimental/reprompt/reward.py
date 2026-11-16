"""RePrompt's reward (https://github.com/microsoft/DKI_LLM/tree/main/RePrompt, `train_reprompt.py`) as a Miles
`--custom-rm-path examples.experimental.reprompt.reward.reprompt_rm`.

RePrompt's TRL run sums three rewards with weight 1 each, on the completion text with the assistant prefix the
prompt ends in put back in front of it:

- `reward_t2i`: the text inside the first single-line `<answer>...</answer>` (or "a photo" when there is none)
  rendered to an image and scored with ImageReward against the user's original prompt, as `0.5 * IR / 2.0` (the GPT-4V
  half of RePrompt's reward is left out; with no API key its call returned 0, so the formula is unchanged).
- `format_reward_func`: 1 when the whole completion is `<think>` reasoning on one line, a newline and one `<answer>`,
  else -1.
- `reward_len`: 1 when the answer has 16 to 76 words, -1 outside that, 0 with no answer.

The image and its ImageReward score come from `reward_server.py`, which runs RePrompt's own rendering and scoring
calls, so this reward and the TRL run's differ only in the policy that wrote the completion.
"""

import asyncio
import itertools
import json
import logging
import os
import re

import httpx
import numpy as np

from miles.utils.args.schema import A, Arg, BaseConfig
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.retry_utils import NonRetryableError
from miles.utils.tracking_utils import tracking
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# The assistant turn RePrompt's prompts end in; TRL prepends it to the completion before scoring.
ASSISTANT_PREFIX = "Let me solve this step by step.\n<think>"
ORI_PROMPT_KEY = "ori_prompt"
REWARD_KEY = "reprompt_reward"
_FORMAT = re.compile(r"Let me solve this step by step.\n<think>.*?</think>\n<answer>.*?</answer>$")
_ANSWER = re.compile(r"<answer>(.*?)</answer>")
_RETRIES = 3

_next_server = itertools.count()


class RePromptRewardConfig(BaseConfig):
    reprompt_reward_urls: A[
        str,
        Arg(help="Comma-separated base URLs of reward_server.py processes, one per GPU; requests rotate over them."),
    ]


class RePromptLogConfig(BaseConfig):
    reprompt_sample_log: A[
        str | None, Arg(help="Directory for one JSONL of scored samples per rollout, for comparing runs.")
    ] = None


def format_reward(completion: str) -> float:
    return 1.0 if _FORMAT.match(completion) else -1.0


def length_reward(completion: str) -> float:
    match = _ANSWER.search(completion)
    if not match:
        return 0.0
    return 1.0 if 15 < len(match.group(1).split()) < 77 else -1.0


def t2i_prompt(completion: str) -> str:
    match = _ANSWER.search(completion)
    return match.group(1) if match else "a photo"


async def reprompt_rm(args, sample: Sample, **pools) -> float | None:
    if not args.rollout_skip_special_tokens:
        raise NonRetryableError("reprompt_rm scores the completion text: run with --rollout-skip-special-tokens")
    completion = ASSISTANT_PREFIX + sample.response
    prompt = t2i_prompt(completion)
    image_reward = await _image_reward(args, prompt, sample.metadata[ORI_PROMPT_KEY])
    if image_reward is None:
        return None
    parts = {
        "t2i": 0.5 * image_reward / 2.0,
        "format": format_reward(completion),
        "length": length_reward(completion),
        "image_reward": image_reward,
        "t2i_prompt": prompt,
    }
    sample.metadata[REWARD_KEY] = parts
    return parts["t2i"] + parts["format"] + parts["length"]


reprompt_rm.config_class = RePromptRewardConfig


async def _image_reward(args, prompt: str, ori_prompt: str) -> float | None:
    urls = args.reprompt_reward_urls.split(",")
    payload = {"prompt": prompt, "ori_prompt": ori_prompt}
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0)) as client:
        for attempt in range(_RETRIES):
            url = urls[next(_next_server) % len(urls)]
            try:
                response = await client.post(f"{url}/score", json=payload)
                response.raise_for_status()
                return float(response.json()["image_reward"])
            except httpx.HTTPError as error:
                logger.warning(f"reward server {url} failed (attempt {attempt + 1}/{_RETRIES}): {error!r}")
                await asyncio.sleep(2**attempt)
    return None


def log_rollout_data(rollout_id, args, samples: list[Sample], rollout_extra_metrics, rollout_time) -> bool:
    """`--custom-rollout-log-function-path`: the three reward terms' means, and the scored samples when asked."""
    scored = [s for s in samples if REWARD_KEY in s.metadata]
    if not scored:
        return False
    metrics = {
        f"rollout/reprompt/{key}": float(np.mean([s.metadata[REWARD_KEY][key] for s in scored]))
        for key in ("t2i", "format", "length", "image_reward")
    }
    metrics["rollout/reprompt/reward"] = float(np.mean([s.get_reward_value(args) for s in scored]))
    metrics["rollout/reprompt/completion_tokens"] = float(np.mean([s.response_length for s in scored]))
    metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
    logger.info(f"reprompt {rollout_id}: {json.dumps(metrics)}")
    tracking.log(args, metrics, step_key="rollout/step")
    if args.reprompt_sample_log:
        os.makedirs(args.reprompt_sample_log, exist_ok=True)
        with open(os.path.join(args.reprompt_sample_log, f"rollout_{rollout_id:05d}.jsonl"), "w") as f:
            for s in scored:
                row = {
                    "prompt": s.metadata[ORI_PROMPT_KEY],
                    "completion": ASSISTANT_PREFIX + s.response,
                    "group": s.group_index,
                } | s.metadata[REWARD_KEY]
                f.write(json.dumps(row) + "\n")
    return False


log_rollout_data.config_class = RePromptLogConfig
