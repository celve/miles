"""RePrompt's reward terms, with RePrompt's own regexes as the reference, and `reprompt_rm` against a fake reward server."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from examples.experimental.reprompt import reward
from examples.experimental.reprompt.reward import (
    ASSISTANT_PREFIX,
    ORI_PROMPT_KEY,
    REWARD_KEY,
    format_reward,
    length_reward,
    log_rollout_data,
    reprompt_rm,
    t2i_prompt,
)
from fastapi import FastAPI, Request

from miles.utils.http_utils import find_available_port
from miles.utils.retry_utils import NonRetryableError
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

_ANSWER_20 = " ".join(["word"] * 20)


@pytest.mark.parametrize(
    "response, expected",
    [
        (f" plan </think>\n<answer>{_ANSWER_20}</answer>", 1.0),
        (f" plan </think>\n<answer>{_ANSWER_20}</answer>\n", 1.0),
        # RePrompt's pattern has no DOTALL, so reasoning that spans lines fails the format.
        (f" plan\nmore </think>\n<answer>{_ANSWER_20}</answer>", -1.0),
        (f" plan </think><answer>{_ANSWER_20}</answer>", -1.0),
        (f" plan </think>\n<answer>{_ANSWER_20}</answer> trailing", -1.0),
    ],
)
def test_format_needs_one_line_of_reasoning_then_the_answer(response, expected):
    assert format_reward(ASSISTANT_PREFIX + response) == expected


@pytest.mark.parametrize(
    "answer, expected",
    [
        (" ".join(["w"] * 15), -1.0),
        (" ".join(["w"] * 16), 1.0),
        (" ".join(["w"] * 76), 1.0),
        (" ".join(["w"] * 77), -1.0),
    ],
)
def test_length_rewards_answers_of_16_to_76_words(answer, expected):
    assert length_reward(f"<think> x </think>\n<answer>{answer}</answer>") == expected


def test_no_answer_scores_zero_length_and_renders_a_photo():
    completion = ASSISTANT_PREFIX + " still thinking"
    assert length_reward(completion) == 0.0
    assert t2i_prompt(completion) == "a photo"
    assert t2i_prompt("<answer>first</answer><answer>second</answer>") == "first"


@pytest.fixture
def reward_server():
    seen = []
    app = FastAPI()

    @app.post("/score")
    async def score(request: Request):
        body = await request.json()
        seen.append(body)
        return {"image_reward": 0.8}

    server = UvicornThreadServer(app, host="127.0.0.1", port=find_available_port(31000))
    server.start()
    yield server.url, seen
    server.stop()


def _sample(response: str) -> Sample:
    return Sample(prompt="p", response=response, metadata={ORI_PROMPT_KEY: "a photo of a cat"})


def test_the_reward_sums_reprompts_three_terms(reward_server):
    url, seen = reward_server
    args = SimpleNamespace(rollout_skip_special_tokens=True, reprompt_reward_urls=url)
    sample = _sample(f" plan </think>\n<answer>{_ANSWER_20}</answer>")

    value = asyncio.run(reprompt_rm(args, sample))

    assert seen == [{"prompt": _ANSWER_20, "ori_prompt": "a photo of a cat"}]
    assert value == pytest.approx(0.5 * 0.8 / 2.0 + 1.0 + 1.0)
    assert sample.metadata[REWARD_KEY]["t2i_prompt"] == _ANSWER_20


def test_an_unreachable_server_drops_the_sample(monkeypatch):
    monkeypatch.setattr(reward, "_RETRIES", 1)
    port = find_available_port(32000)
    args = SimpleNamespace(rollout_skip_special_tokens=True, reprompt_reward_urls=f"http://127.0.0.1:{port}")

    assert asyncio.run(reprompt_rm(args, _sample(" x </think>\n<answer>a</answer>"))) is None


def test_the_stop_token_must_be_stripped():
    args = SimpleNamespace(rollout_skip_special_tokens=False, reprompt_reward_urls="http://unused")
    with pytest.raises(NonRetryableError, match="rollout-skip-special-tokens"):
        asyncio.run(reprompt_rm(args, _sample("")))


def test_the_log_hook_logs_the_terms_and_writes_each_scored_sample(monkeypatch, tmp_path, reward_server):
    logged = []
    monkeypatch.setattr(reward.tracking, "log", lambda args, metrics, step_key: logged.append(metrics))
    url, _ = reward_server
    rm_args = SimpleNamespace(rollout_skip_special_tokens=True, reprompt_reward_urls=url)
    sample = _sample(f" plan </think>\n<answer>{_ANSWER_20}</answer>")
    sample.group_index = 3
    sample.reward = asyncio.run(reprompt_rm(rm_args, sample))
    log_args = SimpleNamespace(reprompt_sample_log=str(tmp_path), reward_key=None, wandb_always_use_train_step=False)

    assert log_rollout_data(7, log_args, [sample], {}, 0.0) is False

    (metrics,) = logged
    assert metrics["rollout/step"] == 7 and metrics["rollout/reprompt/image_reward"] == 0.8
    assert metrics["rollout/reprompt/reward"] == pytest.approx(2.2)
    (row,) = [json.loads(line) for line in (tmp_path / "rollout_00007.jsonl").read_text().splitlines()]
    assert row["prompt"] == "a photo of a cat" and row["group"] == 3
    assert row["completion"].startswith(ASSISTANT_PREFIX) and row["format"] == 1.0
