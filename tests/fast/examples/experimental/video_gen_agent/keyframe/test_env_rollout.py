"""The keyframe env in a real rollout: miles router, v2 session server, a scripted mock engine, the fake image and
clip generators and a stub pool. The mock engine serves text only, so a run that shows images is
``test_image_observations``'s."""

import json

import pytest
from examples.experimental.video_gen_agent.reward import StubClipScorer
from tests.fast.fixtures.generation_fixtures import extra_argv_for_variant
from tests.fast.fixtures.rollout_fixtures import RolloutEnvConfig
from tests.fast.rollout.inference_rollout.integration.utils import MODULAR_ROLLOUT_BASE_ARGV, load_and_call_rollout

from miles.utils.test_utils.mock_sglang_server import ProcessResult

REQUEST = "A paper boat drifts down a rain gutter."
_FAKES = "examples.experimental.video_gen_agent.generator"


class _StubPool:
    async def score(self, outputs, prompts):
        return StubClipScorer().score_batch(outputs, prompts)


_POOLS = {"judge": _StubPool()}


def _config(mode: str) -> RolloutEnvConfig:
    argv = extra_argv_for_variant(
        "agentic_tool_call", custom_agent_function_path="examples.experimental.video_gen_agent.keyframe.env.run_agent"
    )
    argv += ["--custom-agent-function-mode", mode]
    argv += [
        "--custom-rm-path",
        "examples.experimental.video_gen_agent.reward.reward_func",
        "--video-reward-pool",
        "judge",
    ]
    argv += ["--max-consecutive-aborted-groups", "1"]
    return RolloutEnvConfig(
        extra_argv=MODULAR_ROLLOUT_BASE_ARGV + argv,
        data_rows=[{"input": [{"role": "user", "content": REQUEST}], "label": ""}],
    )


def _tool_call(name: str, **arguments) -> str:
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call><|im_end|>\n"


def _policy(prompts: list[str]):
    """Make an image, then the video from the image the env named."""

    def process_fn(prompt: str) -> ProcessResult:
        prompts.append(prompt)
        if "Generated image_1" in prompt:
            return ProcessResult(text=_tool_call("generate_video", image_id="image_1", prompt="the boat spins"))
        return ProcessResult(text=_tool_call("generate_image", prompt="paper boat in a gutter, rain", seed=3))

    return process_fn


@pytest.fixture
def env_vars(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_VIDEO_GEN_GENERATOR", f"{_FAKES}.FakeClipGenerator")
    monkeypatch.setenv("AGENT_VIDEO_GEN_IMAGE_GENERATOR", f"{_FAKES}.FakeImageGenerator")
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path / "run_media"))
    monkeypatch.setenv("AGENT_VIDEO_GEN_SUBMITTED_DIR", str(tmp_path / "submitted"))
    return monkeypatch, tmp_path


@pytest.mark.parametrize("rollout_env", [pytest.param(_config("subproc"), id="subproc")], indirect=True)
def test_an_image_generator_outage_aborts_the_group(rollout_env, env_vars):
    monkeypatch, tmp_path = env_vars
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_UNAVAILABLE", "1")
    rollout_env.mock_server.process_fn = _policy([])

    with pytest.raises(RuntimeError, match="exit_status 'GeneratorUnavailable'"):
        load_and_call_rollout(rollout_env.args, rollout_env.data_source, mode="train", reward_pools=_POOLS)

    assert not (tmp_path / "submitted").exists()
