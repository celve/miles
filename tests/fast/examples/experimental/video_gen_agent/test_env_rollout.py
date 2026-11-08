"""The video-generation env in a real rollout: miles router, v2 session server, a scripted mock engine, the fake
generator and a stub pool."""

import json
import os
from pathlib import Path

import pytest
from examples.experimental.video_gen_agent.generator import Clip, ClipRequest
from examples.experimental.video_gen_agent.multiturn.generator import URL_ENV
from examples.experimental.video_gen_agent.reward import StubClipScorer, stub_score
from tests.fast.fixtures.generation_fixtures import extra_argv_for_variant
from tests.fast.fixtures.rollout_fixtures import RolloutEnvConfig
from tests.fast.rollout.inference_rollout.integration.utils import MODULAR_ROLLOUT_BASE_ARGV, load_and_call_rollout

from miles.utils.test_utils.mock_sglang_server import ProcessResult

REQUEST = "A paper boat drifts down a rain gutter."


class _StubPool:
    async def score(self, outputs, prompts):
        return StubClipScorer().score_batch(outputs, prompts)


class _BrokenGenerator:
    async def generate(self, request: ClipRequest) -> Clip:
        raise KeyError("fps")


_POOLS = {"judge": _StubPool()}
_MULTITURN_GENERATE = "examples.experimental.video_gen_agent.multiturn.generate.generate"


def _config(mode: str, generate_path: str | None = None) -> RolloutEnvConfig:
    argv = extra_argv_for_variant(
        "agentic_tool_call",
        custom_generate_function_path=generate_path,
        custom_agent_function_path="examples.experimental.video_gen_agent.env.run_agent",
    )
    argv += [
        "--custom-agent-function-mode",
        mode,
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
    """Generate a clip, then submit the clip the env named."""

    def process_fn(prompt: str) -> ProcessResult:
        prompts.append(prompt)
        if "Generated clip_1" in prompt:
            return ProcessResult(text=_tool_call("submit", clip_id="clip_1"))
        return ProcessResult(text=_tool_call("generate", prompt="paper boat, rain, close-up", seed=3, seconds=9))

    return process_fn


@pytest.fixture
def env_vars(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "AGENT_VIDEO_GEN_GENERATOR", "examples.experimental.video_gen_agent.generator.FakeClipGenerator"
    )
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_DIR", str(tmp_path / "run_media"))
    monkeypatch.setenv("AGENT_VIDEO_GEN_SUBMITTED_DIR", str(tmp_path / "submitted"))
    return monkeypatch, tmp_path


@pytest.mark.parametrize(
    "rollout_env",
    [pytest.param(_config("inline"), id="inline"), pytest.param(_config("subproc"), id="subproc")],
    indirect=True,
)
def test_submitted_clip_is_scored_in_the_reward_step(rollout_env, env_vars):
    _, tmp_path = env_vars
    prompts: list[str] = []
    rollout_env.mock_server.process_fn = _policy(prompts)

    out = load_and_call_rollout(rollout_env.args, rollout_env.data_source, mode="train", reward_pools=_POOLS)

    samples = [s for group in out.samples for sub in group for s in (sub if isinstance(sub, list) else [sub])]
    assert samples
    assert "3 generations and 8 turns" in prompts[0] and REQUEST in prompts[0]
    assert "Generated clip_1: 5 s (cut to the 5 s limit), seed 3. Generations left: 2. Turns left: 7." in prompts[1]
    (generated,) = (tmp_path / "run_media").iterdir()
    request = json.loads(generated.read_bytes())
    assert (request["seconds"], request["width"], request["height"], request["fps"]) == (5.0, 832, 480, 16)
    (kept,) = (tmp_path / "submitted").iterdir()
    assert kept.read_bytes() == generated.read_bytes()
    for sample in samples:
        submitted = sample.metadata["video_gen"]["submitted"]
        assert submitted == {"clip_id": "clip_1", "path": str(kept), "prompt": "paper boat, rain, close-up"}
        assert sample.metadata["video_gen"]["request"] == REQUEST
        assert sample.reward == stub_score(str(kept))


@pytest.mark.parametrize("rollout_env", [pytest.param(_config("subproc"), id="subproc")], indirect=True)
def test_generator_outage_aborts_the_group(rollout_env, env_vars):
    monkeypatch, tmp_path = env_vars
    monkeypatch.setenv("AGENT_VIDEO_GEN_FAKE_UNAVAILABLE", "1")
    rollout_env.mock_server.process_fn = _policy([])

    with pytest.raises(RuntimeError, match="exit_status 'GeneratorUnavailable'"):
        load_and_call_rollout(rollout_env.args, rollout_env.data_source, mode="train", reward_pools=_POOLS)

    assert not Path(tmp_path / "submitted").exists()


@pytest.mark.parametrize("rollout_env", [pytest.param(_config("subproc", _MULTITURN_GENERATE))], indirect=True)
def test_the_multiturn_hook_hands_the_env_the_generators_router(rollout_env, env_vars):
    monkeypatch, _ = env_vars
    monkeypatch.delenv(URL_ENV, raising=False)
    rollout_env.mock_server.process_fn = _policy([])

    out = load_and_call_rollout(rollout_env.args, rollout_env.data_source, mode="train", reward_pools=_POOLS)

    args = rollout_env.args
    assert os.environ[URL_ENV] == f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    assert out.samples


@pytest.mark.parametrize("rollout_env", [pytest.param(_config("inline"), id="inline")], indirect=True)
def test_a_turn_cut_at_the_token_limit_ends_the_episode_unsubmitted(rollout_env, env_vars):
    prompts: list[str] = []

    def process_fn(prompt: str) -> ProcessResult:
        prompts.append(prompt)
        return ProcessResult(text="Let me picture the gutter first", finish_reason="length")

    rollout_env.mock_server.process_fn = process_fn

    out = load_and_call_rollout(rollout_env.args, rollout_env.data_source, mode="train", reward_pools=_POOLS)

    samples = [s for group in out.samples for sub in group for s in (sub if isinstance(sub, list) else [sub])]
    assert len(prompts) == 1 and samples
    for sample in samples:
        assert sample.metadata["video_gen"]["submitted"] is None and sample.metadata["video_gen"]["turns"] == 1
        assert sample.reward == 0.0


@pytest.mark.parametrize("rollout_env", [pytest.param(_config("inline"), id="inline")], indirect=True)
def test_an_env_fault_aborts_the_group_instead_of_training_it(rollout_env, env_vars):
    monkeypatch, _ = env_vars
    monkeypatch.setenv("AGENT_VIDEO_GEN_GENERATOR", f"{__name__}._BrokenGenerator")
    rollout_env.mock_server.process_fn = _policy([])

    with pytest.raises(RuntimeError, match="exit_status 'EnvError'"):
        load_and_call_rollout(rollout_env.args, rollout_env.data_source, mode="train", reward_pools=_POOLS)
