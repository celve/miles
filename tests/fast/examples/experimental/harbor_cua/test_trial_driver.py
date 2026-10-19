"""The computer-1 agent block: the kwargs that decide whether a trial works at all."""

import inspect

import pytest
from examples.experimental.harbor_cua.trial_driver import AGENT_IMPORT_PATH, PROVIDER_NAME, _agent_kwargs


@pytest.fixture
def kwargs():
    return _agent_kwargs(
        api_base="http://127.0.0.1:41234/r/abc/v1",
        session_id="abc",
        sampling_params={"temperature": 0.9, "top_p": 0.95, "max_new_tokens": 512, "ignored": 1},
        max_turns=9,
    )


def test_max_seq_len_is_never_passed(kwargs):
    """Computer1.__init__ has no such parameter, and the TypeError would read as reward 0."""
    assert "max_seq_len" not in kwargs


def test_images_are_enabled_explicitly(kwargs):
    """Left unset, the capability is resolved from litellm's model info, which can say no."""
    assert kwargs["enable_images"] is True
    assert kwargs["model_info"]["supports_vision"] is True


def test_the_endpoint_rides_the_agent_kwargs(kwargs):
    assert kwargs["api_base"] == "http://127.0.0.1:41234/r/abc/v1"
    assert kwargs["llm_kwargs"] == {"api_key": "dummy"}


def test_the_session_id_is_set_so_the_compactors_calls_are_identifiable(kwargs):
    assert kwargs["session_id"] == "abc"
    assert kwargs["collect_rollout_details"] is True


def test_sampling_params_are_filtered_and_renamed(kwargs):
    assert kwargs["temperature"] == 0.9
    assert kwargs["llm_call_kwargs"] == {"top_p": 0.95, "max_tokens": 512}


def test_turn_cap_and_desktop_geometry_are_set(kwargs):
    assert kwargs["max_turns"] == 9
    assert (kwargs["desktop_width"], kwargs["desktop_height"]) == (1024, 900)


def test_every_kwarg_is_a_real_computer1_parameter(kwargs):
    harbor = pytest.importorskip("harbor", reason="pip install harbor -- the recipe's one extra dependency")
    assert harbor
    from harbor.agents.computer_1 import Computer1

    accepted = set(inspect.signature(Computer1.__init__).parameters)
    assert not set(kwargs) - accepted


def test_the_generic_json_provider_is_the_one_selected(kwargs):
    pytest.importorskip("harbor", reason="pip install harbor -- the recipe's one extra dependency")
    from harbor.agents.computer_1.providers.base import _PROVIDER_REGISTRY

    assert _PROVIDER_REGISTRY[PROVIDER_NAME].endswith("GenericJsonProvider")
    assert kwargs["provider"] == PROVIDER_NAME


def test_harbor_builds_the_qwen3_vl_computer_1_from_the_trial_config(kwargs, tmp_path):
    pytest.importorskip("harbor", reason="pip install harbor -- the recipe's one extra dependency")
    from examples.experimental.harbor_cua.qwen_computer import Qwen3VLComputer1, Qwen3VLJsonProvider
    from harbor.agents.factory import AgentFactory
    from harbor.models.agent.name import AgentName
    from harbor.models.trial.config import AgentConfig

    config = AgentConfig(import_path=AGENT_IMPORT_PATH, model_name="openai/model", kwargs=kwargs)
    agent = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)

    assert isinstance(agent, Qwen3VLComputer1)
    assert agent.name() == AgentName.COMPUTER_1.value
    assert isinstance(agent._build_provider(), Qwen3VLJsonProvider)


def test_the_deterministic_inference_seed_is_forwarded():
    """Miles sends sampling_seed per request; the OpenAI surface names it seed."""
    kwargs = _agent_kwargs(
        api_base="http://x/v1", session_id="s", sampling_params={"sampling_seed": 1234}, max_turns=2
    )

    assert kwargs["llm_call_kwargs"]["seed"] == 1234


def test_a_zero_seed_is_still_a_seed():
    kwargs = _agent_kwargs(api_base="http://x/v1", session_id="s", sampling_params={"sampling_seed": 0}, max_turns=2)

    assert kwargs["llm_call_kwargs"]["seed"] == 0


def test_miles_own_seed_name_wins_over_the_openai_one():
    kwargs = _agent_kwargs(
        api_base="http://x/v1", session_id="s", sampling_params={"sampling_seed": 7, "seed": 9}, max_turns=2
    )

    assert kwargs["llm_call_kwargs"]["seed"] == 7
