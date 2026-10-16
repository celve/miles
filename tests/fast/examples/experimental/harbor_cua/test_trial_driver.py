"""The computer-1 agent block: the kwargs that decide whether a trial works at all."""

import inspect

import pytest

from examples.experimental.harbor_cua.trial_driver import AGENT_NAME, PROVIDER_NAME, _agent_kwargs


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


def test_the_agent_name_is_the_registered_one():
    pytest.importorskip("harbor", reason="pip install harbor -- the recipe's one extra dependency")
    from harbor.models.agent.name import AgentName

    assert AgentName(AGENT_NAME) is AgentName.COMPUTER_1
