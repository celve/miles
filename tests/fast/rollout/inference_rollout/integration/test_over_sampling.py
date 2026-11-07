import pytest
from tests.fast.fixtures.rollout_fixtures import RolloutEnvConfig
from tests.fast.rollout.inference_rollout.integration.utils import (
    filter_by_reward,
    integration_env_config,
    load_and_call_train,
)

from miles.utils.function_registry import function_registry

_DATA_ROWS = [
    {"input": "What is 1+7?", "label": "8"},
    {"input": "What is 1+8?", "label": "wrong"},
    {"input": "What is 1+9?", "label": "wrong"},
    {"input": "What is 1+6?", "label": "wrong"},
]

_BASE_ARGV = [
    "--over-sampling-batch-size",
    "4",
    "--dynamic-sampling-filter-path",
    "test:filter_by_reward",
]


def _over_sampling_config(rollout_batch_size: int):
    return integration_env_config(["--rollout-batch-size", str(rollout_batch_size)] + _BASE_ARGV, data_rows=_DATA_ROWS)


@pytest.mark.parametrize(
    "rollout_env,expected_rounds",
    [
        pytest.param(_over_sampling_config(1), 1, id="one_round"),
        pytest.param(_over_sampling_config(2), 2, id="two_rounds"),
    ],
    indirect=["rollout_env"],
)
def test_over_sampling_rounds(rollout_env, expected_rounds):
    env = rollout_env

    with function_registry.temporary("test:filter_by_reward", filter_by_reward):
        out = load_and_call_train(env.args, env.data_source)

    assert len(out.samples) == env.args.rollout_batch_size
    assert all(group[0].reward == 1 for group in out.samples)

    requests_count = len(env.mock_server.request_log)
    expected_requests = expected_rounds * env.args.over_sampling_batch_size
    assert requests_count == expected_requests, f"Expected {expected_rounds} round(s) = {expected_requests} requests"


_ONE_WRONG_ROWS = [
    {"input": "What is 1+7?", "label": "8"},
    {"input": "What is 1+8?", "label": "9"},
    {"input": "What is 1+9?", "label": "wrong"},
    {"input": "What is 1+6?", "label": "7"},
]


_NO_OVER_SAMPLING_ARGV = ["--rollout-batch-size", "3", "--dynamic-sampling-filter-path", "test:filter_by_reward"]
_LEGACY_DRIVER_ARGV = [
    "--rollout-function-path",
    "miles.rollout.sglang_rollout.generate_rollout",
    "--eval-function-path",
    "miles.rollout.sglang_rollout.generate_rollout",
    "--custom-generate-function-path",
    "miles.rollout.sglang_rollout.generate",
]


@pytest.mark.parametrize(
    "rollout_env",
    [
        pytest.param(integration_env_config(_NO_OVER_SAMPLING_ARGV, data_rows=_ONE_WRONG_ROWS), id="class_driver"),
        pytest.param(
            RolloutEnvConfig(extra_argv=_LEGACY_DRIVER_ARGV + _NO_OVER_SAMPLING_ARGV, data_rows=_ONE_WRONG_ROWS),
            id="legacy_driver",
        ),
    ],
    indirect=True,
)
def test_a_dropped_group_is_replaced_by_one_group_not_a_new_batch(rollout_env):
    env = rollout_env

    with function_registry.temporary("test:filter_by_reward", filter_by_reward):
        out = load_and_call_train(env.args, env.data_source)

    assert len(out.samples) == 3 and all(group[0].reward == 1 for group in out.samples)
    assert len(env.mock_server.request_log) == 4, "the filtered group needs one replacement, and none is left to abort"
