from argparse import Namespace
from unittest.mock import Mock

import pytest
from tests.fast.fixtures.sglang_config_fixtures import make_sglang_config
from tests.fast.rollout.inference_rollout.conftest import make_eval_args, make_state

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput, RolloutFnConstructorInput, RolloutFnEvalInput
from miles.rollout.inference_rollout import inference_rollout_common as common
from miles.rollout.inference_rollout import inference_rollout_eval as rollout_eval
from miles.rollout.inference_rollout.inference_rollout_common import (
    GenerateState,
    InferenceRolloutFn,
    RolloutGenerationHooks,
)
from miles.utils.data import Dataset
from miles.utils.retry_utils import NonRetryableError
from miles.utils.types import Sample


def test_default_generation_state_keeps_text_loaders_and_sampling(monkeypatch):
    tokenizer, processor = Mock(), Mock()
    monkeypatch.setattr(common, "load_tokenizer", tokenizer)
    monkeypatch.setattr(common, "load_processor", processor)
    args = Namespace(
        hf_checkpoint="text-model",
        chat_template_path="template",
        rollout_temperature=0.8,
        rollout_top_p=0.9,
        rollout_top_k=25,
        rollout_max_response_len=64,
        rollout_stop=["end"],
        rollout_stop_token_ids=[42],
        rollout_skip_special_tokens=True,
        custom_generate_function_path=None,
    )

    state = GenerateState(args)

    tokenizer.assert_called_once_with("text-model", chat_template_path="template", trust_remote_code=True)
    processor.assert_called_once_with("text-model", trust_remote_code=True)
    assert state.tokenizer is tokenizer.return_value
    assert state.processor is processor.return_value
    assert state.sampling_params == {
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 25,
        "max_new_tokens": 64,
        "stop": ["end"],
        "stop_token_ids": [42],
        "skip_special_tokens": True,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }


def test_subclass_supplies_hooks_to_the_state_it_builds(monkeypatch):
    tokenizer, processor = Mock(), Mock()
    monkeypatch.setattr(common, "load_tokenizer", tokenizer)
    monkeypatch.setattr(common, "load_processor", processor)
    sampling = Mock(return_value={"num_inference_steps": 12})

    class NonTextRollout(InferenceRolloutFn):
        def generation_hooks(self) -> RolloutGenerationHooks:
            return RolloutGenerationHooks(load_tokenizer=False, sampling_params=sampling)

    args = Namespace(custom_generate_function_path=None)
    fn = NonTextRollout(RolloutFnConstructorInput(args=args, data_source=Mock()))

    tokenizer.assert_not_called()
    processor.assert_not_called()
    assert fn.state.tokenizer is None
    assert fn.state.processor is None
    sampling.assert_called_once_with(args)
    assert fn.state.sampling_params == {"num_inference_steps": 12}


@pytest.mark.parametrize("custom_prompts", [False, True])
@pytest.mark.parametrize("custom_sampling", [False, True])
async def test_eval_prompt_and_sampling_hooks_are_independent(monkeypatch, tmp_path, custom_prompts, custom_sampling):
    args = Namespace(
        **make_eval_args(namespaced_radix_cache=False),
        custom_generate_function_path=None,
        sglang_server_concurrency=2,
        inference_runtime_mut_state=Namespace(engine_count=1),
        multimodal_token_space_processing=False,
        eval_max_prompt_len=None,
        multimodal_keys=None,
        apply_chat_template_kwargs=None,
        shared_media_dir=None,
    )
    [config] = args.eval_datasets
    config.input_key = "text"
    config.label_key = "label"
    config.max_response_len = 32
    path = tmp_path / "eval.jsonl"
    path.write_text('{"text": "text prompt", "label": "text label"}\n')
    config.path = str(path)
    tokenizer, processor = Mock(), Mock()
    monkeypatch.setattr(rollout_eval, "load_tokenizer", tokenizer)
    monkeypatch.setattr(rollout_eval, "load_processor", processor)
    original = Sample(prompt="custom prompt", label="custom label")
    load_prompts = Mock(return_value=[original])
    sampling = Mock(return_value={"num_inference_steps": 20})
    captured: list[GenerateFnInput] = []

    async def generate(input: GenerateFnInput) -> GenerateFnOutput:
        captured.append(input)
        input.sample.status = Sample.Status.COMPLETED
        input.sample.reward = 1.0
        return GenerateFnOutput(samples=input.sample)

    class EvalRollout(InferenceRolloutFn):
        def generation_hooks(self) -> RolloutGenerationHooks:
            return RolloutGenerationHooks(
                load_tokenizer=False,
                sampling_params=lambda args: {},
                eval_prompts=load_prompts if custom_prompts else None,
                eval_sampling_params=sampling if custom_sampling else None,
            )

    monkeypatch.setattr(common, "generate", generate)
    fn = EvalRollout(RolloutFnConstructorInput(args=args, data_source=Mock()))

    output = await fn(RolloutFnEvalInput(rollout_id=0))

    if custom_prompts:
        load_prompts.assert_called_once_with(args, config)
        tokenizer.assert_not_called()
        processor.assert_not_called()
        assert fn.eval_prompt_dataset_cache == {}
    else:
        [dataset] = fn.eval_prompt_dataset_cache.values()
        assert isinstance(dataset, Dataset)
        [original] = dataset.samples
        assert (original.prompt, original.label) == ("text prompt", "text label")
        tokenizer.assert_called_once_with("fake-checkpoint", chat_template_path=None, trust_remote_code=True)
        processor.assert_called_once_with("fake-checkpoint", trust_remote_code=True)
    if custom_sampling:
        sampling.assert_called_once_with(args, config)
        assert all(call.sampling_params == {"num_inference_steps": 20} for call in captured)
    else:
        assert all(call.sampling_params["max_new_tokens"] == 32 for call in captured)
    result = output.data[config.name]
    assert result["rewards"] == [1.0, 1.0]
    assert result["truncated"] == [False, False]
    assert [sample.index for sample in result["samples"]] == [0, 1]
    assert [(sample.prompt, sample.label) for sample in result["samples"]] == [(original.prompt, original.label)] * 2
    assert original.status == Sample.Status.PENDING


@pytest.mark.parametrize(
    ("seed_key", "deterministic", "expected"),
    [
        (None, False, [{}, {}]),
        (None, True, [{"sampling_seed": 42}, {"sampling_seed": 43}]),
        ("seed", False, [{"seed": 42}, {"seed": 43}]),
    ],
)
async def test_eval_seed_key_preserves_text_defaults(monkeypatch, seed_key, deterministic, expected):
    args = Namespace(**make_eval_args(namespaced_radix_cache=False), rollout_seed=42)
    args.sglang = make_sglang_config(enable_deterministic_inference=deterministic)
    sampling_params = {}
    state = make_state(None, args=args)
    state.hooks = RolloutGenerationHooks(
        eval_prompts=lambda args, cfg: [Sample(prompt="prompt")],
        eval_sampling_params=lambda args, cfg: sampling_params,
        eval_seed_key=seed_key,
    )
    captured = []

    async def generate_and_rm(state, sample, sampling_params, **kwargs):
        captured.append(sampling_params)
        sample.reward = 1.0
        return sample

    monkeypatch.setattr(rollout_eval, "generate_and_rm", generate_and_rm)

    await rollout_eval.eval_rollout_single_dataset(state, args.eval_datasets[0], {})

    assert captured == expected
    assert sampling_params == {}


@pytest.mark.parametrize("retry", [True, False])
async def test_group_reward_error_uses_the_hook_policy(monkeypatch, retry):
    scorer_error = RuntimeError("scorer failed")

    async def generate_and_rm(state, sample, sampling_params, **kwargs):
        return sample

    async def batched_async_rm(args, group, **kwargs):
        raise scorer_error

    monkeypatch.setattr(common, "generate_and_rm", generate_and_rm)
    monkeypatch.setattr(common, "batched_async_rm", batched_async_rm)
    state = make_state(None)
    if not retry:
        state.hooks = RolloutGenerationHooks(retry_group_reward_errors=False)

    expected_error = RuntimeError if retry else NonRetryableError
    with pytest.raises(expected_error, match="scorer failed") as error:
        await common.generate_and_rm_group(state, [Sample(index=0)], {})

    assert (error.value if retry else error.value.__cause__) is scorer_error
