from __future__ import annotations

from argparse import Namespace

import pytest
from pydantic import ValidationError
from tests.fast.backends.sglang_utils.test_sglang_config import _make_args as _make_launch_args

from miles.rollout.session.config import (
    SessionServerConfig,
    compute_session_server_config,
    validate_session_image_args,
)

_ARGS_TO_CONFIG_FIELD = {
    "miles_router_timeout": "timeout",
    "hf_checkpoint": "hf_checkpoint",
    "chat_template_path": "chat_template_path",
    "tito_model": "tito_model",
    "apply_chat_template_kwargs": "apply_chat_template_kwargs",
    "use_rollout_routing_replay": "use_rollout_routing_replay",
    "use_rollout_indexer_replay": "use_rollout_indexer_replay",
    "use_sampling_support_replay": "use_sampling_support_replay",
    "sglang_speculative_algorithm": "sglang_speculative_algorithm",
    "num_layers": "num_layers",
    "moe_router_topk": "moe_router_topk",
    "save_debug_trajectory_data": "save_debug_trajectory_data",
    "lora_rank": "lora_rank",
    "lora_adapter_path": "lora_adapter_path",
    "lora_train_only": "lora_train_only",
    "use_session_server": "use_session_server",
    "session_image_support": "session_image_support",
    "session_image_engine_settings": "session_image_engine_settings",
    "session_image_text_only_leaf": "session_image_text_only_leaf",
    "session_message_matcher": "session_message_matcher",
    "pause_generation_mode": "pause_generation_mode",
    "session_sample_picker_path": "session_sample_picker_path",
    "session_sample_postprocessor_path": "session_sample_postprocessor_path",
    "rollout_top_logprobs_num": "rollout_top_logprobs_num",
    "rollout_sampling_logprobs_mode": "rollout_sampling_logprobs_mode",
    "rollout_temperature": "rollout_temperature",
}

_CALL_SITE_FIELDS = ("host", "port", "instance_id", "backend_url")
_OPTIONAL_ARGS_ATTRS = (
    "num_layers",
    "pause_generation_mode",
    "moe_router_topk",
    "use_session_server",
    "session_sample_picker_path",
    "session_sample_postprocessor_path",
)

_DISTINCT_ARGS_VALUES = dict(
    miles_router_timeout=31.5,
    hf_checkpoint="/fake/model",
    chat_template_path="/fake/chat_template.jinja",
    tito_model="tito-xyz",
    apply_chat_template_kwargs={"enable_thinking": True},
    use_rollout_routing_replay=True,
    use_rollout_indexer_replay=True,
    use_sampling_support_replay=True,
    sglang_speculative_algorithm="EAGLE",
    num_layers=61,
    moe_router_topk=8,
    save_debug_trajectory_data="/fake/traj",
    lora_rank=32,
    lora_adapter_path="/fake/adapters/x",
    lora_train_only=True,
    use_session_server="v2",
    session_image_support="fake_vlm",
    session_image_engine_settings={"model_path": "/fake/ckpt", "image_processor_backend": "auto"},
    session_image_text_only_leaf="keep",
    session_message_matcher="fake.matcher",
    pause_generation_mode="in_place",
    session_sample_picker_path="fake.picker",
    session_sample_postprocessor_path="fake.postprocessor",
    rollout_top_logprobs_num=32,
    rollout_sampling_logprobs_mode="support",
    rollout_temperature=0.7,
)


def _make_args(**overrides) -> Namespace:
    defaults = dict(_DISTINCT_ARGS_VALUES)
    defaults.update(overrides)
    return Namespace(**defaults)


class TestComputeSessionServerConfig:
    def test_every_config_field_has_a_known_source(self):
        """Adding a config field without extending this test's mapping must fail here."""
        covered = set(_CALL_SITE_FIELDS) | set(_ARGS_TO_CONFIG_FIELD.values())
        assert covered == set(SessionServerConfig.model_fields)

    def test_call_site_fields_are_copied(self):
        """Host, port, instance id, and backend url come from the caller, not from args."""
        config = compute_session_server_config(
            _make_args(), host="10.0.0.1", port=5001, instance_id="abc", backend_url="http://10.0.0.2:3000"
        )
        assert config.host == "10.0.0.1"
        assert config.port == 5001
        assert config.instance_id == "abc"
        assert config.backend_url == "http://10.0.0.2:3000"

    @pytest.mark.parametrize("args_attr,config_field", sorted(_ARGS_TO_CONFIG_FIELD.items()))
    def test_each_args_field_is_copied(self, args_attr: str, config_field: str):
        """Every args-derived field maps one-to-one, using values no hardcoded default would produce."""
        config = compute_session_server_config(
            _make_args(), host="10.0.0.1", port=5001, instance_id="abc", backend_url="http://10.0.0.2:3000"
        )
        assert getattr(config, config_field) == _DISTINCT_ARGS_VALUES[args_attr]

    @pytest.mark.parametrize("flag_value", [True, False])
    def test_a_boolean_session_server_flag_reaches_the_config_unchanged(self, flag_value: bool):
        """A bare --use-session-server (True) and an omitted one (False) both survive the launch-time config build."""
        config = compute_session_server_config(
            _make_args(use_session_server=flag_value),
            host="10.0.0.1",
            port=5001,
            instance_id="abc",
            backend_url="http://10.0.0.2:3000",
        )
        assert config.use_session_server is flag_value

    def test_missing_optional_args_fall_back_to_none(self):
        """An args object that omits the optional attributes yields None for them instead of failing."""
        present = {name: value for name, value in _DISTINCT_ARGS_VALUES.items() if name not in _OPTIONAL_ARGS_ATTRS}
        config = compute_session_server_config(
            Namespace(**present), host="10.0.0.1", port=5001, instance_id="abc", backend_url="http://10.0.0.2:3000"
        )
        assert [getattr(config, name) for name in _OPTIONAL_ARGS_ATTRS] == [None] * len(_OPTIONAL_ARGS_ATTRS)

    def test_older_args_default_rollout_temperature(self):
        args = Namespace(
            **{key: value for key, value in _DISTINCT_ARGS_VALUES.items() if key != "rollout_temperature"}
        )
        config = compute_session_server_config(
            args, host="127.0.0.1", port=5001, instance_id=None, backend_url="http://127.0.0.1:3000"
        )
        assert config.rollout_temperature == 1.0


_COMPLETE_CONFIG_KWARGS = dict(
    host="127.0.0.1",
    port=5001,
    instance_id=None,
    backend_url="http://127.0.0.1:3000",
    timeout=None,
    hf_checkpoint=None,
    chat_template_path=None,
    tito_model="default",
    apply_chat_template_kwargs=None,
    use_rollout_routing_replay=False,
    use_rollout_indexer_replay=False,
    use_sampling_support_replay=False,
    sglang_speculative_algorithm=None,
    num_layers=None,
    moe_router_topk=None,
    save_debug_trajectory_data=None,
    lora_rank=0,
    lora_adapter_path=None,
    lora_train_only=False,
    use_session_server=None,
    session_image_support="off",
    session_image_engine_settings=None,
    session_image_text_only_leaf="drop",
    session_message_matcher="strict",
    pause_generation_mode=None,
    session_sample_picker_path=None,
    session_sample_postprocessor_path=None,
    rollout_top_logprobs_num=0,
    rollout_sampling_logprobs_mode="selected",
    rollout_temperature=1.0,
)


class TestSessionServerConfig:
    def test_complete_kwargs_cover_every_field(self):
        """The per-field omission cases are only meaningful if the full kwargs validate."""
        assert set(_COMPLETE_CONFIG_KWARGS) == set(SessionServerConfig.model_fields)
        assert SessionServerConfig(**_COMPLETE_CONFIG_KWARGS).port == 5001

    def test_the_bare_session_server_flag_is_accepted(self):
        """--use-session-server written without a version arrives as True, which used to fail validation."""
        config = SessionServerConfig(**{**_COMPLETE_CONFIG_KWARGS, "use_session_server": True})

        assert config.use_session_server is True

    def test_the_omitted_session_server_flag_is_accepted(self):
        """Leaving --use-session-server off yields False, which the str-only annotation used to reject."""
        config = SessionServerConfig(**{**_COMPLETE_CONFIG_KWARGS, "use_session_server": False})

        assert config.use_session_server is False

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_a_named_session_server_version_stays_a_string(self, version: str):
        """Widening the flag to accept booleans must not turn a named version into a bool."""
        config = SessionServerConfig(**{**_COMPLETE_CONFIG_KWARGS, "use_session_server": version})

        assert config.use_session_server == version

    @pytest.mark.parametrize("value", [3.5, ["v2"], {"version": "v2"}])
    def test_a_session_server_flag_of_another_type_is_rejected(self, value: object):
        """The flag accepts only booleans and version strings, so other types must still fail validation."""
        with pytest.raises(ValidationError):
            SessionServerConfig(**{**_COMPLETE_CONFIG_KWARGS, "use_session_server": value})

    @pytest.mark.parametrize("missing", sorted(SessionServerConfig.model_fields))
    def test_every_field_is_required(self, missing: str):
        """Omitting any single field must fail, so none can be silently defaulted."""
        kwargs = {name: value for name, value in _COMPLETE_CONFIG_KWARGS.items() if name != missing}
        with pytest.raises(ValidationError):
            SessionServerConfig(**kwargs)

    def test_unknown_field_is_rejected(self):
        """An unrecognized keyword must fail validation instead of being silently ignored."""
        with pytest.raises(ValidationError):
            SessionServerConfig(**_COMPLETE_CONFIG_KWARGS, unknown_field="whatever")

    def test_field_assignment_is_rejected(self):
        """The config is frozen, so assigning to a field after construction must fail."""
        config = SessionServerConfig(**_COMPLETE_CONFIG_KWARGS)
        with pytest.raises(ValidationError):
            config.port = 5002


def _image_args(tmp_path=None, yaml_text: str | None = None, **overrides) -> Namespace:
    """Launch args as `miles_validate_args` leaves them, with SGLang 0.5.22's server defaults."""
    if yaml_text is not None:
        (tmp_path / "sglang.yaml").write_text(yaml_text)
        overrides["sglang_config"] = str(tmp_path / "sglang.yaml")
    values = dict(
        session_image_support="qwen3_vl",
        use_session_server="v2",
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        rollout_external=False,
        sglang_revision=None,
        sglang_tokenizer_path=None,
        sglang_mm_process_config=None,
        sglang_image_processor_backend="auto",
        sglang_disable_fast_image_processor=False,
        sglang_sampling_defaults="model",
        sglang_tool_call_parser=None,
        sglang_reasoning_parser=None,
    )
    values.update(overrides)
    return _make_launch_args(**values)


def _groups(*groups: str) -> str:
    return "sglang:\n  - name: actor\n    server_groups:\n" + "".join(groups)


def _group(num_gpus: int, worker_type: str = "regular", **overrides) -> str:
    text = f"      - worker_type: {worker_type}\n        num_gpus: {num_gpus}\n"
    if overrides:
        text += "        overrides:\n" + "".join(f"          {key}: {value}\n" for key, value in overrides.items())
    return text


class TestSessionImageEngine:
    """The image settings come from the args Miles launches the policy engines with."""

    def test_the_default_launch_args_resolve_to_the_engines_defaults(self):
        args = _image_args()
        validate_session_image_args(args)
        assert args.session_image_engine_settings == {
            "model_path": "/ckpt/actor",
            "revision": None,
            "tokenizer_path": "/ckpt/actor",
            "mm_process_config": None,
            "image_processor_backend": "auto",
            "disable_fast_image_processor": False,
            "sampling_defaults": "model",
            "tool_call_parser": None,
            "reasoning_parser": None,
        }

    def test_support_off_resolves_nothing(self):
        args = _image_args(session_image_support="off", rollout_external=True)
        validate_session_image_args(args)
        assert args.session_image_engine_settings is None

    def test_a_base_flag_is_checked(self):
        with pytest.raises(ValueError, match="mm_process_config"):
            validate_session_image_args(_image_args(sglang_mm_process_config={"image": {"do_normalize": False}}))

    def test_a_group_override_beats_the_base_flag(self, tmp_path):
        args = _image_args(
            tmp_path, _groups(_group(4, tool_call_parser="qwen25"), _group(4, tool_call_parser="qwen25"))
        )
        validate_session_image_args(args)
        assert args.session_image_engine_settings["tool_call_parser"] == "qwen25"

    def test_an_unsupported_group_override_is_refused(self, tmp_path):
        yaml_text = _groups(_group(8, mm_process_config="{image: {do_normalize: false}}"))
        with pytest.raises(ValueError, match="mm_process_config"):
            validate_session_image_args(_image_args(tmp_path, yaml_text))

    def test_groups_that_launch_differently_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match=r"differ in \['tool_call_parser'\]"):
            validate_session_image_args(
                _image_args(tmp_path, _groups(_group(4, tool_call_parser="qwen25"), _group(4)))
            )

    def test_a_placeholder_group_launches_no_engine_and_is_ignored(self, tmp_path):
        args = _image_args(tmp_path, _groups(_group(4), _group(4, worker_type="placeholder")))
        validate_session_image_args(args)
        assert args.session_image_engine_settings["image_processor_backend"] == "auto"

    def test_another_checkpoint_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="the two processors would disagree"):
            validate_session_image_args(_image_args(tmp_path, _groups(_group(8, model_path="/ckpt/other"))))

    def test_external_engines_are_refused(self):
        with pytest.raises(ValueError, match="engines Miles launches itself"):
            validate_session_image_args(_image_args(rollout_external=True))

    def test_prefill_decode_disaggregation_is_refused(self, tmp_path):
        yaml_text = _groups(_group(4, worker_type="prefill"), _group(4, worker_type="decode"))
        with pytest.raises(ValueError, match="disaggregation"):
            validate_session_image_args(_image_args(tmp_path, yaml_text))

    def test_a_tokenizer_path_flag_naming_the_checkpoint_is_the_default(self):
        args = _image_args(sglang_tokenizer_path="/ckpt/actor")
        validate_session_image_args(args)
        assert args.session_image_engine_settings["tokenizer_path"] == "/ckpt/actor"

    def test_a_processor_root_other_than_the_checkpoint_is_refused(self):
        """SGLang loads the image processor from tokenizer_path, which the token ids cannot vouch for."""
        with pytest.raises(ValueError, match="load their tokenizer and image processor from"):
            validate_session_image_args(_image_args(sglang_tokenizer_path="/ckpt/renormalized"))

    def test_a_processor_root_override_on_every_group_is_refused(self, tmp_path):
        yaml_text = _groups(
            _group(4, tokenizer_path="/ckpt/renormalized"), _group(4, tokenizer_path="/ckpt/renormalized")
        )
        with pytest.raises(ValueError, match="load their tokenizer and image processor from"):
            validate_session_image_args(_image_args(tmp_path, yaml_text))

    def test_a_processor_root_override_on_one_group_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match=r"differ in \['tokenizer_path'\]"):
            validate_session_image_args(
                _image_args(tmp_path, _groups(_group(4, tokenizer_path="/ckpt/renormalized"), _group(4)))
            )

    def test_the_legacy_slow_processor_flag_is_refused(self):
        """It forces SGLang's 'pil' backend whatever image_processor_backend says."""
        with pytest.raises(ValueError, match="disable_fast_image_processor"):
            validate_session_image_args(_image_args(sglang_disable_fast_image_processor=True))

    def test_an_auto_parser_is_refused(self):
        with pytest.raises(ValueError, match="name the parser explicitly"):
            validate_session_image_args(_image_args(sglang_reasoning_parser="auto"))
