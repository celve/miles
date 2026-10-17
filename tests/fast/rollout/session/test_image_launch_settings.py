from __future__ import annotations

from argparse import Namespace

import pytest
from tests.fast.backends.sglang_utils.test_sglang_config import _make_args as _make_launch_args

from miles.rollout.session.images.launch_settings import validate_session_image_args


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
        }

    def test_support_off_resolves_nothing(self):
        args = _image_args(session_image_support="off", rollout_external=True)
        validate_session_image_args(args)
        assert args.session_image_engine_settings is None

    def test_a_base_flag_is_checked(self):
        with pytest.raises(ValueError, match="mm_process_config"):
            validate_session_image_args(_image_args(sglang_mm_process_config={"image": {"do_normalize": False}}))

    def test_a_group_override_beats_the_base_flag(self, tmp_path):
        yaml_text = _groups(_group(4, tokenizer_path="/ckpt/actor"), _group(4, tokenizer_path="/ckpt/actor"))
        args = _image_args(tmp_path, yaml_text, sglang_tokenizer_path="/ckpt/renormalized")
        validate_session_image_args(args)
        assert args.session_image_engine_settings["tokenizer_path"] == "/ckpt/actor"

    def test_an_unsupported_group_override_is_refused(self, tmp_path):
        yaml_text = _groups(_group(8, mm_process_config="{image: {do_normalize: false}}"))
        with pytest.raises(ValueError, match="mm_process_config"):
            validate_session_image_args(_image_args(tmp_path, yaml_text))

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

    def test_parsers_and_sampling_defaults_are_left_to_the_engines(self):
        args = _image_args(
            sglang_reasoning_parser="auto", sglang_tool_call_parser="auto", sglang_sampling_defaults="openai"
        )
        validate_session_image_args(args)
        assert "reasoning_parser" not in args.session_image_engine_settings
