from __future__ import annotations

from argparse import Namespace

import pydantic
import pytest
from tests.fast.fixtures.sglang_config_fixtures import resolve_sglang_config

from miles.backends.sglang_utils.sglang_config import EngineKind

_POLICY = """\
  - name: policy
    server_groups:
      - worker_type: regular
        num_gpus: 1
"""


def _make_args(**overrides) -> Namespace:
    defaults = dict(
        sglang_config=None,
        prefill_num_servers=None,
        rollout_num_gpus=3,
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=8,
        eval_num_gpus=0,
        eval_num_gpus_per_engine=1,
        hf_checkpoint="/ckpt/policy",
        offload_rollout=False,
        debug_train_only=False,
        debug_rollout_only=False,
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        multi_lora=False,
        router="miles",
        starts_inference_engines=True,
    )
    return Namespace(**(defaults | overrides))


def _resolve(tmp_path, yaml_text: str, **overrides):
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)
    return resolve_sglang_config(_make_args(sglang_config=str(path), **overrides))


def _generator(*, update_weights: str = "false", groups: str | None = None) -> str:
    groups = groups or ("      - worker_type: regular\n" "        engine_kind: diffusion\n" "        num_gpus: 2\n")
    return (
        "  - name: generator\n"
        "    model_path: /models/Z-Image-Turbo\n"
        f"    update_weights: {update_weights}\n"
        "    server_groups:\n" + groups
    )


class TestEngineKindParsing:
    def test_a_group_without_the_key_is_a_text_engine(self, tmp_path):
        """Every config written before the key existed must keep launching sglang.launch_server."""
        cfg = _resolve(tmp_path, "sglang:\n" + _POLICY.replace("num_gpus: 1", "num_gpus: 3"))
        assert [g.engine_kind for g in cfg.models[0].server_groups] == [EngineKind.TEXT]
        assert cfg.models[0].engine_kind == EngineKind.TEXT

    def test_a_diffusion_group_resolves_beside_the_policy(self, tmp_path):
        cfg = _resolve(tmp_path, "sglang:\n" + _POLICY + _generator())
        policy, generator = cfg.models
        assert policy.engine_kind == EngineKind.TEXT
        assert generator.engine_kind == EngineKind.DIFFUSION
        assert generator.server_groups[0].overrides == {"model_path": "/models/Z-Image-Turbo"}
        assert generator.update_weights is False

    def test_an_unknown_engine_kind_is_rejected(self, tmp_path):
        groups = "      - worker_type: regular\n        engine_kind: video\n        num_gpus: 2\n"
        with pytest.raises(pydantic.ValidationError, match="engine_kind"):
            _resolve(tmp_path, "sglang:\n" + _POLICY + _generator(groups=groups))

    def test_offload_rollout_gives_a_diffusion_group_no_text_server_arg(self, tmp_path):
        """sglang.multimodal_gen rejects enable_memory_saver, which offload_rollout adds to resident text groups."""
        cfg = _resolve(tmp_path, "sglang:\n" + _POLICY + _generator(), offload_rollout=True)
        policy, generator = cfg.models
        assert policy.server_groups[0].overrides["enable_memory_saver"] is False
        assert "enable_memory_saver" not in generator.server_groups[0].overrides


_RULES = {
    "update_weights": "update_weights must be false",
    "first": "must not be the first model",
    "all_diffusion": "every server group must set engine_kind: diffusion",
    "router": "--router miles is required",
    "regular": "every server group must be worker_type: regular",
    "one_gpu": "num_gpus_per_engine must be 1",
    "colocate": "--colocate is not supported",
}


class TestDiffusionModelValidation:
    def test_a_valid_generator_passes(self, tmp_path):
        _resolve(tmp_path, "sglang:\n" + _POLICY + _generator())

    @pytest.mark.parametrize(
        ("rule", "yaml_text", "overrides"),
        [
            ("update_weights", "sglang:\n" + _POLICY + _generator(update_weights="true"), {}),
            ("first", "sglang:\n" + _generator() + _POLICY, {}),
            (
                "all_diffusion",
                "sglang:\n"
                + _POLICY
                + _generator(
                    groups="      - worker_type: regular\n        engine_kind: diffusion\n        num_gpus: 1\n"
                    "      - worker_type: regular\n        num_gpus: 1\n"
                ),
                {},
            ),
            ("router", "sglang:\n" + _POLICY + _generator(), {"router": "sglang"}),
            (
                "regular",
                "sglang:\n"
                + _POLICY
                + _generator(
                    groups="      - worker_type: placeholder\n        engine_kind: diffusion\n        num_gpus: 2\n"
                ),
                {},
            ),
            (
                "one_gpu",
                "sglang:\n"
                + _POLICY
                + _generator(
                    groups="      - worker_type: regular\n        engine_kind: diffusion\n        num_gpus: 2\n"
                    "        num_gpus_per_engine: 2\n"
                ),
                {},
            ),
            ("colocate", "sglang:\n" + _POLICY + _generator(), {"colocate": True}),
        ],
    )
    def test_each_rule_alone_is_named(self, tmp_path, rule, yaml_text, overrides):
        with pytest.raises(ValueError) as exc_info:
            _resolve(tmp_path, yaml_text, **overrides)
        message = str(exc_info.value)
        assert "Diffusion model 'generator'" in message
        assert [name for name, text in _RULES.items() if text in message] == [rule]

    def test_every_broken_rule_is_named_in_one_error(self, tmp_path):
        groups = (
            "      - worker_type: prefill\n        engine_kind: diffusion\n        num_gpus: 2\n"
            "        num_gpus_per_engine: 2\n"
            "      - worker_type: regular\n        num_gpus: 1\n"
        )
        yaml_text = "sglang:\n" + _generator(update_weights="true", groups=groups) + _POLICY
        with pytest.raises(ValueError) as exc_info:
            _resolve(tmp_path, yaml_text, rollout_num_gpus=4, router="sglang", colocate=True)
        message = str(exc_info.value)
        assert [name for name, text in _RULES.items() if text in message] == list(_RULES)

    def test_a_frozen_text_model_is_not_held_to_the_diffusion_rules(self, tmp_path):
        """A frozen text ref model beside the policy predates diffusion groups and must keep resolving."""
        ref = (
            "  - name: ref\n    model_path: /models/ref\n    update_weights: false\n    server_groups:\n"
            "      - worker_type: regular\n        num_gpus: 2\n        num_gpus_per_engine: 2\n"
        )
        cfg = _resolve(tmp_path, "sglang:\n" + _POLICY + ref, router="sglang")
        assert [m.engine_kind for m in cfg.models] == [EngineKind.TEXT, EngineKind.TEXT]
