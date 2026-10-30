"""A text run neither imports the diffusion dependencies nor sees the diffusion additions."""

import subprocess
import sys
import textwrap

from miles.utils.args.configs.diffusion import DiffusionConfig
from miles.utils.types import Sample

_DIFFUSION_ONLY_PACKAGES = ["diffusers", "peft", "paddleocr", "paddle", "Levenshtein"]


def test_text_modules_import_and_parse_without_diffusion_packages():
    script = textwrap.dedent(
        f"""
        import importlib.abc, sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in {_DIFFUSION_ONLY_PACKAGES!r}:
                    raise ImportError(f"diffusion-only package {{name}} imported on the text path")

        sys.meta_path.insert(0, Block())
        import miles.backends.training_utils.weight_update.protocol
        import miles.ray.rollout.server_cell
        import miles.ray.rollout.train_data_conversion
        import miles.ray.specs.inference
        import miles.ray.specs.train
        from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

        assert parse_fsdp_test_config().model_kind == "text"
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_text_samples_serialize_as_before():
    sample = Sample(index=3, prompt="p", tokens=[1, 2], response_length=1)

    value = sample.to_dict()

    assert list(Sample.__dataclass_fields__)[-1] == "diffusion"
    assert "diffusion" not in value
    assert set(value) == set(Sample.__dataclass_fields__) - {"diffusion"}
    assert Sample.from_dict(value).to_dict() == value


def test_text_runs_keep_every_diffusion_flag_at_its_default():
    from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

    args = parse_fsdp_test_config()

    for name, field in DiffusionConfig.model_fields.items():
        assert getattr(args, name) == field.get_default(call_default_factory=True), name
    assert args.model_kind == "text"
