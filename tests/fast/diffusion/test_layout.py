"""Guards for the diffusion layout: the paths resolved only at run time import, and the trainer stays a leaf."""

import ast
import importlib
import re
import shlex
from pathlib import Path

import pytest

from miles.backends.sglang_utils.sglang_diffusion_engine import compute_diffusion_engine_launch_cmd
from miles.ray.specs.train import _DIFFUSION_TRAINER_ACTOR_CLASSES
from miles.utils.args.configs.diffusion import DiffusionConfig
from miles.utils.diffusion.arguments import _HOOKS

REPO = Path(__file__).resolve().parents[3]
TRAINER_PACKAGE = "miles.backends.fsdp_utils.diffusion"
PREPARE_SNAPSHOT = REPO / "tests/snapshots/launch_scripts/py/scripts/run_sd35_medium_flow_grpo_ocr.py/prepare.txt"


def _engine_main_module():
    argv = shlex.split(
        compute_diffusion_engine_launch_cmd(
            _EngineArgs(),
            interpreter_prefix=["python"],
            base_gpu_id=0,
            num_gpus_per_engine=1,
            sglang_overrides={},
            nccl_port=1,
            host="h",
            port=2,
        )
    )
    return argv[argv.index("-m") + 1]


class _EngineArgs:
    hf_checkpoint = "/m"
    diffusion_lora_target_modules = ["attn.to_q"]
    diffusion_dit_precision = None
    diffusion_forward_dtype = "bf16"


def _recipe_prepare_imports():
    return re.findall(r"from ([\w.]+) import (\w+)", PREPARE_SNAPSHOT.read_text())


RUNTIME_OBJECTS = [
    *_HOOKS.values(),
    *_DIFFUSION_TRAINER_ACTOR_CLASSES.values(),
    DiffusionConfig.model_fields["diffusion_step_strategy_path"].default,
    *(f"{module}.{name}" for module, name in _recipe_prepare_imports()),
]


@pytest.mark.parametrize("path", RUNTIME_OBJECTS)
def test_each_path_resolved_at_run_time_names_an_object(path):
    module, name = path.rsplit(".", 1)

    assert hasattr(importlib.import_module(module), name), path


def test_the_engine_launch_module_imports():
    importlib.import_module(_engine_main_module())


def _module_level_imports(tree):
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Import):
            yield node.lineno, [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, [node.module]
        elif isinstance(node, (ast.If, ast.Try)):
            pending += [*node.body, *node.orelse, *(n for h in getattr(node, "handlers", []) for n in h.body)]


def _names(module, prefix):
    return module == prefix or module.startswith(prefix + ".")


def test_only_the_trainer_package_imports_itself_at_module_level():
    trainer_dir = REPO / TRAINER_PACKAGE.replace(".", "/")
    offenders = [
        f"{path.relative_to(REPO)}:{lineno}"
        for path in sorted((REPO / "miles").rglob("*.py"))
        if trainer_dir not in path.parents
        for lineno, modules in _module_level_imports(ast.parse(path.read_text()))
        if any(_names(module, TRAINER_PACKAGE) for module in modules)
    ]

    assert offenders == []


def test_the_shared_sample_payload_imports_no_trainer_or_rollout_code():
    tree = ast.parse((REPO / "miles/utils/diffusion/types.py").read_text())
    imported = [
        module
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for module in ([node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names])
        if module
    ]

    assert not [module for module in imported if _names(module, TRAINER_PACKAGE) or _names(module, "miles.rollout")]
