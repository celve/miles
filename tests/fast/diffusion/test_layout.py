"""Guards for the diffusion layout: the paths resolved only at run time import, and the trainer stays a leaf."""

import ast
import importlib
import re
import shlex
from pathlib import Path

import pytest

from miles.backends.sglang_utils.sglang_diffusion_engine import compute_diffusion_engine_launch_cmd
from miles.ray.specs.reward import REWARD_WORKER_CLASS
from miles.ray.specs.train import _DIFFUSION_TRAINER_ACTOR_CLASSES
from miles.utils.args.configs.diffusion import DiffusionConfig
from miles.utils.diffusion.arguments import _HOOKS
from miles.utils.diffusion.rewards import OCR_SCORER, PICKSCORE_SCORER

REPO = Path(__file__).resolve().parents[3]
TRAINER_PACKAGE = "miles.backends.fsdp_utils.diffusion"
PREPARE_SNAPSHOT = REPO / "tests/snapshots/launch_scripts/py/scripts/run_sd35_medium_flow_grpo_ocr.py/prepare.txt"
NFT_EXECUTE_SNAPSHOT = REPO / "tests/snapshots/launch_scripts/py/scripts/run_sd35_medium_nft_ocr.py/execute.txt"
WAN_EXECUTE_SNAPSHOT = (
    REPO / "tests/snapshots/launch_scripts/py/scripts/run_wan22_a14b_flow_grpo_pickscore.py/execute.txt"
)


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
    diffusion_flow_shift = None


def _recipe_prepare_imports():
    return re.findall(r"from ([\w.]+) import (\w+)", PREPARE_SNAPSHOT.read_text())


def _recipe_step_strategy(snapshot):
    (path,) = re.findall(r"--diffusion-step-strategy-path (\S+)", snapshot.read_text())
    return path


RUNTIME_OBJECTS = [
    *_HOOKS.values(),
    *_DIFFUSION_TRAINER_ACTOR_CLASSES.values(),
    DiffusionConfig.model_fields["diffusion_step_strategy_path"].default,
    DiffusionConfig.model_fields["diffusion_nft_timestep_strategy_path"].default,
    _recipe_step_strategy(NFT_EXECUTE_SNAPSHOT),
    _recipe_step_strategy(WAN_EXECUTE_SNAPSHOT),
    # rm_hub imports PickScore inside the call, so text runs never load it.
    "miles.rollout.rm_hub.pickscore.pickscore_rm",
    # Reward workers construct these by path, inside the worker process.
    REWARD_WORKER_CLASS,
    OCR_SCORER,
    PICKSCORE_SCORER,
    *(f"{module}.{name}" for module, name in _recipe_prepare_imports()),
]


@pytest.mark.parametrize("path", RUNTIME_OBJECTS)
def test_each_path_resolved_at_run_time_names_an_object(path):
    module, name = path.rsplit(".", 1)

    assert hasattr(importlib.import_module(module), name), path


def test_the_engine_launch_module_imports():
    importlib.import_module(_engine_main_module())


def _module_name(path):
    parts = list(path.relative_to(REPO).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imported(node, module_name, is_package):
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    base = node.module
    if node.level:
        package = module_name.split(".")[: None if is_package else -1]
        base = ".".join([*package[: len(package) - node.level + 1], *filter(None, [node.module])])
    return [f"{base}.{alias.name}" for alias in node.names]


def _eager_imports(tree, module_name, is_package=False):
    """Imports that run when the module is imported: everywhere but inside function bodies."""
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node.lineno, _imported(node, module_name, is_package)
        pending += ast.iter_child_nodes(node)


def _names(module, prefix):
    return module == prefix or module.startswith(prefix + ".")


def _offenders(source, module_name, is_package=False):
    return [
        lineno
        for lineno, modules in _eager_imports(ast.parse(source), module_name, is_package)
        if any(_names(module, TRAINER_PACKAGE) for module in modules)
    ]


@pytest.mark.parametrize(
    "source, module_name",
    [
        ("from miles.backends.fsdp_utils import diffusion", "miles.x"),
        ("from .diffusion import actor", "miles.backends.fsdp_utils.x"),
        ("from ..fsdp_utils.diffusion.actor import A", "miles.backends.sglang_utils.x"),
        ("try:\n    pass\nfinally:\n    import miles.backends.fsdp_utils.diffusion", "miles.x"),
        ("class A:\n    from miles.backends.fsdp_utils.diffusion import actor", "miles.x"),
        ("with x:\n    import miles.backends.fsdp_utils.diffusion.sde", "miles.x"),
    ],
)
def test_the_scan_sees_every_form_of_an_eager_trainer_import(source, module_name):
    assert _offenders(source, module_name)


def test_the_scan_allows_a_trainer_import_inside_a_function():
    assert not _offenders("def f():\n    from miles.backends.fsdp_utils.diffusion import actor", "miles.x")


def test_only_the_trainer_package_imports_itself_at_module_level():
    trainer_dir = REPO / TRAINER_PACKAGE.replace(".", "/")
    offenders = [
        f"{path.relative_to(REPO)}:{lineno}"
        for path in sorted((REPO / "miles").rglob("*.py"))
        if trainer_dir not in path.parents
        for lineno in _offenders(path.read_text(), _module_name(path), path.name == "__init__.py")
    ]

    assert offenders == []


SHARED_LEAVES = [
    REPO / "miles/utils/diffusion/types.py",
    *sorted((REPO / "miles/utils/diffusion/algorithms").glob("*.py")),
]


@pytest.mark.parametrize("path", SHARED_LEAVES, ids=lambda path: str(path.relative_to(REPO)))
def test_the_shared_payload_and_algorithms_import_no_trainer_or_rollout_code(path):
    imported = [
        module
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for module in _imported(node, _module_name(path), path.name == "__init__.py")
    ]

    assert not [module for module in imported if _names(module, TRAINER_PACKAGE) or _names(module, "miles.rollout")]
