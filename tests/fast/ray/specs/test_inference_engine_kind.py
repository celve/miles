from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.specs import inference
from miles.ray.specs.inference import InferenceEngineSpec
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.workers.launch_gate import GATE_PORT_NAME
from miles.utils.workers.worker_spec import HostAndPort, LaunchCommandContext

_TEXT_PATH_SNAPSHOT = Path(__file__).parent / "snapshots" / "text_path_engine_specs.json"

# Configs with no engine_kind, covering each path the key touches: default, multi-model with a frozen ref, PD,
# placeholders, and offload_rollout (which injects enable_memory_saver into resident groups).
_TEXT_CASES: dict[str, dict] = {
    "default_layout": dict(rollout_num_gpus=4, rollout_num_gpus_per_engine=2),
    "policy_and_frozen_ref": dict(
        yaml="sglang:\n"
        "  - name: policy\n    server_groups:\n      - {worker_type: regular, num_gpus: 2}\n"
        "  - name: ref\n    model_path: /models/ref\n    update_weights: false\n    server_groups:\n"
        "      - {worker_type: regular, num_gpus: 2, num_gpus_per_engine: 2, overrides: {mem_fraction_static: 0.5}}\n"
        "      - {worker_type: placeholder, num_gpus: 1}\n",
        rollout_num_gpus=5,
    ),
    "pd_disaggregation": dict(
        yaml="sglang:\n  - name: default\n    server_groups:\n"
        "      - {worker_type: prefill, num_gpus: 2}\n      - {worker_type: decode, num_gpus: 2}\n",
        rollout_num_gpus=4,
    ),
    "offload_rollout_resident": dict(
        yaml="sglang:\n  - name: default\n    server_groups:\n      - {worker_type: regular, num_gpus: 2}\n"
        "  - name: ref\n    model_path: /models/ref\n    server_groups:\n      - {worker_type: regular, num_gpus: 1}\n",
        rollout_num_gpus=3,
        offload_rollout=True,
    ),
}

_PORTS = {
    "primary": 30000,
    "dist_init": 9000,
    "nccl": 10000,
    "disaggregation_bootstrap": 11000,
    "engine_info_bootstrap": 12000,
    GATE_PORT_NAME: 13000,
}


def _launch_ctx(args, spec: InferenceEngineSpec) -> LaunchCommandContext:
    """Addresses for exactly the ports the spec declares, as the launcher hands them out."""
    ports = {info.name: _PORTS[info.name] for info in spec.port_infos}
    return LaunchCommandContext(
        args=args,
        scaling=ScalingConfig(sglang_scaling=args.sglang_scaling),
        cell_index=0,
        worker_in_cell_index=0,
        num_workers_per_cell=1,
        gpu_ids=[1],
        local_gpu_ids=[1],
        self_addrs={name: HostAndPort(host="10.0.0.5", port=port) for name, port in ports.items()},
        pool_addrs={},
    )


def _make_case_args(tmp_path: Path, *, yaml: str | None = None, **overrides):
    if yaml is not None:
        path = tmp_path / "sglang.yaml"
        path.write_text(yaml)
        overrides["sglang_config"] = str(path)
    return make_args(hf_checkpoint="/models/policy", **overrides)


def _specs(args) -> list[InferenceEngineSpec]:
    return [spec.model_copy(update=dict(interpreter_prefix=["python3"])) for spec in InferenceEngineSpec.create(args)]


def _record_launch(**kwargs) -> str:
    # Everything Miles decides; the unchanged compute_engine_launch_cmd turns it into argv with sglang's defaults.
    return json.dumps({name: value for name, value in kwargs.items() if name != "args"}, sort_keys=True, default=str)


def _describe_specs(args) -> list[dict]:
    scaling = ScalingConfig(sglang_scaling=args.sglang_scaling)
    return [
        dict(
            name=spec.name,
            port_infos=[info.model_dump(mode="json") for info in spec.port_infos],
            static_meta=spec.compute_static_meta(scaling).resolve(cell_index=0),
            launch_inputs=json.loads(spec.launch_command(_launch_ctx(args, spec))),
        )
        for spec in _specs(args)
    ]


def _capture_text_path(tmp_path: Path) -> dict:
    captured = {}
    original = inference.compute_engine_launch_cmd
    inference.compute_engine_launch_cmd = _record_launch
    try:
        for name, case in _TEXT_CASES.items():
            case_dir = tmp_path / name
            case_dir.mkdir()
            captured[name] = _describe_specs(_make_case_args(case_dir, **case))
    finally:
        inference.compute_engine_launch_cmd = original
    return json.loads(json.dumps(captured))


class TestTextPathUnchanged:
    def test_configs_without_engine_kind_launch_as_main_does(self, tmp_path):
        """Snapshot captured on main a18187f3ef; this test passes there and must keep passing unchanged."""
        assert _capture_text_path(tmp_path) == json.loads(_TEXT_PATH_SNAPSHOT.read_text())


_DIFFUSION_YAML = (
    "sglang:\n"
    "  - name: policy\n    server_groups:\n      - {worker_type: regular, num_gpus: 1}\n"
    "  - name: generator\n    model_path: /models/Z-Image-Turbo\n    update_weights: false\n    server_groups:\n"
    "      - {worker_type: regular, engine_kind: diffusion, num_gpus: 2, overrides: {batching_max_size: 4}}\n"
)


@pytest.fixture
def diffusion_args(tmp_path):
    return _make_case_args(tmp_path, yaml=_DIFFUSION_YAML, rollout_num_gpus=3, router="miles")


class TestDiffusionEngineSpec:
    def test_the_generator_launches_sglang_multimodal_gen_on_its_own_gpu(self, diffusion_args):
        policy, generator = _specs(diffusion_args)
        argv = shlex.split(generator.launch_command(_launch_ctx(diffusion_args, generator)))

        assert argv[:6] == [
            "python3",
            "-m",
            "miles.backends.sglang_utils.sglang_diffusion_engine_main",
            "--base-gpu-id",
            "1",
            "--server-args",
        ]
        assert json.loads(argv[6]) == {
            "model_path": "/models/Z-Image-Turbo",
            "trust_remote_code": True,
            "host": "10.0.0.5",
            "port": 30000,
            "nccl_port": 10000,
            "master_port": 9000,
            "num_gpus": 1,
            "warmup_mode": "off",
            "performance_mode": "manual",
            "batching_max_size": 4,
        }
        assert "sglang.launch_server" in policy.launch_command(_launch_ctx(diffusion_args, policy))

    def test_the_generator_reserves_no_launch_gate(self, diffusion_args):
        """sglang.multimodal_gen serves no gate; a reserved one would make its cell wait on a port nobody opens."""
        policy, generator = _specs(diffusion_args)
        assert GATE_PORT_NAME in [info.name for info in policy.port_infos]
        assert [info.name for info in generator.port_infos] == ["primary", "dist_init", "nccl"]

    def test_only_generator_cells_carry_the_engine_kind(self, diffusion_args):
        scaling = ScalingConfig(sglang_scaling=diffusion_args.sglang_scaling)
        policy, generator = _specs(diffusion_args)
        assert "engine_kind" not in policy.compute_static_meta(scaling).resolve(cell_index=0)
        generator_meta = generator.compute_static_meta(scaling).resolve(cell_index=1)
        assert generator_meta["engine_kind"] == "diffusion"
        assert generator_meta["update_weights"] is False
        assert generator_meta["gpu_offset"] == 2


if __name__ == "__main__":
    # Regenerate the text-path snapshot; run on the base branch only.
    import tempfile

    sys.orig_argv = [sys.executable, "-m", "pytest"]
    with tempfile.TemporaryDirectory() as tmp:
        _TEXT_PATH_SNAPSHOT.parent.mkdir(exist_ok=True)
        _TEXT_PATH_SNAPSHOT.write_text(json.dumps(_capture_text_path(Path(tmp)), indent=2, sort_keys=True) + "\n")
