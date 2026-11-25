from __future__ import annotations

import json
import shlex
import subprocess
import sys
import types

import pytest

from miles.backends.sglang_utils import sglang_diffusion_engine_main
from miles.backends.sglang_utils.sglang_diffusion_engine import compute_diffusion_engine_launch_cmd


def _launch_argv(**overrides) -> list[str]:
    kwargs = dict(
        interpreter_prefix=["/usr/bin/python3"],
        base_gpu_id=2,
        sglang_overrides={"model_path": "/models/Z-Image-Turbo"},
        host="10.0.0.5",
        port=30000,
        nccl_port=10001,
        master_port=9001,
    )
    return shlex.split(compute_diffusion_engine_launch_cmd(**(kwargs | overrides)))


class TestLaunchCommand:
    def test_the_engine_starts_through_the_gpu_pinning_entry_point(self):
        argv = _launch_argv()
        assert argv[:6] == [
            "/usr/bin/python3",
            "-m",
            "miles.backends.sglang_utils.sglang_diffusion_engine_main",
            "--base-gpu-id",
            "2",
            "--server-args",
        ]
        assert json.loads(argv[6]) == {
            "model_path": "/models/Z-Image-Turbo",
            "trust_remote_code": True,
            "host": "10.0.0.5",
            "port": 30000,
            "nccl_port": 10001,
            "master_port": 9001,
            "num_gpus": 1,
            "warmup_mode": "off",
            "performance_mode": "manual",
        }

    def test_the_group_overrides_win_over_miles_defaults(self):
        overrides = {"model_path": "/m", "performance_mode": "speed", "batching_max_size": 4}
        server_args = json.loads(_launch_argv(sglang_overrides=overrides)[6])
        assert server_args["performance_mode"] == "speed"
        assert server_args["batching_max_size"] == 4

    def test_an_ipv6_host_is_passed_without_brackets(self):
        assert json.loads(_launch_argv(host="[fd00::1]")[6])["host"] == "fd00::1"


class TestEntryPoint:
    @pytest.mark.parametrize(
        ("visible", "base_gpu_id", "expected"), [(None, 3, "3"), ("4,5,6", 1, "5"), ("GPU-a,GPU-b", 0, "GPU-a")]
    )
    def test_the_engine_sees_only_its_own_gpu(self, monkeypatch, visible, base_gpu_id, expected):
        if visible is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
        sglang_diffusion_engine_main.pin_gpu(base_gpu_id)
        assert sglang_diffusion_engine_main.os.environ["CUDA_VISIBLE_DEVICES"] == expected

    def test_an_unknown_server_arg_is_rejected_by_name_before_launch(self):
        """Pipeline args such as dit_precision pass; enable_memory_saver, a text-server arg, does not."""
        server_args = {
            "model_path": "/m",
            "dit_precision": "bf16",
            "dit_config.torch_compile_mode": "max-autotune",
            "enable_memory_saver": False,
            "not_an_arg": 1,
            "dit_configs.x": 1,
        }
        result = subprocess.run(
            [sys.executable, "-m", "miles.backends.sglang_utils.sglang_diffusion_engine_main", "--base-gpu-id", "0"]
            + ["--server-args", json.dumps(server_args)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode != 0
        assert "does not take these server args: ['dit_configs.x', 'enable_memory_saver', 'not_an_arg']" in (
            result.stderr
        )


class TestNestedConfigArgs:
    _PIPELINE = types.SimpleNamespace(
        dit_config=types.SimpleNamespace(torch_compile_mode=None),
        vae_config=types.SimpleNamespace(tile_size=None),
        text_encoder_configs=[],
    )

    def test_fields_the_model_config_has_are_accepted(self):
        server_args = {"model_path": "/m", "dit_config.torch_compile_mode": "x", "vae_config.tile_size": 1}
        sglang_diffusion_engine_main.reject_unknown_nested_args(server_args, pipeline_config=self._PIPELINE)

    def test_fields_sglang_would_silently_drop_are_rejected_by_name(self):
        server_args = {"dit_config.no_such_field": 1, "t5_config.x": 1, "dit_config.torch_compile_mode": "x"}
        with pytest.raises(ValueError, match=r"\['dit_config.no_such_field', 't5_config.x'\]"):
            sglang_diffusion_engine_main.reject_unknown_nested_args(server_args, pipeline_config=self._PIPELINE)
