import dataclasses
import json
import shlex
import sys
from types import ModuleType, SimpleNamespace

import pytest

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.diffusion import engine_main
from miles.diffusion.api_client import DiffusionApiClient
from miles.diffusion.engine import compute_diffusion_engine_launch_cmd
from miles.ray.rollout.server_cell import ServerCell


def _args(**overrides):
    values = dict(
        hf_checkpoint="/models/default",
        diffusion_lora_target_modules=["attn.to_q"],
        diffusion_dit_precision="fp16",
        diffusion_forward_dtype="fp16",
        diffusion_flow_shift=None,
    )
    return SimpleNamespace(**(values | overrides))


def _launch(overrides, update_weights=True, **args):
    argv = shlex.split(
        compute_diffusion_engine_launch_cmd(
            _args(**args),
            interpreter_prefix=["python"],
            base_gpu_id=1,
            num_gpus_per_engine=1,
            sglang_overrides=overrides,
            update_weights=update_weights,
            nccl_port=15001,
            host="[::1]",
            port=15000,
        )
    )
    return argv, json.loads(argv[argv.index("--server-args") + 1])


def test_the_engine_is_multimodal_gen_pinned_to_its_gpu():
    argv, server_args = _launch({})

    assert argv[:3] == ["python", "-m", "miles.diffusion.engine_main"]
    assert argv[argv.index("--base-gpu-id") + 1] == "1"
    assert server_args["host"] == "::1" and server_args["master_port"] == 25001
    assert server_args["lora_merge_mode"] == "merge" and server_args["dit_precision"] == "fp16"


def test_the_server_groups_overrides_and_model_path_win():
    _, server_args = _launch({"model_path": "/models/from-sglang-config", "dit_precision": "bf16"})

    assert server_args["model_path"] == "/models/from-sglang-config"
    assert server_args["dit_precision"] == "bf16"


def test_the_flow_shift_reaches_the_engine_only_when_set():
    assert "flow_shift" not in _launch({})[1]
    assert _launch({}, diffusion_flow_shift=3.0)[1]["flow_shift"] == 3.0


def _engine_spec_for(tmp_path, model_kind, **group):
    from tests.fast.ray.rollout.conftest import make_args, make_sglang_config_yaml

    from miles.ray.specs.inference import InferenceEngineSpec

    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(
        make_sglang_config_yaml(
            server_groups=[{"worker_type": "regular", "num_gpus": 2, "num_gpus_per_engine": 1, **group}]
        )
    )
    (spec,) = InferenceEngineSpec.create(
        make_args(sglang_config=str(config_path), rollout_num_gpus=2, model_kind=model_kind)
    )
    return spec


def test_only_diffusion_engines_drop_the_launch_gate(tmp_path):
    assert "gate" in {port.name for port in _engine_spec_for(tmp_path, "text").port_infos}
    assert "gate" not in {port.name for port in _engine_spec_for(tmp_path, "diffusion").port_infos}


def test_each_group_takes_its_engine_kind_from_its_own_config(tmp_path):
    text_group = _engine_spec_for(tmp_path, "diffusion", engine_kind="text")
    diffusion_group = _engine_spec_for(tmp_path, "text", engine_kind="diffusion")

    assert "gate" in {port.name for port in text_group.port_infos}
    assert text_group.partial_static_meta.values["engine_kind"] == "text"
    assert "gate" not in {port.name for port in diffusion_group.port_infos}
    assert diffusion_group.partial_static_meta.values["engine_kind"] == "diffusion"


def test_a_cell_picks_its_client_from_its_engine_kind():
    def client_class(engine_kind):
        cell = SimpleNamespace(meta=SimpleNamespace(engine_kind=engine_kind), args=SimpleNamespace(model_kind="text"))
        return ServerCell._api_client_class.fget(cell)

    assert client_class("diffusion") is DiffusionApiClient
    assert client_class("text") is SGLangApiClient


@dataclasses.dataclass
class _ServerArgs:
    model_path: str = ""
    lora_merge_mode: str = ""

    @classmethod
    def from_kwargs(cls, **kwargs):
        return cls(**kwargs)


def _run_engine_main(monkeypatch, server_args, server_args_cls=_ServerArgs):
    launched = []
    launch_module = ModuleType("sglang.multimodal_gen.runtime.launch_server")
    launch_module.launch_server = launched.append
    args_module = ModuleType("sglang.multimodal_gen.runtime.server_args")
    args_module.ServerArgs = server_args_cls
    monkeypatch.setitem(sys.modules, launch_module.__name__, launch_module)
    monkeypatch.setitem(sys.modules, args_module.__name__, args_module)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(sys, "argv", ["engine_main", "--base-gpu-id", "0", "--server-args", json.dumps(server_args)])
    engine_main.main()
    return launched


def test_an_engine_arg_multimodal_gen_lacks_fails_naming_it(monkeypatch):
    with pytest.raises(ValueError, match="mem_fraction_static"):
        _run_engine_main(monkeypatch, {"model_path": "/m", "mem_fraction_static": 0.8})


def test_only_the_key_miles_injects_into_every_group_is_filtered(monkeypatch):
    launched = _run_engine_main(monkeypatch, {"model_path": "/m", "enable_memory_saver": False})

    assert launched == [_ServerArgs(model_path="/m")]


class _KwargsServerArgs(_ServerArgs):
    @classmethod
    def from_kwargs(cls, **kwargs):
        return kwargs


def test_the_flow_shift_passes_to_multimodal_gen_as_a_pipeline_config_arg(monkeypatch):
    launched = _run_engine_main(monkeypatch, {"model_path": "/m", "flow_shift": 3.0}, _KwargsServerArgs)

    assert launched == [{"model_path": "/m", "flow_shift": 3.0}]
