import json
import shlex
from types import SimpleNamespace

from miles.diffusion.engine import compute_diffusion_engine_launch_cmd


def _args(**overrides):
    values = dict(
        hf_checkpoint="/models/default",
        diffusion_lora_target_modules=["attn.to_q"],
        diffusion_dit_precision="fp16",
        diffusion_forward_dtype="fp16",
    )
    return SimpleNamespace(**(values | overrides))


def _launch(overrides):
    argv = shlex.split(
        compute_diffusion_engine_launch_cmd(
            _args(),
            interpreter_prefix=["python"],
            base_gpu_id=1,
            num_gpus_per_engine=1,
            sglang_overrides=overrides,
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


def _engine_spec_for(tmp_path, model_kind):
    from tests.fast.ray.rollout.conftest import make_args, make_sglang_config_yaml

    from miles.ray.specs.inference import InferenceEngineSpec

    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(
        make_sglang_config_yaml(server_groups=[{"worker_type": "regular", "num_gpus": 2, "num_gpus_per_engine": 1}])
    )
    (spec,) = InferenceEngineSpec.create(
        make_args(sglang_config=str(config_path), rollout_num_gpus=2, model_kind=model_kind)
    )
    return spec


def test_only_diffusion_engines_drop_the_launch_gate(tmp_path):
    assert "gate" in {port.name for port in _engine_spec_for(tmp_path, "text").port_infos}
    assert "gate" not in {port.name for port in _engine_spec_for(tmp_path, "diffusion").port_infos}
