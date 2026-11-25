"""Launch command of an ``engine_kind: diffusion`` inference engine (``sglang.multimodal_gen``)."""

import json
import shlex


def compute_diffusion_engine_launch_cmd(
    *,
    interpreter_prefix: list[str],
    base_gpu_id: int,
    sglang_overrides: dict,
    host: str,
    port: int,
    nccl_port: int,
    master_port: int,
) -> str:
    server_args = compute_diffusion_server_args(
        sglang_overrides=sglang_overrides, host=host, port=port, nccl_port=nccl_port, master_port=master_port
    )
    return shlex.join(
        [
            *interpreter_prefix,
            "-m",
            "miles.backends.sglang_utils.sglang_diffusion_engine_main",
            "--base-gpu-id",
            str(base_gpu_id),
            "--server-args",
            json.dumps(server_args),
        ]
    )


def compute_diffusion_server_args(
    *, sglang_overrides: dict, host: str, port: int, nccl_port: int, master_port: int
) -> dict:
    """Keyword arguments of ``sglang.multimodal_gen``'s ``ServerArgs.from_kwargs``.

    The group's overrides, which carry the model's ``model_path``, win over these defaults, as they do for text
    engines; none of the run-wide ``--sglang-*`` text-server args apply.
    """
    server_args = {
        "trust_remote_code": True,
        "host": host.strip("[]"),
        "port": port,
        "nccl_port": nccl_port,
        "master_port": master_port,
        "num_gpus": 1,
        "warmup_mode": "off",
        # Miles places engines on GPUs of their own; the automatic policy would offload by free memory at startup.
        "performance_mode": "manual",
    }
    return {**server_args, **sglang_overrides}
