"""Entry point of one sglang.multimodal_gen server process, pinned to its GPUs before CUDA initializes."""

import argparse
import dataclasses
import json
import os

# `ServerGroupConfig.resolve` adds this text-engine key to a group's overrides; multimodal_gen sleeps whole engines.
_MILES_INJECTED_TEXT_ARGS = {"enable_memory_saver"}
# PipelineConfig fields, which `ServerArgs.from_kwargs` reads from the same keyword arguments.
_PIPELINE_CONFIG_ARGS = {"dit_precision", "vae_precision", "flow_shift"}


def _pin_gpus(base_gpu_id: int, num_gpus: int) -> None:
    gpu_ids = range(base_gpu_id, base_gpu_id + num_gpus)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = [visible.split(",")[gpu_id] for gpu_id in gpu_ids] if visible else [str(gpu_id) for gpu_id in gpu_ids]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-gpu-id", type=int, required=True)
    parser.add_argument("--server-args", type=json.loads, required=True)
    cli = parser.parse_args()

    _pin_gpus(cli.base_gpu_id, cli.server_args.get("num_gpus", 1))

    from sglang.multimodal_gen.runtime.launch_server import launch_server
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    accepted = {field.name for field in dataclasses.fields(ServerArgs)} | _PIPELINE_CONFIG_ARGS
    if unknown := sorted(cli.server_args.keys() - accepted - _MILES_INJECTED_TEXT_ARGS):
        raise ValueError(f"sglang.multimodal_gen does not take these server args: {unknown}")
    launch_server(ServerArgs.from_kwargs(**{k: v for k, v in cli.server_args.items() if k in accepted}))


if __name__ == "__main__":
    main()
