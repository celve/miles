"""Entry point of one sglang.multimodal_gen server process, pinned to its GPU before CUDA initializes."""

import argparse
import json
import os


def _pin_gpu(base_gpu_id: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = visible.split(",")[base_gpu_id] if visible else str(base_gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = physical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-gpu-id", type=int, required=True)
    parser.add_argument("--server-args", type=json.loads, required=True)
    cli = parser.parse_args()

    _pin_gpu(cli.base_gpu_id)

    from sglang.multimodal_gen.runtime.launch_server import launch_server
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    launch_server(ServerArgs.from_kwargs(**cli.server_args))


if __name__ == "__main__":
    main()
