"""Entry point of one sglang.multimodal_gen server process, pinned to its GPU before CUDA initializes."""

import argparse
import dataclasses
import json
import logging
import os

logger = logging.getLogger(__name__)


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

    accepted = {field.name for field in dataclasses.fields(ServerArgs)} | {"dit_precision", "vae_precision"}
    # Miles adds text-engine keys (e.g. enable_memory_saver) to every group's overrides; drop what this server lacks.
    if dropped := sorted(cli.server_args.keys() - accepted):
        logger.warning(f"sglang.multimodal_gen does not take these server args, dropping them: {dropped}")
    launch_server(ServerArgs.from_kwargs(**{k: v for k, v in cli.server_args.items() if k in accepted}))


if __name__ == "__main__":
    main()
