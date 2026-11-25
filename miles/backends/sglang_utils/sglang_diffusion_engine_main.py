"""Entry point of one ``sglang.multimodal_gen`` server process, pinned to its GPU before CUDA initializes."""

import argparse
import dataclasses
import json
import os


def pin_gpu(base_gpu_id: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[base_gpu_id] if visible else str(base_gpu_id)


def reject_unknown_server_args(server_args: dict, *, accepted: set[str]) -> None:
    if unknown := sorted(server_args.keys() - accepted):
        raise ValueError(f"sglang.multimodal_gen does not take these server args: {unknown}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-gpu-id", type=int, required=True)
    parser.add_argument("--server-args", type=json.loads, required=True)
    cli = parser.parse_args(argv)

    pin_gpu(cli.base_gpu_id)

    # Imported after pinning: sglang.multimodal_gen initializes CUDA on import.
    from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
    from sglang.multimodal_gen.runtime.launch_server import launch_server
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    # ServerArgs.from_kwargs also hands its kwargs to the model's PipelineConfig (dit_precision and the like).
    accepted = {f.name for f in dataclasses.fields(ServerArgs)} | {f.name for f in dataclasses.fields(PipelineConfig)}
    reject_unknown_server_args(cli.server_args, accepted=accepted)
    launch_server(ServerArgs.from_kwargs(**cli.server_args))


if __name__ == "__main__":
    main()
