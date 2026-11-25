"""Entry point of one ``sglang.multimodal_gen`` server process, pinned to its GPU before CUDA initializes."""

import argparse
import dataclasses
import json
import os

# PipelineConfig.update_config_from_dict reads `<name>.<field>` keys into these nested configs.
_NESTED_CONFIGS = ("dit_config", "vae_config", "t5_config")


def pin_gpu(base_gpu_id: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[base_gpu_id] if visible else str(base_gpu_id)


def reject_unknown_server_args(server_args: dict, *, accepted: set[str]) -> None:
    unknown = [
        key
        for key in server_args
        if key not in accepted and not (key.partition(".")[0] in _NESTED_CONFIGS and key.partition(".")[2])
    ]
    if unknown:
        raise ValueError(f"sglang.multimodal_gen does not take these server args: {sorted(unknown)}")


def reject_unknown_nested_args(server_args: dict, *, pipeline_config) -> None:
    """SGLang drops a `<name>.<field>` key whose field the model's config lacks; refuse it instead."""
    unknown = [key for key in server_args if "." in key and not _has_nested_field(pipeline_config, key)]
    if unknown:
        raise ValueError(f"this model's sglang.multimodal_gen pipeline has no such config fields: {sorted(unknown)}")


def _has_nested_field(pipeline_config, key: str) -> bool:
    name, _, field = key.partition(".")
    if name == "t5_config":
        config = next((c for c in pipeline_config.text_encoder_configs if type(c).__name__ == "T5Config"), None)
    else:
        config = getattr(pipeline_config, name)  # config-access-exempt: nested config named by the override key
    return hasattr(config, field)  # config-access-exempt: field named by the override key


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
    server_args = ServerArgs.from_kwargs(**cli.server_args)
    reject_unknown_nested_args(cli.server_args, pipeline_config=server_args.pipeline_config)
    launch_server(server_args)


if __name__ == "__main__":
    main()
