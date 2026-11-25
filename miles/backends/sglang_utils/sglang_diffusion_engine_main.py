"""Entry point of one ``sglang.multimodal_gen`` server process, pinned to its GPU before CUDA initializes."""

import argparse
import json
import os
import runpy
import sys
import tempfile

_SGLANG_LAUNCHER = "sglang.multimodal_gen.runtime.launch_server"


def pin_gpu(base_gpu_id: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[base_gpu_id] if visible else str(base_gpu_id)


def to_sglang_argv(server_args: dict) -> list[str]:
    """The command line SGLang's own launcher parses, so it names unknown args and rejects conflicting topology."""
    from sglang.multimodal_gen.runtime.server_args import ServerArgs
    from sglang.multimodal_gen.runtime.utils.argparse import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    ServerArgs.add_cli_args(parser)
    with tempfile.NamedTemporaryFile("w", suffix=".json") as config_file:
        json.dump(server_args, config_file)
        config_file.flush()
        # SGLang's own expansion of a --config file, booleans and nested configs included.
        argv = parser._load_config_file(config_file.name)
    # Only parse_args turns underscores into dashes; the launcher's parse_known_args does not.
    return ["--" + arg[2:].replace("_", "-") if arg.startswith("--") else arg for arg in argv]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-gpu-id", type=int, required=True)
    parser.add_argument("--server-args", type=json.loads, required=True)
    cli = parser.parse_args(argv)

    # Before any sglang import: sglang.multimodal_gen initializes CUDA on import.
    pin_gpu(cli.base_gpu_id)
    sys.argv = [_SGLANG_LAUNCHER, *to_sglang_argv(cli.server_args)]
    runpy.run_module(_SGLANG_LAUNCHER, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
