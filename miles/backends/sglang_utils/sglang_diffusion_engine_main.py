"""Entry point of one ``sglang.multimodal_gen`` server process, pinned to its GPU before CUDA initializes."""

import argparse
import json
import os
import sys

# Ports and GPUs Miles allocated for this engine; SGLang may move a busy port when it builds its args.
_ALLOCATED_ARGS = ("num_gpus", "port", "master_port")


def pin_gpu(base_gpu_id: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[base_gpu_id] if visible else str(base_gpu_id)


def reject_unknown_server_args(server_args: dict, *, option_names: set[str]) -> None:
    """Accept only the names SGLang's own CLI registers, spelled as its option destinations (no aliases)."""
    if unknown := sorted(server_args.keys() - option_names):
        raise ValueError(f"sglang.multimodal_gen has no server option named: {unknown}")


def reject_moved_allocations(server_args: dict, *, resolved) -> None:
    resolved_values = {key: getattr(resolved, key) for key in _ALLOCATED_ARGS}  # config-access-exempt: SGLang's args
    moved = {key: value for key, value in resolved_values.items() if value != server_args[key]}
    if moved:
        raise ValueError(f"sglang.multimodal_gen resolved {moved}, not what Miles allocated for this engine")


def sglang_option_names(server_args: dict) -> set[str]:
    """The options SGLang's `serve` command takes for this model, its pipeline's own ones included."""
    from sglang.multimodal_gen.runtime.entrypoints.cli.serve import add_multimodal_gen_serve_args
    from sglang.multimodal_gen.runtime.utils.argparse import FlexibleArgumentParser

    # SGLang picks the pipeline's extra options from the command line it is parsing.
    selector = [
        f"--{key.replace('_', '-')}={server_args[key]}"
        for key in ("model_path", "pipeline_class_name")
        if key in server_args
    ]
    saved_argv, sys.argv = sys.argv, [sys.argv[0], *selector]
    try:
        parser = add_multimodal_gen_serve_args(FlexibleArgumentParser())
    finally:
        sys.argv = saved_argv
    return {action.dest for action in parser._actions} - {"help", "config"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-gpu-id", type=int, required=True)
    parser.add_argument("--server-args", type=json.loads, required=True)
    cli = parser.parse_args(argv)

    pin_gpu(cli.base_gpu_id)

    # Imported after pinning: sglang.multimodal_gen initializes CUDA on import.
    from sglang.multimodal_gen.runtime.launch_server import dispatch_launch
    from sglang.multimodal_gen.runtime.server_args import ServerArgs
    from sglang.multimodal_gen.runtime.utils.process import kill_process_tree

    reject_unknown_server_args(cli.server_args, option_names=sglang_option_names(cli.server_args))
    # The dict constructor SGLang's own CLI ends in; it keeps structured values such as warmup_sampling_params.
    server_args = ServerArgs.from_dict(dict(cli.server_args))
    reject_moved_allocations(cli.server_args, resolved=server_args)
    try:
        dispatch_launch(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
