"""Entry point of one ``sglang.multimodal_gen`` server process, pinned to its GPU before CUDA initializes."""

import argparse
import json
import os


def pin_gpu(base_gpu_id: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[base_gpu_id] if visible else str(base_gpu_id)


def reject_unknown_server_args(server_args: dict, *, option_names: set[str]) -> None:
    """Accept only the names SGLang's own CLI registers, spelled as its option destinations (no aliases)."""
    if unknown := sorted(server_args.keys() - option_names):
        raise ValueError(f"sglang.multimodal_gen has no server option named: {unknown}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-gpu-id", type=int, required=True)
    parser.add_argument("--server-args", type=json.loads, required=True)
    cli = parser.parse_args(argv)

    pin_gpu(cli.base_gpu_id)

    # Imported after pinning: sglang.multimodal_gen initializes CUDA on import.
    from sglang.multimodal_gen.runtime.launch_server import dispatch_launch
    from sglang.multimodal_gen.runtime.server_args import ServerArgs
    from sglang.multimodal_gen.runtime.utils.argparse import FlexibleArgumentParser
    from sglang.multimodal_gen.runtime.utils.process import kill_process_tree

    sglang_parser = FlexibleArgumentParser()
    ServerArgs.add_cli_args(sglang_parser)
    option_names = {action.dest for action in sglang_parser._actions} - {"help"}
    reject_unknown_server_args(cli.server_args, option_names=option_names)
    # The dict constructor SGLang's own CLI ends in; it keeps structured values such as warmup_sampling_params.
    server_args = ServerArgs.from_dict(dict(cli.server_args))
    assert server_args.num_gpus == 1, f"a diffusion engine runs on one GPU, but SGLang resolved {server_args.num_gpus}"
    try:
        dispatch_launch(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
