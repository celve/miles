"""Build the served model directory of the keyframe recipe's reference-to-video generator.

The directory sits beside the downloaded diffusers checkpoint and links its components. Its ``model_index.json``
names ``WanVACEReferencePipeline`` and the LoRAs to fuse, and its scheduler takes the flow shift. It also holds
``reference_pipeline.py`` as ``pipeline.py``. sglang's diffusers backend loads the directory as a custom pipeline. The
build is cheap and idempotent: it rewrites everything but the linked weights.

    python -m examples.experimental.video_gen_agent.keyframe.reference_model --src <diffusers dir> --dst <served dir> \\
        --flow-shift 5 --lora <path>:transformer
"""

import argparse
import json
import shutil
from pathlib import Path

PIPELINE_CLASS = "WanVACEReferencePipeline"
_OWN = ("model_index.json", "scheduler", "pipeline.py")


def build(src: Path, dst: Path, *, flow_shift: float, loras: list[tuple[str, str]]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        link = dst / entry.name
        if entry.name in _OWN:
            continue
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(entry.resolve())
    index = json.loads((src / "model_index.json").read_text())
    index |= {"_class_name": PIPELINE_CLASS, "lora_paths": [list(lora) for lora in loras]}
    (dst / "model_index.json").write_text(json.dumps(index, indent=2))
    scheduler = json.loads((src / "scheduler" / "scheduler_config.json").read_text())
    (dst / "scheduler").mkdir(exist_ok=True)
    (dst / "scheduler" / "scheduler_config.json").write_text(json.dumps(scheduler | {"flow_shift": flow_shift}))
    shutil.copyfile(Path(__file__).with_name("reference_pipeline.py"), dst / "pipeline.py")


def _lora(value: str) -> tuple[str, str]:
    path, _, component = value.rpartition(":")
    if component not in ("transformer", "transformer_2") or not path:
        raise argparse.ArgumentTypeError(f"--lora is <path>:transformer or <path>:transformer_2, got {value!r}")
    return path, component


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--flow-shift", type=float, required=True)
    parser.add_argument("--lora", type=_lora, action="append", default=[])
    args = parser.parse_args(argv)
    build(args.src, args.dst, flow_shift=args.flow_shift, loras=args.lora)


if __name__ == "__main__":
    main()
