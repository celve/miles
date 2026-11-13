import json
import os

from miles.utils.tracking_utils import tracking

# Set to a path to also append every logged metric dict as one JSON line (exact floats, for parity checks).
METRICS_JSONL_ENV = "MILES_DIFFUSION_METRICS_JSONL"


def log_metrics(args, metrics: dict[str, float], *, step_key: str) -> None:
    tracking.log(args, metrics, step_key=step_key)
    if path := os.environ.get(METRICS_JSONL_ENV):
        with open(path, "a") as f:
            f.write(json.dumps({"step_key": step_key, **metrics}) + "\n")
