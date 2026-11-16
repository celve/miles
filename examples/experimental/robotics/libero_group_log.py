"""Miles' nonzero-std group filter, plus one JSON line per decided group appended to ``<run dir>/groups.jsonl``.

The filter drops all-success and all-fail groups before anything is dumped, so this log is the only record of
which LIBERO rows they were: it gives training-state success per rollout and lets a run drop rows already solved.
Pass ``--dynamic-sampling-filter-path libero_group_log.filter_and_log``.
"""

import json
import os
import time

from miles.rollout.filter_hub.common_filters import apply_reward_nonzero_std_filter, iter_samples


def _policy_version(sample) -> str | None:
    try:
        return sample.weight_versions[0]["spans"][0]["version"]
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


def filter_and_log(args, samples, **kwargs):
    output = apply_reward_nonzero_std_filter(args, samples, **kwargs)
    flat = list(iter_samples(samples))
    metadata = flat[0].metadata or {}
    record = {
        "t": time.time(),
        "policy_version": _policy_version(flat[0]),
        "suite": metadata.get("suite"),
        "task_id": metadata.get("task_id"),
        "trial_id": metadata.get("trial_id"),
        "rewards": [sample.get_reward_value(args) for sample in flat],
        "keep": bool(output.keep),
    }
    # --save is <run dir>/checkpoints in the LIBERO launcher.
    path = os.path.join(os.path.dirname(os.path.normpath(args.save)), "groups.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return output
