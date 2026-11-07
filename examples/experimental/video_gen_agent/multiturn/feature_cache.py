"""``--custom-rollout-log-function-path examples.experimental.video_gen_agent.multiturn.feature_cache.prune_feature_cache``:
keeps SGLang's disk item feature cache to the clips of the rollout being trained on.

The cache never evicts, and a run's clips are all new, so it would grow by a step's clips every step. Training is
synchronous, so when rollout k has been generated every entry written before it started belongs to a step already
trained; removing it is safe, since a removed entry reads back as a miss and is computed again.
"""

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def prune_feature_cache(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    cache_dir = Path(args.sglang.common_value("token_space_preprocess_cache_dir"))
    started = time.time() - rollout_time
    removed = 0
    # Only finished entries: a ``.tmp`` is a writer's claim, which no other process may remove.
    for entry in cache_dir.glob("*/*.pt"):
        try:
            if entry.stat().st_mtime < started:
                entry.unlink()
                removed += 1
        except FileNotFoundError:
            pass
    logger.info(f"feature cache {rollout_id}: removed {removed} entries from before this rollout")
    return False
