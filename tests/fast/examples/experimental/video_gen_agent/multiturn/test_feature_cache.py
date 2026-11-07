import os
import time
from types import SimpleNamespace

from examples.experimental.video_gen_agent.multiturn.feature_cache import prune_feature_cache


def _args(cache_dir):
    return SimpleNamespace(sglang=SimpleNamespace(common_value=lambda name: str(cache_dir)))


def _entry(cache_dir, name: str, *, age: float):
    path = cache_dir / name[:2] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))
    return path


def test_entries_from_before_the_rollout_are_removed_and_its_own_and_claims_stay(tmp_path):
    old = _entry(tmp_path, "aa11.pt", age=600)
    new = _entry(tmp_path, "bb22.pt", age=10)
    claim = _entry(tmp_path, "cc33.pt.tmp", age=600)

    assert prune_feature_cache(3, _args(tmp_path), [], {}, rollout_time=60) is False

    assert not old.exists() and new.exists() and claim.exists()
