from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_reward_pool_kwargs: ContextVar[dict[str, Any] | None] = ContextVar("reward_pool_kwargs", default=None)


@contextmanager
def use_reward_pools(pools: Mapping[str, Any]):
    token = _reward_pool_kwargs.set({f"{name}_pool": pool for name, pool in pools.items()})
    try:
        yield
    finally:
        _reward_pool_kwargs.reset(token)


def get_reward_pool_kwargs() -> dict[str, Any]:
    return _reward_pool_kwargs.get() or {}
