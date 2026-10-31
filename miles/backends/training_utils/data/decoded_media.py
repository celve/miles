"""Media caches for the trainer-side rebuild: parallel decode, prefetch, and byte-bounded caches keyed by path."""

import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch


def decoded_nbytes(image: Any) -> int:
    if hasattr(image, "nbytes"):
        return int(image.nbytes)
    width, height = image.size
    return width * height * len(image.getbands())


class DecodedImages:
    """Decodes image paths on a thread pool and keeps finished decodes until their bytes exceed ``cache_bytes``.

    ``request`` starts decodes without waiting; ``get`` waits for one. ``trim`` evicts the least recently requested
    finished decodes, never one in ``keep``, so in-flight and prefetched micro-batches survive a zero-byte cache.
    """

    def __init__(
        self,
        decode: Callable[[str], Any],
        *,
        workers: int,
        cache_bytes: int,
        thread_initializer: Callable[[], None] | None = None,
    ) -> None:
        self._decode = decode
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="miles-media-decode", initializer=thread_initializer
        )
        # A round's rebuild owns one pool; stop its threads when the round's data is dropped.
        weakref.finalize(self, self._pool.shutdown, wait=False, cancel_futures=True)
        self._cache_bytes = cache_bytes
        self._entries: OrderedDict[str, Future] = OrderedDict()
        self._lock = threading.Lock()
        self.decodes = 0

    def request(self, paths: Iterable[str]) -> None:
        with self._lock:
            for path in paths:
                if path in self._entries:
                    self._entries.move_to_end(path)
                else:
                    self._entries[path] = self._pool.submit(self._decode, path)
                    self.decodes += 1

    def get(self, path: str) -> Any:
        self.request([path])
        with self._lock:
            future = self._entries[path]
        return future.result()

    def trim(self, keep: set[str]) -> None:
        with self._lock:
            held = sum(decoded_nbytes(f.result()) for f in self._entries.values() if f.done() and not f.exception())
            for path in list(self._entries):
                if held <= self._cache_bytes:
                    break
                future = self._entries[path]
                if path in keep or not future.done():
                    continue
                del self._entries[path]
                if not future.exception():
                    held -= decoded_nbytes(future.result())


class ImageFeatureCache:
    """Each image's processor rows and grid in pinned host memory, keyed by path, least recently used evicted first.

    Not thread-safe: one rebuild thread owns it.
    """

    def __init__(self, cache_bytes: int) -> None:
        self._cache_bytes = cache_bytes
        self._entries: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._held = 0

    def get(self, paths: list[str]) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        """Every path's (rows, grid), or None unless all are cached."""
        if not all(path in self._entries for path in paths):
            return None
        for path in paths:
            self._entries.move_to_end(path)
        return [self._entries[path] for path in paths]

    def put(self, path: str, rows: torch.Tensor, grid: torch.Tensor) -> None:
        if path in self._entries:
            return
        host = torch.empty(rows.shape, dtype=rows.dtype, pin_memory=rows.is_cuda)
        host.copy_(rows, non_blocking=True)
        self._entries[path] = (host, grid.cpu())
        self._held += host.nbytes
        while self._held > self._cache_bytes and self._entries:
            _, (evicted, _) = self._entries.popitem(last=False)
            self._held -= evicted.nbytes
