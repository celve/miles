"""Decoded media for the trainer-side rebuild: parallel decode, prefetch and a byte-bounded cache keyed by path."""

import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any


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

    def __init__(self, decode: Callable[[str], Any], *, workers: int, cache_bytes: int) -> None:
        self._decode = decode
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="miles-media-decode")
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
