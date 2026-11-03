import functools
import importlib

_PATCHED = "_miles_fa4_causal_window"
_FULL_CAUSAL = {(-1, 0), (-1, -1)}


def install_fa4_causal_window_patch(backends=None) -> None:
    """Pass FlashAttention 4 no window for plain causal attention.

    Transformer Engine hands FA4 `window_size=(-1, 0)` beside `causal=True`, which FA4 reads as local attention, and
    its SM100 head_dim-256 backward (Qwen3.5) refuses local attention, so training found no attention backend.
    """
    if backends is None:
        try:
            backends = importlib.import_module("transformer_engine.pytorch.attention.dot_product_attention.backends")
        except ImportError:
            return
    for name in ("flash_attn_func_v4", "flash_attn_varlen_func_v4"):
        func = getattr(backends, name, None)
        if func is None or getattr(func, _PATCHED, False):
            continue

        @functools.wraps(func)
        def wrapped(*args, _func=func, **kwargs):
            if kwargs.get("causal") and tuple(kwargs.get("window_size") or ()) in _FULL_CAUSAL:
                kwargs["window_size"] = (None, None)
            return _func(*args, **kwargs)

        setattr(wrapped, _PATCHED, True)
        setattr(backends, name, wrapped)
