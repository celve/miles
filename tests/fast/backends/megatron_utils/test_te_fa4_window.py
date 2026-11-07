from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils.te_fa4_window import install_fa4_causal_window_patch


def _fake_backends():
    calls = []

    def fa4(*args, **kwargs):
        calls.append(kwargs)

    return SimpleNamespace(flash_attn_func_v4=fa4, flash_attn_varlen_func_v4=fa4), calls


@pytest.mark.parametrize("window", [(-1, 0), (-1, -1)])
def test_full_causal_window_reaches_fa4_as_no_window(window):
    backends, calls = _fake_backends()
    install_fa4_causal_window_patch(backends)

    backends.flash_attn_varlen_func_v4("q", "k", "v", causal=True, window_size=window)

    assert calls[-1]["window_size"] == (None, None)


@pytest.mark.parametrize(("causal", "window"), [(True, (128, 0)), (False, (-1, -1)), (False, (64, 64))])
def test_other_windows_are_left_alone(causal, window):
    backends, calls = _fake_backends()
    install_fa4_causal_window_patch(backends)

    backends.flash_attn_func_v4("q", "k", "v", causal=causal, window_size=window)

    assert calls[-1]["window_size"] == window


def test_installing_twice_wraps_once():
    backends, _ = _fake_backends()
    install_fa4_causal_window_patch(backends)
    first = backends.flash_attn_varlen_func_v4
    install_fa4_causal_window_patch(backends)

    assert backends.flash_attn_varlen_func_v4 is first
