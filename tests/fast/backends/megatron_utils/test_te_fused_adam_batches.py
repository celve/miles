from types import SimpleNamespace

from miles.backends.megatron_utils import te_fused_adam_batches
from miles.backends.megatron_utils.te_fused_adam_batches import install_fused_adam_batches_patch


def _fake_fused_adam(limit):
    calls = []

    def applier(op, noop_flag, tensor_lists, *args):
        assert sum(len(ts) for ts in tensor_lists) <= limit, "Maximum number of tensors reached"
        calls.append([list(ts) for ts in tensor_lists])

    return SimpleNamespace(multi_tensor_applier=applier), calls


def test_a_call_over_the_handle_pool_is_split_and_covers_every_tensor(monkeypatch):
    monkeypatch.setattr(te_fused_adam_batches, "MAX_HANDLES", 20)
    module, calls = _fake_fused_adam(limit=20)
    install_fused_adam_batches_patch(module)
    lists = [[f"{name}{i}" for i in range(9)] for name in "gpmvw"]

    module.multi_tensor_applier("adam", None, lists, 1e-3)

    assert len(calls) == 3
    for k in range(5):
        assert [t for call in calls for t in call[k]] == lists[k]


def test_a_call_within_the_pool_goes_through_once(monkeypatch):
    monkeypatch.setattr(te_fused_adam_batches, "MAX_HANDLES", 20)
    module, calls = _fake_fused_adam(limit=20)
    install_fused_adam_batches_patch(module)

    module.multi_tensor_applier("adam", None, [["a", "b"], ["c", "d"]], 1e-3)

    assert calls == [[["a", "b"], ["c", "d"]]]


def test_installing_twice_wraps_once():
    module, _ = _fake_fused_adam(limit=100)
    install_fused_adam_batches_patch(module)
    first = module.multi_tensor_applier
    install_fused_adam_batches_patch(module)

    assert module.multi_tensor_applier is first
