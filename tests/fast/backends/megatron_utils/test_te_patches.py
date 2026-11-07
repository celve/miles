from argparse import Namespace

import pytest

from miles.backends.megatron_utils import model as model_module


class _Built(Exception):
    pass


@pytest.mark.parametrize("multi_lora", [True, False])
def test_bridge_lora_paths_get_the_te_patches(monkeypatch, multi_lora):
    calls = []
    monkeypatch.setattr(model_module, "install_te_training_patches", lambda: calls.append("installed"))
    monkeypatch.setattr(model_module, "is_multi_lora_enabled", lambda args: multi_lora)
    monkeypatch.setattr(model_module, "is_lora_enabled", lambda args: True)

    def build(args):
        calls.append("built")
        raise _Built

    monkeypatch.setattr(model_module, "_setup_lora_model_via_bridge", build)
    args = Namespace(moe_use_upcycling=False, load="ckpt", pretrained_checkpoint=None, megatron_to_hf_mode="bridge")

    with pytest.raises(_Built):
        model_module.setup_model_and_optimizer(args, role="actor")

    assert calls == ["installed", "built"]
