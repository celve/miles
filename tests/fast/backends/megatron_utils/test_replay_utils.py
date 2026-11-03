from types import SimpleNamespace

import pytest
import torch
from torch import nn

from miles.backends.megatron_utils import replay_utils
from miles.utils.replay_base import BaseReplayManager, Replay


class _Router(nn.Module):
    def __init__(self, manager):
        super().__init__()
        manager.register_to_module(self, "routing_replay")


class _Model(nn.Module):
    """A model whose constructor builds one decoder, then replaces it, as Megatron-Bridge's Qwen3-VL text model does."""

    def __init__(self, manager, num_layers):
        super().__init__()
        self.decoder = nn.ModuleList(_Router(manager) for _ in range(num_layers))
        self.decoder = nn.ModuleList(_Router(manager) for _ in range(num_layers))
        self.config = SimpleNamespace(num_layers=num_layers, moe_layer_freq=1)


class _Wrapped(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model


@pytest.fixture
def manager():
    m = BaseReplayManager()
    m.enabled = True
    return m


@pytest.fixture(autouse=True)
def _local_layout(monkeypatch):
    # Replay.record pins host memory, which needs CUDA.
    monkeypatch.setattr(Replay, "record", lambda self, t: self.top_indices_list.append(t.clone()))
    monkeypatch.setattr(replay_utils, "get_num_layers_to_build", lambda config, vp_stage=None: config.num_layers)
    monkeypatch.setattr(replay_utils, "get_transformer_layer_offset", lambda config, vp_stage=None: 0)


def test_records_into_the_live_routers_when_a_decoder_was_replaced(manager):
    model = _Wrapped(_Model(manager, num_layers=3))
    replay_data = torch.arange(5 * 3 * 2, dtype=torch.int32).reshape(5, 3, 2)

    replay_utils.register_replay_list_moe(manager.replays, replay_data, models=[model])

    assert len(manager.replays) == 6
    for layer, router in enumerate(model.module.decoder):
        assert torch.equal(router.routing_replay.top_indices_list[0], replay_data[:, layer])


def test_refuses_a_layout_that_does_not_match_the_routers(manager):
    model = _Wrapped(_Model(manager, num_layers=3))
    model.module.config.num_layers = 4

    with pytest.raises(ValueError, match="3 routers hold a routing replay"):
        replay_utils.register_replay_list_moe(manager.replays, torch.zeros(5, 4, 2, dtype=torch.int32), models=[model])
