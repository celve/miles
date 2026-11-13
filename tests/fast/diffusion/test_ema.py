import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from miles.backends.fsdp_utils.diffusion import actor as diffusion_actor
from miles.backends.fsdp_utils.diffusion.ema import EMAOptimizer, load_ema
from miles.backends.training_utils.torch_native import actor as base_actor
from miles.backends.training_utils.torch_native.offload import move_train_state


def _model(value=1.0):
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    return model


def _ema(model, **overrides):
    return EMAOptimizer(model, **(dict(decay=0.0, uprate=0.25, uphold=0.5, flat_steps=1) | overrides))


def test_the_decay_holds_for_the_flat_steps_then_ramps_to_its_cap():
    model = _model(0.0)
    ema = _ema(model)
    decays = []
    for step in range(1, 5):
        before = ema.state[model.weight]["ema"].clone()
        with torch.no_grad():
            model.weight.fill_(float(step))
        ema.step(step)
        after = ema.state[model.weight]["ema"]
        decays.append(round(float((step - after[0, 0]) / (step - before[0, 0])), 6))

    assert decays == [0.0, 0.25, 0.5, 0.5]


def test_use_weights_swaps_the_ema_in_and_restores_the_actor():
    model = _model(4.0)
    ema = _ema(model)
    with torch.no_grad():
        model.weight.fill_(8.0)

    with ema.use_weights(model):
        assert model.weight.eq(4.0).all()
    assert model.weight.eq(8.0).all() and ema.state[model.weight]["ema"].eq(4.0).all()


@pytest.fixture
def single_rank_gloo():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_use_weights_reaches_a_root_owned_parameter_after_an_fsdp_forward(single_rank_gloo):
    model = fully_shard(_model(1.0), mesh=init_device_mesh("cpu", (1,)))
    ema = _ema(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(4.0)
    inputs = torch.ones(1, 2)
    # The root keeps its gathered parameters registered after a forward, until it reshards.
    assert model(inputs).item() == 8.0

    with torch.no_grad(), ema.use_weights(model):
        assert model(inputs).item() == 2.0
    assert model(inputs).item() == 8.0


def test_a_checkpoint_without_an_ema_restarts_it_from_the_loaded_weights(tmp_path):
    model = _model(1.0)
    ema = _ema(model)
    with torch.no_grad():
        model.weight.fill_(3.0)

    load_ema(ema, model, tmp_path)

    assert ema.state[model.weight]["ema"].eq(3.0).all()


def test_sleep_and_wake_move_the_ema_with_the_optimizer_state(monkeypatch):
    model = _model()
    ema = _ema(model)
    moved = []
    monkeypatch.setattr(
        "miles.backends.training_utils.torch_native.offload._to_pinned_host", lambda t: moved.append(t) or t
    )

    move_train_state([], [ema], to_host=True)

    assert moved and moved[0] is ema.state[model.weight]["ema"]


def test_the_engines_are_synced_from_the_ema_and_training_resumes_on_the_actor_weights(monkeypatch):
    model = _model(4.0)
    ema = _ema(model)
    with torch.no_grad():
        model.weight.fill_(8.0)
    synced = []
    monkeypatch.setattr(
        base_actor.TorchNativeTrainRayActor,
        "update_weights",
        lambda self, info, debug_weight_update_id, rollout_id: synced.append(model.weight.clone()) or "out",
    )
    actor = object.__new__(diffusion_actor.DiffusionFSDPTrainRayActor)
    actor.model, actor.ema = model, ema

    assert actor.update_weights(None, "id", 0) == "out"
    assert synced[0].eq(4.0).all() and model.weight.eq(8.0).all()


@pytest.mark.parametrize("uses_ema", [True, False])
def test_the_ema_steps_once_per_rollout_after_its_optimizer_steps(monkeypatch, uses_ema):
    model = _model()
    calls = []
    actor = object.__new__(diffusion_actor.DiffusionFSDPTrainRayActor)
    actor.args = SimpleNamespace(
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        global_batch_size=2,
        diffusion_micro_batch_size_sample=1,
        diffusion_micro_batch_size_tstep=1,
    )
    actor.ema = SimpleNamespace(step=lambda step: calls.append(("ema", step))) if uses_ema else None
    actor.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    actor.prof = SimpleNamespace(step=lambda rollout_id: None)
    actor.global_step = 0

    def optimizer_step(rollout_id, micro_batches, *, first_microbatch_id):
        actor.global_step += 1
        calls.append(("step", first_microbatch_id, len(micro_batches)))

    actor._optimizer_step = optimizer_step
    actor._train_core(5, [[{}], [{}], [{}], [{}]])

    assert calls == [("step", 0, 2), ("step", 2, 2)] + ([("ema", 2)] if uses_ema else [])


def _checkpointing_actor(tmp_path, *, no_save_optim=False, no_load_optim=False, value=4.0):
    actor = object.__new__(diffusion_actor.DiffusionFSDPTrainRayActor)
    actor.model = _model(value)
    actor.ema = _ema(actor.model)
    backend = SimpleNamespace(
        save=str(tmp_path), load=str(tmp_path), no_save_optim=no_save_optim, no_load_optim=no_load_optim
    )
    actor.args = SimpleNamespace(backend=backend)
    return actor


def test_the_ema_is_on_disk_before_the_checkpoint_writes_its_tracker(tmp_path, monkeypatch, single_rank_gloo):
    actor = _checkpointing_actor(tmp_path)
    seen = []
    monkeypatch.setattr(
        diffusion_actor.checkpoint,
        "save",
        lambda a, iteration: seen.append((tmp_path / "iter_0000004" / "ema").exists()),
    )

    actor._save_checkpoint(3)

    assert seen == [True]


def test_no_save_optim_leaves_the_ema_out_of_the_checkpoint(tmp_path, monkeypatch, single_rank_gloo):
    actor = _checkpointing_actor(tmp_path, no_save_optim=True)
    saved = []
    monkeypatch.setattr(diffusion_actor.checkpoint, "save", lambda a, iteration: saved.append(iteration))

    actor._save_checkpoint(3)

    assert saved == [3] and not (tmp_path / "iter_0000004" / "ema").exists()


@pytest.mark.parametrize("no_load_optim,expected", [(False, 4.0), (True, 9.0)])
def test_a_resume_loads_the_saved_ema_unless_no_load_optim(
    tmp_path, monkeypatch, single_rank_gloo, no_load_optim, expected
):
    monkeypatch.setattr(diffusion_actor.checkpoint, "save", lambda a, iteration: None)
    _checkpointing_actor(tmp_path, value=4.0)._save_checkpoint(3)
    resumed = _checkpointing_actor(tmp_path, no_load_optim=no_load_optim, value=1.0)
    with torch.no_grad():
        resumed.model.weight.fill_(9.0)

    resumed._load_ema(4)

    assert resumed.ema.state[resumed.model.weight]["ema"].eq(expected).all()
