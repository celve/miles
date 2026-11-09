"""A frozen diffusion cell merges its fixed LoRAs into the engine before the router can send it a request."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI, Request, Response
from tests.fast.ray.rollout.conftest import make_args, track_server_cell

from miles.backends.sglang_utils.sglang_config import DiffusionLora
from miles.ray.rollout import server_cell as server_cell_module
from miles.ray.rollout.cell_state import CellAddrInfo
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

pytestmark = pytest.mark.usefixtures("dispose_tracked_server_cells")

_LORAS = (
    DiffusionLora(path="/loras/high.safetensors", target="transformer"),
    DiffusionLora(path="/loras/low.safetensors", target="transformer_2"),
)


class _Router:
    def __init__(self, log: list):
        self.log = log

    async def add_worker(self, **kwargs):
        self.log.append(("add_worker", kwargs["worker_url"]))

    async def remove_worker(self, **kwargs):
        self.log.append(("remove_worker", kwargs["worker_url"]))


@pytest.fixture
def engine(monkeypatch):
    """A multimodal_gen engine's ``/v1/set_lora``, answering with ``status["code"]``, and the order of events."""
    log, status = [], {"code": 200}
    app = FastAPI()

    @app.post("/v1/set_lora")
    async def set_lora(request: Request):
        log.append(("set_lora", (await request.json())["lora_path"]))
        return Response(status_code=status["code"], content=b"{}")

    server = UvicornThreadServer(app, host="127.0.0.1", port=find_available_port(34000))
    server.start()

    async def _addr_info(self) -> CellAddrInfo:
        return CellAddrInfo(server_url=server.url, bootstrap_port=None, gate_url=None)

    async def _healthy(server_url, api_key, timeout=5.0) -> bool:
        return True

    monkeypatch.setattr(ServerCell, "_compute_addr_info", _addr_info)
    monkeypatch.setattr(server_cell_module, "probe_server_healthy", _healthy)
    yield server.url, log, status
    server.stop()


def _frozen_cell(log: list, loras=_LORAS) -> ServerCell:
    meta = ServerCellMetadata(
        model_id="video",
        worker_type="regular",
        cell_id="inference-engine-2-0-0",
        num_gpus_per_engine=1,
        gpu_offset=2,
        sglang_api_key=None,
        worker_name="inference-engine-2-0-0-0",
        needs_offload=False,
        update_weights=False,
        workers_hash="hash",
        engine_kind="diffusion",
        loras=[lora.model_dump() for lora in loras],
    )
    return track_server_cell(ServerCell(args=make_args(), meta=meta, router_api_client=_Router(log), provider=None))


async def test_the_loras_are_merged_before_the_engine_joins_the_router(engine):
    url, log, _ = engine
    cell = _frozen_cell(log)
    await cell.init()

    await cell.tick()

    assert log == [("set_lora", ["/loras/high.safetensors", "/loras/low.safetensors"]), ("add_worker", url)]
    assert cell.is_serving


async def test_an_engine_that_refuses_its_loras_never_serves(engine):
    _, log, status = engine
    status["code"] = 500
    cell = _frozen_cell(log)
    await cell.init()

    with pytest.raises(httpx.HTTPStatusError):
        await cell.tick()

    assert [event for event, _ in log] == ["set_lora"] and cell.is_initializing


async def test_a_cell_without_loras_sends_no_set_lora(engine):
    url, log, _ = engine
    cell = _frozen_cell(log, loras=())
    await cell.init()

    await cell.tick()

    assert log == [("add_worker", url)]
