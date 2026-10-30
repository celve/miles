import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import FastAPI

from miles.router.config import MilesRouterConfig
from miles.router.router import MilesRouter
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


@asynccontextmanager
async def _router_clients():
    router = MilesRouter(
        MilesRouterConfig(
            host="127.0.0.1",
            port=20000,
            max_connections=100,
            timeout=None,
            health_check_interval=0.01,
            health_check_failure_threshold=3,
        )
    )
    async with router.client, httpx.AsyncClient(
        transport=httpx.ASGITransport(app=router.app), base_url="http://router"
    ) as client:
        yield router, client


async def test_slow_health_recovers_after_all_workers_are_quarantined():
    """Three real read timeouts must not permanently strand an otherwise responsive engine."""
    app = FastAPI()
    health_calls = []
    server = UvicornThreadServer(app, host="127.0.0.1", port=find_available_port(30000))

    @app.get("/health")
    async def health():
        health_calls.append(None)
        if len(health_calls) <= 3:
            await asyncio.sleep(5.1)
        return {"status": "ok"}

    @app.post("/generate")
    async def generate():
        return {"text": "still serving"}

    server.start()
    try:
        async with _router_clients() as (router, client):
            (await client.post("/add_worker", params={"url": server.url})).raise_for_status()
            response = await client.post("/generate", json={})
            assert response.json() == {"text": "still serving"}

            for _ in range(3):
                await router._health_check_once()

            assert len(health_calls) == 3
            assert router.worker_failure_counts[server.url] == 3
            assert router.dead_workers == {server.url}
            response = await client.post("/generate", json={})
            response.raise_for_status()
            assert response.json() == {"text": "still serving"}
            assert router.dead_workers == {server.url}

            await router._health_check_once()

            assert len(health_calls) == 4
            assert router.dead_workers == set()
            assert router.worker_failure_counts[server.url] == 0
            response = await client.post("/generate", json={})
            response.raise_for_status()
            assert response.json() == {"text": "still serving"}
            assert router.worker_request_counts[server.url] == 0
    finally:
        server.stop()


@pytest.mark.parametrize("state", ["busy", "dead", "mixed"], ids=["busy", "dead", "mixed"])
async def test_quarantine_only_blocks_routing_when_another_worker_is_available(state):
    """Slow health probes must not block a responsive pool or displace a worker outside quarantine."""
    attempts = []

    async def upstream_response(request):
        if request.url.path == "/health":
            return httpx.Response(200 if state == "mixed" and request.url.host == "w1" else 503)
        attempts.append(request.url.host)
        if state == "dead" or (state == "mixed" and request.url.host == "w2"):
            raise httpx.ConnectError("engine down", request=request)
        return httpx.Response(200, json={"text": "still serving"})

    async with _router_clients() as (router, client):
        await router.client.aclose()
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream_response)) as upstream:
            router.client = upstream
            for url in ("http://w1", "http://w2"):
                (await client.post("/add_worker", params={"url": url})).raise_for_status()
            for _ in range(3):
                await router._health_check_once()
            router.worker_request_counts = {"http://w1": 3, "http://w2": 1}

            if state == "dead":
                with pytest.raises(httpx.ConnectError, match="engine down"):
                    await client.post("/generate", json={})
            else:
                response = await client.post("/generate", json={})
                response.raise_for_status()
                assert response.json() == {"text": "still serving"}

            assert attempts == (["w1"] if state == "mixed" else ["w2"])
            assert router.worker_request_counts == {"http://w1": 3, "http://w2": 1}
            assert router.dead_workers == ({"http://w2"} if state == "mixed" else {"http://w1", "http://w2"})


async def test_an_empty_pool_cannot_fall_back_to_an_unregistered_worker():
    async with _router_clients() as (router, client):
        with pytest.raises(RuntimeError, match="No workers registered"):
            await client.post("/generate", json={})


@pytest.mark.parametrize("failure", ["timeout", "connection", "status"], ids=["timeout", "connection", "status"])
async def test_recovery_resets_consecutive_failures_and_leaves_dead_workers_excluded(failure, caplog):
    """Recovery must clear the old failure streak without admitting a peer that still fails."""
    recovered = False

    async def health(request):
        if request.url.host == "recovering" and recovered:
            return httpx.Response(200)
        if failure == "timeout":
            raise httpx.ReadTimeout("health delayed", request=request)
        if failure == "connection":
            raise httpx.ConnectError("engine down", request=request)
        return httpx.Response(503)

    async with _router_clients() as (router, client):
        await router.client.aclose()
        async with httpx.AsyncClient(transport=httpx.MockTransport(health)) as upstream:
            router.client = upstream
            for url in ("http://recovering", "http://dead"):
                (await client.post("/add_worker", params={"url": url})).raise_for_status()

            for _ in range(5):
                await router._health_check_once()
            assert router.dead_workers == {"http://recovering", "http://dead"}
            assert caplog.text.count("Marking as DEAD") == 2

            recovered = True
            await router._health_check_once()
            assert router.dead_workers == {"http://dead"}
            assert router.worker_failure_counts == {"http://recovering": 0, "http://dead": 6}
            assert router._use_url() == "http://recovering"
            router._finish_url("http://recovering")

            recovered = False
            for _ in range(2):
                await router._health_check_once()
                assert router.dead_workers == {"http://dead"}
            await router._health_check_once()
            assert router.dead_workers == {"http://recovering", "http://dead"}


@pytest.mark.parametrize("status", [200, 503], ids=["healthy", "unhealthy"])
async def test_removing_a_quarantined_worker_during_a_probe_keeps_it_removed(status):
    """A recovery result must not undo controller deregistration, including after a failed weight update."""
    probing = asyncio.Event()
    finish_probe = asyncio.Event()

    async def health(request):
        probing.set()
        await finish_probe.wait()
        return httpx.Response(status)

    async with _router_clients() as (router, client):
        await router.client.aclose()
        async with httpx.AsyncClient(transport=httpx.MockTransport(health)) as upstream:
            router.client = upstream
            (await client.post("/add_worker", params={"url": "http://removed"})).raise_for_status()
            router.dead_workers.add("http://removed")
            probe = asyncio.create_task(router._health_check_once())
            try:
                await asyncio.wait_for(probing.wait(), timeout=1)
                (await client.post("/remove_worker", params={"url": "http://removed"})).raise_for_status()
            finally:
                finish_probe.set()
                await probe

            assert router.worker_request_counts == {}
            assert router.worker_failure_counts == {}
            assert router.dead_workers == set()
