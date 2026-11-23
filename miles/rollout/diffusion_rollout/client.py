"""HTTP client for sglang.multimodal_gen's msgpack rollout endpoint."""

import asyncio
import logging
from argparse import Namespace
from typing import Any

import httpx

from miles.rollout.diffusion_rollout.dispatch import MicrogroupDispatcher, Reward
from miles.rollout.diffusion_rollout.response import parse_rollout_response
from miles.utils.diffusion.types import DiffusionRollout
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_MAX_RETRIES = 60


class ImageClient:
    def __init__(self, args: Namespace, reward_kwargs: dict[str, Any]) -> None:
        self.url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/rollout/generate"
        self._http = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=0), timeout=httpx.Timeout(None))
        self.dispatcher = MicrogroupDispatcher(args, self._request, reward_kwargs)

    async def generate(self, sample: Sample, sampling_params: dict[str, Any]) -> tuple[DiffusionRollout, Reward]:
        return await self.dispatcher.generate(sample, sampling_params)

    async def _request(self, body: dict[str, Any]) -> list[DiffusionRollout]:
        raw = await self._post(body)
        return await asyncio.to_thread(parse_rollout_response, raw)

    async def _post(self, payload: dict[str, Any]) -> bytes:
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._http.post(self.url, json=payload)
                response.raise_for_status()
                return response.content
            except Exception:
                if attempt + 1 >= _MAX_RETRIES:
                    raise
                logger.warning(f"/rollout/generate attempt {attempt} failed, retrying", exc_info=True)
                await asyncio.sleep(1)
        raise AssertionError("unreachable")

    async def dispose(self) -> None:
        await self._http.aclose()
