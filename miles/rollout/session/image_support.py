"""The session server's image capability, constructed once per process.

``build_image_support`` returns ``None`` unless ``--session-image-support`` names
a family, so nothing here is loaded — and no processor is read off disk — for a
text-only run. The returned object is read-only and shared by every session; the
per-session half is ``SessionMedia``, which owns that session's interned bytes
and dies with it.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import orjson

from miles.rollout.session.config import SESSION_IMAGE_SUPPORT_OFF, SessionServerConfig
from miles.rollout.session.errors import EngineNotReadyError, EngineUnsupportedError, SessionError
from miles.rollout.session.generate_transport import CompletionParsers
from miles.rollout.session.image_processing import (
    ImageTokenCounter,
    ProcessedImage,
    ProcessorFingerprint,
    build_fingerprint,
    process_images,
)
from miles.rollout.session.media import EMPTY_MEDIA, MediaArena, RequestMedia, extract_request_media

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineImageConfig:
    """How the engine says it will preprocess images, read from the engine itself.

    Matching token ids prove the two sides agree on the image *grid*, because the
    expanded length is derived from it — they do not prove they agree on the pixel
    values behind those tokens. A decode or normalization override changes the
    pixels while leaving the grid, and therefore every token id, identical. So the
    engine's own configuration decides whether this run may serve images, and a
    reply that does not state it is refused rather than defaulted.
    """

    model_path: str
    revision: str | None
    mm_process_config: dict[str, Any]
    image_processor_backend: str
    sampling_defaults: str
    tool_call_parser: str | None
    reasoning_parser: str | None

    # The fields a reply must actually carry. Absent is refused, never defaulted: a
    # router-manager reply, an error body and `{}` all used to read as "no override".
    REQUIRED = ("model_path", "mm_process_config", "image_processor_backend", "sampling_defaults")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "revision": self.revision,
            "mm_process_config": self.mm_process_config,
            "image_processor_backend": self.image_processor_backend,
            "sampling_defaults": self.sampling_defaults,
            "tool_call_parser": self.tool_call_parser,
            "reasoning_parser": self.reasoning_parser,
        }

    def to_dict_json(self) -> str:
        """A comparison key: every engine behind the router must agree on all of it."""
        return orjson.dumps(self.to_dict(), option=orjson.OPT_SORT_KEYS).decode()

    def model_sampling_defaults(self, generation_defaults: dict[str, Any]) -> dict[str, Any]:
        """The defaults the engine applies to a field the request omits.

        Mirrors `ModelConfig.get_default_sampling_params`, which returns the
        checkpoint's generation config only when `sampling_defaults == "model"`.
        Without this an omitted field would take SGLang's own fallback on the image
        route and the checkpoint's on the chat route, so one session's two routes
        would sample differently.
        """
        return dict(generation_defaults) if self.sampling_defaults == "model" else {}


class EngineBinding:
    """Resolves the one engine that will serve this run's image turns, and checks it.

    Not at construction: Miles waits for this server's listener *before* it launches
    any engine (`placement_group` waits for the session server, then initializes the
    inference controller), so a constructor-time lookup has nothing to ask.

    And not the backend as a whole: the backend is the router
    (`specs/inference.py` passes the router's address), which picks a worker per
    request. Four review rounds showed that no amount of checking *through* the
    router establishes which worker serves a turn — the session server neither
    chooses nor observes that choice, so membership polling only narrows a window.
    So the router is taken out of the image path instead: one engine is resolved and
    validated, and image generations go straight to it. The validated worker is then
    the serving worker by construction.

    A readiness or transport failure is retried on the next image request; a
    topology or configuration this path cannot serve is cached and refused for good.
    """

    def __init__(self, checkpoint: str, backend_url: str, engine_url: str | None = None) -> None:
        self.checkpoint = checkpoint
        self.backend_url = backend_url.rstrip("/")
        self.configured_engine_url = engine_url.rstrip("/") if engine_url else None
        self._lock = asyncio.Lock()
        self._resolved: ResolvedEngine | None = None
        self._rejected: SessionError | None = None

    @property
    def resolved(self) -> "ResolvedEngine | None":
        return self._resolved

    async def get(self, backend: Any) -> "ResolvedEngine":
        """The engine image turns go to, validated. Raises until one can be."""
        if self._resolved is not None:
            return self._resolved
        if self._rejected is not None:
            raise self._rejected
        async with self._lock:
            if self._resolved is not None:
                return self._resolved
            if self._rejected is not None:
                raise self._rejected
            try:
                url, payload = await self._resolve_engine_url(backend)
                if payload is None:
                    status, body = await backend.do_get(f"{url}/get_server_info")
                    payload = _server_info_payload(url, status, body)
                config = parse_engine_image_config(payload, where=url)
                assert_engine_preprocessing_supported(config, checkpoint=self.checkpoint, where=url)
            except SessionError as exc:
                if isinstance(exc, EngineUnsupportedError):
                    self._rejected = exc
                raise
            logger.info("[session] Image turns pinned to engine %s: %s", url, config.to_dict())
            self._resolved = ResolvedEngine(url=url, config=config)
            return self._resolved

    async def _resolve_engine_url(self, backend: Any) -> tuple[str, dict[str, Any] | None]:
        """Which single engine serves image turns, plus its server info when already read.

        Fail-closed on anything ambiguous.
        """
        if self.configured_engine_url:
            return self.configured_engine_url, None
        workers = await self._enumerate_workers(backend)
        if workers is None:
            return await self._backend_as_direct_engine(backend)
        if not workers:
            raise EngineNotReadyError(
                f"the router at {self.backend_url} reports no registered engines yet, "
                "so no image turn can be served"
            )
        if len(workers) > 1:
            raise EngineUnsupportedError(
                f"the router at {self.backend_url} fronts {len(workers)} engines ({sorted(workers)}) and "
                "picks one per request, so the engine whose image preprocessing was checked need not be "
                "the one that serves a turn. Name the engine image turns must use with "
                "--session-image-engine-url. Image turns then bypass the router; text turns are unaffected."
            )
        return workers[0], None

    async def _enumerate_workers(self, backend: Any) -> list[str] | None:
        """The backend's workers, or ``None`` when it exposes no worker list at all.

        Both router APIs are recognized: `/list_workers` on MilesRouter and
        `/workers` on sglang-router >= 0.3.0, which `sglang_router_api_client` itself
        uses. Reading a 404 from only one of them as "direct engine" is what let a
        modern gateway through in round 4.
        """
        for path, extract in (("list_workers", _urls_from_list_workers), ("workers", _urls_from_workers)):
            status, body = await backend.do_get(f"{self.backend_url}/{path}")
            if status == 0:
                raise EngineNotReadyError(
                    f"the backend at {self.backend_url} could not be reached: "
                    f"{body.decode(errors='replace')[:200]}"
                )
            if status == 404:
                continue
            if status != 200:
                raise EngineNotReadyError(
                    f"the backend's /{path} answered {status}, so the engines behind it are unknown"
                )
            try:
                payload = orjson.loads(body)
            except orjson.JSONDecodeError as exc:
                raise EngineNotReadyError(f"the backend's /{path} is not JSON: {exc}") from exc
            urls = extract(payload)
            if urls is not None:
                return urls
            raise EngineUnsupportedError(
                f"the backend's /{path} answered a shape this path does not recognize "
                f"({sorted(payload)[:12] if isinstance(payload, dict) else type(payload).__name__}), so the "
                "engines behind it cannot be established. Name the engine image turns must use with "
                "--session-image-engine-url."
            )
        return None

    async def _backend_as_direct_engine(self, backend: Any) -> tuple[str, dict[str, Any]]:
        """Require the backend to identify itself as the engine, rather than assuming it.

        A direct engine advertises its own ``host``/``port``, which match the address
        dialed. A gateway proxying a worker's info reports that worker's port instead,
        so it fails this and is refused — the hole that "404 means direct" left open.
        """
        status, body = await backend.do_get(f"{self.backend_url}/get_server_info")
        payload = _server_info_payload(self.backend_url, status, body)
        dialed = _url_authority(self.backend_url)
        advertises = _advertises_dialed_address(payload, dialed)
        if advertises is None:
            raise EngineUnsupportedError(
                f"{self.backend_url} exposes no worker list and states no host/port of its own, so it "
                "cannot be shown to be the engine that serves an image turn rather than a proxy in front "
                "of one. Name the engine with --session-image-engine-url."
            )
        if not advertises:
            raise EngineUnsupportedError(
                f"{self.backend_url} was dialed at {dialed} but advertises "
                f"{payload.get('host')}:{payload.get('port')}, so it is proxying an engine rather than "
                "being one, and the engine that serves an image turn need not be the one checked. "
                "Name the engine with --session-image-engine-url."
            )
        return self.backend_url, payload


@dataclass(frozen=True)
class ResolvedEngine:
    """The engine image turns are sent to, and what it said about itself."""

    url: str
    config: EngineImageConfig


def _urls_from_list_workers(payload: Any) -> list[str] | None:
    """MilesRouter: ``{"urls": [...]}``."""
    if not isinstance(payload, dict):
        return None
    urls = payload.get("urls")
    if isinstance(urls, list) and all(isinstance(url, str) and url for url in urls):
        return [url.rstrip("/") for url in urls]
    return None


def _urls_from_workers(payload: Any) -> list[str] | None:
    """sglang-router >= 0.3.0: ``{"workers": [{"url": ...}, ...]}``."""
    if not isinstance(payload, dict):
        return None
    workers = payload.get("workers")
    if not isinstance(workers, list):
        return None
    urls = []
    for worker in workers:
        url = worker.get("url") if isinstance(worker, dict) else worker
        if not isinstance(url, str) or not url:
            return None
        urls.append(url.rstrip("/"))
    return urls


def _url_authority(url: str) -> str:
    from urllib.parse import urlsplit

    parts = urlsplit(url if "//" in url else f"//{url}")
    return f"{parts.hostname or ''}:{parts.port or ''}"


def _advertises_dialed_address(payload: dict[str, Any], dialed: str) -> bool | None:
    """Whether the engine says it listens on the address dialed; ``None`` when it says nothing.

    The port is what discriminates: a gateway proxying a worker's server info reports
    that worker's port, not the port the request arrived on. A wildcard bind answers
    for whatever address reached it, so only the port is compared then.
    """
    port = payload.get("port")
    if port is None:
        return None
    host = payload.get("host")
    dialed_host, _, dialed_port = dialed.partition(":")
    if str(port) != dialed_port:
        return False
    if host in (None, "", "0.0.0.0", "::", "*"):
        return True
    loopback = {"127.0.0.1", "localhost", "::1"}
    return host == dialed_host or (host in loopback and dialed_host in loopback)


def _server_info_payload(url: str, status: int, body: bytes) -> dict[str, Any]:
    """The engine's own server-info object, or refuse."""
    if status == 0:
        raise EngineNotReadyError(
            f"engine {url} could not be reached for its image configuration: "
            f"{body.decode(errors='replace')[:200]}"
        )
    if status != 200:
        raise EngineNotReadyError(
            f"engine {url} answered {status} for /get_server_info, so its image configuration is unknown"
        )
    try:
        payload = orjson.loads(body)
    except orjson.JSONDecodeError as exc:
        raise EngineNotReadyError(f"engine {url} did not answer /get_server_info with JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EngineNotReadyError(
            f"engine {url} answered /get_server_info with {type(payload).__name__}, not an object"
        )
    return payload


def parse_engine_image_config(payload: dict[str, Any], *, where: str = "the engine") -> EngineImageConfig:
    """Read the fields that decide pixel agreement, refusing a reply that omits any."""
    missing = [key for key in EngineImageConfig.REQUIRED if key not in payload]
    if missing:
        raise EngineNotReadyError(
            f"{where} does not state {', '.join(missing)} in /get_server_info, so it cannot be checked. "
            f"It answered with keys {sorted(payload)[:12]}. A router or gateway that aggregates workers "
            "answers for none of them, and a router still electing one can start to."
        )
    return EngineImageConfig(
        model_path=str(payload["model_path"] or ""),
        revision=payload.get("revision"),
        mm_process_config=payload["mm_process_config"] or {},
        image_processor_backend=str(payload["image_processor_backend"] or ""),
        sampling_defaults=str(payload["sampling_defaults"] or ""),
        tool_call_parser=payload.get("tool_call_parser"),
        reasoning_parser=payload.get("reasoning_parser"),
    )


def assert_engine_preprocessing_supported(
    engine: EngineImageConfig, *, checkpoint: str, where: str = "the engine"
) -> None:
    """Require the one engine image configuration whose pixels this slice has verified."""
    if not engine.model_path:
        raise EngineUnsupportedError(
            f"{where} states an empty model_path, so the checkpoint it serves cannot be established"
        )
    if engine.model_path != checkpoint:
        raise EngineUnsupportedError(
            f"{where} serves {engine.model_path!r} but --hf-checkpoint is {checkpoint!r}; "
            "the two processors would disagree about every image"
        )
    if engine.revision is not None:
        raise EngineUnsupportedError(
            f"{where} pins revision {engine.revision!r} while this server loads the processor unpinned, "
            "so the two could resize or normalize differently with every token id still matching; "
            "serve the checkpoint's default revision"
        )
    if engine.mm_process_config:
        raise EngineUnsupportedError(
            f"{where} was launched with --mm-process-config {engine.mm_process_config!r}, which changes "
            "how it decodes or normalizes images; its pixels would then differ from the trainer's while "
            "every token id stayed identical. Launch it with the checkpoint's default image preprocessing."
        )
    if engine.image_processor_backend != "auto":
        raise EngineUnsupportedError(
            f"{where} has --image-processor-backend {engine.image_processor_backend!r}; this path has "
            "only verified pixel agreement for the default 'auto' backend"
        )


@dataclass(frozen=True)
class SessionImageSupport:
    """What every image-bearing turn in this process shares.

    The processor and its fingerprint are known at construction. The engine's
    configuration is not: it is resolved on first use through ``engine``, because
    no engine exists when this server is built.
    """

    processor: Any
    fingerprint: ProcessorFingerprint
    tokenizer: Any
    text_only_leaf: str  # "drop" | "keep"
    binding: EngineBinding
    generation_defaults: dict[str, Any]

    @property
    def placeholder_id(self) -> int:
        return self.fingerprint.image_token_id

    def new_session_media(self) -> "SessionMedia":
        return SessionMedia(counter=ImageTokenCounter(self.processor, self.fingerprint))

    async def engine(self, backend: Any) -> "ResolvedEngine":
        return await self.binding.get(backend)

    def parsers(self, engine: EngineImageConfig) -> CompletionParsers:
        """The parsers the engine applies, so both routes in one session agree."""
        return CompletionParsers(
            reasoning_parser=engine.reasoning_parser,
            tool_call_parser=engine.tool_call_parser,
            tokenizer=self.tokenizer,
        )

    def sampling_defaults(self, engine: EngineImageConfig) -> dict[str, Any]:
        return engine.model_sampling_defaults(self.generation_defaults)

    def provenance(self) -> dict[str, Any]:
        """What bound this run's pixels, recorded with the session and every sample.

        The engine half is present once it has been resolved, which is on the first
        image request; a text-only session never resolves it and records none.
        """
        resolved = self.binding.resolved
        return {
            "processor": self.fingerprint.to_dict(),
            "engine": resolved.config.to_dict() if resolved is not None else None,
            "engine_url": resolved.url if resolved is not None else None,
        }

    def process(self, asset_ids: list[str], counts: list[int], media: "SessionMedia") -> list[ProcessedImage]:
        assets = [media.arena.get(asset_id) for asset_id in asset_ids]
        return process_images(self.processor, assets, self.fingerprint, counts)


@dataclass
class SessionMedia:
    """One session's immutable image bytes and their expanded token counts."""

    counter: ImageTokenCounter
    arena: MediaArena = field(default_factory=MediaArena)

    def ingest(self, messages: list[dict[str, Any]]) -> tuple[RequestMedia, list[int]]:
        """Snapshot a request's images and say how many tokens each expands into."""
        media = extract_request_media(messages, self.arena)
        if not media.occurrences:
            return EMPTY_MEDIA, []
        return media, self.counts_for(media)

    def counts_for(self, media: RequestMedia) -> list[int]:
        return self.counter.count_all([self.arena.get(asset_id) for asset_id in media.asset_ids])


# The fields SGLang's ModelConfig.get_default_sampling_params reads from a checkpoint.
_GENERATION_DEFAULT_KEYS = ("repetition_penalty", "temperature", "top_k", "top_p", "min_p")


def build_image_support(config: SessionServerConfig, tokenizer: Any) -> SessionImageSupport | None:
    """Load the processor for an image-enabled run, or ``None`` for text.

    Nothing here touches the engine: it does not exist yet when this runs.
    """
    if config.session_image_support == SESSION_IMAGE_SUPPORT_OFF:
        return None

    from miles.utils.processing_utils import load_processor

    processor = load_processor(config.hf_checkpoint, trust_remote_code=True)
    if processor is None:
        raise ValueError(
            f"--session-image-support={config.session_image_support} needs an image processor, but "
            f"{config.hf_checkpoint!r} provides none"
        )
    if not hasattr(processor, "_get_num_multimodal_tokens"):
        raise ValueError(
            f"{config.hf_checkpoint!r}'s processor has no _get_num_multimodal_tokens, so per-image token "
            "counts cannot be derived without processing every image on every turn"
        )
    fingerprint = build_fingerprint(processor, tokenizer, checkpoint=config.hf_checkpoint)
    logger.info(
        "[session] Image support=%s fingerprint=%s text_only_leaf=%s; the engine's own image "
        "configuration is checked on the first image request",
        config.session_image_support,
        fingerprint.to_dict(),
        config.session_image_text_only_leaf,
    )
    return SessionImageSupport(
        processor=processor,
        fingerprint=fingerprint,
        tokenizer=tokenizer,
        text_only_leaf=config.session_image_text_only_leaf,
        binding=EngineBinding(config.hf_checkpoint, config.backend_url, config.session_image_engine_url),
        generation_defaults=read_generation_defaults(config.hf_checkpoint),
    )


def read_generation_defaults(checkpoint: str) -> dict[str, Any]:
    """The checkpoint's own sampling defaults, for a field the request omits."""
    from transformers import GenerationConfig

    try:
        config = GenerationConfig.from_pretrained(checkpoint).to_dict()
    except Exception as exc:
        logger.warning("[session] No generation config for %s (%s); omitted sampling fields "
                       "will take the engine's own fallbacks", checkpoint, exc)
        return {}
    return {key: config[key] for key in _GENERATION_DEFAULT_KEYS if config.get(key) is not None}
