"""The env's view of the frozen generators: one call that turns a request into a clip file, or into an image file.

The real clients wrap a diffusion engine group behind the Miles router; ``FakeClipGenerator`` and
``FakeImageGenerator`` stand in for them in tests and CPU smoke runs.
"""

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo


@dataclass(frozen=True)
class ClipRequest:
    prompt: str
    negative_prompt: str
    seed: int
    seconds: float
    width: int
    height: int
    fps: int
    # A frame to continue from, for image-to-video; None generates from text alone.
    image_path: str | None = None
    # Images the clip shows as subjects and look rather than starts from, for reference-to-video.
    reference_paths: tuple[str, ...] = ()

    @property
    def num_frames(self) -> int:
        return max(1, round(self.seconds * self.fps))


@dataclass(frozen=True)
class Clip:
    path: str


class GeneratorUnavailable(Exception):
    """The generator could not serve a request for reasons outside the policy's control: down, overloaded, timed out."""


class ClipGenerator(Protocol):
    async def generate(self, request: ClipRequest) -> Clip:
        """Return the generated clip, or raise ``GeneratorUnavailable``."""
        ...


class FakeClipGenerator:
    """Writes a small file per request, named by the request's hash, as the generator writes clips by content hash.

    The file holds the request as JSON, so a test can read back what was asked for. ``AGENT_VIDEO_GEN_FAKE_DIR``
    names the directory (default: a temp dir); ``AGENT_VIDEO_GEN_FAKE_UNAVAILABLE=1`` makes every call fail as an
    outage would.
    """

    def __init__(self):
        self.clip_dir = Path(os.environ.get("AGENT_VIDEO_GEN_FAKE_DIR") or tempfile.mkdtemp(prefix="fake_clips_"))
        self.unavailable = os.environ.get("AGENT_VIDEO_GEN_FAKE_UNAVAILABLE", "") == "1"

    async def generate(self, request: ClipRequest) -> Clip:
        if self.unavailable:
            raise GeneratorUnavailable("the fake generator is set to be unavailable")
        body = json.dumps(asdict(request), sort_keys=True).encode()
        path = self.clip_dir / f"{hashlib.sha256(body).hexdigest()}.mp4"
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return Clip(path=str(path))


@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int


@dataclass(frozen=True)
class Image:
    path: str


class ImageGenerator(Protocol):
    async def generate(self, request: ImageRequest) -> Image:
        """Return the generated image, or raise ``GeneratorUnavailable``."""
        ...


class FakeImageGenerator:
    """Writes a real PNG of the requested size per request, one colour from the request's hash, named by its hash.

    The planner's processor reads it as it would a generated image; the PNG's ``request`` text chunk holds the
    request as JSON. ``AGENT_VIDEO_GEN_FAKE_DIR`` and ``AGENT_VIDEO_GEN_FAKE_UNAVAILABLE`` work as for
    ``FakeClipGenerator``.
    """

    def __init__(self):
        self.image_dir = Path(os.environ.get("AGENT_VIDEO_GEN_FAKE_DIR") or tempfile.mkdtemp(prefix="fake_images_"))
        self.unavailable = os.environ.get("AGENT_VIDEO_GEN_FAKE_UNAVAILABLE", "") == "1"

    async def generate(self, request: ImageRequest) -> Image:
        if self.unavailable:
            raise GeneratorUnavailable("the fake generator is set to be unavailable")
        body = json.dumps(asdict(request), sort_keys=True)
        digest = hashlib.sha256(body.encode()).digest()
        path = self.image_dir / f"{digest.hex()}.png"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        info = PngInfo()
        info.add_text("request", body)
        partial = path.with_name(f"{path.name}.{uuid.uuid4().hex}.part")
        PILImage.new("RGB", (request.width, request.height), tuple(digest[:3])).save(
            partial, format="PNG", pnginfo=info
        )
        os.replace(partial, path)
        return Image(path=str(path))
