"""The env's view of the frozen video generator: one call that turns a request into a clip file.

The real client wraps the diffusion engine group behind the Miles router; ``FakeClipGenerator`` stands in
for it in tests and CPU smoke runs.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


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
