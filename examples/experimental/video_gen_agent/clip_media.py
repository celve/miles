"""Frame-level operations on the env's mp4 clips: a clip's last frame, and a chain of shots joined into one clip.

Both write files that a path names forever, as the generator does: SGLang's disk feature cache keys on the path.
"""

import collections
import hashlib
import os
import uuid
from pathlib import Path

import av


def last_frame(clip_path: str) -> str:
    """The clip's last frame as ``<clip>.last.png``, written once; a clip's path only ever holds the same bytes."""
    destination = Path(f"{clip_path}.last.png")
    if not destination.exists():
        with av.open(clip_path) as container:
            tail = collections.deque(container.decode(video=0), maxlen=1)
        if not tail:
            raise ValueError(f"{clip_path} has no video frames")
        _write_atomically(destination, lambda path: tail[0].to_image().save(path, format="PNG"))
    return str(destination)


def join_shots(clip_paths: list[str], *, fps: int, output_dir: str) -> str:
    """Join shots into ``<output_dir>/<sha256>.mp4``; each later shot loses its first frame, the one it continues from."""
    partial = Path(output_dir) / f"{uuid.uuid4().hex}.mp4.part"
    partial.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(partial), mode="w", format="mp4") as output:
        stream = output.add_stream("libx264", rate=fps)
        for index, path in enumerate(clip_paths):
            with av.open(path) as shot:
                source = shot.streams.video[0]
                if index == 0:
                    stream.width, stream.height, stream.pix_fmt = source.width, source.height, "yuv420p"
                for position, frame in enumerate(shot.decode(source)):
                    if index and not position:
                        continue
                    pixels = frame.to_ndarray(format="rgb24")
                    output.mux(stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")))
        output.mux(stream.encode())
    destination = partial.with_name(f"{hashlib.sha256(partial.read_bytes()).hexdigest()}.mp4")
    os.replace(partial, destination)
    return str(destination)


def _write_atomically(destination: Path, write) -> None:
    # Episodes of one group often reach the same clip at once; a reader must never see a half-written file.
    partial = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.part")
    write(partial)
    os.replace(partial, destination)
