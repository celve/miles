"""Store a generated clip as an mp4 named by the sha256 of its bytes."""

import hashlib
import os
import tempfile

import av
import torch


def save_clip(video: torch.Tensor, *, fps: int, output_dir: str) -> str:
    """Encode a uint8 ``[C, F, H, W]`` RGB clip to ``<output_dir>/<sha256>.mp4`` and return that path."""
    os.makedirs(output_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".mp4.part")
    os.close(fd)
    try:
        _encode_mp4(video, fps=fps, path=tmp_path)
        with open(tmp_path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        path = os.path.join(output_dir, f"{digest}.mp4")
        # The planner's engines cache features by file path forever, so a name only ever holds these bytes.
        os.replace(tmp_path, path)
    except BaseException:
        os.remove(tmp_path)
        raise
    return path


def _encode_mp4(video: torch.Tensor, *, fps: int, path: str) -> None:
    channels, _, height, width = video.shape
    if video.dtype != torch.uint8 or channels != 3:
        raise ValueError(f"expected a uint8 RGB clip [3, F, H, W], got {video.dtype} {tuple(video.shape)}")
    with av.open(path, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for frame in video.permute(1, 2, 3, 0).contiguous().numpy():
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())
