"""Store a generated clip as an mp4 named by the sha256 of its bytes."""

import hashlib
import os
import tempfile

import imageio_ffmpeg
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
    frames = video.permute(1, 2, 3, 0).contiguous().numpy()
    writer = imageio_ffmpeg.write_frames(
        path,
        size=(width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-f", "mp4"],
    )
    writer.send(None)
    for frame in frames:
        writer.send(frame)
    writer.close()
