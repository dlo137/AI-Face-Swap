"""
rebuild.py — Frame reassembly using ffmpeg-python

Public API:
  rebuild_video(frame_dir, audio_path, output_path) → None
"""
from __future__ import annotations

import os
import glob
import ffmpeg


FPS_META_FILE = "fps.txt"   # written by extract.py, read back here


def rebuild_video(
    frame_dir: str,
    audio_path: str,
    output_path: str,
    crf: int = 17,
    preset: str = "slow",
) -> None:
    """
    Reassemble sorted PNG frames into an H.264 MP4 and mux the original audio.

    FPS is read from fps.txt written by extract_frames() into frame_dir.
    Falls back to 30.0 if the file is missing.

    Args:
        frame_dir:    Directory containing frame_000001.png … (from restore.py output).
        audio_path:   Path to audio.aac written by extract_frames().
                      If the file is empty or missing the output is silent.
        output_path:  Destination .mp4 path.
        crf:          H.264 quality [0=lossless, 51=worst]. 17 is near-lossless.
        preset:       FFmpeg speed/compression trade-off.
    """
    fps = _read_fps(frame_dir)
    frame_pattern = os.path.join(frame_dir, "frame_%06d.png")
    has_audio = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0

    n_frames = len(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    print(f"[rebuild] {n_frames} frames @ {fps:.3f} fps | audio={has_audio}")

    video_in = (
        ffmpeg
        .input(frame_pattern, framerate=fps, pattern_type="sequence")
        .video
    )

    if has_audio:
        audio_in = ffmpeg.input(audio_path).audio

        (
            ffmpeg
            .output(
                video_in,
                audio_in,
                output_path,
                vcodec="libx264",
                crf=crf,
                preset=preset,
                pix_fmt="yuv420p",      # broad player compatibility
                acodec="aac",
                audio_bitrate="192k",
                ac=2,
                shortest=None,          # trim to shortest stream
                movflags="+faststart",  # web-friendly: moov atom at front
            )
            .overwrite_output()
            .run(quiet=True)
        )
    else:
        (
            ffmpeg
            .output(
                video_in,
                output_path,
                vcodec="libx264",
                crf=crf,
                preset=preset,
                pix_fmt="yuv420p",
                movflags="+faststart",
            )
            .overwrite_output()
            .run(quiet=True)
        )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[rebuild] output: {output_path} ({size_mb:.1f} MB)")


# ── FPS metadata helpers (used by extract.py + rebuild.py) ───────────────────

def write_fps(frame_dir: str, fps: float) -> None:
    """Write fps to a small metadata file in the frame directory."""
    with open(os.path.join(frame_dir, FPS_META_FILE), "w") as f:
        f.write(str(fps))


def _read_fps(frame_dir: str, default: float = 30.0) -> float:
    """Read fps written by extract_frames(). Returns default if file missing."""
    meta = os.path.join(frame_dir, FPS_META_FILE)
    if os.path.exists(meta):
        try:
            return float(open(meta).read().strip())
        except ValueError:
            pass
    print(f"[rebuild] fps.txt not found in {frame_dir}, defaulting to {default}")
    return default
