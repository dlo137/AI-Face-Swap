"""
rebuild.py — FFmpeg video reassembly
Takes the restored PNG frames + extracted audio and produces a final MP4.
"""
import os
import subprocess


def rebuild_video(
    frames_dir: str,
    audio_path: str,
    fps: float,
    output_path: str,
    crf: int = 18,
    preset: str = "slow",
):
    """
    Reassemble PNG frames into an MP4, then mux in the original audio.

    Args:
        frames_dir:   Directory containing frame_000001.png …
        audio_path:   Path to the extracted AAC audio file.
        fps:          Frame rate to encode at (from extract.py).
        output_path:  Destination MP4 path.
        crf:          H.264 quality [0=lossless, 51=worst]. 18 is near-lossless.
        preset:       FFmpeg encoding speed/compression trade-off.
    """
    frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
    has_audio = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0

    if has_audio:
        _run([
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-i", audio_path,
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", "yuv420p",   # broad compatibility
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",             # trim to shortest stream
            "-movflags", "+faststart",
            output_path,
        ])
    else:
        # Silent video — no audio stream
        _run([
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ])

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[rebuild] output: {output_path} ({size_mb:.1f} MB)")


def _run(cmd: list):
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {result.stderr[-2000:]}"
        )
