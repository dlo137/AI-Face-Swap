"""
extract.py — FFmpeg frame extraction
Splits a video into PNG frames and pulls the audio track.
Returns the video's FPS so rebuild.py can reassemble correctly.
"""
import os
import subprocess


def extract_frames(video_path: str, frames_dir: str, audio_path: str) -> float:
    """
    Extract every frame as a PNG and the audio as AAC.

    Args:
        video_path:  Path to the input video file.
        frames_dir:  Directory to write frame_0001.png, frame_0002.png, …
        audio_path:  Where to write the extracted audio (AAC).

    Returns:
        fps (float) — frame rate of the source video.
    """
    fps = _probe_fps(video_path)

    # Extract frames
    frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
    _run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "1",          # lossless-ish PNG quality
        frame_pattern,
    ])

    # Extract audio (best-effort — silent videos are fine)
    _run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "aac",
        "-b:a", "192k",
        audio_path,
    ], check=False)

    n_frames = len([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    print(f"[extract] {n_frames} frames @ {fps} fps | audio: {os.path.exists(audio_path)}")
    return fps


def _probe_fps(video_path: str) -> float:
    """Use ffprobe to read the exact frame rate."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    raw = result.stdout.strip()  # e.g. "30000/1001" or "30/1"
    if "/" in raw:
        num, den = raw.split("/")
        return float(num) / float(den)
    return float(raw)


def _run(cmd: list, check: bool = True):
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg command failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {result.stderr[-2000:]}"
        )
