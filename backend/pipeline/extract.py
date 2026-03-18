"""
extract.py — Frame extraction + audio strip using ffmpeg-python
"""
import os
import glob
import ffmpeg


def extract_frames(video_path: str, output_dir: str) -> tuple[list[str], str]:
    """
    Extract every frame from a video as PNG files and strip the audio track.

    Handles videos up to 2 minutes / 1080p. Frames are named
    frame_000001.png, frame_000002.png, … for stable sorted ordering.

    Args:
        video_path: Absolute path to the input video file (MP4, MOV, etc.).
        output_dir: Directory to write frames and audio into. Created if missing.

    Returns:
        (frame_paths, audio_path) where:
          - frame_paths is a sorted list of absolute paths to every PNG frame.
          - audio_path  is the absolute path to audio.aac (may be an empty file
            if the source video has no audio track — always safe to pass to
            rebuild.py which handles the silent case).
    """
    os.makedirs(output_dir, exist_ok=True)

    audio_path = os.path.join(output_dir, "audio.aac")
    frame_pattern = os.path.join(output_dir, "frame_%06d.png")

    # ── Probe ──────────────────────────────────────────────────────────────────
    probe = ffmpeg.probe(video_path)
    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"), None
    )
    audio_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "audio"), None
    )

    if video_stream is None:
        raise ValueError(f"No video stream found in: {video_path}")

    fps = _parse_fps(video_stream.get("r_frame_rate", "30/1"))
    duration = float(probe["format"].get("duration", 0))
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    print(f"[extract] {width}x{height} @ {fps:.3f} fps | duration={duration:.1f}s")

    # ── Frames ─────────────────────────────────────────────────────────────────
    (
        ffmpeg
        .input(video_path)
        .video
        .filter("fps", fps=fps)           # lock to exact source fps
        .filter("scale",                  # cap at 1080p, preserve AR
                w="min(iw,1920)",
                h="min(ih,1080)",
                force_original_aspect_ratio="decrease")
        .output(
            frame_pattern,
            vcodec="png",
            vsync="vfr",                  # drop dupes without renumbering
            **{"qscale:v": 1},            # near-lossless PNG compression
        )
        .overwrite_output()
        .run(quiet=True)
    )

    frame_paths = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
    print(f"[extract] extracted {len(frame_paths)} frames")

    # ── Audio ──────────────────────────────────────────────────────────────────
    if audio_stream:
        (
            ffmpeg
            .input(video_path)
            .audio
            .output(
                audio_path,
                acodec="aac",
                audio_bitrate="192k",
                ac=2,                     # force stereo
            )
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"[extract] audio stripped → {audio_path}")
    else:
        # Write an empty placeholder so callers don't need to guard for None
        open(audio_path, "wb").close()
        print("[extract] no audio track found — empty placeholder written")

    return frame_paths, audio_path


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_fps(r_frame_rate: str) -> float:
    """Convert ffprobe r_frame_rate string (e.g. '30000/1001') to float."""
    if "/" in r_frame_rate:
        num, den = r_frame_rate.split("/")
        den = int(den)
        return float(num) / den if den else 30.0
    return float(r_frame_rate)
