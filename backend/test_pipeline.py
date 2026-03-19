"""
test_pipeline.py — End-to-end smoke test for the face swap Modal endpoint.

Usage:
    cd backend
    python test_pipeline.py

Set ENDPOINT_URL and API_TOKEN below, or export them as env vars before running.
"""
import os
import sys
import time
import requests

# ── Config ────────────────────────────────────────────────────────────────────
# Fill these in after `modal deploy` prints your endpoint URL.

ENDPOINT_URL = os.environ.get(
    "ENDPOINT_URL",
    "https://dlo137--face-swap-pipeline-api-endpoint.modal.run",
)
API_TOKEN = os.environ.get(
    "API_TOKEN",
    "key_69cefa816fba63fa95a1cb0726114604ba4abf7bbed638129e6cb6b1787040361d8bacd3cf97840932757312169581d846bed7e2b94955ffa6a0a3d8d63227f7",
)

SOURCE_IMAGE = os.path.join("test_assets", "source.jpg")
TARGET_VIDEO = os.path.join("test_assets", "target.mov")
OUTPUT_DIR   = "test_output"
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "result.mp4")

# Modal web endpoint stays open for the full job — allow up to 12 min
REQUEST_TIMEOUT = 720


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_assets():
    missing = [p for p in (SOURCE_IMAGE, TARGET_VIDEO) if not os.path.exists(p)]
    if missing:
        print("ERROR: missing test assets:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)
    src_kb = os.path.getsize(SOURCE_IMAGE) / 1024
    vid_mb = os.path.getsize(TARGET_VIDEO) / (1024 * 1024)
    print(f"  source image : {SOURCE_IMAGE} ({src_kb:.1f} KB)")
    print(f"  target video : {TARGET_VIDEO} ({vid_mb:.1f} MB)")


def _post_job(source_bytes: bytes, video_bytes: bytes) -> dict:
    print(f"\n  POST {ENDPOINT_URL}")
    resp = requests.post(
        ENDPOINT_URL,
        headers={"Authorization": f"Bearer {API_TOKEN}"},
        files={
            "source_image": ("source.jpg", source_bytes, "image/jpeg"),
            "target_video": ("target.mov", video_bytes, "video/quicktime"),
        },
        timeout=REQUEST_TIMEOUT,
    )
    print(f"  HTTP {resp.status_code}")
    try:
        return resp.json()
    except Exception:
        print(f"  raw response: {resp.text[:500]}")
        raise


def _download(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"\n  downloading result → {dest}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def _video_duration(path: str) -> float | None:
    """Return duration in seconds using ffprobe, or None if unavailable."""
    import subprocess, json
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            stderr=subprocess.DEVNULL,
        )
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return None


def _has_audio(path: str) -> bool:
    import subprocess
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Face Swap Pipeline — Smoke Test")
    print("=" * 60)

    # 1. Verify assets exist
    print("\n[1] checking test assets...")
    _check_assets()

    # 2. Read files
    with open(SOURCE_IMAGE, "rb") as f: source_bytes = f.read()
    with open(TARGET_VIDEO, "rb") as f: video_bytes  = f.read()

    # 3. Call endpoint
    print("\n[2] submitting job to Modal endpoint...")
    start = time.time()

    try:
        result = _post_job(source_bytes, video_bytes)
    except requests.exceptions.Timeout:
        print(f"\nERROR: request timed out after {REQUEST_TIMEOUT}s")
        sys.exit(1)
    except requests.exceptions.ConnectionError as e:
        print(f"\nERROR: could not connect to endpoint: {e}")
        sys.exit(1)

    elapsed = time.time() - start

    if result.get("status") != "complete":
        print(f"\nERROR: job failed")
        print(f"  message: {result.get('message', 'unknown')}")
        sys.exit(1)

    output_url = result["url"]
    print(f"  elapsed   : {elapsed:.1f}s")
    print(f"  output URL: {output_url}")

    # 4. Download result
    print("\n[3] downloading output video...")
    _download(output_url, OUTPUT_FILE)

    # 5. Report
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    in_dur  = _video_duration(TARGET_VIDEO)
    out_dur = _video_duration(OUTPUT_FILE)
    audio   = _has_audio(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("  Smoke Test Results")
    print("=" * 60)
    print(f"  elapsed          : {elapsed:.1f}s")
    print(f"  output file      : {OUTPUT_FILE}")
    print(f"  output size      : {size_mb:.1f} MB")
    print(f"  input duration   : {in_dur:.2f}s"  if in_dur  else "  input duration   : unknown")
    print(f"  output duration  : {out_dur:.2f}s" if out_dur else "  output duration  : unknown")

    dur_ok    = in_dur and out_dur and abs(in_dur - out_dur) < 0.5
    audio_ok  = audio
    size_ok   = size_mb > 0.1

    print(f"\n  [{'✓' if dur_ok   else '✗'}] duration matches input (within 0.5s)")
    print(f"  [{'✓' if audio_ok else '✗'}] audio track present")
    print(f"  [{'✓' if size_ok  else '✗'}] output file not empty")
    print()
    print("  Manual checks still required:")
    print("  [ ] face swap visible in first 3 frames")
    print("  [ ] no black frames or corrupted segments")
    print("  [ ] CodeFormer didn't wash out source identity")
    print("  [ ] audio is synced to video")
    print()

    if dur_ok and audio_ok and size_ok:
        print("  ✓ automated checks passed — open result.mp4 for visual review")
    else:
        print("  ✗ one or more automated checks failed — see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
