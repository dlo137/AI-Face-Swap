"""
download_weights.py — Run this once locally to seed the Modal Volume.

Usage:
    cd backend
    python download_weights.py

Requires: modal (pip install modal) + an authenticated token (modal token new)
"""

import os
import sys
import urllib.request
import zipfile
import tempfile
import shutil

import modal

# ── Volume ────────────────────────────────────────────────────────────────────

VOLUME_NAME = "face-swap-weights"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ── Model manifest ─────────────────────────────────────────────────────────────
# Each entry: (display_name, url, remote_path, is_zip, zip_extract_dir)
# is_zip=True  → download zip, extract, upload the *contents* of zip_extract_dir
# is_zip=False → download file directly, upload to remote_path

MODELS = [
    # ── InsightFace buffalo_l — already uploaded, skip ───────────────────────
    # (
    #     "InsightFace buffalo_l",
    #     "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    #     "/weights/insightface/models/buffalo_l",
    #     True,
    #     "buffalo_l",
    # ),

    # ── ArcFace — already bundled inside buffalo_l as w600k_r50.onnx ─────────
    # No separate download needed; pipeline references it from insightface/models/buffalo_l/

    # ── BiSeNet face parser (ONNX, FaceFusion 3.3.0 assets) ──────────────────
    (
        "BiSeNet face parser",
        "https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/bisenet_resnet_34.onnx",
        "/weights/bisenet/bisenet_resnet_34.onnx",
        False,
        None,
    ),

    # ── HyperSwap 1A 256 — already uploaded, skip ────────────────────────────
    # (
    #     "HyperSwap 1A 256",
    #     "https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1a_256.onnx",
    #     "/weights/hyperswap/hyperswap_1a_256.onnx",
    #     False,
    #     None,
    # ),

    # ── CodeFormer checkpoint — already uploaded, skip ────────────────────────
    # (
    #     "CodeFormer weights",
    #     "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    #     "/weights/codeformer/codeformer.pth",
    #     False,
    #     None,
    # ),

    # ── CodeFormer face detection — already uploaded, skip ────────────────────
    # (
    #     "CodeFormer — RetinaFace detection",
    #     "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    #     "/weights/codeformer/detection_Resnet50_Final.pth",
    #     False,
    #     None,
    # ),

    # ── CodeFormer face parsing (BiSeNet PyTorch) ─────────────────────────────
    (
        "CodeFormer — BiSeNet parsing",
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        "/weights/codeformer/parsing_parsenet.pth",
        False,
        None,
    ),
]

# ── Helpers ────────────────────────────────────────────────────────────────────

class _Progress:
    """Simple download progress bar."""
    def __init__(self, label: str):
        self.label = label
        self._last = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if total_size <= 0:
            return
        pct = min(100, int(block_num * block_size * 100 / total_size))
        if pct != self._last and pct % 10 == 0:
            print(f"    {pct}%", end="\r", flush=True)
            self._last = pct


def _download(url: str, dest: str, label: str) -> None:
    print(f"  Downloading {label} ...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest, reporthook=_Progress(label))
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"  Downloaded  {size_mb:.1f} MB → {dest}")


def _upload_file(batch, local_path: str, remote_path: str) -> None:
    print(f"  Uploading   {remote_path} ...", end=" ", flush=True)
    batch.put_file(local_path, remote_path)
    print("queued")


def _upload_dir(batch, local_dir: str, remote_dir: str) -> None:
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_file = os.path.join(root, fname)
            relative = os.path.relpath(local_file, local_dir)
            remote_file = remote_dir.rstrip("/") + "/" + relative.replace("\\", "/")
            print(f"  Uploading   {remote_file} ...", end=" ", flush=True)
            batch.put_file(local_file, remote_file)
            print("queued")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Seeding Modal Volume: {VOLUME_NAME}")
    print(f"{'='*60}\n")

    with tempfile.TemporaryDirectory() as tmp:
        with vol.batch_upload(force=True) as batch:
            for name, url, remote_path, is_zip, zip_subdir in MODELS:
                print(f"\n[{name}]")

                local_dest = os.path.join(tmp, os.path.basename(url))
                try:
                    _download(url, local_dest, name)
                except Exception as e:
                    print(f"  ERROR downloading {name}: {e}")
                    print("  Skipping — fix the URL and re-run.")
                    continue

                if is_zip:
                    extract_dir = os.path.join(tmp, f"extracted_{os.path.basename(url)}")
                    os.makedirs(extract_dir, exist_ok=True)
                    print(f"  Extracting zip ...")
                    with zipfile.ZipFile(local_dest, "r") as z:
                        z.extractall(extract_dir)

                    # Upload the specific subfolder from inside the zip
                    if zip_subdir:
                        src_dir = os.path.join(extract_dir, zip_subdir)
                        if not os.path.isdir(src_dir):
                            # Try top-level (some zips don't nest)
                            src_dir = extract_dir
                    else:
                        src_dir = extract_dir

                    _upload_dir(batch, src_dir, remote_path)
                else:
                    _upload_file(batch, local_dest, remote_path)

                print(f"  ✓ {name} queued for upload")

    print(f"\n{'='*60}")
    print("  All uploads committed to Modal Volume.")
    print(f"  Volume: {VOLUME_NAME}")
    print(f"{'='*60}\n")
    print("Next step:")
    print("  python -m modal run main.py::swap_face\n")


if __name__ == "__main__":
    main()
