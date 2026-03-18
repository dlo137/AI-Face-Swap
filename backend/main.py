"""
Face Swap Pipeline — Modal entrypoint
Accepts: source_image (bytes), target_video (bytes)
Returns: Supabase public URL of the swapped video
"""
import modal

# ── Image ─────────────────────────────────────────────────────────────────────

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    # System deps
    .apt_install(
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "git",
        "wget",
        "unzip",
    )
    # Core Python packages
    .pip_install(
        "torch==2.2.1",
        "torchvision==0.17.1",
        "torchaudio==2.2.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "insightface==0.7.3",
        "onnxruntime-gpu==1.17.1",
        "opencv-python-headless==4.9.0.80",
        "ffmpeg-python==0.2.0",
        "Pillow",
        "numpy",
        "scipy",
        "tqdm",
        "gfpgan",  # fallback restorer
        "basicsr",
        "facexlib",
        "supabase",
    )
    # CodeFormer from source
    .run_commands(
        "git clone https://github.com/sczhou/CodeFormer.git /opt/CodeFormer",
        "cd /opt/CodeFormer && pip install -r requirements.txt",
        "cd /opt/CodeFormer && python basicsr/setup.py develop 2>/dev/null || true",
    )
    # Add CodeFormer to Python path
    .env({"PYTHONPATH": "/opt/CodeFormer"})
)

# ── Volume (persistent model weights) ─────────────────────────────────────────

weights_volume = modal.Volume.from_name("face-swap-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"

# ── App ───────────────────────────────────────────────────────────────────────

app = modal.App("face-swap-pipeline", image=image)

# ── Weight download (run once to seed the volume) ─────────────────────────────

@app.function(
    volumes={WEIGHTS_DIR: weights_volume},
    timeout=600,
)
def download_weights():
    """
    Download all model weights into the persistent volume.
    Run once with: modal run main.py::download_weights
    """
    import os
    import urllib.request

    def fetch(url: str, dest: str):
        if os.path.exists(dest):
            print(f"[weights] already exists: {dest}")
            return
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"[weights] downloading {url} → {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"[weights] done: {dest}")

    # InsightFace buffalo_l (face detection + recognition)
    fetch(
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        f"{WEIGHTS_DIR}/insightface/buffalo_l.zip",
    )
    _unzip(f"{WEIGHTS_DIR}/insightface/buffalo_l.zip", f"{WEIGHTS_DIR}/insightface/models")

    # hyperswap_1a_256.onnx — already downloaded via download_weights.py

    # CodeFormer weights
    fetch(
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        f"{WEIGHTS_DIR}/codeformer/codeformer.pth",
    )
    fetch(
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        f"{WEIGHTS_DIR}/codeformer/detection_Resnet50_Final.pth",
    )
    fetch(
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/parsing_parsenet.pth",
        f"{WEIGHTS_DIR}/codeformer/parsing_parsenet.pth",
    )

    weights_volume.commit()
    print("[weights] all weights ready.")


def _unzip(zip_path: str, dest_dir: str):
    import zipfile
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)


# ── Main pipeline function ────────────────────────────────────────────────────

@app.function(
    gpu="A100",
    volumes={WEIGHTS_DIR: weights_volume},
    secrets=[modal.Secret.from_name("supabase-secret")],
    timeout=600,
    memory=32768,
)
def swap_face(source_image: bytes, target_video: bytes) -> str:
    """
    Full face-swap pipeline.

    Args:
        source_image: Raw bytes of a JPEG/PNG containing the source face.
        target_video: Raw bytes of an MP4/MOV to swap faces in.

    Returns:
        Public Supabase URL of the output video.
    """
    import os
    import tempfile

    from pipeline.extract import extract_frames
    from pipeline.detect import detect_face, extract_identity
    from pipeline.swap import load_swapper, swap_faces_in_frames
    from pipeline.restore import restore_frames
    from pipeline.rebuild import rebuild_video
    from storage import upload_video

    ARCFACE_PATH = os.path.join(WEIGHTS_DIR, "insightface", "models", "buffalo_l", "w600k_r50.onnx")

    with tempfile.TemporaryDirectory() as work_dir:
        # ── 1. Write inputs to disk ────────────────────────────────────────────
        src_img_path = os.path.join(work_dir, "source.jpg")
        tgt_vid_path = os.path.join(work_dir, "target.mp4")
        with open(src_img_path, "wb") as f:
            f.write(source_image)
        with open(tgt_vid_path, "wb") as f:
            f.write(target_video)

        frames_dir = os.path.join(work_dir, "frames")
        swapped_dir = os.path.join(work_dir, "swapped")
        restored_dir = os.path.join(work_dir, "restored")
        output_path = os.path.join(work_dir, "output.mp4")

        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(swapped_dir, exist_ok=True)
        os.makedirs(restored_dir, exist_ok=True)

        # ── 2. Extract frames + audio ──────────────────────────────────────────
        print("[pipeline] extracting frames...")
        frame_paths, audio_path = extract_frames(tgt_vid_path, frames_dir)

        # ── 3. Load swap model ─────────────────────────────────────────────────
        print("[pipeline] loading HyperSwap model...")
        swapper = load_swapper(WEIGHTS_DIR)

        # ── 4. Extract source identity embedding (once per job) ────────────────
        print("[pipeline] extracting source identity...")
        identity_embedding = extract_identity(src_img_path, ARCFACE_PATH)

        # ── 5. Swap faces in every frame ───────────────────────────────────────
        print("[pipeline] swapping faces...")
        swap_faces_in_frames(swapper, identity_embedding, frame_paths, swapped_dir)

        # ── 6. Restore / enhance frames ────────────────────────────────────────
        print("[pipeline] restoring frames with CodeFormer...")
        restore_frames(swapped_dir, restored_dir, WEIGHTS_DIR)

        # ── 7. Reassemble video ────────────────────────────────────────────────
        print("[pipeline] rebuilding video...")
        rebuild_video(restored_dir, audio_path, fps, output_path)

        # ── 8. Upload to Supabase ──────────────────────────────────────────────
        print("[pipeline] uploading to Supabase...")
        url = upload_video(output_path)
        print(f"[pipeline] done → {url}")
        return url
