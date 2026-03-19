"""
Face Swap Pipeline — Modal entrypoint (single-file, no local imports)
"""
from __future__ import annotations
import os
import modal

# ── Modal image ────────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.10")
    .run_commands(
        "apt-get update && apt-get install -y g++ build-essential ffmpeg "
        "libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 git wget unzip"
    )
    .pip_install(
        "torch==2.2.1",
        "torchvision==0.17.1",
        "torchaudio==2.2.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy<2",
        "insightface==0.7.3",
        "onnxruntime-gpu==1.17.1",
        "opencv-python-headless==4.9.0.80",
        "ffmpeg-python==0.2.0",
        "Pillow",
        "scipy",
        "tqdm",
        "basicsr",
        "facexlib",
        "supabase",
        "fastapi",
        "python-multipart",
    )
    .run_commands(
        "git clone https://github.com/sczhou/CodeFormer.git /opt/CodeFormer",
        "cd /opt/CodeFormer && pip install -r requirements.txt",
        "cd /opt/CodeFormer && python basicsr/setup.py develop 2>/dev/null || true",
    )
    .env({"PYTHONPATH": "/opt/CodeFormer"})
)

# ── Volume + app ───────────────────────────────────────────────────────────────

weights_volume = modal.Volume.from_name("face-swap-weights", create_if_missing=True)
WEIGHTS_DIR  = "/weights"
ARCFACE_PATH = f"{WEIGHTS_DIR}/insightface/models/buffalo_l/w600k_r50.onnx"

app = modal.App("face-swap-pipeline", image=image)

# ── Weight download (run once) ─────────────────────────────────────────────────

@app.function(volumes={WEIGHTS_DIR: weights_volume}, timeout=600)
def download_weights():
    """modal run main.py::download_weights"""
    import urllib.request, zipfile

    def fetch(url, dest):
        if os.path.exists(dest):
            print(f"[weights] exists: {dest}")
            return
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"[weights] downloading → {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"[weights] done: {dest}")

    fetch("https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
          f"{WEIGHTS_DIR}/insightface/buffalo_l.zip")
    target = f"{WEIGHTS_DIR}/insightface/models/buffalo_l"
    if not os.path.exists(f"{target}/w600k_r50.onnx"):
        os.makedirs(target, exist_ok=True)
        with zipfile.ZipFile(f"{WEIGHTS_DIR}/insightface/buffalo_l.zip") as z:
            z.extractall(target)
        print(f"[weights] buffalo_l extracted → {target}")

    fetch("https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1a_256.onnx",
          f"{WEIGHTS_DIR}/hyperswap/hyperswap_1a_256.onnx")
    fetch("https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
          f"{WEIGHTS_DIR}/codeformer/codeformer.pth")
    fetch("https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
          f"{WEIGHTS_DIR}/codeformer/detection_Resnet50_Final.pth")
    fetch("https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
          f"{WEIGHTS_DIR}/codeformer/parsing_parsenet.pth")

    weights_volume.commit()
    print("[weights] all done.")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS (inlined — no local imports)
# ══════════════════════════════════════════════════════════════════════════════

# ── Extract ────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, output_dir: str):
    import glob, ffmpeg
    os.makedirs(output_dir, exist_ok=True)
    audio_path    = os.path.join(output_dir, "audio.aac")
    frame_pattern = os.path.join(output_dir, "frame_%06d.png")

    probe = ffmpeg.probe(video_path)
    video_stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    audio_stream = next((s for s in probe["streams"] if s["codec_type"] == "audio"), None)
    if video_stream is None:
        raise ValueError(f"No video stream in: {video_path}")

    fps      = _parse_fps(video_stream.get("r_frame_rate", "30/1"))
    duration = float(probe["format"].get("duration", 0))
    w, h     = int(video_stream.get("width", 0)), int(video_stream.get("height", 0))
    print(f"[extract] {w}x{h} @ {fps:.3f} fps | duration={duration:.1f}s")

    # Persist fps for rebuild
    with open(os.path.join(output_dir, "fps.txt"), "w") as f:
        f.write(str(fps))

    (ffmpeg.input(video_path).video
     .filter("fps", fps=fps)
     .filter("scale", w="min(iw,1920)", h="min(ih,1080)", force_original_aspect_ratio="decrease")
     .output(frame_pattern, vcodec="png", vsync="vfr", **{"qscale:v": 1})
     .overwrite_output().run(quiet=True))

    if audio_stream:
        (ffmpeg.input(video_path).audio
         .output(audio_path, acodec="aac", audio_bitrate="192k", ac=2)
         .overwrite_output().run(quiet=True))
        print(f"[extract] audio → {audio_path}")
    else:
        open(audio_path, "wb").close()

    frame_paths = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
    print(f"[extract] {len(frame_paths)} frames extracted")
    return frame_paths, audio_path


def _parse_fps(r: str) -> float:
    if "/" in r:
        n, d = r.split("/")
        return float(n) / int(d) if int(d) else 30.0
    return float(r)


# ── Detect ─────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
import numpy as np

@dataclass
class FaceDetection:
    bbox:       np.ndarray
    landmarks:  np.ndarray
    confidence: float
    _raw:       object = field(default=None, repr=False)

_analyser      = None
_analyser_wdir = ""

def _get_analyser(weights_dir: str):
    global _analyser, _analyser_wdir
    if _analyser is None or _analyser_wdir != weights_dir:
        from insightface.app import FaceAnalysis
        root = os.path.join(weights_dir, "insightface")
        _analyser = FaceAnalysis(name="buffalo_l", root=root,
                                 providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        _analyser.prepare(ctx_id=0, det_size=(640, 640))
        _analyser_wdir = weights_dir
        print("[detect] buffalo_l loaded")
    return _analyser

def detect_face(frame: np.ndarray, weights_dir: str = "/weights",
                min_confidence: float = 0.6):
    analyser = _get_analyser(weights_dir)
    faces = analyser.get(frame)
    if not faces:
        return None
    best = max(faces, key=lambda f: _bbox_area(f.bbox))
    if float(best.det_score) < min_confidence:
        return None
    return FaceDetection(
        bbox=np.array(best.bbox, dtype=np.float32),
        landmarks=np.array(best.kps, dtype=np.float32),
        confidence=float(best.det_score),
        _raw=best,
    )

def extract_identity(image_path: str, arcface_model_path: str) -> np.ndarray:
    import cv2, onnxruntime
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Source image not found: {image_path}")
    if not os.path.exists(arcface_model_path):
        raise FileNotFoundError(f"ArcFace model not found: {arcface_model_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not decode: {image_path}")
    face = detect_face(frame)
    if face is None:
        raise ValueError(f"No face detected in source image: {image_path}")
    aligned = _align_to(frame, face.landmarks, 112)
    session = onnxruntime.InferenceSession(
        arcface_model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    blob = _preprocess_arcface(aligned)
    raw  = session.run(None, {session.get_inputs()[0].name: blob})[0]
    emb  = _l2_norm(raw[0])
    print(f"[detect] embedding extracted | norm={np.linalg.norm(emb):.4f}")
    return emb

_ARCFACE_TPL_112 = np.array([
    [38.2946, 51.6963],[73.5318, 51.5014],[56.0252, 71.7366],
    [41.5493, 92.3655],[70.7299, 92.2041]], dtype=np.float32)

def _align_to(frame, kps, size):
    import cv2
    tpl = _ARCFACE_TPL_112 * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(kps, tpl, method=cv2.LMEDS)
    return cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR)

def _align_to_M(frame, kps, size):
    import cv2
    tpl = _ARCFACE_TPL_112 * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(kps, tpl, method=cv2.LMEDS)
    return cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR), M

def _preprocess_arcface(face_112):
    import cv2
    rgb = cv2.cvtColor(face_112, cv2.COLOR_BGR2RGB).astype(np.float32)
    return ((rgb - 127.5) / 128.0).transpose(2, 0, 1)[np.newaxis]

def _l2_norm(v):
    return v / (np.linalg.norm(v) + 1e-8)

def _bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0., x2 - x1) * max(0., y2 - y1)


# ── Swap ───────────────────────────────────────────────────────────────────────

# BiSeNet CelebAMask-HQ labels included in the swap mask.
# 1=skin 2=l_brow 3=r_brow 4=l_eye 5=r_eye 10=nose
# 12=u_lip 13=i_mouth 14=l_lip 16=chin 17=neck
_FACE_LABELS   = {1, 2, 3, 4, 5, 10, 12, 13, 14, 16, 17}
_BISENET_MEAN  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_BISENET_STD   = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_models(weights_dir: str):
    import onnxruntime
    providers      = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    hyperswap_path = os.path.join(weights_dir, "hyperswap", "hyperswap_1a_256.onnx")
    bisenet_path   = os.path.join(weights_dir, "bisenet",   "bisenet_resnet_34.onnx")
    if not os.path.exists(hyperswap_path):
        raise FileNotFoundError(f"HyperSwap model not found: {hyperswap_path}")
    hyperswap = onnxruntime.InferenceSession(hyperswap_path, providers=providers)
    _log_io(hyperswap, "hyperswap")
    bisenet = None
    if os.path.exists(bisenet_path):
        bisenet = onnxruntime.InferenceSession(bisenet_path, providers=providers)
        _log_io(bisenet, "bisenet")
    else:
        print("[swap] BiSeNet not found — using ellipse mask fallback")
    return hyperswap, bisenet

def swap_frame(frame, identity_embedding, hyperswap_model, bisenet_model):
    import cv2
    face = detect_face(frame)
    if face is None:
        return frame
    aligned, M = _align_to_M(frame, face.landmarks, 256)
    swapped_crop = _run_hyperswap(hyperswap_model, identity_embedding, aligned)

    # Landmark correction: HyperSwap output can drift slightly from the
    # canonical 256-px grid, causing the mouth to double-expose.
    # Detect landmarks in the swapped crop and re-align to the canonical
    # template so M_inv lands it precisely on the target face.
    swap_face = detect_face(swapped_crop)
    if swap_face is not None:
        tpl = (_ARCFACE_TPL_112 * (256.0 / 112.0)).astype(np.float32)
        M_corr, _ = cv2.estimateAffinePartial2D(
            swap_face.landmarks, tpl, method=cv2.LMEDS)
        if M_corr is not None:
            swapped_crop = cv2.warpAffine(
                swapped_crop, M_corr, (256, 256), flags=cv2.INTER_LINEAR)

    mask_256 = (_bisenet_face_mask(bisenet_model, aligned)
                if bisenet_model is not None else _ellipse_mask(256))
    h, w = frame.shape[:2]
    M_inv = cv2.invertAffineTransform(M)
    swapped_full = cv2.warpAffine(swapped_crop, M_inv, (w, h), flags=cv2.INTER_LINEAR)
    mask_full    = cv2.warpAffine(mask_256,     M_inv, (w, h), flags=cv2.INTER_LINEAR)
    mask_full    = cv2.GaussianBlur(mask_full, (21, 21), 0).clip(0., 1.)
    alpha = mask_full[:, :, np.newaxis].astype(np.float32)
    return (swapped_full.astype(np.float32) * alpha +
            frame.astype(np.float32) * (1. - alpha)).clip(0, 255).astype(np.uint8)

def _run_hyperswap(session, identity_embedding, aligned_256):
    import cv2
    src_emb = identity_embedding.reshape(1, -1).astype(np.float32)
    rgb  = cv2.cvtColor(aligned_256, cv2.COLOR_BGR2RGB).astype(np.float32)
    blob = ((rgb / 127.5) - 1.0).transpose(2, 0, 1)[np.newaxis]
    feed = {}
    for inp in session.get_inputs():
        feed[inp.name] = src_emb if len(inp.shape) == 2 else blob
    out = session.run(None, feed)[0]
    result = (out[0].transpose(1, 2, 0) + 1.0) * 127.5
    return cv2.cvtColor(result.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def _bisenet_run(session, img_512):
    """Run BiSeNet on a 512-px BGR crop; return label map and unique label set."""
    rgb    = cv2.cvtColor(img_512, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob   = ((rgb - _BISENET_MEAN) / _BISENET_STD).transpose(2, 0, 1)[np.newaxis]
    logits = session.run(None, {session.get_inputs()[0].name: blob})[0]
    labels = logits[0].argmax(axis=0)
    return labels, set(np.unique(labels).tolist())


def _extend_mask_down(mask, frac=0.20):
    """Extend the bottom edge of the mask bbox downward by frac of its height."""
    rows = np.where(mask > 0.5)
    if len(rows[0]) == 0:
        return mask
    y_min, y_max = int(rows[0].min()), int(rows[0].max())
    x_min, x_max = int(rows[1].min()), int(rows[1].max())
    ext      = int((y_max - y_min) * frac)
    y_bottom = min(mask.shape[0], y_max + ext)
    extended = mask.copy()
    extended[y_max:y_bottom, x_min:x_max] = 1.0
    return extended


def _bisenet_face_mask(session, face_crop_256):
    import cv2
    resized = cv2.resize(face_crop_256, (512, 512), interpolation=cv2.INTER_LINEAR)
    labels, unique = _bisenet_run(session, resized)
    print(f"[bisenet] labels detected: {sorted(unique)}")
    mask = np.zeros_like(labels, dtype=np.float32)
    for lbl in _FACE_LABELS:
        mask[labels == lbl] = 1.0
    mask  = _extend_mask_down(mask, frac=0.20)
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask  = cv2.dilate(mask, dil_k, iterations=3)
    mask  = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(mask, (15, 15), 7).clip(0., 1.)

def _ellipse_mask(size: int):
    import cv2
    mask = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    cv2.ellipse(mask, (cx, cy), (cx - 12, cy - 10), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(mask, (31, 31), 15)

def _log_io(session, name):
    ins  = [(i.name, i.shape) for i in session.get_inputs()]
    outs = [(o.name, o.shape) for o in session.get_outputs()]
    print(f"[swap] {name} | inputs={ins} | outputs={outs}")


# ── Restore ────────────────────────────────────────────────────────────────────

CODEFORMER_DIR      = "/opt/CodeFormer"
CODEFORMER_FIDELITY = 0.15

def load_codeformer(weights_dir: str):
    import sys, importlib, torch
    if CODEFORMER_DIR not in sys.path:
        sys.path.insert(0, CODEFORMER_DIR)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cf_arch = importlib.import_module("basicsr.archs.codeformer_arch")
    net = cf_arch.CodeFormer(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
        connect_list=["32", "64", "128", "256"]).to(device)
    ckpt_path = os.path.join(weights_dir, "codeformer", "codeformer.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"CodeFormer checkpoint missing: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt["params_ema"])
    net.eval()
    print(f"[restore] CodeFormer loaded (device={device})")
    return net

def _sharpen_face(face_crop: np.ndarray) -> np.ndarray:
    """Unsharp-style sharpening applied to the face crop only."""
    import cv2
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(face_crop, -1, kernel)


def restore_frame(swapped_frame, original_frame, bisenet_model=None, codeformer_net=None):
    import cv2
    face = detect_face(swapped_frame)
    if face is None:
        return swapped_frame
    colour_corrected = _match_lighting(swapped_frame, original_frame, face)
    aligned_512, M   = _align_to_M(colour_corrected, face.landmarks, 512)
    restored_512     = _run_codeformer(codeformer_net, aligned_512) if codeformer_net else aligned_512
    restored_512     = _sharpen_face(restored_512)
    mask_512 = (_bisenet_mask_512(bisenet_model, aligned_512)
                if bisenet_model is not None else _ellipse_mask_r(512))
    return _paste_back(swapped_frame, restored_512, mask_512, M)

def _match_lighting(swapped, original, face):
    import cv2
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(swapped.shape[1], x2), min(swapped.shape[0], y2)
    sl = cv2.cvtColor(swapped,  cv2.COLOR_BGR2LAB).astype(np.float32)
    ol = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    rl = sl.copy()
    for c in range(3):
        sm, ss = sl[y1:y2, x1:x2, c].mean(), sl[y1:y2, x1:x2, c].std() + 1e-6
        rm, rs = ol[y1:y2, x1:x2, c].mean(), ol[y1:y2, x1:x2, c].std() + 1e-6
        rl[y1:y2, x1:x2, c] = (sl[y1:y2, x1:x2, c] - sm) * (rs / ss) + rm
    return cv2.cvtColor(rl.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

def _run_codeformer(net, face_512):
    import cv2, torch
    from torchvision.transforms.functional import normalize as tv_norm
    device = next(net.parameters()).device
    rgb    = cv2.cvtColor(face_512, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    tv_norm(tensor, [0.5]*3, [0.5]*3, inplace=True)
    with torch.no_grad():
        out = net(tensor, w=CODEFORMER_FIDELITY, adain=True)[0]
    out_np = out.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    out_np = (out_np * 0.5 + 0.5).clip(0, 1)
    return cv2.cvtColor((out_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def _bisenet_mask_512(session, face_512):
    import cv2
    labels, unique = _bisenet_run(session, face_512)
    print(f"[bisenet512] labels detected: {sorted(unique)}")
    mask = np.zeros_like(labels, dtype=np.float32)
    for lbl in _FACE_LABELS:
        mask[labels == lbl] = 1.0
    mask  = _extend_mask_down(mask, frac=0.20)
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask  = cv2.dilate(mask, dil_k, iterations=3)
    return cv2.GaussianBlur(mask, (31, 31), 11).clip(0., 1.)

def _ellipse_mask_r(size: int):
    import cv2
    mask = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, int(size * 0.45)
    ax, ay = int(size * 0.38), int(size * 0.46)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
    k = size // 16 * 2 + 1
    return cv2.GaussianBlur(mask, (k, k), k // 3).clip(0., 1.)

def _paste_back(frame, restored_512, mask_512, M):
    import cv2
    h, w  = frame.shape[:2]
    M_inv = cv2.invertAffineTransform(M)
    rf    = cv2.warpAffine(restored_512, M_inv, (w, h), flags=cv2.INTER_LINEAR)
    mf    = cv2.warpAffine(mask_512,     M_inv, (w, h), flags=cv2.INTER_LINEAR)
    mf    = cv2.GaussianBlur(mf, (21, 21), 0).clip(0., 1.)
    alpha = mf[:, :, np.newaxis].astype(np.float32)
    return (rf.astype(np.float32) * alpha +
            frame.astype(np.float32) * (1. - alpha)).clip(0, 255).astype(np.uint8)


# ── Temporal smoothing ─────────────────────────────────────────────────────────

def _temporal_smooth(frame_dir: str) -> None:
    """Blend each frame's face bbox with its neighbours to reduce flicker.

    smoothed_roi = 0.8 * current + 0.1 * prev + 0.1 * next
    Applied only inside the detected face bounding box.
    """
    import cv2, glob
    paths = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    n = len(paths)
    if n < 3:
        return
    print(f"[smooth] temporal smoothing {n} frames...")
    for i, path in enumerate(paths):
        curr = cv2.imread(path)
        if curr is None:
            continue
        face = detect_face(curr)
        if face is None:
            continue
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(curr.shape[1], x2), min(curr.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        prev = cv2.imread(paths[max(0, i - 1)])
        nxt  = cv2.imread(paths[min(n - 1, i + 1)])
        if prev is None or nxt is None:
            continue
        roi = (
            0.8 * curr[y1:y2, x1:x2].astype(np.float32) +
            0.1 * prev[y1:y2, x1:x2].astype(np.float32) +
            0.1 * nxt [y1:y2, x1:x2].astype(np.float32)
        ).clip(0, 255).astype(np.uint8)
        curr[y1:y2, x1:x2] = roi
        cv2.imwrite(path, curr)
    print("[smooth] done")


# ── Rebuild ────────────────────────────────────────────────────────────────────

def rebuild_video(frame_dir: str, audio_path: str, output_path: str,
                  crf: int = 17, preset: str = "slow"):
    import glob, ffmpeg
    fps_file = os.path.join(frame_dir, "fps.txt")
    fps = 30.0
    if os.path.exists(fps_file):
        try: fps = float(open(fps_file).read().strip())
        except ValueError: pass

    frame_pattern = os.path.join(frame_dir, "frame_%06d.png")
    n = len(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    has_audio = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
    print(f"[rebuild] {n} frames @ {fps:.3f} fps | audio={has_audio}")

    video_in = ffmpeg.input(frame_pattern, framerate=fps, pattern_type="sequence").video
    out_args  = dict(vcodec="libx264", crf=crf, preset=preset,
                     pix_fmt="yuv420p", movflags="+faststart")

    if has_audio:
        (ffmpeg.output(video_in, ffmpeg.input(audio_path).audio, output_path,
                       acodec="aac", audio_bitrate="192k", ac=2,
                       shortest=None, **out_args)
         .overwrite_output().run(quiet=True))
    else:
        (ffmpeg.output(video_in, output_path, **out_args)
         .overwrite_output().run(quiet=True))

    print(f"[rebuild] {output_path} ({os.path.getsize(output_path)/1e6:.1f} MB)")


# ── Upload ─────────────────────────────────────────────────────────────────────

def upload_to_supabase(file_path: str, supabase_url: str,
                       supabase_key: str, bucket: str) -> str:
    import uuid
    from supabase import create_client
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    client     = create_client(supabase_url, supabase_key)
    ext        = os.path.splitext(file_path)[1] or ".mp4"
    obj_path   = f"outputs/{uuid.uuid4()}{ext}"
    size_mb    = os.path.getsize(file_path) / 1e6
    print(f"[storage] uploading {size_mb:.1f} MB → {bucket}/{obj_path}")
    ctype = {".mp4":"video/mp4",".mov":"video/quicktime"}.get(ext.lower(),"application/octet-stream")
    with open(file_path, "rb") as f:
        client.storage.from_(bucket).upload(
            path=obj_path, file=f, file_options={"content-type": ctype})
    url = client.storage.from_(bucket).get_public_url(obj_path)
    print(f"[storage] public URL: {url}")
    return url


# ══════════════════════════════════════════════════════════════════════════════
# MODAL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    gpu="A100",
    volumes={WEIGHTS_DIR: weights_volume},
    timeout=600,
    memory=32768,
)
def run_face_swap(source_image_bytes: bytes, target_video_bytes: bytes,
                  supabase_url: str, supabase_key: str) -> str:
    import shutil, tempfile, time, cv2

    job_start = time.time()
    work_dir  = tempfile.mkdtemp(prefix="faceswap_")

    def _e(): return f"{time.time()-job_start:.1f}s"

    try:
        src_img_path = os.path.join(work_dir, "source.jpg")
        tgt_vid_path = os.path.join(work_dir, "target.mp4")
        output_path  = os.path.join(work_dir, "output.mp4")
        frames_dir   = os.path.join(work_dir, "frames")
        output_dir   = os.path.join(work_dir, "processed")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        with open(src_img_path, "wb") as f: f.write(source_image_bytes)
        with open(tgt_vid_path, "wb") as f: f.write(target_video_bytes)
        print(f"[{_e()}] step 1/7 — inputs written")

        print(f"[{_e()}] step 2/7 — extracting frames + audio...")
        frame_paths, audio_path = extract_frames(tgt_vid_path, frames_dir)
        total = len(frame_paths)
        print(f"[{_e()}] extracted {total} frames")

        print(f"[{_e()}] step 3/7 — loading models...")
        hyperswap_model, bisenet_model = load_models(WEIGHTS_DIR)
        codeformer_net                 = load_codeformer(WEIGHTS_DIR)
        print(f"[{_e()}] models loaded")

        print(f"[{_e()}] step 4/7 — extracting source identity...")
        identity_embedding = extract_identity(src_img_path, ARCFACE_PATH)
        print(f"[{_e()}] embedding ready shape={identity_embedding.shape}")

        print(f"[{_e()}] step 5/7 — processing {total} frames...")
        skipped = 0
        for i, fp in enumerate(frame_paths):
            frame = cv2.imread(fp)
            if frame is None:
                skipped += 1
                continue
            swapped  = swap_frame(frame, identity_embedding, hyperswap_model, bisenet_model)
            restored = restore_frame(swapped, frame, bisenet_model, codeformer_net)
            cv2.imwrite(os.path.join(output_dir, os.path.basename(fp)), restored)
            if (i + 1) % 50 == 0 or (i + 1) == total:
                print(f"[{_e()}] frames {i+1}/{total} ({skipped} skipped)")
        print(f"[{_e()}] step 5/7 complete")

        print(f"[{_e()}] step 5b/7 — temporal smoothing...")
        _temporal_smooth(output_dir)
        print(f"[{_e()}] temporal smoothing complete")

        print(f"[{_e()}] step 6/7 — rebuilding video...")
        rebuild_video(output_dir, audio_path, output_path)

        print(f"[{_e()}] step 7/7 — uploading to Supabase...")
        bucket = os.environ.get("SUPABASE_BUCKET", "faceswap-uploads")
        url    = upload_to_supabase(output_path, supabase_url, supabase_key, bucket)
        print(f"[{_e()}] done → {url}")
        return url

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"[{_e()}] /tmp cleaned up")


# ── Web endpoint ───────────────────────────────────────────────────────────────

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

fastapi_app = FastAPI()

@fastapi_app.post("/")
async def api_swap(request: Request):
    expected_token = os.environ.get("API_TOKEN", "")
    auth_header    = request.headers.get("Authorization", "")

    if not expected_token:
        return JSONResponse({"status":"error","message":"API_TOKEN not configured"}, status_code=500)
    if not auth_header.startswith("Bearer ") or auth_header[7:] != expected_token:
        return JSONResponse({"status":"error","message":"Unauthorised"}, status_code=401)

    try:
        form        = await request.form()
        source_file = form.get("source_image")
        video_file  = form.get("target_video")
        if source_file is None or video_file is None:
            return JSONResponse({"status":"error","message":"source_image and target_video required"}, status_code=422)
        source_bytes = await source_file.read()
        video_bytes  = await video_file.read()
        if not source_bytes:
            return JSONResponse({"status":"error","message":"source_image is empty"}, status_code=422)
        if not video_bytes:
            return JSONResponse({"status":"error","message":"target_video is empty"}, status_code=422)
    except Exception as exc:
        return JSONResponse({"status":"error","message":f"Form parse error: {exc}"}, status_code=422)

    supabase_url = os.environ.get("SUPABASE_URL","")
    supabase_key = os.environ.get("SUPABASE_KEY","")
    if not supabase_url or not supabase_key:
        return JSONResponse({"status":"error","message":"Supabase credentials not configured"}, status_code=500)

    try:
        url = await run_face_swap.remote.aio(source_bytes, video_bytes, supabase_url, supabase_key)
        return JSONResponse({"status":"complete","url":url}, status_code=200)
    except Exception as exc:
        return JSONResponse({"status":"error","message":str(exc)}, status_code=500)


@app.function(
    secrets=[modal.Secret.from_name("supabase-secret"), modal.Secret.from_name("api-secret")],
    timeout=660,
)
@modal.asgi_app()
def api_endpoint():
    return fastapi_app
