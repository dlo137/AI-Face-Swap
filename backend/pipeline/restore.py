"""
restore.py — Post-processing: lighting correction + CodeFormer restoration

Public API:
  load_codeformer(weights_dir)                    → nn.Module
  restore_frame(swapped_frame, original_frame,
                bisenet_model, codeformer_net)    → np.ndarray
  restore_frames(...)                             → None   (batch driver)
"""
from __future__ import annotations

import os
import sys
import glob
import cv2
import numpy as np
import torch

import onnxruntime

from pipeline.detect import detect_face, FaceDetection

CODEFORMER_DIR = "/opt/CodeFormer"
CODEFORMER_FIDELITY = 0.6   # w=0 → max quality/hallucination, w=1 → max fidelity

# ArcFace 5-pt template scaled to 512 for CodeFormer input
_KPS_TEMPLATE_512 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32) * (512 / 112.0)

# BiSeNet face labels (same set as swap.py)
_FACE_LABELS = {1, 2, 3, 4, 5, 10, 11, 12, 13}

# ImageNet normalisation for BiSeNet
_BISENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_BISENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_codeformer(weights_dir: str) -> torch.nn.Module:
    """
    Load the CodeFormer network from the persistent weights volume.

    Returns:
        CodeFormer model in eval mode on CUDA (or CPU if unavailable).

    Raises:
        FileNotFoundError: if the checkpoint is missing.
    """
    _ensure_codeformer_path()
    import importlib

    device = _device()
    cf_arch = importlib.import_module("basicsr.archs.codeformer_arch")

    net = cf_arch.CodeFormer(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)

    ckpt_path = os.path.join(weights_dir, "codeformer", "codeformer.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"CodeFormer checkpoint missing: {ckpt_path}\n"
            "Run download_weights.py first."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt["params_ema"])
    net.eval()
    print(f"[restore] CodeFormer loaded (device={device})")
    return net


# ── Public: restore_frame ──────────────────────────────────────────────────────

def restore_frame(
    swapped_frame: np.ndarray,
    original_frame: np.ndarray,
    bisenet_model: onnxruntime.InferenceSession | None = None,
    codeformer_net: torch.nn.Module | None = None,
) -> np.ndarray:
    """
    Post-process a face-swapped frame:

      1. Detect the face region in the swapped frame.
      2. Histogram-match the swapped face's LAB lighting to the original scene.
      3. Align the colour-corrected face to 512×512 and run CodeFormer (w=0.6)
         to recover skin texture and fix swap artefacts.
      4. Paste the restored face back using a soft BiSeNet (or elliptical)
         boundary mask for seamless blending.

    Args:
        swapped_frame:   BGR uint8 — output of swap_frame().
        original_frame:  BGR uint8 — the unmodified source video frame.
        bisenet_model:   Optional pre-loaded BiSeNet ONNX session.
        codeformer_net:  Optional pre-loaded CodeFormer torch.nn.Module.

    Returns:
        BGR uint8 ndarray, same shape as swapped_frame.
    """
    # ── 1. Detect face ─────────────────────────────────────────────────────────
    face = detect_face(swapped_frame)
    if face is None:
        return swapped_frame            # no face detected — return as-is

    # ── 2. Histogram match: align swapped face lighting to original scene ───────
    colour_corrected = _match_lighting(swapped_frame, original_frame, face)

    # ── 3. Align to 512×512 for CodeFormer ────────────────────────────────────
    aligned_512, M = _align_512(colour_corrected, face.landmarks)

    # ── 3b. Run CodeFormer ─────────────────────────────────────────────────────
    if codeformer_net is not None:
        restored_512 = _run_codeformer(codeformer_net, aligned_512)
    else:
        restored_512 = aligned_512      # skip if model not loaded

    # ── 4. Build blending mask ─────────────────────────────────────────────────
    if bisenet_model is not None:
        mask_512 = _bisenet_mask_512(bisenet_model, aligned_512)
    else:
        mask_512 = _ellipse_mask(512)

    # ── 5. Paste restored face back into the full frame ────────────────────────
    result = _paste_back(swapped_frame, restored_512, mask_512, M)
    return result


# ── Batch driver ───────────────────────────────────────────────────────────────

def restore_frames(
    swapped_dir: str,
    original_dir: str,
    output_dir: str,
    bisenet_model: onnxruntime.InferenceSession | None = None,
    codeformer_net: torch.nn.Module | None = None,
) -> None:
    """
    Run restore_frame() over every PNG in swapped_dir.
    original_dir must contain matching filenames (same frame names from extract.py).
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(swapped_dir, "*.png")))
    total = len(paths)
    print(f"[restore] restoring {total} frames (CodeFormer w={CODEFORMER_FIDELITY})...")

    for i, swapped_path in enumerate(paths):
        fname = os.path.basename(swapped_path)
        original_path = os.path.join(original_dir, fname)

        swapped = cv2.imread(swapped_path)
        original = cv2.imread(original_path) if os.path.exists(original_path) else swapped

        if swapped is None:
            print(f"[restore] warning: unreadable {swapped_path}, skipping")
            continue

        result = restore_frame(swapped, original, bisenet_model, codeformer_net)

        cv2.imwrite(os.path.join(output_dir, fname), result)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"[restore] {i + 1}/{total} done")

    print("[restore] complete")


# ── Step 2: Histogram / lighting match ────────────────────────────────────────

def _match_lighting(
    swapped: np.ndarray,
    original: np.ndarray,
    face: FaceDetection,
) -> np.ndarray:
    """
    Per-channel LAB mean/std transfer: align the face region of `swapped`
    to the lighting statistics of the same region in `original`.

    Only the face bounding-box region is sampled for statistics — this
    prevents the global scene (e.g. bright sky) from pulling the face colour.
    """
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(swapped.shape[1], x2), min(swapped.shape[0], y2)

    # Work in LAB
    swapped_lab  = cv2.cvtColor(swapped,  cv2.COLOR_BGR2LAB).astype(np.float32)
    original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)

    result_lab = swapped_lab.copy()

    for c in range(3):
        src_patch = swapped_lab [y1:y2, x1:x2, c]
        ref_patch = original_lab[y1:y2, x1:x2, c]

        src_mean, src_std = src_patch.mean(), src_patch.std() + 1e-6
        ref_mean, ref_std = ref_patch.mean(), ref_patch.std() + 1e-6

        # Shift only the face bbox region, leave the rest of the frame intact
        corrected = (swapped_lab[y1:y2, x1:x2, c] - src_mean) * (ref_std / src_std) + ref_mean
        result_lab[y1:y2, x1:x2, c] = corrected

    result_lab = result_lab.clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


# ── Step 3: CodeFormer inference ───────────────────────────────────────────────

def _run_codeformer(net: torch.nn.Module, face_512: np.ndarray) -> np.ndarray:
    """
    Run CodeFormer on a 512×512 face crop.

    Args:
        net:      CodeFormer in eval mode.
        face_512: BGR uint8 (512, 512, 3).

    Returns:
        BGR uint8 (512, 512, 3).
    """
    from torchvision.transforms.functional import normalize as tv_normalize

    device = next(net.parameters()).device

    # BGR → RGB, HWC → CHW, [0,255] → [0,1]
    rgb = cv2.cvtColor(face_512, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    # Normalise to [-1, 1] (CodeFormer training convention)
    tv_normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)

    with torch.no_grad():
        output = net(tensor, w=CODEFORMER_FIDELITY, adain=True)[0]  # (1, 3, 512, 512)

    # [-1,1] → [0,1] → [0,255], CHW → HWC, RGB → BGR
    out_np = output.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    out_np = (out_np * 0.5 + 0.5).clip(0, 1)
    out_np = (out_np * 255).astype(np.uint8)
    return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)


# ── Step 4: BiSeNet mask at 512 ───────────────────────────────────────────────

def _bisenet_mask_512(
    session: onnxruntime.InferenceSession,
    face_512: np.ndarray,
) -> np.ndarray:
    """
    Run BiSeNet on the 512×512 crop; return a soft float32 face mask (512,512).
    Hair, neck, background are excluded — only skin/facial features are 1.0.
    """
    rgb = cv2.cvtColor(face_512, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = ((rgb - _BISENET_MEAN) / _BISENET_STD).transpose(2, 0, 1)[np.newaxis]

    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: blob})[0]   # (1, 19, 512, 512)

    labels = logits[0].argmax(axis=0)   # (512, 512)

    mask = np.zeros_like(labels, dtype=np.float32)
    for lbl in _FACE_LABELS:
        mask[labels == lbl] = 1.0

    # Use a larger kernel here (σ=11) to widen the feather zone at 512 res
    return cv2.GaussianBlur(mask, (31, 31), 11).clip(0.0, 1.0)


def _ellipse_mask(size: int) -> np.ndarray:
    """Soft elliptical fallback mask, float32 (size, size) in [0,1]."""
    mask = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, int(size * 0.45)
    ax, ay = int(size * 0.38), int(size * 0.46)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
    k = size // 16 * 2 + 1      # always odd
    return cv2.GaussianBlur(mask, (k, k), k // 3).clip(0.0, 1.0)


# ── Step 5: paste-back ─────────────────────────────────────────────────────────

def _paste_back(
    frame: np.ndarray,
    restored_512: np.ndarray,
    mask_512: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """
    Inverse-warp the restored 512×512 crop back to full-frame coords
    and alpha-blend using the soft mask.
    """
    h, w = frame.shape[:2]
    M_inv = cv2.invertAffineTransform(M)

    restored_full = cv2.warpAffine(restored_512, M_inv, (w, h), flags=cv2.INTER_LINEAR)
    mask_full     = cv2.warpAffine(mask_512,     M_inv, (w, h), flags=cv2.INTER_LINEAR)

    alpha = mask_full[:, :, np.newaxis].astype(np.float32)
    result = (
        restored_full.astype(np.float32) * alpha +
        frame.astype(np.float32)         * (1.0 - alpha)
    ).clip(0, 255).astype(np.uint8)

    return result


# ── Alignment ──────────────────────────────────────────────────────────────────

def _align_512(frame: np.ndarray, kps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Affine-align face to 512×512 canonical space. Returns (crop, M)."""
    M, _ = cv2.estimateAffinePartial2D(kps, _KPS_TEMPLATE_512, method=cv2.LMEDS)
    aligned = cv2.warpAffine(frame, M, (512, 512), flags=cv2.INTER_LINEAR)
    return aligned, M


# ── Misc ───────────────────────────────────────────────────────────────────────

def _ensure_codeformer_path():
    if CODEFORMER_DIR not in sys.path:
        sys.path.insert(0, CODEFORMER_DIR)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
