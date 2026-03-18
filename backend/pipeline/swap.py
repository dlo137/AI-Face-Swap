"""
swap.py — HyperSwap core swap function with BiSeNet face masking

Public API:
  load_models(weights_dir)                          → (hyperswap_session, bisenet_session)
  swap_frame(frame, identity_embedding,
             hyperswap_model, bisenet_model)        → np.ndarray
  swap_faces_in_frames(...)                         → None   (batch driver)
"""
from __future__ import annotations

import os
import cv2
import numpy as np
import onnxruntime

from pipeline.detect import detect_face, FaceDetection

# ── BiSeNet face-region label map ─────────────────────────────────────────────
# Standard 19-class CelebAMask-HQ parsing labels
# We swap only the core face skin region; preserve hair, neck, background.
_FACE_LABELS = {1, 2, 3, 4, 5, 10, 11, 12, 13}
#  1=skin  2=l-brow  3=r-brow  4=l-eye  5=r-eye
#  10=nose  11=mouth  12=upper-lip  13=lower-lip
_PRESERVE_LABELS = {0, 14, 15, 16, 17, 18}
#  0=bg  14=neck  15=necklace  16=cloth  17=hair  18=hat

# ImageNet normalisation for BiSeNet input
_BISENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_BISENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ArcFace 5-point template at 256×256 (HyperSwap crop size)
_KPS_TEMPLATE_256 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32) * (256 / 112.0)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_models(weights_dir: str) -> tuple:
    """
    Load HyperSwap and BiSeNet ONNX sessions.

    Returns:
        (hyperswap_session, bisenet_session)
        bisenet_session may be None if the weight file is absent —
        swap_frame will fall back to an elliptical mask.
    """
    hyperswap_path = os.path.join(weights_dir, "hyperswap", "hyperswap_1a_256.onnx")
    bisenet_path   = os.path.join(weights_dir, "bisenet",   "bisenet_resnet_34.onnx")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if not os.path.exists(hyperswap_path):
        raise FileNotFoundError(f"HyperSwap model not found: {hyperswap_path}")

    hyperswap = onnxruntime.InferenceSession(hyperswap_path, providers=providers)
    _log_io(hyperswap, "hyperswap")

    bisenet = None
    if os.path.exists(bisenet_path):
        bisenet = onnxruntime.InferenceSession(bisenet_path, providers=providers)
        _log_io(bisenet, "bisenet")
    else:
        print("[swap] BiSeNet ONNX not found — will use elliptical mask fallback")

    return hyperswap, bisenet


# Kept for backwards-compat with main.py which calls load_swapper()
def load_swapper(weights_dir: str):
    hyperswap, _ = load_models(weights_dir)
    return hyperswap


# ── Public: swap_frame ─────────────────────────────────────────────────────────

def swap_frame(
    frame: np.ndarray,
    identity_embedding: np.ndarray,
    hyperswap_model: onnxruntime.InferenceSession,
    bisenet_model: onnxruntime.InferenceSession | None,
) -> np.ndarray:
    """
    Replace the face in `frame` with the identity described by `identity_embedding`.

    Strategy:
      1. detect_face()  — if None, return original frame unchanged (Strategy B)
      2. Align face to 256×256 for HyperSwap
      3. Run HyperSwap → swapped 256×256 crop
      4. Build a face-region mask via BiSeNet (or ellipse fallback)
      5. Inverse-warp swapped crop back to frame coordinates
      6. Alpha-blend using the mask — hair/neck/background untouched

    Args:
        frame:               BGR uint8 ndarray, any resolution.
        identity_embedding:  (512,) float32 from extract_identity().
        hyperswap_model:     Loaded onnxruntime.InferenceSession.
        bisenet_model:       Loaded onnxruntime.InferenceSession, or None.

    Returns:
        BGR uint8 ndarray, same shape as `frame`.
    """
    # ── Step 1: detect ─────────────────────────────────────────────────────────
    face = detect_face(frame)
    if face is None:
        return frame                        # Strategy B: no face → pass through

    # ── Step 2: align to 256×256 ───────────────────────────────────────────────
    aligned, M = _align_256(frame, face.landmarks)

    # ── Step 3: HyperSwap inference ────────────────────────────────────────────
    swapped_crop = _run_hyperswap(hyperswap_model, identity_embedding, aligned)

    # ── Step 4: build face mask ────────────────────────────────────────────────
    if bisenet_model is not None:
        mask_256 = _bisenet_face_mask(bisenet_model, aligned)
    else:
        mask_256 = _ellipse_mask(size=256)

    # ── Step 5: inverse-warp both swapped crop and mask back to frame space ────
    h, w = frame.shape[:2]
    M_inv = cv2.invertAffineTransform(M)

    swapped_full = cv2.warpAffine(
        swapped_crop, M_inv, (w, h), flags=cv2.INTER_LINEAR
    )
    mask_full = cv2.warpAffine(
        mask_256, M_inv, (w, h), flags=cv2.INTER_LINEAR
    )

    # ── Step 6: alpha blend — face region only ─────────────────────────────────
    alpha = mask_full[:, :, np.newaxis].astype(np.float32)
    result = (
        swapped_full.astype(np.float32) * alpha +
        frame.astype(np.float32) * (1.0 - alpha)
    ).clip(0, 255).astype(np.uint8)

    return result


# ── Batch driver ───────────────────────────────────────────────────────────────

def swap_faces_in_frames(
    swapper: onnxruntime.InferenceSession,
    identity_embedding: np.ndarray,
    frame_paths: list[str],
    output_dir: str,
    bisenet_model: onnxruntime.InferenceSession | None = None,
) -> None:
    """
    Run swap_frame() over a sorted list of frame paths and write results.
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(frame_paths)
    print(f"[swap] processing {total} frames...")

    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"[swap] warning: unreadable frame {frame_path}, skipping")
            continue

        result = swap_frame(frame, identity_embedding, swapper, bisenet_model)

        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        cv2.imwrite(out_path, result)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"[swap] {i + 1}/{total} done")

    print("[swap] complete")


# ── HyperSwap inference ────────────────────────────────────────────────────────

def _run_hyperswap(
    session: onnxruntime.InferenceSession,
    identity_embedding: np.ndarray,   # (512,)
    aligned_256: np.ndarray,          # BGR 256×256
) -> np.ndarray:
    """
    Run a single HyperSwap_1a_256 pass.

    Input layout (inferred from model metadata):
      - embedding input : (1, 512)          — source ArcFace embedding
      - image input     : (1, 3, 256, 256)  — target face crop, [-1, 1]
    Output:
      - (1, 3, 256, 256) — swapped face, [-1, 1]

    Returns BGR uint8 256×256.
    """
    src_emb = identity_embedding.reshape(1, -1).astype(np.float32)

    # BGR → RGB, HWC → NCHW, [0,255] → [-1,1]
    rgb = cv2.cvtColor(aligned_256, cv2.COLOR_BGR2RGB).astype(np.float32)
    blob = ((rgb / 127.5) - 1.0).transpose(2, 0, 1)[np.newaxis]  # (1,3,256,256)

    # Route inputs by name — HyperSwap uses "source" / "target" or similar
    feed = _build_feed(session, src_emb, blob)
    output = session.run(None, feed)[0]   # (1, 3, 256, 256)

    # [-1,1] → [0,255], NCHW → HWC, RGB → BGR
    out = (output[0].transpose(1, 2, 0) + 1.0) * 127.5
    return cv2.cvtColor(out.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _build_feed(
    session: onnxruntime.InferenceSession,
    embedding: np.ndarray,   # (1, 512)
    image_blob: np.ndarray,  # (1, 3, 256, 256)
) -> dict:
    """
    Map inputs to the session's declared input names by shape heuristic.
    Logs a warning if the heuristic is uncertain.
    """
    feed = {}
    for inp in session.get_inputs():
        shape = inp.shape
        # A (1,512) or (1,×) 2-D tensor → embedding
        if len(shape) == 2:
            feed[inp.name] = embedding
        # A (1,3,H,W) 4-D tensor → image crop
        elif len(shape) == 4:
            feed[inp.name] = image_blob
        else:
            print(f"[swap] warning: unexpected input shape {shape} for '{inp.name}'")
    return feed


# ── BiSeNet masking ────────────────────────────────────────────────────────────

def _bisenet_face_mask(
    session: onnxruntime.InferenceSession,
    face_crop_256: np.ndarray,
) -> np.ndarray:
    """
    Run BiSeNet on the 256×256 face crop and return a soft float32 mask
    of shape (256, 256) where 1.0 = swap this pixel, 0.0 = preserve.

    Only labels in _FACE_LABELS are included; hair, neck, background
    are excluded so they are never overwritten by the swapped face.
    """
    # Resize crop to BiSeNet's expected 512×512 input
    resized = cv2.resize(face_crop_256, (512, 512), interpolation=cv2.INTER_LINEAR)

    # BGR → RGB, normalise with ImageNet stats, NCHW
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = ((rgb - _BISENET_MEAN) / _BISENET_STD).transpose(2, 0, 1)[np.newaxis]

    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: blob})[0]  # (1, 19, 512, 512)

    labels = logits[0].argmax(axis=0)   # (512, 512) int label map

    # Binary mask: 1 where we want to swap, 0 where we preserve
    face_mask_512 = np.zeros_like(labels, dtype=np.float32)
    for lbl in _FACE_LABELS:
        face_mask_512[labels == lbl] = 1.0

    # Shrink back to 256×256
    face_mask_256 = cv2.resize(face_mask_512, (256, 256), interpolation=cv2.INTER_LINEAR)

    # Feather edges with a Gaussian blur so the blend is seamless
    face_mask_256 = cv2.GaussianBlur(face_mask_256, (15, 15), 7)

    return face_mask_256.clip(0.0, 1.0)


def _ellipse_mask(size: int = 256) -> np.ndarray:
    """
    Soft elliptical fallback mask — used when BiSeNet ONNX is unavailable.
    Returns float32 (size, size) in [0, 1].
    """
    mask = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    # Slightly inset ellipse so hard boundary never shows
    cv2.ellipse(mask, (cx, cy), (cx - 12, cy - 10), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(mask, (31, 31), 15)


# ── Alignment ─────────────────────────────────────────────────────────────────

def _align_256(frame: np.ndarray, kps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Affine-align face to 256×256 canonical space using 5 landmarks.
    Returns (aligned_crop, transform_matrix M).
    """
    M, _ = cv2.estimateAffinePartial2D(kps, _KPS_TEMPLATE_256, method=cv2.LMEDS)
    aligned = cv2.warpAffine(frame, M, (256, 256), flags=cv2.INTER_LINEAR)
    return aligned, M


# ── Misc ───────────────────────────────────────────────────────────────────────

def _log_io(session: onnxruntime.InferenceSession, name: str) -> None:
    ins  = [(i.name, i.shape) for i in session.get_inputs()]
    outs = [(o.name, o.shape) for o in session.get_outputs()]
    print(f"[swap] {name} loaded | inputs={ins} | outputs={outs}")
