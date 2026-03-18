"""
detect.py — Face detection (InsightFace buffalo_l) + identity embedding (ArcFace ONNX)

Two public functions:
  detect_face(frame)                          → FaceDetection | None
  extract_identity(image_path, arcface_path)  → np.ndarray (512,)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis

# ── Singleton analyser (loaded once per worker) ───────────────────────────────

_analyser: FaceAnalysis | None = None
_analyser_weights_dir: str = ""


def _get_analyser(weights_dir: str) -> FaceAnalysis:
    global _analyser, _analyser_weights_dir
    if _analyser is None or _analyser_weights_dir != weights_dir:
        models_root = os.path.join(weights_dir, "insightface")
        _analyser = FaceAnalysis(
            name="buffalo_l",
            root=models_root,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _analyser.prepare(ctx_id=0, det_size=(640, 640))
        _analyser_weights_dir = weights_dir
        print("[detect] buffalo_l loaded")
    return _analyser


# ── Return type ────────────────────────────────────────────────────────────────

@dataclass
class FaceDetection:
    bbox: np.ndarray        # (4,) float32  [x1, y1, x2, y2]
    landmarks: np.ndarray   # (5, 2) float32  5-point kps
    confidence: float       # det_score in [0, 1]
    _raw: object = None     # underlying InsightFace face object (internal use)


# ── Public: detect_face ────────────────────────────────────────────────────────

def detect_face(
    frame: np.ndarray,
    weights_dir: str = "/weights",
    min_confidence: float = 0.6,
) -> FaceDetection | None:
    """
    Detect the most prominent face in a BGR frame.

    Args:
        frame:          BGR numpy array (H, W, 3).
        weights_dir:    Root of the Modal weights volume.
        min_confidence: Detections below this score are discarded.

    Returns:
        FaceDetection with bbox, landmarks, and confidence,
        or None if no face passes the confidence threshold.
    """
    analyser = _get_analyser(weights_dir)
    faces = analyser.get(frame)

    if not faces:
        return None

    # Pick the largest face by bounding-box area
    best = max(faces, key=lambda f: _bbox_area(f.bbox))

    if float(best.det_score) < min_confidence:
        return None

    return FaceDetection(
        bbox=np.array(best.bbox, dtype=np.float32),
        landmarks=np.array(best.kps, dtype=np.float32),
        confidence=float(best.det_score),
        _raw=best,
    )


# ── Public: extract_identity ───────────────────────────────────────────────────

def extract_identity(image_path: str, arcface_model_path: str) -> np.ndarray:
    """
    Load a source image once and return a normalised ArcFace identity embedding.

    Run this once per job (not per frame) — the returned vector is reused
    for every frame in swap.py.

    Args:
        image_path:         Path to the source face image (JPEG/PNG).
        arcface_model_path: Path to the ArcFace ONNX model.
                            Default weight: /weights/insightface/models/buffalo_l/w600k_r50.onnx
                            (downloaded as part of buffalo_l; functionally equivalent
                             to arcface_r100 for 512-d identity embedding)

    Returns:
        Unit-normalised embedding vector, shape (512,), dtype float32.

    Raises:
        ValueError: if no face is detected in the source image.
        FileNotFoundError: if the image or model file does not exist.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Source image not found: {image_path}")
    if not os.path.exists(arcface_model_path):
        raise FileNotFoundError(f"ArcFace model not found: {arcface_model_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not decode image: {image_path}")

    # Detect face to get 5-point landmarks for alignment
    face = detect_face(frame)
    if face is None:
        raise ValueError(f"No face detected (conf < 0.6) in source image: {image_path}")

    # Align to 112×112 ArcFace canonical space
    aligned = _align_112(frame, face.landmarks)

    # Run ArcFace ONNX
    session = onnxruntime.InferenceSession(
        arcface_model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    blob = _preprocess_arcface(aligned)             # (1, 3, 112, 112)
    raw_embedding = session.run(None, {input_name: blob})[0]  # (1, 512)

    embedding = _l2_norm(raw_embedding[0])          # (512,)
    print(f"[detect] identity embedding extracted | norm={np.linalg.norm(embedding):.4f}")
    return embedding


# ── Private helpers ────────────────────────────────────────────────────────────

# Standard 5-point ArcFace template at 112×112
_ARCFACE_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _align_112(frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Affine-warp face to canonical 112×112 using 5 landmarks."""
    M, _ = cv2.estimateAffinePartial2D(kps, _ARCFACE_TEMPLATE_112, method=cv2.LMEDS)
    return cv2.warpAffine(frame, M, (112, 112), flags=cv2.INTER_LINEAR)


def _preprocess_arcface(face_112: np.ndarray) -> np.ndarray:
    """BGR 112×112 → float32 NCHW blob normalised to [-1, 1]."""
    rgb = cv2.cvtColor(face_112, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - 127.5) / 128.0            # ArcFace standard normalisation
    return rgb.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 112, 112)


def _l2_norm(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)


def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
