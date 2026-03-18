"""
detect.py — InsightFace face detection and analysis
Loads the buffalo_l model from the weights volume and returns the best
(largest, highest-confidence) face found in an image.
"""
import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis


def load_face_analyser(weights_dir: str) -> FaceAnalysis:
    """
    Load buffalo_l from the persistent weights volume.
    InsightFace looks for models at: <root>/models/<name>/
    """
    models_root = os.path.join(weights_dir, "insightface")
    analyser = FaceAnalysis(
        name="buffalo_l",
        root=models_root,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    # det_size controls detection resolution — 640x640 is a good default
    analyser.prepare(ctx_id=0, det_size=(640, 640))
    print("[detect] buffalo_l loaded")
    return analyser


def get_source_face(analyser: FaceAnalysis, image_path: str):
    """
    Return the single best face from the source image.
    'Best' = largest bounding-box area (most prominent face).

    Raises:
        ValueError: if no face is detected in the source image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = analyser.get(img)
    if not faces:
        raise ValueError("No face detected in the source image.")

    # Pick the face with the largest bounding box
    best = max(faces, key=lambda f: _bbox_area(f.bbox))
    print(f"[detect] source face found (det_score={best.det_score:.3f})")
    return best


def get_frame_faces(analyser: FaceAnalysis, frame: np.ndarray) -> list:
    """
    Detect all faces in a single video frame.
    Returns an empty list if none are found.
    """
    return analyser.get(frame)


def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
