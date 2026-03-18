"""
swap.py — Face swapping with hyperswap_1a_256.onnx
For each frame: detect target faces, swap the most prominent one to match
the source face embedding, write result to output directory.
"""
import os
import glob
import cv2
import numpy as np
import onnxruntime

from pipeline.detect import get_frame_faces, _bbox_area


def load_swapper(weights_dir: str):
    """
    Load hyperswap_1a_256.onnx via ONNX Runtime.
    """
    model_path = os.path.join(weights_dir, "hyperswap", "hyperswap_1a_256.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"hyperswap_1a_256.onnx not found at {model_path}. "
            "Run download_weights.py first."
        )

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = onnxruntime.InferenceSession(model_path, providers=providers)

    # Log input/output names for debugging
    inputs = [i.name for i in session.get_inputs()]
    outputs = [o.name for o in session.get_outputs()]
    print(f"[swap] hyperswap loaded | inputs={inputs} | outputs={outputs}")

    return session


def swap_faces_in_frames(
    analyser,
    swapper,           # onnxruntime.InferenceSession
    source_face,
    frames_dir: str,
    output_dir: str,
    swap_all_faces: bool = True,
):
    """
    Iterate over every PNG in frames_dir, swap faces, write to output_dir.
    """
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    total = len(frame_paths)
    print(f"[swap] processing {total} frames...")

    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"[swap] warning: could not read {frame_path}, skipping")
            continue

        target_faces = get_frame_faces(analyser, frame)

        if not target_faces:
            out_path = os.path.join(output_dir, os.path.basename(frame_path))
            cv2.imwrite(out_path, frame)
            continue

        if not swap_all_faces:
            target_faces = [max(target_faces, key=lambda f: _bbox_area(f.bbox))]

        result = frame.copy()
        for target_face in target_faces:
            result = _run_hyperswap(swapper, source_face, target_face, result)

        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        cv2.imwrite(out_path, result)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"[swap] {i + 1}/{total} frames done")

    print("[swap] face swap complete")


# ── HyperSwap inference ────────────────────────────────────────────────────────

def _run_hyperswap(
    session: "onnxruntime.InferenceSession",
    source_face,
    target_face,
    frame: np.ndarray,
) -> np.ndarray:
    """
    Run a single HyperSwap inference pass.

    HyperSwap_1a expects:
      - source_embedding : (1, 512)  float32  — ArcFace embedding of source face
      - target_face      : (1, 3, 256, 256) float32  — cropped+aligned target face

    Returns the full frame with the swapped face pasted back.
    """
    input_names = [i.name for i in session.get_inputs()]

    # 1. Crop and align the target face region (256×256)
    aligned, M = _align_face(frame, target_face.kps, size=256)

    # 2. Preprocess: BGR→RGB, HWC→NCHW, [0,255]→[-1,1]
    blob = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
    blob = (blob / 127.5) - 1.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1,3,256,256)

    # 3. Source embedding from InsightFace (normed 512-d)
    src_emb = source_face.normed_embedding.reshape(1, -1).astype(np.float32)

    # Build feed dict — order matches model's declared inputs
    feed = {}
    for name in input_names:
        if "embed" in name.lower() or "source" in name.lower() and "face" not in name.lower():
            feed[name] = src_emb
        else:
            feed[name] = blob

    # 4. Run inference
    outputs = session.run(None, feed)
    swapped = outputs[0]  # (1,3,256,256)

    # 5. Postprocess: [-1,1]→[0,255], NCHW→HWC, RGB→BGR
    swapped = (swapped[0].transpose(1, 2, 0) + 1.0) * 127.5
    swapped = swapped.clip(0, 255).astype(np.uint8)
    swapped = cv2.cvtColor(swapped, cv2.COLOR_RGB2BGR)

    # 6. Paste swapped face back onto the full frame
    result = _paste_back(frame, swapped, M)
    return result


def _align_face(frame: np.ndarray, kps: np.ndarray, size: int = 256) -> tuple:
    """
    Affine-align the face to a canonical 256×256 crop using 5-point landmarks.
    Returns (aligned_crop, transform_matrix).
    """
    # Standard 5-point arcface template scaled to `size`
    TEMPLATE = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32) * (size / 112.0)

    M, _ = cv2.estimateAffinePartial2D(kps, TEMPLATE, method=cv2.LMEDS)
    aligned = cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR)
    return aligned, M


def _paste_back(frame: np.ndarray, swapped: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Inverse-warp the swapped face back onto the original frame using a
    soft alpha mask to blend edges.
    """
    h, w = frame.shape[:2]

    # Inverse transform
    M_inv = cv2.invertAffineTransform(M)
    swapped_full = cv2.warpAffine(swapped, M_inv, (w, h), flags=cv2.INTER_LINEAR)

    # Build a soft elliptical mask in face-crop space, then inverse-warp it too
    mask_crop = np.zeros((swapped.shape[0], swapped.shape[1]), dtype=np.float32)
    cx, cy = swapped.shape[1] // 2, swapped.shape[0] // 2
    cv2.ellipse(mask_crop, (cx, cy), (cx - 8, cy - 8), 0, 0, 360, 1.0, -1)
    mask_crop = cv2.GaussianBlur(mask_crop, (21, 21), 11)

    mask_full = cv2.warpAffine(mask_crop, M_inv, (w, h), flags=cv2.INTER_LINEAR)
    mask_full = mask_full[:, :, np.newaxis]  # (H,W,1)

    result = (swapped_full.astype(np.float32) * mask_full +
              frame.astype(np.float32) * (1.0 - mask_full))
    return result.clip(0, 255).astype(np.uint8)
