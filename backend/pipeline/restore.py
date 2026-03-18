"""
restore.py — CodeFormer face restoration + histogram colour matching
Enhances each swapped frame to fix artefacts and blending issues.
Colour correction matches the swapped face region back to the original
frame's colour distribution to avoid hue drift.
"""
import os
import glob
import sys
import cv2
import numpy as np
import torch


# ── CodeFormer setup ──────────────────────────────────────────────────────────

CODEFORMER_DIR = "/opt/CodeFormer"


def _ensure_codeformer_path():
    if CODEFORMER_DIR not in sys.path:
        sys.path.insert(0, CODEFORMER_DIR)


def restore_frames(
    input_dir: str,
    output_dir: str,
    weights_dir: str,
    fidelity_weight: float = 0.7,
):
    """
    Run CodeFormer on every PNG in input_dir and write results to output_dir.

    Args:
        input_dir:        Directory of swapped frames.
        output_dir:       Where to write restored frames.
        weights_dir:      Path to the Modal weights volume.
        fidelity_weight:  CodeFormer w parameter [0=quality, 1=fidelity].
                          0.7 gives a good quality/identity balance.
    """
    _ensure_codeformer_path()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = _load_codeformer(weights_dir, device)

    frame_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    total = len(frame_paths)
    print(f"[restore] restoring {total} frames (fidelity={fidelity_weight})...")

    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"[restore] warning: could not read {frame_path}, skipping")
            continue

        restored = _restore_single(net, frame, device, fidelity_weight)

        # Histogram match the face regions back to original colours
        restored = _histogram_match_faces(frame, restored)

        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        cv2.imwrite(out_path, restored)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"[restore] {i + 1}/{total} frames done")

    print("[restore] restoration complete")


# ── Private helpers ───────────────────────────────────────────────────────────

def _load_codeformer(weights_dir: str, device: torch.device):
    _ensure_codeformer_path()
    from basicsr.utils.registry import ARCH_REGISTRY  # noqa: F401 — registers archs

    # Import after path is set
    from basicsr.archs.rrdbnet_arch import RRDBNet  # kept for reference
    import importlib
    codeformer_arch = importlib.import_module("basicsr.archs.codeformer_arch")

    net = codeformer_arch.CodeFormer(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)

    ckpt_path = os.path.join(weights_dir, "codeformer", "codeformer.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"CodeFormer checkpoint not found: {ckpt_path}. "
            "Run `modal run main.py::download_weights` first."
        )

    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint["params_ema"])
    net.eval()
    print("[restore] CodeFormer loaded")
    return net


def _restore_single(net, frame: np.ndarray, device, w: float) -> np.ndarray:
    """Run CodeFormer on a single BGR frame. Returns BGR frame."""
    from torchvision.transforms.functional import normalize

    # BGR→RGB, HWC→CHW, [0,255]→[0,1]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # Normalise to [-1, 1]
    normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)

    with torch.no_grad():
        output = net(tensor, w=w, adain=True)[0]

    # Back to [0,255] BGR
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = (output * 0.5 + 0.5).clip(0, 1)
    output = (output * 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # Resize to original dims if CodeFormer changed them
    if output.shape[:2] != frame.shape[:2]:
        output = cv2.resize(output, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    return output


def _histogram_match_faces(original: np.ndarray, restored: np.ndarray) -> np.ndarray:
    """
    Match the colour histogram of `restored` to `original` using the LAB
    colour space. This corrects any hue/saturation drift introduced by the
    face swap or CodeFormer while keeping the restored detail.
    """
    original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    restored_lab = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Match each channel independently
    result_lab = restored_lab.copy()
    for c in range(3):
        orig_mean, orig_std = original_lab[:, :, c].mean(), original_lab[:, :, c].std() + 1e-6
        rest_mean, rest_std = restored_lab[:, :, c].mean(), restored_lab[:, :, c].std() + 1e-6
        result_lab[:, :, c] = (restored_lab[:, :, c] - rest_mean) * (orig_std / rest_std) + orig_mean

    result_lab = result_lab.clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
