"""
quick_test.py — point at ONE image, get the corrected crop back. That's it.

SETUP (same as before):
  This file must sit inside your yolov7 clone (same folder as best.pt),
  because it needs models/experimental.py etc. from that repo.

EDIT THESE TWO LINES, THEN RUN:
    python3 quick_test.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ============== EDIT THESE TWO ==============
IMAGE_PATH = Path(r"C:\Users\Sumankumar.Sahu\.gemini\antigravity\scratch\project1\yolov7\yolov7\testing\guage_204ca6ce-0a82-4977-b18a-e56635a284af.jpg")
WEIGHTS = Path(__file__).resolve().parent / "best.pt"
# ==============================================

_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device


# ---------- orientation.py logic, pasted in directly ----------

def _segment_gauge(bgr_image):
    h, w = bgr_image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    margin = max(2, int(0.05 * min(h, w)))
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)
    try:
        cv2.grabCut(bgr_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        fg = np.zeros((h, w), np.uint8)
    fg_fraction = fg.sum() / 255 / (h * w)
    if fg_fraction < 0.05 or fg_fraction > 0.95:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return fg


def _largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def _axis_angle_degrees(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    edges = [(box[(i + 1) % 4] - box[i]) for i in range(4)]
    longest = max(edges, key=lambda e: np.hypot(*e))
    angle = np.degrees(np.arctan2(longest[1], longest[0]))
    while angle <= -90:
        angle += 180
    while angle > 90:
        angle -= 180
    return angle


def _rotation_matrix_no_crop(h, w, angle_degrees):
    cx, cy = w / 2, h / 2
    m = cv2.getRotationMatrix2D((cx, cy), angle_degrees, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    m[0, 2] += (new_w / 2) - cx
    m[1, 2] += (new_h / 2) - cy
    return m, new_w, new_h


def _rotate_no_crop(image, angle_degrees, border_value=(255, 255, 255)):
    h, w = image.shape[:2]
    m, new_w, new_h = _rotation_matrix_no_crop(h, w, angle_degrees)
    return cv2.warpAffine(image, m, (new_w, new_h), borderValue=border_value)


def _transform_point(pt, m):
    x, y = pt
    return (m[0, 0] * x + m[0, 1] * y + m[0, 2], m[1, 0] * x + m[1, 1] * y + m[1, 2])


def normalize_gauge_orientation_from_boxes(raw_bgr_image, out_box, in_box):
    ox1, oy1, ox2, oy2 = out_box
    out_crop = raw_bgr_image[oy1:oy2, ox1:ox2]
    ix1, iy1, ix2, iy2 = in_box
    in_crop = raw_bgr_image[iy1:iy2, ix1:ix2]
    screen_center_local = ((ix1 + ix2) / 2 - ox1, (iy1 + iy2) / 2 - oy1)

    mask = _segment_gauge(out_crop)
    contour = _largest_contour(mask)
    if contour is None:
        return out_crop, in_crop, 0.0, {"confidence": "low (segmentation failed)"}

    axis_angle = _axis_angle_degrees(contour)
    h, w = out_crop.shape[:2]
    m, new_w, new_h = _rotation_matrix_no_crop(h, w, axis_angle)
    rotated_mask = cv2.warpAffine(mask, m, (new_w, new_h))
    _, rotated_mask = cv2.threshold(rotated_mask, 127, 255, cv2.THRESH_BINARY)
    screen_rotated = _transform_point(screen_center_local, m)

    ys, xs = np.where(rotated_mask > 0)
    screen_score, width_score = 0.0, 0.0
    if len(xs):
        lo, hi = xs.min(), xs.max()
        mid, span = (lo + hi) / 2, max(1.0, (hi - lo) / 2)
        screen_score = -((screen_rotated[0] - mid) / span)
        sums = rotated_mask.sum(axis=0)
        present = np.where(sums > 0)[0]
        if len(present) >= 10:
            i0, i1 = present[0], present[-1]
            q = max(1, (i1 - i0) // 4)
            start_w, end_w = sums[i0:i0 + q].mean(), sums[i1 - q:i1].mean()
            width_score = -((end_w - start_w) / max(1.0, start_w + end_w))

    eps = 0.01
    width_usable = abs(width_score) > eps
    screen_usable = abs(screen_score) > eps
    if width_usable:
        flip_needed = width_score > 0
        confidence = "high (width + screen agree)" if screen_usable and (screen_score > 0) == flip_needed \
            else "medium (used width signal)"
    elif screen_usable:
        flip_needed, confidence = screen_score > 0, "low (only weak screen signal)"
    else:
        flip_needed, confidence = False, "low (no usable signal)"

    total_angle = axis_angle + (180 if flip_needed else 0)
    corrected_out = _rotate_no_crop(out_crop, total_angle)
    corrected_in = _rotate_no_crop(in_crop, total_angle)
    info = {"axis_angle": axis_angle, "flip_applied": flip_needed, "confidence": confidence,
            "screen_score": screen_score, "width_score": width_score, "total_rotation": total_angle}
    return corrected_out, corrected_in, total_angle, info


def tight_crop_rotated_screen(rotated_bgr, margin=2):
    gray = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    corners = [gray[0, 0], gray[0, w - 1], gray[h - 1, 0], gray[h - 1, w - 1]]
    pad_val = int(np.median(corners))
    mask = (gray < pad_val - 20).astype(np.uint8) * 255 if pad_val > 127 \
        else (gray > pad_val + 20).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    coords = cv2.findNonZero(mask)
    if coords is None:
        return rotated_bgr
    rx, ry, rw, rh = cv2.boundingRect(coords)
    rx, ry = max(0, rx - margin), max(0, ry - margin)
    rw, rh = min(w - rx, rw + 2 * margin), min(h - ry, rh + 2 * margin)
    return rotated_bgr[ry:ry + rh, rx:rx + rw] if rw > 0 and rh > 0 else rotated_bgr


def enforce_landscape(bgr_image):
    h, w = bgr_image.shape[:2]
    return bgr_image if w >= h else cv2.rotate(bgr_image, cv2.ROTATE_90_CLOCKWISE)


# ---------- run on ONE image ----------

def main():
    if not WEIGHTS.is_file():
        print(f"ERROR: weights not found at {WEIGHTS}")
        sys.exit(1)
    if not IMAGE_PATH.is_file():
        print(f"ERROR: image not found at {IMAGE_PATH}")
        sys.exit(1)

    set_logging()
    device = select_device("")
    model = attempt_load(str(WEIGHTS), map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(960, s=stride)
    names = model.module.names if hasattr(model, "module") else model.names

    original = cv2.imread(str(IMAGE_PATH))
    dataset = LoadImages(str(IMAGE_PATH), img_size=imgsz, stride=stride)

    candidates = []
    for _, image, _, _ in dataset:
        tensor = torch.from_numpy(image).to(device).float() / 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            pred = model(tensor, augment=False)[0]
        det = non_max_suppression(pred, 0.25, 0.45)[0]
        if det is None or not len(det):
            continue
        det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], original.shape).round()
        for *xyxy, conf, cls in det:
            candidates.append((float(conf), names[int(cls)], tuple(map(int, xyxy))))

    print("Detections:")
    for conf, label, box in sorted(candidates, key=lambda c: -c[0]):
        print(f"  {label:20s} conf={conf:.3f} box={box}")

    def best(pred):
        m = [c for c in candidates if pred(c[1].lower())]
        return max(m, key=lambda c: c[0]) if m else None

    best_in = best(lambda l: "in_digital" in l)
    best_out = best(lambda l: "out_digital" in l)

    out_dir = IMAGE_PATH.parent
    stem = IMAGE_PATH.stem

    if best_in is None:
        print("\nNo in_digital detected -- nothing to orient. Stopping here.")
        return

    if best_out is None:
        print("\nNo out_digital detected -- can't run the full pipeline (needs both boxes).")
        return

    in_box = best_in[2]
    out_box = best_out[2]
    corrected_out, corrected_in, angle, info = normalize_gauge_orientation_from_boxes(original, out_box, in_box)
    tight = tight_crop_rotated_screen(corrected_in)
    final = enforce_landscape(tight)

    print(f"\nangle={angle:.1f}  flip={info['flip_applied']}  confidence={info['confidence']}")
    print(f"screen_score={info['screen_score']:.4f}  width_score={info['width_score']:.4f}")
    print(f"final screen size: {final.shape[1]}x{final.shape[0]} (w x h)")

    out_path = out_dir / f"{stem}_FINAL.jpg"
    cv2.imwrite(str(out_path), final)
    cv2.imwrite(str(out_dir / f"{stem}_raw_in_crop.jpg"), original[in_box[1]:in_box[3], in_box[0]:in_box[2]])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
