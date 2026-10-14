"""
test_orientation_merged.py

Single-file test harness: YOLO detection + orientation.py's logic pasted
in directly (orientation.py itself has no yolov7 dependency, so merging
costs nothing).

REQUIRED SETUP (this is why your editor shows squiggly import lines):
  `models.experimental`, `utils.datasets`, `utils.general`, `utils.torch_utils`
  are NOT pip packages -- they're files inside the yolov7 repo
  (https://github.com/WongKinYiu/yolov7). This script must live inside a
  clone of that repo (or that repo must be on PYTHONPATH), e.g.:

      git clone https://github.com/WongKinYiu/yolov7.git
      cp test_orientation_merged.py yolov7/
      cd yolov7
      pip install -r requirements.txt
      python3 test_orientation_merged.py

FOLDER LAYOUT (edit TEST_ROOT below to point at this):
  test_run/
    claim1/image.jpg
    claim2/image.jpg
    ...
  Each subfolder under TEST_ROOT is treated as one "claim" -- mirrors the
  claim_id folder structure gauge_identification_two.py expects in prod.

IMG_SIZE:
  960 matches opt_img_size=960 in gauge_identification_two.py (the
  production inference size). Keep it matching production, don't treat
  it as a free variable -- check_img_size() just rounds it to the
  nearest multiple of the model stride, it doesn't "optimize" it for you.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Newer torch defaults torch.load(weights_only=True), which breaks yolov7's
# attempt_load() since the checkpoint pickles full model classes. We trust
# this checkpoint (it's your own best.pt), so force weights_only=False.
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device


# ============================================================
# orientation.py logic, pasted in directly (v3 -- always-landscape)
# ============================================================

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
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


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
    nx = m[0, 0] * x + m[0, 1] * y + m[0, 2]
    ny = m[1, 0] * x + m[1, 1] * y + m[1, 2]
    return nx, ny


def normalize_gauge_orientation_from_boxes(raw_bgr_image, out_box, in_box, debug=False):
    ox1, oy1, ox2, oy2 = out_box
    out_crop = raw_bgr_image[oy1:oy2, ox1:ox2]
    ix1, iy1, ix2, iy2 = in_box
    in_crop = raw_bgr_image[iy1:iy2, ix1:ix2]
    screen_center_local = ((ix1 + ix2) / 2 - ox1, (iy1 + iy2) / 2 - oy1)

    mask = _segment_gauge(out_crop)
    contour = _largest_contour(mask)
    if contour is None:
        info = {"status": "segmentation_failed", "confidence": "low (segmentation failed)"}
        return out_crop, in_crop, 0.0, info

    axis_angle = _axis_angle_degrees(contour)
    rotation_amount = axis_angle

    h, w = out_crop.shape[:2]
    m, new_w, new_h = _rotation_matrix_no_crop(h, w, rotation_amount)
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
            denom = max(1.0, start_w + end_w)
            width_score = -((end_w - start_w) / denom)

    eps = 0.01
    width_usable = abs(width_score) > eps
    screen_usable = abs(screen_score) > eps

    if width_usable:
        flip_needed = width_score > 0
        if screen_usable and (screen_score > 0) == flip_needed:
            confidence = "high (width + screen signals agree)"
        else:
            confidence = "medium (used width signal)"
    elif screen_usable:
        flip_needed = screen_score > 0
        confidence = "low (width signal empty -- only weak screen signal available)"
    else:
        flip_needed = False
        confidence = "low (no usable signal -- review this one)"

    total_angle = rotation_amount + (180 if flip_needed else 0)
    corrected_out = _rotate_no_crop(out_crop, total_angle)
    corrected_in = _rotate_no_crop(in_crop, total_angle)

    info = {
        "axis_angle": axis_angle,
        "flip_applied": flip_needed,
        "confidence": confidence,
        "screen_score": screen_score,
        "width_score": width_score,
        "total_rotation": total_angle,
    }
    if debug:
        info["mask"] = mask
    return corrected_out, corrected_in, total_angle, info


def tight_crop_rotated_screen(rotated_bgr, margin=2):
    gray = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    corners = [gray[0, 0], gray[0, w - 1], gray[h - 1, 0], gray[h - 1, w - 1]]
    pad_val = int(np.median(corners))

    if pad_val > 127:
        mask = (gray < pad_val - 20).astype(np.uint8) * 255
    else:
        mask = (gray > pad_val + 20).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    coords = cv2.findNonZero(mask)
    if coords is None:
        return rotated_bgr

    rx, ry, rw, rh = cv2.boundingRect(coords)
    rx, ry = max(0, rx - margin), max(0, ry - margin)
    rw = min(w - rx, rw + 2 * margin)
    rh = min(h - ry, rh + 2 * margin)
    if rw <= 0 or rh <= 0:
        return rotated_bgr
    return rotated_bgr[ry:ry + rh, rx:rx + rw]


def enforce_landscape(bgr_image):
    h, w = bgr_image.shape[:2]
    if w >= h:
        return bgr_image
    return cv2.rotate(bgr_image, cv2.ROTATE_90_CLOCKWISE)


def normalize_screen_only(in_bgr, debug=False):
    mask = _segment_gauge(in_bgr)
    contour = _largest_contour(mask)
    if contour is None:
        info = {"status": "segmentation_failed", "confidence": "low (segmentation failed)"}
        return in_bgr, info

    axis_angle = _axis_angle_degrees(contour)
    rotated = _rotate_no_crop(in_bgr, axis_angle)

    rh, rw = rotated.shape[:2]
    forced_90 = False
    if rh > rw:
        rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
        rh, rw = rotated.shape[:2]
        forced_90 = True

    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    left_half = ink[:, :rw // 2]
    right_half = ink[:, rw // 2:]
    ink_score = (int(right_half.sum()) - int(left_half.sum())) / max(1, int(ink.sum()))

    top_half = gray[:rh // 2, :]
    bot_half = gray[rh // 2:, :]
    brightness_score = (float(bot_half.mean()) - float(top_half.mean())) / 255.0

    info = {
        "axis_angle": axis_angle,
        "forced_90_for_landscape": forced_90,
        "ink_score": ink_score,
        "brightness_score": brightness_score,
        "confidence": "low (screen-only fallback -- no device box to anchor against)",
    }
    if debug:
        info["mask"] = mask
    return rotated, info


# ============================================================
# YOLO detection + test driver
# ============================================================

WEIGHTS = Path(__file__).resolve().parent / "best.pt"  # assumes best.pt sits next to this script
TEST_ROOT = Path(r"C:\Users\harsh\Downloads\WORK\rotation\yolov7\testing")    # <-- point this at your test claim folders
IMG_SIZE = 960                               # matches opt_img_size in gauge_identification_two.py
CONF_THRES = 0.25
IOU_THRES = 0.45


def load_model():
    set_logging()
    device = select_device("")
    model = attempt_load(WEIGHTS, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(IMG_SIZE, s=stride)
    model.eval()
    return model, device, stride, imgsz


def detect_boxes(model, device, stride, imgsz, image_path):
    original = cv2.imread(str(image_path))
    dataset = LoadImages(str(image_path), img_size=imgsz, stride=stride)
    names = model.module.names if hasattr(model, "module") else model.names

    candidates = []
    for _, image, _, _ in dataset:
        tensor = torch.from_numpy(image).to(device).float()
        tensor /= 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            prediction = model(tensor, augment=False)[0]
        detections = non_max_suppression(prediction, CONF_THRES, IOU_THRES)[0]
        if detections is None or not len(detections):
            continue

        detections[:, :4] = scale_coords(tensor.shape[2:], detections[:, :4], original.shape).round()
        for *xyxy, confidence, class_id in detections:
            class_index = int(class_id)
            label_name = str(names[class_index])
            candidates.append((float(confidence), label_name, tuple(map(int, xyxy))))

    return original, candidates


def best_match(candidates, predicate):
    matches = [c for c in candidates if predicate(c[1].lower())]
    return max(matches, key=lambda item: item[0]) if matches else None


def run_for_claim(model, device, stride, imgsz, claim_dir: Path):
    image_path = claim_dir / "image.jpg"
    out_dir = claim_dir / "orientation_debug"
    out_dir.mkdir(exist_ok=True)

    print(f"\n=== {claim_dir.name} ({image_path}) ===")
    original, candidates = detect_boxes(model, device, stride, imgsz, image_path)

    if not candidates:
        print("No detections at all.")
        return

    print("All detections:")
    for conf, label, box in sorted(candidates, key=lambda c: -c[0]):
        print(f"  {label:20s} conf={conf:.3f} box={box}")

    best_in = best_match(candidates, lambda l: "in_digital" in l)
    best_out = best_match(candidates, lambda l: "out_digital" in l)

    if best_in is None:
        print("No in_digital box found -- can't test screen orientation for this image.")
        return

    _, _, (ix1, iy1, ix2, iy2) = best_in
    h, w = original.shape[:2]
    ix1, ix2 = max(0, ix1), min(w, ix2)
    iy1, iy2 = max(0, iy1), min(h, iy2)
    raw_in_crop = original[iy1:iy2, ix1:ix2]
    cv2.imwrite(str(out_dir / "01_raw_in_crop.jpg"), raw_in_crop)

    if best_out is not None:
        _, _, (ox1, oy1, ox2, oy2) = best_out
        ox1, ox2 = max(0, ox1), min(w, ox2)
        oy1, oy2 = max(0, oy1), min(h, oy2)
        raw_out_crop = original[oy1:oy2, ox1:ox2]
        cv2.imwrite(str(out_dir / "01_raw_out_crop.jpg"), raw_out_crop)

        in_box = (ix1, iy1, ix2, iy2)
        out_box = (ox1, oy1, ox2, oy2)

        corrected_out, corrected_in, total_angle, info = normalize_gauge_orientation_from_boxes(
            original, out_box, in_box
        )
        print(f"[dual-box] angle={total_angle:.1f} deg  flip={info.get('flip_applied')}  "
              f"confidence={info.get('confidence')}")
        if "screen_score" in info:
            print(f"           screen_score={info['screen_score']:.4f}  "
                  f"width_score={info['width_score']:.4f}")

        cv2.imwrite(str(out_dir / "02_corrected_in.jpg"), corrected_in)
        cv2.imwrite(str(out_dir / "02_corrected_out.jpg"), corrected_out)

        tight = tight_crop_rotated_screen(corrected_in)
        cv2.imwrite(str(out_dir / "03_tight_crop.jpg"), tight)

        final = enforce_landscape(tight)
        cv2.imwrite(str(out_dir / "04_final_landscape.jpg"), final)
        print(f"Final screen size: {final.shape[1]}x{final.shape[0]} (w x h)")
    else:
        print("No out_digital box found -- testing screen-only fallback path.")
        screen, info = normalize_screen_only(raw_in_crop)
        print(f"[screen-only] axis_angle={info.get('axis_angle', 0):.1f}  "
              f"confidence={info.get('confidence')}")
        cv2.imwrite(str(out_dir / "02_screen_only_rotated.jpg"), screen)

        tight = tight_crop_rotated_screen(screen)
        cv2.imwrite(str(out_dir / "03_tight_crop.jpg"), tight)

        final = enforce_landscape(tight)
        cv2.imwrite(str(out_dir / "04_final_landscape.jpg"), final)
        print(f"Final screen size: {final.shape[1]}x{final.shape[0]} (w x h)")

    print(f"Debug crops saved to: {out_dir}")


def main():
    model, device, stride, imgsz = load_model()
    claim_dirs = sorted([p for p in TEST_ROOT.iterdir() if p.is_dir()])
    if not claim_dirs:
        print(f"No claim folders found under {TEST_ROOT}")
        sys.exit(1)

    for claim_dir in claim_dirs:
        if (claim_dir / "image.jpg").is_file():
            run_for_claim(model, device, stride, imgsz, claim_dir)


if __name__ == "__main__":
    main()
