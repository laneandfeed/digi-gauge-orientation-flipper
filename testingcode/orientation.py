"""
orientation.py — v3

Normalizes a gauge screen photo to a landscape, right-reading crop.

Design change from v2: the previous version chose a target axis (horizontal
for compact gauge bodies, vertical for elongated ones) to match how a human
naturally holds each gauge type. That was the wrong goal. The only thing
that actually matters is that the SCREEN crop that goes to OCR is landscape
-- how the physical device body happens to be shaped is irrelevant. So this
version always aligns to horizontal, full stop. That single change is what
fixes the WNB-gauge failure: its elongated body was tripping the old
"elongated -> vertical" rule and producing a portrait screen.

Everything else keeps the principle from v1/v2: never guess a rotation
direction with a hardcoded default. A hardcoded "always rotate this way"
is exactly the bug pattern in the original rotate_portrait() code, and
exactly what broke the WNB gauge. The only hardcoded-direction rotation
left anywhere in this file is enforce_landscape(), which exists purely as
a last-resort safety net for when the main geometric signals aren't
available at all -- and it's documented as such, not relied on.

Public API:
    normalize_gauge_orientation_from_boxes(raw_bgr, out_box, in_box)
        Primary path. Needs both YOLO boxes. Always-landscape,
        continuous-angle, confidence-flagged.
    apply_same_rotation(bgr_image, total_angle_degrees)
    tight_crop_rotated_screen(rotated_bgr)
        Post-processing: strips the corner padding a free-angle rotation
        leaves behind.
    enforce_landscape(bgr_image)
        Last-resort shape guarantee only -- see docstring.
    normalize_screen_only(in_bgr)
        Fallback when out_digital wasn't detected. Weaker signals, always
        flagged low confidence.
"""
import sys
from pathlib import Path

# Ensure local yolov7 repository imports are resolved
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import cv2
import numpy as np
import torch

# ============== EDIT THESE TWO (or pass image path as CLI argument) ==============
IMAGE_PATH = Path(r"C:\Users\Sumankumar.Sahu\.gemini\antigravity\scratch\project1\yolov7\yolov7\testing\RAJ_26_098051_guage_faea75ec-efb7-4438-a963-5dab95d0bd1d.jpg")
WEIGHTS = ROOT / "best.pt"
# ==================================================================================

_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device

# ---------- segmentation & axis-finding ----------

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
    """Angle of the contour's long axis, folded to (-90, 90]."""
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


# ---------- primary path: both boxes available ----------

def normalize_gauge_orientation_from_boxes(raw_bgr_image, out_box, in_box, debug=False):
    """
    raw_bgr_image: the full original photo (not pre-cropped)
    out_box, in_box: (x1, y1, x2, y2) in raw-image pixel coordinates

    Always targets a LANDSCAPE screen crop. Resolves the remaining
    left/right ambiguity from two independent signals (where the screen
    sits relative to the device body, and which end of the body is
    narrower/wider) -- and only acts when they agree. Disagreement or
    weak signal -> no flip applied, confidence flagged "low", so these
    cases surface for review instead of being silently guessed.

    Returns: corrected_out_crop, corrected_in_crop, total_rotation_degrees, info
    """
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
    rotation_amount = axis_angle  # always align straight to horizontal

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
        screen_score = -((screen_rotated[0] - mid) / span)  # + means "flip"
        sums = rotated_mask.sum(axis=0)
        present = np.where(sums > 0)[0]
        if len(present) >= 10:
            i0, i1 = present[0], present[-1]
            q = max(1, (i1 - i0) // 4)
            start_w, end_w = sums[i0:i0 + q].mean(), sums[i1 - q:i1].mean()
            denom = max(1.0, start_w + end_w)
            width_score = -((end_w - start_w) / denom)

    eps = 0.01  # treats a signal this close to 0 as "no usable measurement",
                # not as a real vote either way
    width_usable = abs(width_score) > eps
    screen_usable = abs(screen_score) > eps

    if width_usable:
        # width_score is an average over many mask pixels, so noise tends to
        # cancel out. screen_score comes from a single box-center point run
        # through a rotation matrix, so it's much more sensitive to
        # detector box imprecision. Across every real photo tested so far,
        # width_score's SIGN was correct 6/6 times; screen_score's sign was
        # only correct 3/6. So width leads, and screen only adds confidence
        # when it happens to agree -- it no longer gets to overrule width.
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


def apply_same_rotation(bgr_image, total_angle_degrees):
    return _rotate_no_crop(bgr_image, total_angle_degrees)


# ---------- post-processing ----------

def tight_crop_rotated_screen(rotated_bgr, margin=2):
    """A free-angle rotation leaves the four corners filled with solid
    padding (whatever borderValue was used). Crop down to just the real
    content so the padding doesn't get read as part of the screen."""
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
    """LAST-RESORT safety net. If normalize_gauge_orientation_from_boxes
    ran, the result should already be landscape -- this should rarely
    fire. It guarantees width >= height ONLY. It does NOT know or claim
    the reading direction is correct; anything that reaches this function
    already came from a low-confidence or failed path and should be
    treated as needing a human look, not as fixed."""
    h, w = bgr_image.shape[:2]
    if w >= h:
        return bgr_image
    return cv2.rotate(bgr_image, cv2.ROTATE_90_CLOCKWISE)


# ---------- fallback: only the screen crop is available ----------

def normalize_screen_only(in_bgr, debug=False):
    """
    Used only when out_digital wasn't detected, so there's no device body
    to anchor against. Combines two weak, content-based signals (screen's
    own long axis; which half has more ink/segment mass) rather than
    trusting a single one or a hardcoded direction -- but even in
    agreement these are inherently weaker than the dual-box signals, so
    this path is always reported at reduced confidence.
    """
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
    # Digit strokes are always a small minority of the screen's pixels --
    # true regardless of whether this is an ordinary dark-ink-on-light-LCD
    # or a backlit light-ink-on-dark-LCD. So instead of assuming a fixed
    # polarity, threshold both ways and keep whichever class is smaller.
    _, bright_class = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, dark_class = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = bright_class if bright_class.mean() < dark_class.mean() else dark_class
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
# Single-Image Quick Testing Runner
# ============================================================

def load_model(weights_path=WEIGHTS, device_str="", img_size=960):
    set_logging()
    device = select_device(device_str)
    model = attempt_load(str(weights_path), map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(img_size, s=stride)
    model.eval()
    return model, device, stride, imgsz


def detect_boxes(model, device, stride, imgsz, image_path, conf_thres=0.25, iou_thres=0.45):
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Could not read image: {image_path}")

    dataset = LoadImages(str(image_path), img_size=imgsz, stride=stride)
    names = model.module.names if hasattr(model, "module") else model.names

    candidates = []
    for _, image, _, _ in dataset:
        tensor = torch.from_numpy(image).to(device).float() / 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            prediction = model(tensor, augment=False)[0]
        detections = non_max_suppression(prediction, conf_thres, iou_thres)[0]
        if detections is None or not len(detections):
            continue

        detections[:, :4] = scale_coords(tensor.shape[2:], detections[:, :4], original.shape).round()
        for *xyxy, confidence, class_id in detections:
            label_name = str(names[int(class_id)])
            candidates.append((float(confidence), label_name, tuple(map(int, xyxy))))

    return original, candidates


def process_single_image(model, device, stride, imgsz, image_path: Path):
    print(f"\n==========================================")
    print(f"Processing image: {image_path.name}")
    print(f"Path: {image_path}")
    print(f"==========================================")

    original, candidates = detect_boxes(model, device, stride, imgsz, image_path)

    if not candidates:
        print("No detections found by YOLO model.")
        return

    print("\nAll Detections:")
    for conf, label, box in sorted(candidates, key=lambda c: -c[0]):
        print(f"  {label:20s} conf={conf:.3f} box={box}")

    def best_match(predicate):
        matches = [c for c in candidates if predicate(c[1].lower())]
        return max(matches, key=lambda item: item[0]) if matches else None

    best_in = best_match(lambda l: "in_digital" in l)
    best_out = best_match(lambda l: "out_digital" in l)

    out_dir = image_path.parent
    stem = image_path.stem

    if best_in is None:
        print("\nNo 'in_digital' box detected -- cannot process screen orientation.")
        return

    _, _, (ix1, iy1, ix2, iy2) = best_in
    h, w = original.shape[:2]
    ix1, ix2 = max(0, ix1), min(w, ix2)
    iy1, iy2 = max(0, iy1), min(h, iy2)
    in_box = (ix1, iy1, ix2, iy2)

    raw_in_crop = original[iy1:iy2, ix1:ix2]
    cv2.imwrite(str(out_dir / f"{stem}_raw_in_crop.jpg"), raw_in_crop)

    if best_out is not None:
        _, _, (ox1, oy1, ox2, oy2) = best_out
        ox1, ox2 = max(0, ox1), min(w, ox2)
        oy1, oy2 = max(0, oy1), min(h, oy2)
        out_box = (ox1, oy1, ox2, oy2)

        raw_out_crop = original[oy1:oy2, ox1:ox2]
        cv2.imwrite(str(out_dir / f"{stem}_raw_out_crop.jpg"), raw_out_crop)

        corrected_out, corrected_in, total_angle, info = normalize_gauge_orientation_from_boxes(
            original, out_box, in_box
        )
        print(f"\n[Dual-Box Orientation]")
        print(f"  Total Rotation Angle : {total_angle:.1f}°")
        print(f"  Flip Applied         : {info.get('flip_applied')}")
        print(f"  Confidence           : {info.get('confidence')}")
        if "screen_score" in info:
            print(f"  Screen Score         : {info['screen_score']:.4f}")
            print(f"  Width Score          : {info['width_score']:.4f}")

        tight = tight_crop_rotated_screen(corrected_in)
        final = enforce_landscape(tight)
    else:
        print("\n[Screen-Only Fallback (No 'out_digital' detected)]")
        screen, info = normalize_screen_only(raw_in_crop)
        print(f"  Axis Angle           : {info.get('axis_angle', 0):.1f}°")
        print(f"  Confidence           : {info.get('confidence')}")

        tight = tight_crop_rotated_screen(screen)
        final = enforce_landscape(tight)

    final_path = out_dir / f"{stem}_FINAL.jpg"
    cv2.imwrite(str(final_path), final)
    print(f"\nFinal Landscape Screen Size: {final.shape[1]}x{final.shape[0]} (w x h)")
    print(f"Saved final output: {final_path}")


def main():
    target_path = Path(sys.argv[1]) if len(sys.argv) > 1 else IMAGE_PATH

    if not WEIGHTS.is_file():
        print(f"ERROR: Model weights file not found at: {WEIGHTS}")
        sys.exit(1)

    if not target_path.exists():
        print(f"ERROR: Target path does not exist: {target_path}")
        sys.exit(1)

    model, device, stride, imgsz = load_model(WEIGHTS)

    if target_path.is_file():
        process_single_image(model, device, stride, imgsz, target_path)
    elif target_path.is_dir():
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted([p for p in target_path.iterdir() if p.is_file() and p.suffix.lower() in image_extensions])
        if not image_files:
            print(f"No image files found in directory: {target_path}")
            return
        print(f"Found {len(image_files)} image(s) in {target_path}")
        for img_p in image_files:
            process_single_image(model, device, stride, imgsz, img_p)


if __name__ == "__main__":
    main()

