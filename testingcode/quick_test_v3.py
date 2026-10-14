"""
quick_test.py — point at ONE image, get the corrected crop back. That's it.

SETUP: this file must sit inside your yolov7 clone (same folder as best.pt).

EDIT THESE TWO LINES, THEN RUN:
    python3 quick_test.py
"""

import math
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


# ---------- orientation.py logic, pasted in directly (current version) ----------

def _segment_gauge(bgr_image):
    # GrabCut's k-means init is randomized and unseeded by default --
    # verified this makes the SAME crop give different angles run to run
    # (up to 26 degrees on one real photo). Fixing the seed makes results
    # reproducible.
    cv2.setRNGSeed(42)
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

    # Cross-check against the screen's own axis -- a genuine LCD is
    # mounted flush in the body, so body and screen axes should agree
    # closely. Big disagreement means the (larger) out_digital mask
    # likely picked up background clutter; the tighter in_digital crop
    # is less exposed to that.
    angle_source = "out_digital body"
    in_mask_for_angle = _segment_gauge(in_crop)
    in_contour_for_angle = _largest_contour(in_mask_for_angle)
    if in_contour_for_angle is not None:
        in_angle = _axis_angle_degrees(in_contour_for_angle)
        d = abs(axis_angle - in_angle) % 180
        disagreement = min(d, 180 - d)
        if disagreement > 15:
            axis_angle = in_angle
            angle_source = f"in_digital screen (disagreed with body by {disagreement:.0f} deg)"

    rotation_amount = axis_angle

    # For backlit (light-on-dark) screens specifically, the lit digits are
    # the brightest thing in the crop and their left/right position is a
    # real signal (confirmed: +0.94 on a correct crop, -0.79 on the same
    # gauge rotated 180). Does NOT apply to ordinary dark-on-light screens
    # -- gated on is_light_on_dark so it can't fire there. Also gated on
    # signal strength: is_light_on_dark itself is noisy on loose
    # segmentations (SAN and "last" both read True but with near-zero
    # bright_side_score, 0.001/0.017) -- those correctly fall through to
    # the width/screen signals below instead of being trusted.
    aligned_in_crop = _rotate_no_crop(in_crop, rotation_amount)
    gray_in = cv2.cvtColor(aligned_in_crop, cv2.COLOR_BGR2GRAY)
    _, bright_class = cv2.threshold(gray_in, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, dark_class = cv2.threshold(gray_in, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    is_light_on_dark = bright_class.mean() < dark_class.mean()

    bright_side_score = 0.0
    if is_light_on_dark:
        ih, iw = gray_in.shape
        bright_cut = np.percentile(gray_in, 90)
        bright_mask = (gray_in >= bright_cut).astype(np.float64)
        bl = bright_mask[:, :iw // 2].sum()
        br = bright_mask[:, iw // 2:].sum()
        bright_side_score = (br - bl) / max(1.0, br + bl)

    BRIGHT_SIDE_EPS = 0.1
    if is_light_on_dark and abs(bright_side_score) > BRIGHT_SIDE_EPS:
        flip_needed = bright_side_score < 0
        total_angle = rotation_amount + (180 if flip_needed else 0)
        corrected_out = _rotate_no_crop(out_crop, total_angle)
        corrected_in = _rotate_no_crop(in_crop, total_angle)
        info = {"axis_angle": axis_angle, "angle_source": angle_source, "flip_applied": flip_needed,
                "confidence": f"medium (backlit screen, bright-pixel side check, score={bright_side_score:+.2f})",
                "screen_score": 0.0, "width_score": 0.0, "total_rotation": total_angle}
        return corrected_out, corrected_in, total_angle, info

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

    total_angle = rotation_amount + (180 if flip_needed else 0)
    corrected_out = _rotate_no_crop(out_crop, total_angle)
    corrected_in = _rotate_no_crop(in_crop, total_angle)
    info = {"axis_angle": axis_angle, "angle_source": angle_source, "flip_applied": flip_needed,
            "confidence": confidence, "screen_score": screen_score, "width_score": width_score,
            "total_rotation": total_angle}
    return corrected_out, corrected_in, total_angle, info


def _largest_inscribed_rect(w, h, angle_degrees):
    """Largest axis-aligned rectangle (same center) that fits entirely
    inside a w x h rectangle rotated by angle_degrees -- avoids the
    padding triangles a free-angle rotation leaves. Replaces an earlier
    version that guessed padding from corner pixel colors, which was
    proven to never remove anything at any angle (a rotated rectangle's
    own corners always touch all four edges of its tightly-fit bounding
    canvas, so that kind of crop is a mathematical no-op)."""
    angle = math.radians(angle_degrees % 180)
    if angle > math.pi / 2:
        angle = math.pi - angle
    if angle < 1e-6 or abs(angle - math.pi / 2) < 1e-6:
        return float(w), float(h)
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long + 1e-9 or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a
    return abs(wr), abs(hr)


def tight_crop_rotated_screen(rotated_bgr, original_size, angle_degrees, margin=1):
    """original_size: (width, height) of the crop BEFORE rotation (e.g.
    in_crop.shape[1], in_crop.shape[0]). angle_degrees: total_angle
    returned by normalize_gauge_orientation_from_boxes."""
    orig_w, orig_h = original_size
    wr, hr = _largest_inscribed_rect(orig_w, orig_h, angle_degrees)
    rh, rw = rotated_bgr.shape[:2]
    cx, cy = rw / 2.0, rh / 2.0
    half_w, half_h = max(1.0, wr / 2.0 - margin), max(1.0, hr / 2.0 - margin)
    x1, x2 = max(0, int(cx - half_w)), min(rw, int(cx + half_w))
    y1, y2 = max(0, int(cy - half_h)), min(rh, int(cy + half_h))
    return rotated_bgr[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else rotated_bgr


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

    ix1, iy1, ix2, iy2 = in_box
    original_in_size = (ix2 - ix1, iy2 - iy1)
    tight = tight_crop_rotated_screen(corrected_in, original_in_size, angle)
    final = enforce_landscape(tight)

def _post_correct_brightness(screen_bgr):
        gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cx = w // 2
        
        # Detect screen type
        _, bright_class = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, dark_class = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        is_light_on_dark = bright_class.mean() < dark_class.mean()
        
        # Bright pixel distribution (90th percentile isolates digits)
        bright_cut = np.percentile(gray, 90)
        bright_mask = (gray >= bright_cut).astype(np.uint8) * 255
        
        left_count = int(bright_mask[:, :cx].sum() / 255)
        right_count = int(bright_mask[:, cx:].sum() / 255)
        total = left_count + right_count
        side_score = (right_count - left_count) / max(1.0, total)
        
        EPS = 0.15
        flip_needed = False
        confidence = "none"
        
        if is_light_on_dark and abs(side_score) > EPS:
            flip_needed = side_score < 0  # bright on left → flip
            confidence = f"high (light-on-dark, score={side_score:+.3f})"
        elif not is_light_on_dark and abs(side_score) > EPS:
            dark_cut = np.percentile(gray, 10)
            dark_mask = (gray <= dark_cut).astype(np.uint8) * 255
            dl = int(dark_mask[:, :cx].sum() / 255)
            dr = int(dark_mask[:, cx:].sum() / 255)
            dark_score = (dr - dl) / max(1.0, dl + dr)
            if abs(dark_score) > EPS:
                flip_needed = dark_score < 0
                confidence = f"high (dark-on-light, score={dark_score:+.3f})"
            else:
                confidence = f"low (dark-on-light, weak={dark_score:+.3f})"
        else:
            confidence = f"low (weak signal={side_score:+.3f})"
        
        if flip_needed:
            corrected = cv2.rotate(screen_bgr, cv2.ROTATE_180)
        else:
            corrected = screen_bgr
            
        return corrected, {
            "flip_needed": flip_needed,
            "confidence": confidence,
            "side_score": side_score,
            "left_count": left_count,
            "right_count": right_count,
        }
    
final_v2, post_info = _post_correct_brightness(final)
    
print(f"\n[POST-CORRECTION V2] flip={post_info['flip_needed']}  {post_info['confidence']}")
print(f"  left_bright={post_info['left_count']}  right_bright={post_info['right_count']}  score={post_info['side_score']:+.3f}")
    
# Save v2 result
out_path_v2 = out_dir / f"{stem}_FINAL_v2.jpg"
cv2.imwrite(str(out_path_v2), final_v2)
print(f"Saved v2: {out_path_v2}")

print(f"\nangle={angle:.1f}  flip={info['flip_applied']}  confidence={info['confidence']}")
print(f"angle_source={info['angle_source']}")
print(f"screen_score={info['screen_score']:.4f}  width_score={info['width_score']:.4f}")
print(f"final screen size: {final.shape[1]}x{final.shape[0]} (w x h)")
    
    

out_path = out_dir / f"{stem}_FINAL.jpg"
cv2.imwrite(str(out_path), final)
cv2.imwrite(str(out_dir / f"{stem}_raw_in_crop.jpg"), original[in_box[1]:in_box[3], in_box[0]:in_box[2]])
print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
