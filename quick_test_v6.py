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

# ============== EDIT THESE ==============
IMAGE_PATH = Path(r"C:\Users\Sumankumar.Sahu\.gemini\antigravity\scratch\project1\yolov7\yolov7\testing\image 12.jpg")
WEIGHTS = Path(__file__).resolve().parent / "best.pt"
AUTO_APPLY_V2 = True      # True = auto-save _FINAL_v2 when confident
# =======================================

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



def post_correct_by_brightness(final_bgr):
    """
    Post-processing sanity check -- runs only on the truly final (tight-
    cropped, landscape) image. Checks if digits are on the correct side
    of the X-axis and suggests 180° flip if not.

    Handles BOTH screen types:
    - Light-on-dark (backlit): digits are BRIGHT, check bright pixels
    - Dark-on-light (standard): digits are DARK, check dark pixels

    In both cases, digits should be concentrated on the RIGHT half (+X).
    """
    gray = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cx, cy = w // 2, h // 2

    _, bright_class = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, dark_class = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    is_light_on_dark = bright_class.mean() < dark_class.mean()

    info = {
        "is_light_on_dark": is_light_on_dark,
        "left_count": 0, "right_count": 0, "score": 0.0,
        "suggest_flip": False, "confident": False,
        "check_type": "none",
    }
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- Determine which pixels to check (foreground = digits) ---
    if is_light_on_dark:
        # Backlit LCD: digits are BRIGHT → simple percentile works well
        fg_cut = np.percentile(gray, 90)
        fg_mask = (gray >= fg_cut).astype(np.uint8)
        check_type = "bright"
    else:
        # Standard LCD: digits are DARK → adaptive threshold works better
        # than percentile (which catches border noise instead of digits)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        fg_mask = cv2.adaptiveThreshold(gray_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        # Clean up small noise
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        check_type = "dark"

    # --- Remove border margin (ignore edge artifacts) ---
    margin_x = max(3, int(w * 0.05))
    margin_y = max(3, int(h * 0.05))
    fg_mask[:margin_y, :] = 0
    fg_mask[-margin_y:, :] = 0
    fg_mask[:, :margin_x] = 0
    fg_mask[:, -margin_x:] = 0

    # --- Count foreground pixels in each half ---
    left = int(fg_mask[:, :cx].sum())
    right = int(fg_mask[:, cx:].sum())
    total = max(1, left + right)
    score = (right - left) / total

    EPS = 0.15
    confident = abs(score) > EPS
    suggest_flip = confident and score < 0

    info.update({
        "left_count": left, "right_count": right, "score": score,
        "suggest_flip": suggest_flip, "confident": confident,
        "check_type": check_type,
    })

    suggested = cv2.rotate(final_bgr, cv2.ROTATE_180) if suggest_flip else final_bgr.copy()

    # --- Build analysis overlay with graph ---
    graph_h = h
    canvas_h = h + graph_h + 40
    canvas_w = max(w + 20, 420)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 25

    y_img = 20
    x_off = 10
    canvas[y_img:y_img+h, x_off:x_off+w] = final_bgr
    img_ol = canvas[y_img:y_img+h, x_off:x_off+w].copy()

    cv2.line(img_ol, (cx, 0), (cx, h), (0, 255, 255), 1)
    cv2.line(img_ol, (0, cy), (w, cy), (0, 255, 255), 1)
    cv2.putText(img_ol, "Q2 (-X,+Y)", (5, 15), font, 0.35, (255, 255, 255), 1)
    cv2.putText(img_ol, "Q1 (+X,+Y)", (w-80, 15), font, 0.35, (255, 255, 255), 1)
    cv2.putText(img_ol, "Q3 (-X,-Y)", (5, h-5), font, 0.35, (255, 255, 255), 1)
    cv2.putText(img_ol, "Q4 (+X,-Y)", (w-80, h-5), font, 0.35, (255, 255, 255), 1)

    # Color-code halves: red = where fg pixels ARE (if wrong), green = target side
    if left > right:
        hl = np.zeros_like(img_ol[:, :cx]); hl[:] = (0, 0, 80)
        img_ol[:, :cx] = cv2.addWeighted(img_ol[:, :cx], 0.8, hl, 0.2, 0)
        hr = np.zeros_like(img_ol[:, cx:]); hr[:] = (0, 80, 0)
        img_ol[:, cx:] = cv2.addWeighted(img_ol[:, cx:], 0.8, hr, 0.2, 0)
    else:
        hl = np.zeros_like(img_ol[:, :cx]); hl[:] = (0, 80, 0)
        img_ol[:, :cx] = cv2.addWeighted(img_ol[:, :cx], 0.8, hl, 0.2, 0)
        hr = np.zeros_like(img_ol[:, cx:]); hr[:] = (0, 0, 80)
        img_ol[:, cx:] = cv2.addWeighted(img_ol[:, cx:], 0.8, hr, 0.2, 0)

    canvas[y_img:y_img+h, x_off:x_off+w] = img_ol

    # Bottom: pixel distribution graph
    y_graph = y_img + h + 30
    gh = graph_h - 20
    gw = w
    cv2.rectangle(canvas, (x_off, y_graph), (x_off+gw, y_graph+gh), (40, 40, 40), -1)
    cv2.line(canvas, (x_off, y_graph+gh//2), (x_off+gw, y_graph+gh//2), (180, 180, 180), 1)
    cv2.line(canvas, (x_off+cx, y_graph), (x_off+cx, y_graph+gh), (0, 255, 255), 1)

    col_sums = fg_mask.sum(axis=0)
    max_val = col_sums.max() if col_sums.max() > 0 else 1
    for i, val in enumerate(col_sums):
        bar_h = int((val / max_val) * (gh * 0.85))
        color = (255, 200, 100) if i < cx else (100, 255, 100)
        if bar_h > 0:
            cv2.line(canvas, (x_off+i, y_graph+gh-5), (x_off+i, y_graph+gh-5-bar_h), color, 1)

    cv2.putText(canvas, f"{check_type.upper()} Pixel Distribution along X-axis", (x_off, y_graph-5), font, 0.4, (200, 200, 200), 1)
    cv2.putText(canvas, "-X (Left)", (x_off+5, y_graph+gh//2+15), font, 0.35, (150, 150, 150), 1)
    cv2.putText(canvas, "+X (Right)", (x_off+gw-70, y_graph+gh//2+15), font, 0.35, (150, 150, 150), 1)
    cv2.putText(canvas, f"L:{left}  R:{right}  score:{score:+.2f}", (x_off, y_graph+gh-5), font, 0.35, (255, 255, 0), 1)

    if not confident:
        label, color = "WEAK SIGNAL -- NO CHANGE", (200, 200, 200)
    elif suggest_flip:
        label, color = "FLIP 180 DEGREES", (0, 0, 255)
    else:
        label, color = "KEEP ORIENTATION", (0, 255, 0)
    cv2.putText(canvas, label, (x_off, 15), font, 0.5, color, 2)
    screen_type = "LIGHT-ON-DARK" if is_light_on_dark else "DARK-ON-LIGHT"
    cv2.putText(canvas, screen_type, (canvas_w-130, 15), font, 0.35, (200, 200, 200), 1)

    return suggested, info, canvas



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

    print(f"\nangle={angle:.1f}  flip={info['flip_applied']}  confidence={info['confidence']}")
    print(f"angle_source={info['angle_source']}")
    print(f"screen_score={info['screen_score']:.4f}  width_score={info['width_score']:.4f}")
    print(f"final screen size: {final.shape[1]}x{final.shape[0]} (w x h)")

    # Save original pipeline output
    out_path = out_dir / f"{stem}_FINAL.jpg"
    cv2.imwrite(str(out_path), final)
    cv2.imwrite(str(out_dir / f"{stem}_raw_in_crop.jpg"), original[in_box[1]:in_box[3], in_box[0]:in_box[2]])
    print(f"\nSaved: {out_path}")

    # ---- POST-CORRECTION V2 (brightness check) ----
    suggested, b_info, overlay = post_correct_by_brightness(final)
    overlay_path = out_dir / f"{stem}_brightness_check.jpg"
    cv2.imwrite(str(overlay_path), overlay)

    print(f"\n[POST-CORRECTION V2]")
    print(f"  is_light_on_dark={b_info['is_light_on_dark']}")

    if not b_info["is_light_on_dark"]:
        print("  Dark-on-light screen -- check skipped.")
    else:
        print(f"  L:{b_info['left_count']} R:{b_info['right_count']} score:{b_info['score']:+.3f}")
        print(f"  Analysis: {overlay_path}")

        if not b_info["confident"]:
            print("  Signal too weak -- keeping original.")
        elif b_info["suggest_flip"]:
            print("  SUGGESTION: Screen appears upside-down (digits on left).")
            if AUTO_APPLY_V2:
                out_path_v2 = out_dir / f"{stem}_FINAL_v2.jpg"
                cv2.imwrite(str(out_path_v2), suggested)
                print(f"  AUTO-SAVED flipped: {out_path_v2}")
            else:
                answer = input("  Flip 180 and save as _FINAL_v2.jpg? [y/n]: ").strip().lower()
                if answer == "y":
                    out_path_v2 = out_dir / f"{stem}_FINAL_v2.jpg"
                    cv2.imwrite(str(out_path_v2), suggested)
                    print(f"  Saved: {out_path_v2}")
                else:
                    print("  Skipped -- original kept.")
        else:
            print("  Looks correct -- no flip needed.")


if __name__ == "__main__":
    main()
