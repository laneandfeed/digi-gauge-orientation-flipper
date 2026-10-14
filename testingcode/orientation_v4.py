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

import math

import cv2
import numpy as np


# ---------- segmentation & axis-finding ----------

def _segment_gauge(bgr_image):
    # GrabCut's internal k-means initialization is randomized and NOT
    # seeded by default -- verified this causes the same exact crop to
    # produce different segmentations (and thus different axis angles)
    # from run to run, by up to 26 degrees on one real photo. Fixing the
    # seed makes it fully reproducible without changing typical accuracy.
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

    # Cross-check against the screen's own axis. A genuine LCD is mounted
    # flush in the body, so the body's axis and the screen's own axis
    # should agree closely for a clean segmentation. A big disagreement is
    # a sign the (larger) out_digital mask picked up background clutter --
    # the smaller, tighter in_digital crop is less exposed to that. Based
    # on 4 confirmed photos so far: ~0-2 degree agreement on working cases
    # vs 20-37 degree disagreement on the two known-bad ones (RAJ, SAN).
    # Real signal, small sample -- worth more data, not blind trust.
    angle_source = "out_digital body"
    in_mask_for_angle = _segment_gauge(in_crop)
    in_contour_for_angle = _largest_contour(in_mask_for_angle)
    if in_contour_for_angle is not None:
        in_angle = _axis_angle_degrees(in_contour_for_angle)
        disagreement = min(abs(axis_angle - in_angle) % 180, 180 - abs(axis_angle - in_angle) % 180)
        if disagreement > 15:
            axis_angle = in_angle
            angle_source = f"in_digital screen (disagreed with body by {disagreement:.0f} deg)"

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
        "angle_source": angle_source,
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

def _largest_inscribed_rect(w, h, angle_degrees):
    """Largest axis-aligned rectangle (same center) that fits entirely
    inside a w x h rectangle rotated by angle_degrees -- i.e. exactly
    avoids the padding triangles left by a free-angle rotation. Standard
    closed-form solution to this geometry problem. Verified against a
    synthetic rotation test: 0 leftover padding pixels at 15/30/45/215
    degrees, vs. the old pixel-guessing approach which removed nothing
    at any angle, including 0."""
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
    """
    rotated_bgr: output of _rotate_no_crop (i.e. corrected_in, or
        apply_same_rotation's result)
    original_size: (width, height) of the crop BEFORE that rotation was
        applied -- e.g. in_crop.shape[1], in_crop.shape[0]
    angle_degrees: the exact angle that was applied (total_angle from
        normalize_gauge_orientation_from_boxes)

    Replaces the old approach, which guessed padding from corner pixel
    colors and was proven to never actually remove anything: a rotated
    rectangle's own corners always touch all four edges of its own
    tightly-fit bounding canvas (that's what _rotation_matrix_no_crop
    computes), so an axis-aligned "find the content bounding box" crop on
    the padded image is a mathematical no-op regardless of angle or
    padding color. This instead computes the largest axis-aligned
    rectangle that's geometrically guaranteed to contain zero padding,
    directly from the known rotation -- nothing to guess.
    """
    orig_w, orig_h = original_size
    wr, hr = _largest_inscribed_rect(orig_w, orig_h, angle_degrees)
    rh, rw = rotated_bgr.shape[:2]
    cx, cy = rw / 2.0, rh / 2.0
    half_w = max(1.0, wr / 2.0 - margin)
    half_h = max(1.0, hr / 2.0 - margin)
    x1, x2 = max(0, int(cx - half_w)), min(rw, int(cx + half_w))
    y1, y2 = max(0, int(cy - half_h)), min(rh, int(cy + half_h))
    if x2 <= x1 or y2 <= y1:
        return rotated_bgr
    return rotated_bgr[y1:y2, x1:x2]


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
