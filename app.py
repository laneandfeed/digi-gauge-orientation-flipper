"""
Gradio Web Application for Digital Gauge Orientation Normalizer
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import gradio as gr

# Patched torch load for PyTorch 2.6+
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device
from utils.datasets import letterbox

from quick_test_v6 import (
    normalize_gauge_orientation_from_boxes,
    tight_crop_rotated_screen,
    enforce_landscape,
    post_correct_by_brightness
)

WEIGHTS = Path(__file__).resolve().parent / "best.pt"
IMG_SIZE = 960
CONF_THRES = 0.25
IOU_THRES = 0.45

# Load model globally on startup
set_logging()
device = select_device("")
model = attempt_load(WEIGHTS, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(IMG_SIZE, s=stride)
model.eval()
names = model.module.names if hasattr(model, "module") else model.names


def process_image(input_img):
    if input_img is None:
        return None, None, None, "Please upload an image."

    # input_img from gradio is RGB
    original = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    # Preprocess with letterbox
    img = letterbox(original, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    tensor = torch.from_numpy(img).to(device).float()
    tensor /= 255.0
    if tensor.ndimension() == 3:
        tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = model(tensor, augment=False)[0]
    detections = non_max_suppression(prediction, CONF_THRES, IOU_THRES)[0]

    if detections is None or not len(detections):
        return None, None, None, "No gauge detected in the image."

    detections[:, :4] = scale_coords(tensor.shape[2:], detections[:, :4], original.shape).round()
    candidates = []
    annotated = original.copy()

    for *xyxy, confidence, class_id in detections:
        c_id = int(class_id)
        label = str(names[c_id])
        x1, y1, x2, y2 = map(int, xyxy)
        candidates.append((float(confidence), label, (x1, y1, x2, y2)))
        
        # Draw on annotated image
        color = (0, 255, 0) if "in_digital" in label else (255, 0, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(annotated, f"{label} {confidence:.2f}", (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def best_match(pred):
        m = [c for c in candidates if pred(c[1].lower())]
        return max(m, key=lambda c: c[0]) if m else None

    best_in = best_match(lambda l: "in_digital" in l)
    best_out = best_match(lambda l: "out_digital" in l)

    if best_in is None or best_out is None:
        return (
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            None,
            None,
            "Could not detect both outer casing and LCD screen for orientation correction."
        )

    in_box = best_in[2]
    out_box = best_out[2]
    corrected_out, corrected_in, angle, info = normalize_gauge_orientation_from_boxes(original, out_box, in_box)

    ix1, iy1, ix2, iy2 = in_box
    original_in_size = (ix2 - ix1, iy2 - iy1)
    tight = tight_crop_rotated_screen(corrected_in, original_in_size, angle)
    final = enforce_landscape(tight)

    # Post correction check
    suggested, b_info, overlay = post_correct_by_brightness(final)
    if b_info.get("suggest_flip") and b_info.get("confident"):
        final = suggested

    status_log = (
        f"Angle Detected: {angle:.1f}°\n"
        f"Flip Applied: {info.get('flip_applied')}\n"
        f"Confidence: {info.get('confidence')}\n"
        f"Screen Score: {info.get('screen_score', 0):.4f}\n"
        f"Width Score: {info.get('width_score', 0):.4f}\n"
        f"Final Output Size: {final.shape[1]}x{final.shape[0]} px"
    )

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return annotated_rgb, final_rgb, overlay_rgb, status_log


demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Upload Gauge Image"),
    outputs=[
        gr.Image(type="numpy", label="YOLOv7 Detections"),
        gr.Image(type="numpy", label="Oriented LCD Screen (Final)"),
        gr.Image(type="numpy", label="Ink/Brightness Analysis"),
        gr.Textbox(label="Orientation Details & Diagnostics", lines=7),
    ],
    title="Digital Gauge Orientation Normalizer",
    description="Automated tire depth gauge LCD orientation normalizer using custom YOLOv7 detection, GrabCut segmentation, and ink-density flip detection.",
    examples=[
        ["testing/testing_failed/recorrect_unsuccessful/guage_204ca6ce-0a82-4977-b18a-e56635a284af.jpg"],
    ] if Path("testing/testing_failed/recorrect_unsuccessful/guage_204ca6ce-0a82-4977-b18a-e56635a284af.jpg").exists() else None
)

if __name__ == "__main__":
    demo.launch()
