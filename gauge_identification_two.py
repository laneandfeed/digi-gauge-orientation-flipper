import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def plot_one_box_guage(claim ,name , x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.imwrite(claim + "/cropped_gauge.jpg",img)

def detect(save_img=False):
    t11=time.time()
    label_dict={}
    #source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    opt_weights="app/Gauge_det_2/best.pt"
    opt_source="/root/Downloads/odometer/odo_testing_early_loading/res3"  
    opt_view_img=False
    opt_save_txt=False
    opt_img_size=960
    opt_no_trace=True
    source, weights, view_img, save_txt, imgsz, trace = opt_source, opt_weights, opt_view_img, opt_save_txt, opt_img_size, not opt_no_trace

    # Initialize
    set_logging()
    opt_device=""
    device = select_device(opt_device)
    # device='cuda'
    # half = device.type != 'cpu'  # half precision only supported on CUDA
    # half=True

    # Half precision is supported only on CUDA.  Keeping this conditional also
    # makes CPU fallback safe.
    half = device.type != 'cpu'

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    opt_img_size=960
    if trace:
        model = TracedModel(model, device, opt_img_size)

    model.eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    print(stride,imgsz)
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    t22=time.time()-t11
    print("model early loading time taken is:",t22)
    vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = check_imshow()
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    # else:
    return model,device

with torch.no_grad():
    return_model,return_device=detect()
    
# model= return_model
# device=return_device

@torch.inference_mode()
def gauge_prediction_two(claim_id):
    """Identify the gauge and persist only the best crop for each digital region."""
    claim_path = Path(claim_id)
    source_path = claim_path / "image.jpg"
    input_crop_path = claim_path / "cropped_gauge.jpg"
    output_crop_path = claim_path / "cropped_out_digital.jpg"

    if not source_path.is_file():
        print(f"Gauge source image is missing: {source_path}")
        return "Gauge Not Detected"

    try:
        # A claim directory may be reused.  Never allow a previous request's crop
        # to be consumed when this request has no matching detection.
        for crop_path in (input_crop_path, output_crop_path):
            if crop_path.exists():
                crop_path.unlink()

        original = cv2.imread(str(source_path))
        if original is None:
            print(f"Unable to read gauge source image: {source_path}")
            return "Gauge Not Detected"

        model, device = return_model, return_device
        stride = int(model.stride.max()) if hasattr(model, "stride") else 32
        dataset = LoadImages(str(source_path), img_size=960, stride=stride)
        names = model.module.names if hasattr(model, "module") else model.names
        candidates = []

        for _, image, _, _ in dataset:
            tensor = torch.from_numpy(image).to(device)
            tensor = tensor.half() if device.type != "cpu" else tensor.float()
            tensor /= 255.0
            if tensor.ndimension() == 3:
                tensor = tensor.unsqueeze(0)

            prediction = model(tensor, augment=False)[0]
            detections = non_max_suppression(prediction, 0.3, 0.45)[0]
            if detections is None or not len(detections):
                continue

            detections[:, :4] = scale_coords(tensor.shape[2:], detections[:, :4], original.shape).round()
            for *xyxy, confidence, class_id in detections:
                class_index = int(class_id)
                if class_index < 0 or class_index >= len(names):
                    continue
                label_name = str(names[class_index])
                print(f"Gauge label detected: {label_name} (confidence: {float(confidence):.2f})")
                candidates.append((float(confidence), label_name, tuple(map(int, xyxy))))

        if not candidates:
            return "Gauge Not Detected"

        def best_match(predicate):
            matches = [item for item in candidates if predicate(item[1].lower())]
            return max(matches, key=lambda item: item[0]) if matches else None

        def save_crop(match, destination, rotate_portrait=False):
            _, _, (x1, y1, x2, y2) = match
            height, width = original.shape[:2]
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                print(f"Skipping invalid gauge crop: {(x1, y1, x2, y2)}")
                return False
            crop = original[y1:y2, x1:x2]
            if rotate_portrait and crop.shape[0] >= crop.shape[1]:
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            return bool(cv2.imwrite(str(destination), crop))

        # Keep the highest-confidence input and output independently.  This
        # handles any number of in_digital/out_digital detections deterministically.
        best_input = best_match(lambda label: "in_digital" in label)
        best_output = best_match(lambda label: "out_digital" in label)
        if best_input and not save_crop(best_input, input_crop_path, rotate_portrait=True):
            best_input = None
        if best_output and not save_crop(best_output, output_crop_path):
            best_output = None

        if best_input or best_output:
            return "digital_gauge"

        # For non-digital gauges, retain the single most confident detection.
        best_gauge = max(candidates, key=lambda item: item[0])
        if save_crop(best_gauge, input_crop_path):
            return best_gauge[1]
        return "Gauge Not Detected"
    except Exception as exc:
        # The API caller receives the standard no-detection result rather than a
        # 500 error; the exception is still logged for investigation.
        print(f"Gauge identification failed for {claim_id}: {exc}")
        return "Gauge Not Detected"
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
