import cv2
import numpy as np
import torch

from .config import get_smoke_info

def detect_device():
    """
    Checks if GPU is available.
    If not, the app uses CPU.
    """
    use_cuda = torch.cuda.is_available()
    inference_device = "cuda:0" if use_cuda else "cpu"
    use_half_precision = False
    device_label = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    cuda_version = torch.version.cuda if use_cuda else "Not available"
    precision_label = "FP32"

    if use_cuda:
        # Keep GPU inference in FP32 for CPU/GPU consistency.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")

    return {
        "use_cuda": use_cuda,
        "inference_device": inference_device,
        "use_half_precision": use_half_precision,
        "precision_label": precision_label,
        "device_label": device_label,
        "cuda_version": cuda_version,
    }


def cut_image_inside_box(image, box, padding=0.0):
    """Cuts the image area inside the given box to go to vlm"""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    return image[y1:y2, x1:x2], [x1, y1, x2, y2]


def make_smoke_box_smaller(box, shrink_pct=0.08, top_shrink_pct=0.18):
    """Makes the displayed box tighter, especially from the top."""
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    shrink_x = int(box_w * shrink_pct)
    shrink_y = int(box_h * shrink_pct)
    top_shrink_y = int(box_h * top_shrink_pct)
    return [x1 + shrink_x, y1 + top_shrink_y, x2 - shrink_x, y2 - shrink_y]


def box_area(box): #It is used to know which smoke box is bigger.
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_inside_ratio(inner_box, outer_box): #This checks how much one box is inside another box.
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    overlap_x1 = max(ix1, ox1)
    overlap_y1 = max(iy1, oy1)
    overlap_x2 = min(ix2, ox2)
    overlap_y2 = min(iy2, oy2)
    overlap_area = box_area([overlap_x1, overlap_y1, overlap_x2, overlap_y2])
    inner_area = box_area(inner_box)
    if inner_area <= 0:
        return 0.0
    return overlap_area / inner_area


def remove_nested_smoke_boxes(detections):
    """
    Removes small duplicate boxes inside bigger smoke boxes.
    This keeps the output image cleaner.
    """
    if len(detections) < 2:
        return detections

    sorted_detections = sorted(detections, key=lambda item: box_area(item["xyxy"]), reverse=True)
    kept = []
    for detection in sorted_detections:
        is_nested_duplicate = any(
            box_inside_ratio(detection["xyxy"], kept_detection["xyxy"]) >= 0.75
            for kept_detection in kept
        )
        if not is_nested_duplicate:
            kept.append(detection)

    return sorted(kept, key=lambda item: item["detector_conf"], reverse=True)


def isolate_smoke_pixels(crop_bgr):
    """
    Keeps smoke-looking pixels inside the YOLO crop and dulls the background. this is for counter checking because our florence was not working properly somtimes it did niot give colour so we needed this.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, _value = cv2.split(hsv)

    low_color_mask = saturation < 60
    soft_blue_mask = (hue > 95) & (hue < 135) & (saturation < 120)
    smoke_mask = (low_color_mask | soft_blue_mask).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)

    neutral_background = np.full_like(crop_bgr, 128)
    mask_3channel = cv2.cvtColor(smoke_mask, cv2.COLOR_GRAY2BGR)
    isolated_crop = np.where(mask_3channel > 0, crop_bgr, neutral_background)

    return isolated_crop.astype(np.uint8)


def calculate_smoke_color_percentages(crop_bgr):
    """
    Counts smoke-like pixels as black, white, grey, or blue.
    """
    empty_result = {
        "black_smoke": 0.0,
        "white_smoke": 0.0,
        "grey_smoke": 0.0,
        "blue_smoke": 0.0,
    }
    if crop_bgr is None or crop_bgr.size == 0:
        return empty_result

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    is_neutral_background = (saturation < 3) & (np.abs(value.astype(int) - 128) < 3)
    is_too_colorful = saturation > 110
    usable_pixels = ~is_neutral_background & ~is_too_colorful

    usable_count = int(np.count_nonzero(usable_pixels))
    if usable_count < 50:
        return empty_result

    blue_pixels = usable_pixels & (hue >= 95) & (hue <= 135) & (saturation > 35)
    black_pixels = usable_pixels & ~blue_pixels & (value < 115)
    white_pixels = usable_pixels & ~blue_pixels & (value > 150) & (saturation < 40)
    grey_pixels = usable_pixels & ~blue_pixels & ~black_pixels & ~white_pixels

    return {
        "black_smoke": round(np.count_nonzero(black_pixels) / usable_count * 100, 1),
        "white_smoke": round(np.count_nonzero(white_pixels) / usable_count * 100, 1),
        "grey_smoke": round(np.count_nonzero(grey_pixels) / usable_count * 100, 1),
        "blue_smoke": round(np.count_nonzero(blue_pixels) / usable_count * 100, 1),
    }


def render_display_status(detections):
    """Builds the small smoke/no-smoke status badge."""
    if detections:
        return '<div class="display-status alert">SMOKE DETECTED</div>'
    return '<div class="display-status">NO SMOKE</div>'


def format_color_percentages(color_percentages):
    if not color_percentages:
        return "Not available"

    labels = {
        "black_smoke": "Black",
        "white_smoke": "White",
        "grey_smoke": "Grey",
        "blue_smoke": "Blue",
    }
    return " | ".join(
        f"{label}: {color_percentages.get(class_name, 0.0):.1f}%"
        for class_name, label in labels.items()
    )
