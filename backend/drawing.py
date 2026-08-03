import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def load_label_font():
    """
    Loads a small font for box labels.
    If Calibri is missing, it uses a default font.
    """
    candidate_paths = [
        "calibri.ttf",
        "Calibri.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/Calibri.ttf",
    ]
    for path in candidate_paths:
        try:
            return ImageFont.truetype(path, 10)
        except Exception:
            continue
    return ImageFont.load_default()

def draw_detections(frame_bgr, detections, label_font):
    """
    Draws red smoke boxes and confidence labels on an image.
    """
    annotated = frame_bgr.copy()

    box_thickness = 2
    generic_smoke_color_bgr = (0, 0, 230)
    generic_smoke_color_rgb = (230, 0, 0)

    frame_height, frame_width = annotated.shape[:2]

    label_jobs = []
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["xyxy"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), generic_smoke_color_bgr, box_thickness)
        label = f"#{index} Smoke {detection['detector_conf'] * 100:.0f}%"
        label_jobs.append((label, x1, y1))

    if not label_jobs:
        return annotated

    pil_image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    for label, x1, y1 in label_jobs:
        text_left, text_top, text_right, text_bottom = draw.textbbox((0, 0), label, font=label_font)
        text_w = text_right - text_left
        text_h = text_bottom - text_top

        label_x = min(x1, frame_width - text_w - 6)
        label_x = max(0, label_x)
        y_top = max(0, y1 - text_h - 6)

        draw.rectangle(
            [label_x, y_top, label_x + text_w + 6, y_top + text_h + 6],
            fill=generic_smoke_color_rgb,
        )
        draw.text((label_x + 3, y_top + 2), label, font=label_font, fill=(255, 255, 255))

    annotated = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return annotated

def draw_inference_boxes(frame_bgr, smoke_detections, label_font):
    return draw_detections(frame_bgr, smoke_detections, label_font)

def draw_fast_video_detections(frame_bgr, detections):
    """Draws faster red smoke boxes for video frames."""
    if not detections:
        return frame_bgr

    annotated = frame_bgr.copy()
    generic_smoke_color_bgr = (0, 0, 230)

    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["xyxy"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), generic_smoke_color_bgr, 2)
        label = f"#{index} Smoke {detection['detector_conf'] * 100:.0f}%"
        label_y = max(18, y1 - 6)
        cv2.putText(
            annotated,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            generic_smoke_color_bgr,
            2,
            cv2.LINE_AA,
        )

    return annotated
