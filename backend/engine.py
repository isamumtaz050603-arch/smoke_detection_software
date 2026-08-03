import cv2
import numpy as np
import torch
from PIL import Image

from .config import (
    FLORENCE_CAPTION_TASK,
    SMOKE_COLOR_NEAR_WORD_PATTERN,
    SMOKE_COLOR_KEYWORDS
)

from .utils import (
    cut_image_inside_box,
    make_smoke_box_smaller,
    remove_nested_smoke_boxes,
    isolate_smoke_pixels,
    calculate_smoke_color_percentages
)

class SmokeDetectionEngine:
    """
    Main AI engine.
    It keeps the loaded models and current settings together.
    """
    def __init__(self, detector_model, classifier_model, classifier_processor, inference_device, use_half_precision, config):
        self.detector_model = detector_model
        self.classifier_model = classifier_model
        self.classifier_processor = classifier_processor
        self.inference_device = inference_device
        self.use_half_precision = use_half_precision
        self.config = config

    def prepare_florence_inputs(self, inputs):
        """
        Moves Florence inputs to the selected device.
        On GPU, image pixels must be float16 because Florence is loaded as float16.
        Text tokens stay as integer tensors.
        """
        prepared_inputs = {}
        for key, value in inputs.items():
            value = value.to(self.inference_device)
            if key == "pixel_values" and self.use_half_precision:
                value = value.half()
            prepared_inputs[key] = value
        return prepared_inputs

    # ------------------------------------------------------------------
    # Detection: YOLO finds smoke boxes.
    # ------------------------------------------------------------------
    def detect_smoke_boxes(self, frame_bgr):
        """
        Runs YOLO only.
        Returns smoke boxes and crop data for later classification.
        """
        config = self.config
        predict_args = {
            "source": frame_bgr,
            "conf": config.detector_conf,
            "iou": config.iou_threshold,
            "imgsz": config.detector_image_size,
            "agnostic_nms": True, #is a YOLO setting. It removes duplicate overlapping boxes.
            "device": self.inference_device,
            "half": self.use_half_precision,
            "verbose": False, #Should YOLO print extra details in the terminal
        }

        try:
            detector_results = self.detector_model.predict(**predict_args)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or not str(self.inference_device).startswith("cuda"):
                raise
            torch.cuda.empty_cache()
            predict_args["device"] = "cpu"
            predict_args["half"] = False
            detector_results = self.detector_model.predict(**predict_args)

        boxes = detector_results[0].boxes
        detections = []

        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            detector_score = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            display_box = make_smoke_box_smaller([x1, y1, x2, y2], shrink_pct=0.08)

            # The classifier sees the same smaller smoke area that is drawn on screen.
            smoke_crop, _crop_box = cut_image_inside_box(frame_bgr, display_box)

            detections.append(
                {
                    "xyxy": display_box,
                    "detector_conf": detector_score,
                    "class_name": "smoke",  # Filled with color later.
                    "classifier_conf": 0.0,
                    "_classification_crop": smoke_crop,
                }
            )

        return remove_nested_smoke_boxes(detections)

    # ------------------------------------------------------------------
    # Classification: Florence decides smoke color from the crop caption.
    # ------------------------------------------------------------------
    def classify_smoke_crops(self, crops):
        """
        Classifies smoke crops as black, white, grey, or blue.
        Returns the final class, confidence, Florence caption, and pixel percentages.
        """
        results = [("smoke", 0.0, "", calculate_smoke_color_percentages(None))] * len(crops)
        if not crops:
            return results

        valid_indices = [i for i, crop in enumerate(crops) if crop is not None and crop.size > 0]
        if not valid_indices:
            return results

        for original_index in valid_indices:
            crop_bgr = crops[original_index]
            pixel_crop = isolate_smoke_pixels(crop_bgr)
            color_percentages = calculate_smoke_color_percentages(pixel_crop)
            percentage_class = max(color_percentages, key=color_percentages.get)
            percentage_confidence = color_percentages[percentage_class] / 100.0

            crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

            inputs = self.classifier_processor(text=FLORENCE_CAPTION_TASK, images=crop_pil, return_tensors="pt")
            inputs = self.prepare_florence_inputs(inputs)

            with torch.no_grad():
                generated_ids = self.classifier_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=150,
                    num_beams=3,
                )
            generated_text = self.classifier_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = self.classifier_processor.post_process_generation(
                generated_text, task=FLORENCE_CAPTION_TASK, image_size=(crop_pil.width, crop_pil.height)
            )
            caption_raw = parsed[FLORENCE_CAPTION_TASK].strip()
            caption = caption_raw.lower()

            strict_match = SMOKE_COLOR_NEAR_WORD_PATTERN.search(caption)
            if strict_match:
                color_word = strict_match.group(1)
                florence_class = next(
                    (cname for keyword, cname in SMOKE_COLOR_KEYWORDS if keyword == color_word), None
                )
                if percentage_confidence >= 0.30 and florence_class == percentage_class:
                    class_name = percentage_class
                    confidence = percentage_confidence
                else:
                    class_name = "smoke"
                    confidence = 0.0
            elif percentage_confidence >= 0.30:
                class_name = percentage_class
                confidence = percentage_confidence
            else:
                class_name = "smoke"
                confidence = 0.0

            if class_name:
                results[original_index] = (class_name, confidence, caption_raw, color_percentages)
            else:
                results[original_index] = ("smoke", 0.0, caption_raw, color_percentages)

        return results

    def classify_detected_smoke(self, detections):
        """
        Adds smoke color results to YOLO detections.
        """
        if not detections:
            return detections

        crops = [d["_classification_crop"] for d in detections]
        classifications = self.classify_smoke_crops(crops)

        for detection, (class_name, classifier_conf, florence_caption, color_percentages) in zip(detections, classifications):
            detection["class_name"] = class_name
            detection["classifier_conf"] = classifier_conf
            detection["florence_caption"] = florence_caption
            detection["color_percentages"] = color_percentages
            detection.pop("_classification_crop", None)

        return detections

    def classify_one_detected_smoke(self, detection):
        """
        Classifies one smoke box.
        Used so the frontend can update one result at a time.
        """
        crop = detection.get("_classification_crop")

        class_name, classifier_conf, florence_caption, color_percentages = self.classify_smoke_crops([crop])[0]

        detection["class_name"] = class_name
        detection["classifier_conf"] = classifier_conf
        detection["florence_caption"] = florence_caption
        detection["color_percentages"] = color_percentages
        detection.pop("_classification_crop", None)
        return detection

    def describe_frame_with_vlm(self, frame_image_path):
        """
        Uses Florence to describe a saved video frame.
        Video mode uses this to check whether a recent smoke event is probably
        from the same vehicle before reusing the previous smoke classification.
        """
        frame_pil = Image.open(frame_image_path).convert("RGB")
        inputs = self.classifier_processor(text=FLORENCE_CAPTION_TASK, images=frame_pil, return_tensors="pt")
        inputs = self.prepare_florence_inputs(inputs)

        with torch.no_grad():
            generated_ids = self.classifier_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=120,
                num_beams=3,
            )

        generated_text = self.classifier_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.classifier_processor.post_process_generation(
            generated_text,
            task=FLORENCE_CAPTION_TASK,
            image_size=(frame_pil.width, frame_pil.height),
        )
        return parsed[FLORENCE_CAPTION_TASK].strip()

    def run(self, frame_bgr):
        """Runs detection and classification together."""
        detections = self.detect_smoke_boxes(frame_bgr)
        return self.classify_detected_smoke(detections)
