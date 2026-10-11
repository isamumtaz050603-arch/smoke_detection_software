from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import sys
import os

import easyocr
import pathlib

from .config import DETECTOR_CANDIDATES, FLORENCE_MODEL_DIR, FLORENCE_MODEL_FALLBACK

class ModelNotFoundError(Exception):
    """Used when the YOLO model file is missing."""

def load_yolo_model():
    """
    Finds and loads the YOLO smoke detector model.
    """
    detector_path = next((path for path in DETECTOR_CANDIDATES if path.exists()), None)
    if detector_path is None:
        raise ModelNotFoundError(
            "Missing smoke detector model. Place your one-class smoke model in the models folder "
            "or project root and name it one of: "
            + ", ".join(str(p) for p in DETECTOR_CANDIDATES)
        )

    model = YOLO(str(detector_path))
    return model, detector_path.name

def load_vehicle_model():
    """
    Loads the standard YOLOv8 model for vehicle tracking.
    """
    model = YOLO("yolov8m.pt")
    return model, "yolov8m.pt"

def load_license_plate_model():
    """Loads the YOLO license plate detector."""
    model_path = pathlib.Path("models/license_plate_detector.pt")
    if not model_path.exists():
        raise ModelNotFoundError("Missing license_plate_detector.pt in models folder.")
    model = YOLO(str(model_path))
    return model, model_path.name

def load_ocr_reader():
    """Loads the EasyOCR reader for license plate text extraction."""
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    return reader

def load_florence_model(inference_device):
    """
    Loads Florence-2 for smoke color classification.
    Also returns the processor needed to prepare images for Florence.
    """
    has_weights = (FLORENCE_MODEL_DIR / "model.safetensors").exists() or (FLORENCE_MODEL_DIR / "pytorch_model.bin").exists()
    model_source = str(FLORENCE_MODEL_DIR) if (FLORENCE_MODEL_DIR.exists() and has_weights) else FLORENCE_MODEL_FALLBACK

    model_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True)
    model.to(inference_device)
    model.eval()

    return model, processor, model_source
