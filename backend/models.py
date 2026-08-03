from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

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

def load_florence_model(inference_device):
    """
    Loads Florence-2 for smoke color classification.
    Also returns the processor needed to prepare images for Florence.
    """
    model_source = str(FLORENCE_MODEL_DIR) if FLORENCE_MODEL_DIR.exists() else FLORENCE_MODEL_FALLBACK

    model_dtype = torch.float16 if str(inference_device).startswith("cuda") else torch.float32
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
