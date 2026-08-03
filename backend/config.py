import re
from dataclasses import dataclass
from pathlib import Path

# -----------------------------------------------
# Model file locations
# -----------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DETECTOR_CANDIDATES = [
    PROJECT_ROOT / "models" / "best.pt",
]

# Florence-2 is the vision-language model used after YOLO finds smoke.
# YOLO tells us where the smoke is; Florence classifies the cropped smoke area.
# If this local folder exists, the app loads Florence from your computer.
# If it does not exist, the app tries to load Florence from Hugging Face.
FLORENCE_MODEL_DIR = PROJECT_ROOT / "models" / "florence2_smoke_classifier"
FLORENCE_MODEL_FALLBACK = "microsoft/Florence-2-base"  # used only if local folder is missing (needs internet)
FLORENCE_CAPTION_TASK = "<MORE_DETAILED_CAPTION>"  # asks Florence to describe the crop

# -----------------------------------------------
# Smoke type report details.
# The frontend uses this to show label, status, cause, and action.
# -----------------------------------------------
SMOKE_INFO = {
    "black_smoke": {
        "label": "Black Smoke",
        "color": "#cc0000",
        "severity": "Severe",
        "status": "UNFIT",
        "cause": "Incomplete combustion or excess fuel",
        "action": "Immediate engine and fuel system inspection required",
    },
    "white_smoke": {
        "label": "White Smoke",
        "color": "#777777",
        "severity": "Moderate",
        "status": "REVIEW",
        "cause": "Coolant or water may be burning in engine",
        "action": "Inspect cooling system and head gasket",
    },
    "grey_smoke": {
        "label": "Grey Smoke",
        "color": "#555555",
        "severity": "Moderate to High",
        "status": "REVIEW",
        "cause": "Oil burning or transmission fluid issue",
        "action": "Check engine oil and transmission system",
    },
    "blue_smoke": {
        "label": "Blue Smoke",
        "color": "#1a4f8f",
        "severity": "High",
        "status": "UNFIT",
        "cause": "Engine oil burning in combustion chamber",
        "action": "Inspect piston rings and valve seals",
    },
    "smoke": {
        "label": "Smoke",
        "color": "#8b0000",
        "severity": "High",
        "status": "REVIEW",
        "cause": "Visible exhaust emission detected",
        "action": "Vehicle emission inspection required",
    }
}

def get_smoke_info(class_name):
    clean_name = class_name.lower().replace(" ", "_")
    return SMOKE_INFO.get(clean_name, SMOKE_INFO["smoke"])

# -----------------------------------------------
# Words Florence might use for smoke color.
# -----------------------------------------------
SMOKE_COLOR_KEYWORDS = [
    ("black", "black_smoke"),
    ("white", "white_smoke"),
    ("grey", "grey_smoke"),
    ("gray", "grey_smoke"),   # Same color, different spelling.
    ("blue", "blue_smoke"),
]

# Finds phrases like "black smoke", "white exhaust", or "grey fumes".
SMOKE_COLOR_NEAR_WORD_PATTERN = re.compile(
    r"\b(black|white|grey|gray|blue)\b\s+(smoke|exhaust|fumes)\b"
)

# -----------------------------------------------
# Settings sent from the Streamlit sidebar into the backend.
# -----------------------------------------------
@dataclass
class DetectionConfig:
    detector_conf: float = 0.50
    iou_threshold: float = 0.60
    video_analysis_fps: int = 10
    detector_image_size: int = 512
    report_interval_seconds: float = 1.0
