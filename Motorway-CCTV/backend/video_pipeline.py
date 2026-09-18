import cv2
import numpy as np
import re
from datetime import datetime
from pathlib import Path

from .config import PROJECT_ROOT, get_smoke_info
from .utils import format_color_percentages

VIDEO_REPORT_FRAME_DIR = PROJECT_ROOT / "outputs" / "video_report_frames"



def make_safe_folder_name(text):
    clean_text = re.sub(r"[^A-Za-z0-9_-]+", "_", text or "uploaded_video")
    return clean_text.strip("_")[:80] or "uploaded_video"


def create_video_report_frame_folder(source_name):
    """Creates one folder for the frames saved from this video run."""
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_stem = make_safe_folder_name(Path(source_name).stem)
    frame_folder = VIDEO_REPORT_FRAME_DIR / f"{source_stem}_{run_time}"
    frame_folder.mkdir(parents=True, exist_ok=True)
    return frame_folder


def save_video_report_frame(frame_bgr, frame_folder, frame_number, video_seconds):
    """Saves the exact video frame used for a smoke report."""
    frame_folder.mkdir(parents=True, exist_ok=True)
    file_name = f"frame_{frame_number:06d}_{video_seconds:.2f}s.jpg"
    save_path = frame_folder / file_name
    cv2.imwrite(str(save_path), frame_bgr)
    return str(save_path)


def get_video_frame_stride(video_fps, video_analysis_fps):
    """Decides how many video frames to skip between YOLO checks."""
    if video_fps <= 0:
        video_fps = 25
    return max(1, int(round(video_fps / video_analysis_fps)))







def format_video_time(seconds):
    minutes = int(seconds // 60)
    whole_seconds = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def build_smoke_event(
    frame_number,
    fps,
    detections,
    source_name="Uploaded video",
    classification_time_seconds=None,
    frame_image_path="",
):
    """Builds one report row for smoke found in a video."""
    video_seconds = frame_number / max(fps, 1)
    avg_confidence = np.mean([d["detector_conf"] for d in detections]) * 100
    smoke_types = [get_smoke_info(d.get("class_name", "smoke"))["label"] for d in detections]
    color_percentages = [
        format_color_percentages(d.get("color_percentages"))
        for d in detections
        if d.get("color_percentages")
    ]
    florence_captions = [d.get("florence_caption", "") for d in detections if d.get("florence_caption")]
    now = datetime.now()
    return {
        "Alert": "Smoke detected",
        "Date": now.strftime("%Y-%m-%d"),
        "First Seen Time": now.strftime("%H:%M:%S"),
        "Real Date/Time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "Video Time": format_video_time(video_seconds),
        "Video Seconds": round(video_seconds, 2),
        "Frame": frame_number,
        "Smoke Boxes": len(detections),
        "Smoke Detection %": f"{avg_confidence:.1f}%",
        "Smoke Type": ", ".join(smoke_types),
        "Color Percentages": " || ".join(color_percentages),
        "Average Confidence": f"{avg_confidence:.1f}%",
        "Average Smoke Confidence": f"{avg_confidence:.1f}%",
        "Classification Time": f"{classification_time_seconds:.2f}s" if classification_time_seconds is not None else "",
        "Florence Caption": " || ".join(florence_captions),
        "Frame Image": frame_image_path,
        "Source": source_name,
        "Message": "Smoke emission detected.",
    }


def create_video_writer(output_path, fps, width, height):
    """Creates the output video file writer."""
    codecs = ["mp4v", "avc1", "H264"]
    for codec in codecs:
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError("Could not create the output video writer. Try using an MP4 input video.")
