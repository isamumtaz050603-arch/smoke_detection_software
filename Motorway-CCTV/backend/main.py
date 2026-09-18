import asyncio
import cv2
import json
import base64
import numpy as np
import threading
import queue
import copy
import time
import csv
import os
import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import torch
import easyocr
from transformers import AutoProcessor, AutoModelForCausalLM
import re
from concurrent.futures import ThreadPoolExecutor

from backend.engine import SmokeDetectionEngine
from backend.config import DetectionConfig, FLORENCE_MODEL_FALLBACK
from backend.drawing import draw_fast_video_detections, draw_vehicle_tracks
from backend.video_pipeline import build_smoke_event

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
engine_instance = None
stream_running = False
events_queue = queue.Queue()
latest_frames = {} # dict mapping cam_id -> base64 string
inference_lock = threading.Lock()

# Define your cameras here!
CAMERA_URLS = [0, "mock1", "mock2", "mock3"]

def initialize_engine():
    global engine_instance
    if engine_instance is not None:
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = True if device == "cuda" else False
    print(f"Initializing AI Engine on {device}...")
    
    detector = YOLO("models/best.pt")
    vehicle_model = YOLO("yolov8n.pt") 
    try:
        plate_model = YOLO("models/license_plate_detector.pt")
    except:
        plate_model = None
        
    ocr_reader = easyocr.Reader(['en'], gpu=(device=="cuda"))
    
    processor = AutoProcessor.from_pretrained(FLORENCE_MODEL_FALLBACK, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(FLORENCE_MODEL_FALLBACK, trust_remote_code=True).to(device)
    if half:
        model = model.half()
        
    config = DetectionConfig(tracker_type="ByteTrack", video_analysis_fps=5)
    
    engine_instance = SmokeDetectionEngine(
        detector_model=detector,
        classifier_model=model,
        classifier_processor=processor,
        inference_device=device,
        use_half_precision=half,
        config=config,
        vehicle_model=vehicle_model,
        license_plate_model=plate_model,
        ocr_reader=ocr_reader
    )
    engine_instance.warmup_models()
    print("Engine Ready!")

def background_classification_task(det, frame_count_snap, fps_snap, cam_id, plate_str, plate_crop, frame_copy):
    global engine_instance, events_queue
    classified_det = engine_instance.classify_one_detected_smoke(det)
    event = build_smoke_event(
        frame_number=frame_count_snap,
        fps=fps_snap,
        detections=[classified_det],
        source_name=f"Camera {cam_id}",
    )
    
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    smoke_type = event.get("Smoke Type", "Unknown")
    
    event["Timestamp"] = timestamp_str
    event["Master ID"] = classified_det.get("associated_master_id", "Unknown")
    event["Vehicle Type"] = classified_det.get("associated_class_name", "Unknown")
    event["License Plate"] = plate_str
    event["Cam ID"] = cam_id
    
    # Save the crop for frontend
    crop_bgr = classified_det.get("color_percentages") # Actually we want the crop. Wait, classify_one_detected_smoke pops it.
    # Let's draw it on the full frame instead or crop the bounding box from frame_copy
    x1, y1, x2, y2 = classified_det['xyxy']
    
    # Expand box slightly for a nice visual
    h, w = frame_copy.shape[:2]
    cx1, cy1 = max(0, x1-50), max(0, y1-50)
    cx2, cy2 = min(w, x2+50), min(h, y2+50)
    visual_crop = frame_copy[cy1:cy2, cx1:cx2]
    
    if visual_crop.size > 0:
        _, buffer = cv2.imencode('.jpg', visual_crop)
        event["CropBase64"] = base64.b64encode(buffer).decode('utf-8')
    else:
        event["CropBase64"] = ""

    # FullFrameBase64
    _, full_buffer = cv2.imencode('.jpg', frame_copy, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    event["FullFrameBase64"] = base64.b64encode(full_buffer).decode('utf-8')

    # CSV Logging
    csv_file = "cctv_report.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Camera", "Master ID", "Vehicle Type", "License Plate", "Smoke Color"])
        writer.writerow([
            timestamp_str, f"Camera {cam_id}", event["Master ID"], 
            event["Vehicle Type"], plate_str, smoke_type
        ])

    events_queue.put(event)

def camera_thread(cam_id, url):
    global engine_instance, stream_running, latest_frames, inference_lock
    
    # Handle mock strings for testing UI
    is_mock = isinstance(url, str) and url.startswith("mock")
    cap = cv2.VideoCapture(url) if not is_mock else None
    
    fps = cap.get(cv2.CAP_PROP_FPS) if cap else 25.0
    if fps <= 0: fps = 25.0
    
    frame_stride = max(1, int(fps / engine_instance.config.video_analysis_fps))
    frame_count = 0
    classification_executor = ThreadPoolExecutor(max_workers=1)
    processed_track_ids = set()
    reported_plates = set()
    
    while stream_running:
        if is_mock or not cap.isOpened():
            # Create a mock error frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Cam {cam_id} Disconnected/Mock", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            ret = True
            time.sleep(0.1) # slow down mock
        else:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                cap = cv2.VideoCapture(url)
                continue
                
        frame_count += 1
        detections = []
        tracks = []
        
        # Only analyze real frames
        if not is_mock and (frame_count % frame_stride == 0):
            with inference_lock:
                detections = engine_instance.detect_smoke_boxes(frame)
                tracks = engine_instance.track_vehicles(frame)
                if detections and tracks:
                    detections = engine_instance.associate_smoke_to_vehicles(detections, tracks)
                    
            for det in detections:
                master_id = det.get('associated_master_id')
                if master_id and master_id in processed_track_ids:
                    continue
                    
                rep_plate = "UNREADABLE"
                plate_crop = None
                if master_id:
                    with inference_lock:
                        plate_result = engine_instance.get_plate_for_track(master_id)
                    if isinstance(plate_result, tuple):
                        rep_plate, plate_crop = plate_result
                    else:
                        rep_plate = plate_result
                
                plate_str = str(rep_plate).upper()
                
                if master_id:
                    processed_track_ids.add(master_id)
                    normalized_plate = re.sub(r'[^A-Z0-9]', '', plate_str)
                    
                    if normalized_plate and "UNREADABLE" not in normalized_plate:
                        if normalized_plate in reported_plates:
                            # DUPLICATE PLATE: Skip VLM and skip event
                            continue
                        reported_plates.add(normalized_plate)
                        
                det['associated_plate'] = plate_str
                classification_executor.submit(
                    background_classification_task,
                    copy.deepcopy(det), frame_count, fps, cam_id, plate_str, plate_crop, frame.copy()
                )
                
        # Drawing
        annotated = draw_fast_video_detections(frame.copy(), detections)
        if tracks:
            emitting = set([d.get('associated_master_id') for d in detections if d.get('associated_master_id')])
            annotated = draw_vehicle_tracks(annotated, tracks, emitting)
            
        _, buffer = cv2.imencode('.jpg', annotated)
        b64 = base64.b64encode(buffer).decode('utf-8')
        
        latest_frames[cam_id] = b64
        
    if cap: cap.release()
    classification_executor.shutdown(wait=False)

@app.on_event("startup")
async def startup_event():
    threading.Thread(target=initialize_engine, daemon=True).start()

@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    global stream_running, latest_frames, events_queue, engine_instance
    
    if engine_instance is None:
        await websocket.send_json({"status": "loading models..."})
        while engine_instance is None:
            await asyncio.sleep(1)
            
    if not stream_running:
        stream_running = True
        for i, url in enumerate(CAMERA_URLS):
            threading.Thread(target=camera_thread, args=(i, url), daemon=True).start()
    
    try:
        while True:
            payload = {}
            if latest_frames:
                payload["frames"] = latest_frames.copy()
                
            new_events = []
            while not events_queue.empty():
                new_events.append(events_queue.get())
            if new_events:
                payload["events"] = new_events
                
            if payload:
                await websocket.send_json(payload)
                
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        print("WebSocket disconnected, keeping cameras running for other clients")
