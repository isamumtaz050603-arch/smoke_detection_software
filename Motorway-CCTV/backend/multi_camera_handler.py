import cv2
import time
import numpy as np
import streamlit as st
import threading
import queue
import copy
import difflib
import pandas as pd
import csv
import os
import datetime
from concurrent.futures import ThreadPoolExecutor

from backend.engine import SmokeDetectionEngine
from backend.video_pipeline import get_video_frame_stride, build_smoke_event, create_video_report_frame_folder, save_video_report_frame
from backend.drawing import draw_detections, draw_fast_video_detections, draw_vehicle_tracks
from backend.utils import render_display_status
from backend.config import get_smoke_info

def run_multi_camera_stream(
    urls,
    base_engine,
    config,
    display_video_slots,
    display_status_slots,
    event_frame_slot,
    report_slot,
    session_state,
    track_lookback_frames
):
    """
    Handles multiple CCTV streams concurrently.
    Spawns a thread for each camera to read frames, runs YOLO sequentially via a lock,
    and updates the Streamlit UI in the main thread.
    """
    num_cameras = len(urls)
    
    # 1. Create separate Engine instances to isolate tracking state!
    engines = []
    for _ in range(num_cameras):
        e = SmokeDetectionEngine(
            detector_model=base_engine.detector_model,
            classifier_model=base_engine.classifier_model,
            classifier_processor=base_engine.classifier_processor,
            inference_device=base_engine.inference_device,
            use_half_precision=base_engine.use_half_precision,
            config=config,
            vehicle_model=base_engine.vehicle_model,
            license_plate_model=base_engine.license_plate_model,
            ocr_reader=base_engine.ocr_reader,
        )
        engines.append(e)
        
    # 2. Shared Resources & Thread Safety
    frame_queue = queue.Queue(maxsize=30 * num_cameras)
    inference_lock = threading.Lock()
    
    if "smoke_events" not in session_state:
        session_state["smoke_events"] = []
        
    reported_vids = {i: {} for i in range(num_cameras)}
    reported_plates = {i: {} for i in range(num_cameras)}
    
    # Create folder for saving frames
    report_frame_folder = create_video_report_frame_folder("Multi-Camera-Session")
    
    classification_executor = ThreadPoolExecutor(max_workers=1)
    classification_futures = []
    
    def background_classification_task(det, frame_count_snap, fps_snap, url_snap, plate_str, plate_crop, frame_copy):
        classified_det = base_engine.classify_one_detected_smoke(det)
        event = build_smoke_event(
            frame_number=frame_count_snap,
            fps=fps_snap,
            detections=[classified_det],
            source_name=f"Cam: {url_snap}",
        )
        event["Master ID"] = classified_det.get("associated_master_id", "Unknown")
        event["Vehicle Type"] = classified_det.get("associated_class_name", "Unknown")
        event["License Plate"] = plate_str
        event["Source"] = f"Cam: {url_snap}"
        return event, classified_det, plate_crop, frame_copy
        
    def camera_reader_thread(cam_idx, url, engine_inst):
        cap = cv2.VideoCapture(url if not str(url).isdigit() else int(url))
        reconnect_attempts = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0: fps = 25.0
        frame_stride = get_video_frame_stride(fps, config.video_analysis_fps)
        frame_count = 0
        last_detections = []
        last_tracks = []
        
        while session_state.get("stream_running", False):
            ret, frame = cap.read()
            if not ret:
                reconnect_attempts += 1
                if reconnect_attempts > 10: break
                time.sleep(2)
                cap = cv2.VideoCapture(url if not str(url).isdigit() else int(url))
                continue
                
            reconnect_attempts = 0
            frame_count += 1
            should_analyze = (frame_count % frame_stride == 0)
            
            if should_analyze:
                with inference_lock:
                    detections = engine_inst.detect_smoke_boxes(frame)
                    if config.tracker_type != "None":
                        tracks = engine_inst.track_vehicles(frame)
                        if detections:
                            detections = engine_inst.associate_smoke_to_vehicles(
                                detections, tracks, lookback_frames=track_lookback_frames
                            )
                    else:
                        tracks = []
                        
                # Handle ANPR and Deduplication (Outside lock to save GPU time)
                for det in detections:
                    raw_master_id = det.get('associated_master_id')
                    master_id_str = str(raw_master_id) if raw_master_id is not None else None
                    
                    if master_id_str is not None and master_id_str in reported_vids[cam_idx]:
                        continue
                        
                    rep_plate = "UNREADABLE"
                    plate_crop = None
                    if raw_master_id is not None:
                        with inference_lock:
                            plate_result = engine_inst.get_plate_for_track(raw_master_id)
                        if isinstance(plate_result, tuple):
                            rep_plate, plate_crop = plate_result
                        else:
                            rep_plate = plate_result
                            
                    plate_str = str(rep_plate).upper() if rep_plate is not None else "UNREADABLE"
                    skip = False
                    if plate_str != "UNREADABLE" and plate_str != "NONE":
                        for known_plate in list(reported_plates[cam_idx].keys()):
                            if difflib.SequenceMatcher(None, plate_str, known_plate).ratio() > 0.85:
                                skip = True
                                break
                        if not skip:
                            reported_plates[cam_idx][plate_str] = True
                            
                    if skip: continue
                    if master_id_str is not None:
                        reported_vids[cam_idx][master_id_str] = True
                        
                    det['associated_plate'] = plate_str
                    frame_copy = frame.copy()
                    
                    # Submit to Florence-2 queue
                    future = classification_executor.submit(
                        background_classification_task,
                        copy.deepcopy(det), frame_count, fps, url, plate_str,
                        copy.deepcopy(plate_crop) if plate_crop is not None else None, frame_copy
                    )
                    classification_futures.append(future)
                    
                if detections:
                    last_detections = detections
                last_tracks = tracks
                
            try:
                frame_queue.put_nowait({
                    "cam_idx": cam_idx,
                    "frame": frame,
                    "detections": last_detections,
                    "tracks": last_tracks
                })
            except queue.Full:
                pass # Drop frames if UI is too slow
                
        cap.release()
        frame_queue.put({"cam_idx": cam_idx, "stop": True})
        
    # Start threads
    threads = []
    active_cameras = num_cameras
    for i, url in enumerate(urls):
        t = threading.Thread(target=camera_reader_thread, args=(i, url, engines[i]), daemon=True)
        t.start()
        threads.append(t)
        
    # Main Streamlit UI Loop
    while session_state.get("stream_running", False) and active_cameras > 0:
        try:
            item = frame_queue.get(timeout=1.0)
            cam_idx = item["cam_idx"]
            
            if item.get("stop"):
                active_cameras -= 1
                display_status_slots[cam_idx].error("Camera Disconnected.")
                continue
                
            frame = item["frame"]
            detections = item["detections"]
            tracks = item["tracks"]
            
            # Draw
            annotated_frame = draw_fast_video_detections(frame.copy(), detections)
            emitting_ids = set([d.get('associated_master_id') for d in detections if d.get('associated_master_id') is not None])
            if config.tracker_type != "None" and tracks:
                annotated_frame = draw_vehicle_tracks(annotated_frame, tracks, emitting_ids)
                
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            display_video_slots[cam_idx].image(annotated_rgb, channels="RGB", use_container_width=True)
            display_status_slots[cam_idx].markdown(render_display_status(detections), unsafe_allow_html=True)
            
        except queue.Empty:
            pass
            
        # Process completed Florence-2 results
        completed = [f for f in classification_futures if f.done()]
        for future in completed:
            try:
                event, classified_det, plate_crop, frame_snap = future.result()
                session_state["smoke_events"].append(event)
                
                # Write to CSV
                timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                smoke_color = event.get("Smoke Type", "Unknown")
                try:
                    severity = get_smoke_info(smoke_color).get("severity", "Unknown")
                except:
                    severity = "Unknown"
                    
                # Save the frame to disk
                rep_bgr = draw_fast_video_detections(frame_snap, [classified_det])
                frame_path = save_video_report_frame(
                    rep_bgr, report_frame_folder, event.get("Frame", 0), event.get("Video Seconds", 0.0)
                )
                
                with open("cctv_report.csv", "a", newline="") as f:
                    csv_writer = csv.writer(f)
                    if os.path.getsize("cctv_report.csv") == 0:
                        csv_writer.writerow(["Timestamp", "Camera", "Master Vehicle ID", "Vehicle Type", "License Plate", "Smoke Color", "Severity Level", "Frame Path"])
                    csv_writer.writerow([
                        timestamp_str, event.get("Source", "Unknown"), event.get("Master ID", "Unknown"), 
                        event.get("Vehicle Type", "Unknown"), event.get("License Plate", "Unknown"), 
                        smoke_color, severity, frame_path
                    ])
                
                # Update UI
                with event_frame_slot.container():
                    c1, c2 = st.columns(2)
                    c1.image(cv2.cvtColor(rep_bgr, cv2.COLOR_BGR2RGB), caption=f"Event from {event.get('Source')}")
                    if plate_crop is not None and plate_crop.size > 0:
                        c2.image(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB), caption=f"Plate: {event['License Plate']}")
                    else:
                        c2.info(f"Plate: {event['License Plate']} (No crop)")
                        
                with report_slot.container():
                    df = pd.DataFrame(session_state["smoke_events"])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
            except Exception as e:
                print(f"Florence-2 Error: {e}")
            classification_futures.remove(future)
            
    classification_executor.shutdown(wait=False)
