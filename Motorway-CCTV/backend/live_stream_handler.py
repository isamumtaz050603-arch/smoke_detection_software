import cv2
import time
import numpy as np
import streamlit as st
import difflib
import pandas as pd
import copy
from concurrent.futures import ThreadPoolExecutor

# Import existing backend utilities
from backend.video_pipeline import get_video_frame_stride, build_smoke_event
from backend.drawing import draw_detections, draw_fast_video_detections
from backend.utils import render_display_status

def run_live_stream(
    url, 
    engine, 
    config, 
    raw_video_slot, 
    display_video_slot, 
    display_status_slot, 
    event_frame_slot,
    report_slot,
    session_state, 
    track_lookback_frames
):
    """
    Handles the continuous webcam/RTSP stream loop.
    Applies the full YOLO + Tracking + ANPR + Florence-2 pipeline dynamically without locking the UI.
    Uses asynchronous background threading for Florence-2 to prevent stream stuttering.
    """
    cap = cv2.VideoCapture(url)
    
    # Auto-reconnect settings
    max_reconnect_attempts = 10
    reconnect_attempts = 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
        
    frame_stride = get_video_frame_stride(fps, config.video_analysis_fps)
    frame_count = 0
    
    last_detections = []
    
    # Ensure our session state list for the report exists
    if "smoke_events" not in session_state:
        session_state["smoke_events"] = []

    # Local caches to deduplicate events so we don't classify the same vehicle emitting smoke over and over
    reported_vids = {}
    reported_plates = {}
    
    # Asynchronous Florence-2 classification executor (like Video Mode)
    classification_executor = ThreadPoolExecutor(max_workers=1)
    classification_futures = []

    def background_classification_task(det, frame_count_snap, fps_snap, url_snap, plate_str, plate_crop, frame_copy):
        """ Runs Florence-2 classification in the background thread. """
        classified_det = engine.classify_one_detected_smoke(det)
        
        event = build_smoke_event(
            frame_number=frame_count_snap,
            fps=fps_snap,
            detections=[classified_det],
            source_name=f"Webcam {url_snap}",
        )
        
        event["Master ID"] = classified_det.get("associated_master_id", "Unknown")
        event["Vehicle Type"] = classified_det.get("associated_class_name", "Unknown")
        event["License Plate"] = plate_str
        
        return event, classified_det, plate_crop, frame_copy

    # Streamlit will interrupt this loop if the user clicks "Stop Stream" and st.session_state changes
    while session_state.get("stream_running", False):
        try:
            ret, frame = cap.read()
            if not ret:
                reconnect_attempts += 1
                if reconnect_attempts > max_reconnect_attempts:
                    display_status_slot.error("Connection lost. Max reconnect attempts reached.")
                    break
                
                display_status_slot.warning(f"Connection lost. Reconnecting... (Attempt {reconnect_attempts}/{max_reconnect_attempts})")
                time.sleep(2)
                cap = cv2.VideoCapture(url)
                continue
                
            reconnect_attempts = 0
            frame_count += 1
            
            # 1. Update the Raw Video Feed dynamically (Non-blocking)
            raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_video_slot.image(raw_rgb, channels="RGB", use_container_width=True)
            
            # Determine if this frame should be passed to the YOLO inference engine
            should_analyze = (frame_count % frame_stride == 0)
            
            if should_analyze:
                # 2. Detect Smoke (YOLO)
                detections = engine.detect_smoke_boxes(frame)
                
                # 3. Tracking & Association
                if config.tracker_type != "None":
                    tracks = engine.track_vehicles(frame)
                    if detections:
                        detections = engine.associate_smoke_to_vehicles(
                            detections, 
                            tracks, 
                            lookback_frames=track_lookback_frames
                        )
                
                for det in detections:
                    raw_master_id = det.get('associated_master_id')
                    master_id_str = str(raw_master_id) if raw_master_id is not None else None
                    
                    # Deduplication guard: Skip if we already reported this vehicle
                    if master_id_str is not None and master_id_str in reported_vids:
                        continue
                        
                    # Trigger ANPR Backtrack Logic!
                    rep_plate = "UNREADABLE"
                    plate_crop = None
                    if raw_master_id is not None:
                        plate_result = engine.get_plate_for_track(raw_master_id)
                        if isinstance(plate_result, tuple):
                            rep_plate, plate_crop = plate_result
                        else:
                            rep_plate = plate_result
                            
                    plate_str = str(rep_plate).upper() if rep_plate is not None else "UNREADABLE"
                    
                    # Deduplication guard 2: Fuzzy License Plate Check
                    skip = False
                    if plate_str != "UNREADABLE" and plate_str != "NONE":
                        for known_plate in list(reported_plates.keys()):
                            if difflib.SequenceMatcher(None, plate_str, known_plate).ratio() > 0.85:
                                skip = True
                                break
                        if not skip:
                            reported_plates[plate_str] = True
                            
                    if skip:
                        continue
                        
                    if master_id_str is not None:
                        reported_vids[master_id_str] = True
                        
                    # Secure plate onto detection metadata
                    det['associated_plate'] = plate_str
                    
                    # 4. Submit Florence-2 to Background Thread (Asynchronous!)
                    frame_copy = frame.copy()
                    future = classification_executor.submit(
                        background_classification_task,
                        copy.deepcopy(det),
                        frame_count,
                        fps,
                        url,
                        plate_str,
                        copy.deepcopy(plate_crop) if plate_crop is not None else None,
                        frame_copy
                    )
                    classification_futures.append(future)
                    
                # Persist the current detections for drawing across frames
                if detections:
                    last_detections = detections
                
            # Process completed background tasks (Florence-2 results) continuously
            completed_futures = [f for f in classification_futures if f.done()]
            for future in completed_futures:
                try:
                    event, classified_det, plate_crop, frame_snap = future.result()
                    
                    # 1. Update session state
                    session_state["smoke_events"].append(event)
                    
                    # 2. Real-Time UI Updates for Event Frames
                    with event_frame_slot.container():
                        st.markdown(f"**New Event: {classified_det.get('class_name', 'smoke')}**")
                        c1, c2 = st.columns(2)
                        
                        # Show the full detection frame highlighting the smoke
                        report_frame_bgr = draw_fast_video_detections(frame_snap, [classified_det])
                        report_frame_rgb = cv2.cvtColor(report_frame_bgr, cv2.COLOR_BGR2RGB)
                        c1.image(report_frame_rgb, caption=f"Smoke Frame {event['Frame']}")
                        
                        # Show the ANPR Crop if available
                        if plate_crop is not None and plate_crop.size > 0:
                            plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                            c2.image(plate_rgb, caption=f"Plate Read: {event['License Plate']}")
                        else:
                            c2.info(f"Plate Read: {event['License Plate']} (No Clear Frame Available)")
                            
                    # 3. Real-Time Live Report UI Update
                    with report_slot.container():
                        df = pd.DataFrame(session_state["smoke_events"])
                        cols = ["Real Date/Time", "Master ID", "Vehicle Type", "License Plate", "Smoke Type"]
                        available_cols = [c for c in cols if c in df.columns]
                        st.dataframe(df[available_cols], use_container_width=True, hide_index=True)
                        
                except Exception as e:
                    print(f"Background classification error: {e}")
                
                classification_futures.remove(future)
                
            # 5. Draw Detections on Live Feed
            annotated_frame = draw_detections(frame.copy(), last_detections, label_font=None)
            
            # 6. Update Processed Video Feed and Status (Non-blocking updates to placeholders)
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            display_video_slot.image(annotated_rgb, channels="RGB", use_container_width=True)
            
            status_html = render_display_status(last_detections)
            display_status_slot.markdown(status_html, unsafe_allow_html=True)
            
        except Exception as e:
            display_status_slot.error(f"Error in stream loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(2)
            cap = cv2.VideoCapture(url)
            
    classification_executor.shutdown(wait=False)
    cap.release()
