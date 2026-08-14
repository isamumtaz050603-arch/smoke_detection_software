# Vehicle Smoke Detection System

An advanced AI-powered pipeline designed to automatically detect, track, and classify smoke emissions from vehicles in real-time video streams. 

## How to Run

Start the app by double-clicking:
```text
run_app.bat
```
*(Note: Do not move or delete the `.venv` folder, as it contains the required local Python environment.)*

### Core Dependencies
- `streamlit`, `streamlit-webrtc` (for Live Webcam Streaming), `av`
- `torch`, `ultralytics` (YOLOv8), `transformers` (Florence-2)
- `opencv-python`, `easyocr`

---

## Detailed System Pipeline Architecture

The system operates through a highly optimized, asynchronous multi-stage pipeline, processing raw video frames into actionable, deduplicated emission reports.

### Phase 1: Ingestion & Asynchronous Dispatch
The pipeline supports two major input modes:
1. **Live Webcam (WebRTC):** Uses `streamlit-webrtc` to establish a direct peer-to-peer UDP/TCP connection to the browser. It bypasses Streamlit's WebSocket latency and processes YOLO inference directly on the incoming live video stream frames (processing every 5th frame to maintain real-time performance).
2. **Video Upload (Async Queue):** When a video file is uploaded, a background Producer thread extracts the frames via OpenCV and runs the YOLO AI models. The annotated frames and metadata are pushed to a `queue.Queue`, allowing the main Streamlit Consumer thread to render the UI and write the video file to disk without freezing.

### Phase 2: AI Detection & Tracking (`backend/engine.py`)
1. **Smoke Detection:** Each analyzed frame is passed through a custom YOLOv8 model (`best.pt`). This model predicts bounding boxes specifically for smoke clouds.
2. **Vehicle Tracking (BoT-SORT):** Simultaneously, a YOLOv8 object detection model (`yolov8n.pt`) detects vehicles (Cars, Trucks, Buses, Motorcycles). 
   - We utilize **BoT-SORT** (`custom_botsort.yaml`) to assign a persistent Local Track ID to each vehicle. 
3. **Spatial Association:** The system attempts to link each smoke cloud to a specific vehicle. 
   - It expands the vehicle's bounding box by 150 pixels horizontally and vertically to create a large "catch zone" for trailing exhaust.
   - It calculates a **Spatial Composite Score** (`0.6 * IoS + 0.4 * Distance_Penalty`).
   - If the score is **> 0.20**, the smoke is successfully "parented" to the vehicle's Master ID.
4. **License Plate OCR:** When a vehicle track is generated, the system crops the vehicle, runs a lightweight YOLO Plate detector, and applies EasyOCR to extract the license plate text.

### Phase 3: Temporal Fallback & Strict Deduplication
1. **Orphaned Smoke (Temporal Fallback):** If smoke is detected but no vehicle is found immediately, it checks the last 15 frames in history (requiring a lower threshold of **> 0.08**). If still not found, it is held in a "pending buffer" for up to 15 future frames. If no vehicle ever appears, the smoke is silently dropped.
2. **Track Stitching (Backend):** If occlusion causes a vehicle ID to switch, the system merges the IDs if the old/new license plates match **(> 85% similarity)** or if their historical spatial bounding boxes overlap **(> 50% IoU)**.
3. **Global Deduplication (Frontend):** Before generating a report, the system ensures we don't spam reports for the same vehicle.
   - **Guard 1 (Master ID):** Has this Local Track ID already been reported?
   - **Guard 2 (License Plate):** Does this plate (fuzzy matched **> 85% similarity**) match a previously reported plate?
   - *If either guard triggers, the detection is silently dropped.*

### Phase 4: Vision-Language Model Classification
1. **Background Dispatch:** Valid, deduplicated smoke events are sent to a background `ThreadPoolExecutor`. 
2. **Florence-2 VLM:** The frame crop containing the smoke is fed into Microsoft's 774M parameter Florence-2 Vision-Language Model (isolated from the main thread to prevent UI stalling).
3. **Color & Severity Analysis:** We use the `<CAPTION>` task in Florence-2 to generate a natural language description of the smoke. A heuristic parser then determines the Smoke Color (Black, White, Grey, Blue) and assigns an Environmental Severity score.

### Phase 5: Reporting & Export
1. **Live UI Updates:** The Streamlit frontend dynamically receives the completed `Future` from the thread pool and injects it into the "Smoke Detection Frame Table" and the rich Report Cards.
2. **Visual Overlays:** Bounding boxes are drawn onto the live video feed. Normal vehicles receive Blue boxes, while actively emitting vehicles receive Red boxes. 
3. **Thread-Safe CSV Logging:** To prevent race conditions from the background threads, the pipeline acquires a `threading.Lock()` mutex to safely append the final event to `cctv_report.csv` (containing timestamps, Vehicle Type, ID, Plate, and Smoke Color).

---

## Directory Structure
- `frontend/` - Streamlit UI components, queues, and layout scripts.
- `backend/` - Core logic for YOLO, Florence-2 inference, tracking, drawing, and video stream handling.
- `models/` - Trained neural network weights (`best.pt`, `yolov8n.pt`, and local Florence-2 files).
- `outputs/` - Saved snapshots of emitting vehicles generated during processing.
- `support/` - Runtime cache and temporary files.
