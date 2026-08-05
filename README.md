# Vehicle Smoke Detection System

An advanced AI-powered pipeline designed to automatically detect, track, and classify smoke emissions from vehicles in real-time video streams. 

## How to Run

Start the app by double-clicking:
```text
run_app.bat
```
*(Note: Do not move or delete the `.venv` folder, as it contains the required local Python environment.)*

---

## Detailed System Pipeline Architecture

The system operates through a multi-stage pipeline, processing raw video frames into actionable, deduplicated emission reports.

### Phase 1: Ingestion & Frame Extraction
1. **Video Upload:** The user uploads an MP4/MOV file via the Streamlit frontend (`frontend/app.py`).
2. **Frame Sampling:** The system reads the video using OpenCV. Based on the selected "Analysis FPS" setting, it calculates a `frame_stride` to selectively process frames (e.g., analyzing 5 FPS out of a 30 FPS video to save compute).

### Phase 2: AI Detection & Tracking (`backend/engine.py`)
1. **Smoke Detection:** Each sampled frame is passed through a custom YOLOv8 model (`best.pt`). This model predicts bounding boxes specifically for smoke clouds.
2. **Vehicle Tracking (BoT-SORT):** Simultaneously, a YOLOv8 object detection model (`yolov8n.pt`) detects vehicles (Cars, Trucks, Buses, Motorcycles). 
   - We utilize **BoT-SORT** (`custom_botsort.yaml`) to assign a persistent Local Track ID to each vehicle. 
   - BoT-SORT uses a temporal buffer and a visual Re-ID feature extractor to maintain the ID even if the vehicle is temporarily occluded.
3. **Spatial Association:** The system attempts to link each smoke cloud to a specific vehicle. 
   - It expands the vehicle's bounding box by 150 pixels in all directions to create a large "catch zone" for trailing exhaust.
   - It calculates a **Spatial Composite Score** based on Intersection Over Smoke (IoS) and Box-to-Box pixel distance.
   - If the smoke aligns with a vehicle, the smoke detection inherits the vehicle's Track ID.
4. **License Plate OCR:** When a vehicle track is generated, the system crops the vehicle, runs a lightweight YOLO Plate detector, and applies EasyOCR to extract the license plate text. It stores the clearest plate reading in a `global_vehicle_map`.

### Phase 3: Temporal Fallback & Deduplication
1. **Pending Buffer:** If smoke is detected but no vehicle is found immediately (e.g., tracker warm-up delay), the smoke candidate is held in a queue for up to 15 frames. It is re-evaluated on subsequent frames until a vehicle ID is successfully bound.
2. **Strict Deduplication:** Before generating a report, the system ensures we don't spam reports for the same vehicle.
   - **Guard 1 (Master ID):** Has this Local Track ID already been reported?
   - **Guard 2 (License Plate):** Does this plate (fuzzy matched >85% similarity) match a previously reported plate?
   - *If either guard triggers, the detection is silently dropped.*

### Phase 4: Vision-Language Model Classification
1. **Background Dispatch:** Valid, deduplicated smoke events are sent to a background ThreadPoolExecutor. 
2. **Florence-2 VLM:** The frame crop containing the smoke is fed into Microsoft's Florence-2 Vision-Language Model.
3. **Color & Severity Analysis:** We use the `<CAPTION>` task in Florence-2 to generate a natural language description of the smoke. A heuristic parser then determines the Smoke Color (Black, White, Grey, Blue) and assigns an Environmental Severity score.

### Phase 5: Reporting & Export
1. **Live UI Updates:** The Streamlit frontend dynamically receives the VLM results and injects them into the "Smoke Detection Frame Table" and the rich Report Cards.
2. **Visual Overlays:** Bounding boxes are drawn onto the live video feed. Normal vehicles receive Blue boxes, while actively emitting vehicles receive Red boxes. 
3. **CSV Logging:** Events are permanently logged to `cctv_report.csv` with timestamps, Vehicle Type, ID, Plate, and Smoke Color.

---

## Directory Structure
- `frontend/` - Streamlit UI components and layout scripts.
- `backend/` - Core logic for YOLO, Florence-2 inference, tracking, drawing, and video stream handling.
- `models/` - Trained neural network weights (`best.pt`, `yolov8n.pt`, and local Florence-2 files).
- `outputs/` - Saved snapshots of emitting vehicles generated during processing.
- `support/` - Runtime cache and temporary files.
