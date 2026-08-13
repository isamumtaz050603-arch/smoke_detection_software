import cv2
import numpy as np
import torch
import difflib
from concurrent.futures import ThreadPoolExecutor
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
    def __init__(self, detector_model, classifier_model, classifier_processor, inference_device, use_half_precision, config, vehicle_model=None, license_plate_model=None, ocr_reader=None):
        self.detector_model = detector_model
        self.classifier_model = classifier_model
        self.classifier_processor = classifier_processor
        self.inference_device = inference_device
        self.use_half_precision = use_half_precision
        self.config = config
        self.vehicle_model = vehicle_model
        self.license_plate_model = license_plate_model
        self.ocr_reader = ocr_reader
        
        # Dual-Layer Tracking state
        self.global_vehicle_map = {} # Plate Text -> Master Track ID
        self.local_to_master_id = {} # Local YOLO ID -> Master Track ID
        self.unreadable_cache = set() # Master Track IDs that failed ANPR
        import collections
        self.track_history_buffer = collections.defaultdict(lambda: collections.deque(maxlen=150))
        self.global_frame_history = collections.deque(maxlen=150)

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
    # Tracking: YOLO tracks vehicles (ByteTrack / BoT-SORT).
    # ------------------------------------------------------------------
    def track_vehicles(self, frame_bgr):
        """
        Runs tracking for vehicles.
        Returns a list of dictionaries with vehicle boxes and track IDs.
        """
        if not self.vehicle_model or self.config.tracker_type not in ["ByteTrack", "BoT-SORT"]:
            return []

        tracker_yaml = "models/custom_bytetrack.yaml" if self.config.tracker_type == "ByteTrack" else "models/custom_botsort.yaml"
        # COCO vehicle classes: 2: car, 3: motorcycle, 5: bus, 7: truck
        try:
            results = self.vehicle_model.track(
                frame_bgr,
                persist=True,
                tracker=tracker_yaml,
                classes=[2, 3, 5, 7],
                verbose=False,
                conf=0.15, # Lower confidence to track obscured vehicles
                iou=0.45
            )
        except Exception:
            return []

        if not results or not results[0].boxes or results[0].boxes.id is None:
            return []

        boxes = results[0].boxes
        tracks = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            local_id = int(box.id.item())
            class_id = int(box.cls.item())
            class_name = self.vehicle_model.names.get(class_id, "Vehicle")
            
            box_width = x2 - x1
            box_height = y2 - y1
            
            master_id = self.local_to_master_id.get(local_id)
            
            if master_id is None:
                # Fallback to local_id immediately to prevent tracking drops
                master_id = local_id 
                self.local_to_master_id[local_id] = local_id
                
            # Save state to rolling 30-frame buffer for temporal lookback
            vehicle_crop = frame_bgr[y1:y2, x1:x2].copy()
            self.track_history_buffer[master_id].append({
                'crop': vehicle_crop,
                'bbox': [x1, y1, x2, y2],
                'class_name': class_name
            })

            tracks.append({
                "xyxy": [x1, y1, x2, y2],
                "track_id": master_id,
                "class_name": class_name,
            })
        
        self.global_frame_history.append(tracks)
        return tracks

    def get_plate_for_track(self, track_id):
        """
        Attempts ANPR on the current and past 5 crops of a track_id.
        Returns the recognized plate text, or 'UNREADABLE'.
        """
        if not self.license_plate_model or not self.ocr_reader:
            return "UNREADABLE"

        # FAST PATH CACHE: Return immediately if we already know the plate
        for known_plate, data in self.global_vehicle_map.items():
            if data[0] == track_id:
                return known_plate, data[1]

        # UNREADABLE CACHE: Don't repeatedly scan tracks that already failed
        if track_id in self.unreadable_cache:
            return "UNREADABLE", None

        # Check all available history frames, sorted by bounding box area (largest/closest first)
        history = list(self.track_history_buffer.get(track_id, []))
        
        def get_area(entry):
            x1, y1, x2, y2 = entry['bbox']
            return (x2 - x1) * (y2 - y1)
            
        history_to_check = sorted(history, key=get_area, reverse=True)
        
        best_plate_text = ""
        best_vehicle_crop = None
        
        for entry in history_to_check:
            vehicle_crop = entry['crop']
            if vehicle_crop.size == 0:
                continue
                
            plate_results = self.license_plate_model.predict(vehicle_crop, verbose=False)
            if not plate_results or not plate_results[0].boxes:
                continue
                
            # Take the highest confidence plate box
            best_plate_box = max(plate_results[0].boxes, key=lambda b: float(b.conf.item()))
            
            # Confidence threshold guard
            if float(best_plate_box.conf.item()) <= 0.45:
                continue
                
            px1, py1, px2, py2 = map(int, best_plate_box.xyxy[0].cpu().numpy())
            plate_crop = vehicle_crop[py1:py2, px1:px2]
            
            ocr_results = self.ocr_reader.readtext(
                plate_crop, 
                detail=0,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -',
                mag_ratio=2
            )
            if not ocr_results:
                continue
                
            # Join text if multiple blocks are found
            plate_text = "".join(ocr_results).replace(" ", "").replace("-", "").upper()
            
            # Keep the longest and most complete plate read
            if len(plate_text) > len(best_plate_text):
                best_plate_text = plate_text
                best_vehicle_crop = vehicle_crop
                
            # If we found a solidly long plate, we can stop early to save CPU
            if len(best_plate_text) >= 6:
                break
                
        if len(best_plate_text) >= 3:
            # Fuzzy Match
            best_match_id = None
            best_match_crop = None
            best_ratio = 0.0
            
            for known_plate, data in self.global_vehicle_map.items():
                ratio = difflib.SequenceMatcher(None, best_plate_text, known_plate).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match_id = data[0]
                    best_match_crop = data[1]
                    
            if best_ratio > 0.85:
                # It's an existing vehicle (link broken tracks)
                self.local_to_master_id[track_id] = best_match_id
                return known_plate, best_match_crop
            else:
                # It's a new vehicle
                self.global_vehicle_map[best_plate_text] = (track_id, best_vehicle_crop)
                return best_plate_text, best_vehicle_crop
                
        # If we reached here, the plate was completely unreadable for this track_id.
        # Track Stitching Fallback: 
        # If this is a new broken track (new ID, short history), search the global history 
        # for an older track ID that spatially overlaps with where this vehicle first appeared!
        if history:
            oldest_crop_entry = history[0] # The very first time we saw this track_id
            ox1, oy1, ox2, oy2 = oldest_crop_entry['bbox']
            
            for past_tracks in reversed(list(self.global_frame_history)):
                for vehicle in past_tracks:
                    past_id = vehicle['track_id']
                    
                    # Ignore itself, and ignore tracks we already know are unreadable
                    if past_id != track_id and past_id not in self.unreadable_cache:
                        vx1, vy1, vx2, vy2 = vehicle['xyxy']
                        
                        # Calculate Intersection over Union (IoU)
                        ix1 = max(ox1, vx1)
                        iy1 = max(oy1, vy1)
                        ix2 = min(ox2, vx2)
                        iy2 = min(oy2, vy2)
                        
                        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        if intersection > 0:
                            area1 = (ox2 - ox1) * (oy2 - oy1)
                            area2 = (vx2 - vx1) * (vy2 - vy1)
                            iou = intersection / float(area1 + area2 - intersection)
                            
                            if iou > 0.5: # 50% spatial overlap means it's almost certainly the same car
                                # We found the ghost of the previous track ID!
                                # Let's steal its license plate.
                                stitched_plate, stitched_crop = self.get_plate_for_track(past_id)
                                if stitched_plate != "UNREADABLE":
                                    self.local_to_master_id[track_id] = past_id
                                    return stitched_plate, stitched_crop

        self.unreadable_cache.add(track_id)
        return "UNREADABLE", None

    # ------------------------------------------------------------------
    # Association: Link Smoke to Vehicles
    # ------------------------------------------------------------------
    def associate_smoke_to_vehicles(self, smoke_detections, vehicle_tracks, lookback_frames=15):
        """
        Associates smoke boxes to master vehicle IDs based on a Spatial Composite Score.
        """
        for smoke in smoke_detections:
            smoke['associated_master_id'] = None
            smoke['associated_class_name'] = None
            smoke['associated_plate'] = None
            
            sx1, sy1, sx2, sy2 = smoke['xyxy']
            smoke_area = float((sx2 - sx1) * (sy2 - sy1))
            if smoke_area <= 0:
                continue
                
            sx_mid = (sx1 + sx2) / 2.0
            sy_min = sy1 # top of smoke
            
            best_score = 0.0
            best_master_id = None
            best_class_name = None
            
            for vehicle in vehicle_tracks:
                vx1, vy1, vx2, vy2 = vehicle['xyxy']
                
                # Expand vehicle box aggressively to cover exhaust zones and trailing smoke
                vx1_expanded = vx1 - 150
                vy1_expanded = vy1 - 50
                vx2_expanded = vx2 + 150
                vy2_expanded = vy2 + 150
                
                # Calculate Intersection with expanded box
                ix1 = max(sx1, vx1_expanded)
                iy1 = max(sy1, vy1_expanded)
                ix2 = min(sx2, vx2_expanded)
                iy2 = min(sy2, vy2_expanded)
                
                intersection_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                ios = intersection_area / smoke_area
                
                # Box-to-Box distance (shortest distance between the two bounding boxes)
                dx = max(0.0, vx1 - sx2, sx1 - vx2)
                dy = max(0.0, vy1 - sy2, sy1 - vy2)
                box_distance = np.sqrt(dx**2 + dy**2)
                
                composite_score = (0.6 * ios) + (0.4 * (1.0 / (1.0 + (box_distance / 10.0))))
                
                if composite_score > best_score and composite_score > 0.20:
                    best_score = composite_score
                    best_master_id = vehicle['track_id']
                    best_class_name = vehicle['class_name']
            
            if best_master_id is not None:
                smoke['associated_master_id'] = best_master_id
                smoke['associated_class_name'] = best_class_name
            else:
                # Temporal Vehicle Association Fallback
                fallback_best_score = 0.0
                fallback_master_id = None
                fallback_class_name = None
                
                history_list = list(self.global_frame_history)
                frames_to_check = history_list[-lookback_frames:] if lookback_frames > 0 else []
                
                for past_tracks in reversed(frames_to_check):
                    for vehicle in past_tracks:
                        vx1, vy1, vx2, vy2 = vehicle['xyxy']
                        vx1_expanded = vx1 - 250
                        vy1_expanded = vy1 - 100
                        vx2_expanded = vx2 + 250
                        vy2_expanded = vy2 + 250
                        
                        ix1 = max(sx1, vx1_expanded)
                        iy1 = max(sy1, vy1_expanded)
                        ix2 = min(sx2, vx2_expanded)
                        iy2 = min(sy2, vy2_expanded)
                        
                        intersection_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        ios = intersection_area / smoke_area
                        
                        dx = max(0.0, vx1 - sx2, sx1 - vx2)
                        dy = max(0.0, vy1 - sy2, sy1 - vy2)
                        box_distance = np.sqrt(dx**2 + dy**2)
                        
                        composite_score = (0.6 * ios) + (0.4 * (1.0 / (1.0 + (box_distance / 10.0))))
                        
                        if composite_score > fallback_best_score and composite_score > 0.08:
                            fallback_best_score = composite_score
                            fallback_master_id = vehicle['track_id']
                            fallback_class_name = vehicle['class_name']
                    
                    if fallback_master_id is not None:
                        break # Found best match in the most recent valid past frame
                            
                            
                if fallback_master_id is not None:
                    smoke['associated_master_id'] = fallback_master_id
                    smoke['associated_class_name'] = fallback_class_name
            
            if smoke['associated_master_id'] is not None:
                for plate_text, master_id in self.global_vehicle_map.items():
                    if master_id == smoke['associated_master_id']:
                        smoke['associated_plate'] = plate_text
                        break
                        
        return smoke_detections

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
