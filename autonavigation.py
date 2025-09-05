import cv2
import torch
import numpy as np
import time
from collections import deque
import logging

logger = logging.getLogger(__name__)

class AutoNavigationAssistant:
    """
    A silent navigation assistant that processes frames and returns text-based guidance.
    All audio generation is handled by the frontend to ensure synchronization.
    """
    def __init__(self):
        self.running = False

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load models (this is the slowest part of initialization)
        logger.info("Loading AI models...")
        try:
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
            self.yolo_model.conf = 0.4
            
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.midas.to(self.device)
            self.midas.eval()
            self.midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
            logger.info("AI models loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
        
        # --- Navigation parameters and object dictionaries ---
        self.navigation_objects = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
        self.priority_objects = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 6: 'train', 7: 'truck', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 16: 'bird', 17: 'cat', 18: 'dog'}
        self.furniture_objects = {56: 'chair', 57: 'couch', 59: 'bed', 60: 'dining table', 62: 'tv', 69: 'oven', 72: 'refrigerator'}
        self.last_announcement_time = 0
        self.last_urgent_announcement = 0
        self.urgent_interval = 2.0
        self.important_interval = 3.0
        self.general_interval = 4.0
        self.clear_path_interval = 6.0
        self.meters_per_step = 0.6
        self.depth_scale_factor = 4.0
        self.last_frame = None
        self.motion_threshold = 1000
        self.stationary_time = 0
        self.last_motion_time = time.time()
        self.last_scan_time = 0
        self.scan_interval = 3.0
        self.stationary_scan_interval = 8.0

    def start(self):
        if self.running:
            logger.warning("Start command received, but navigation is already active.")
            return
        logger.info("Starting navigation assistant...")
        self.running = True

    def stop(self):
        if not self.running:
            logger.warning("Stop command received, but navigation is already stopped.")
            return
        logger.info("Stopping navigation assistant...")
        self.running = False

    # NOTE: All pyttsx3/audio-related code has been removed from this class.

    def detect_motion(self, current_frame):
        if self.last_frame is None:
            self.last_frame = current_frame.copy()
            return True
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_last = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_current, gray_last)
        motion_pixels = np.sum(diff > 30)
        self.last_frame = current_frame.copy()
        current_time = time.time()
        if motion_pixels > self.motion_threshold:
            self.last_motion_time = current_time
            self.stationary_time = 0
            return True
        else:
            self.stationary_time = current_time - self.last_motion_time
            return False

    def get_position_description(self, x, w):
        p = x / w
        return "far left" if p < 0.15 else "left" if p < 0.35 else "directly ahead" if p < 0.65 else "right" if p < 0.85 else "far right"

    def meters_to_steps(self, d):
        return max(1, round(d / self.meters_per_step))

    def estimate_distance_in_steps(self, d_val, b_area, f_area):
        norm_d = 1.0 - d_val
        dist_m = norm_d * self.depth_scale_factor
        sz_f = (b_area / f_area) * 1.5
        adj_d = max(0.3, dist_m - sz_f)
        return self.meters_to_steps(adj_d)

    def classify_urgency(self, s, c):
        is_p = c in self.priority_objects
        is_f = c in self.furniture_objects
        if s <= 2: return "urgent" if is_p else "warning" if is_f else "caution"
        elif s <= 5: return "caution" if is_p else "info"
        else: return "info"

    def filter_detections(self, dets, shape):
        h, w = shape[:2]
        area = h * w
        filtered = []
        for d in dets:
            conf, cid, box = d[4], int(d[5]), d[:4]
            if cid in self.navigation_objects and conf > 0.35:
                x1, y1, x2, y2 = map(int, box)
                box_area = (x2 - x1) * (y2 - y1)
                if box_area > area * 0.002:
                    filtered.append({'box': (x1, y1, x2, y2), 'confidence': conf, 'class_id': cid, 'class_name': self.navigation_objects[cid], 'box_area': box_area})
        return filtered

    def detect_wall_or_deadend(self, dm, w):
        h = dm.shape[0]
        fc = dm[h // 3:, w // 3:2 * w // 3]
        if fc.size == 0 or np.sum(fc > 0.8) / fc.size <= 0.4: return None
        steps = self.meters_to_steps((1.0 - np.mean(fc[fc > 0.8])) * self.depth_scale_factor)
        l, r = dm[h // 3:, :w // 3], dm[h // 3:, 2 * w // 3:]
        ap = []
        if l.size > 0 and np.sum(l < 0.6) / l.size > 0.3: ap.append("left")
        if r.size > 0 and np.sum(r < 0.6) / r.size > 0.3: ap.append("right")
        if not ap: return f"dead end reached Turn around"
        return f"wall ahead {steps} steps Turn {' or '.join(ap)}"

    def generate_navigation_guidance(self, frame, dets, d_map):
        h, w = frame.shape[:2]
        dn = (d_map - d_map.min()) / (d_map.max() - d_map.min() + 1e-8)
        fd = self.filter_detections(dets, frame.shape)
        uw, io, gi = [], [], []
        for det in fd:
            (x1, y1, x2, y2), xc = det['box'], (det['box'][0] + det['box'][2]) / 2
            obj_depth = dn[y1:y2, x1:x2]
            if obj_depth.size > 0:
                avg_d = np.mean(obj_depth)
                s = self.estimate_distance_in_steps(avg_d, det['box_area'], h * w)
                p = self.get_position_description(xc, w)
                u = self.classify_urgency(s, det['class_id'])
                ss = "1 step" if s == 1 else f"{s} steps"
                desc = f"{det['class_name']} {p} {ss} away"
                if u == "urgent": uw.append(f"Caution {desc}")
                elif u in ("warning", "caution"): io.append(desc)
                else: gi.append(desc)
        ws = self.detect_wall_or_deadend(dn, w)
        if ws: io.append(ws)
        return uw, io, gi

    def should_announce(self, uw, io, gi, im):
        ct = time.time()
        wm = [m for m in io if any(k in m.lower() for k in ['wall', 'dead end'])]
        if uw and ct - self.last_urgent_announcement > self.urgent_interval:
            self.last_urgent_announcement = ct
            return "urgent", uw[:1]
        if wm and ct - self.last_announcement_time > self.urgent_interval * 1.5: return "wall", wm[:1]
        non_wall_io = [m for m in io if m not in wm]
        if non_wall_io and ct - self.last_announcement_time > (self.important_interval if im else self.important_interval * 1.5): return "important", non_wall_io[:2]
        if gi and ct - self.last_announcement_time > (self.general_interval if im else self.general_interval * 2): return "general", gi[:2]
        if not (uw or io or gi) and ct - self.last_announcement_time > self.clear_path_interval: return "clear", ["Path appears clear"]
        return None, []

    def should_scan(self, im):
        return time.time() - self.last_scan_time > (self.scan_interval if im else self.stationary_scan_interval)

    def process_frame_for_navigation(self, frame):
        try:
            is_moving = self.detect_motion(frame)
            if self.should_scan(is_moving):
                self.last_scan_time = time.time()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.yolo_model(rgb_frame)
                detections = results.xyxy[0].cpu().numpy()
                input_tensor = self.midas_transform(rgb_frame).to(self.device)
                with torch.no_grad():
                    prediction = self.midas(input_tensor)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1) if prediction.dim() == 3 else prediction,
                        size=frame.shape[:2], mode="bicubic", align_corners=False,
                    )
                    depth_map = prediction.squeeze().cpu().numpy()
                
                uw, io, gi = self.generate_navigation_guidance(frame, detections, depth_map)
                priority, announcement_list = self.should_announce(uw, io, gi, is_moving)

                if priority and announcement_list:
                    self.last_announcement_time = time.time()
                    announcement = ". ".join(announcement_list)
                    logger.info(f"Generated guidance [{priority.upper()}]: {announcement}")
                    return {'status': 'processed', 'is_moving': is_moving, 'stationary_time': self.stationary_time, 'priority': priority, 'message': announcement}
                
                return {'status': 'processed_no_announcement', 'is_moving': is_moving, 'stationary_time': self.stationary_time}
            
            return {'status': 'skipped', 'is_moving': is_moving, 'stationary_time': self.stationary_time}
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}
