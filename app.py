from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import torch
import numpy as np
import pyttsx3
import time
from collections import deque
import threading
import base64
from PIL import Image
import io
import json
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class AutoNavigationAssistant:
    def __init__(self):
        self.running = False
        self.connected_clients = set()

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize text-to-speech engine with optimized settings
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 200)
            self.engine.setProperty('volume', 0.9)
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            self.engine = None
        
        # Load models
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
        
        # Navigation-relevant object classes (COCO dataset)
        self.navigation_objects = {
            # People and animals
            0: 'person', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
            21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
            
            # Vehicles and transportation
            1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
            6: 'train', 7: 'truck', 8: 'boat',
            
            # Traffic and road objects
            9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 
            12: 'parking meter', 13: 'bench',
            
            # Sports and outdoor equipment
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            
            # Furniture and household items
            56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
            60: 'dining table', 61: 'toilet', 62: 'tv', 
            
            # Kitchen and appliances
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
            
            # Electronics and office items
            63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 
            67: 'cell phone', 68: 'microwave', 73: 'book', 74: 'clock',
            75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush',
            
            # Food and dining items  
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 
            43: 'knife', 44: 'spoon', 45: 'bowl',
            
            # Clothing and accessories
            26: 'backpack', 27: 'umbrella', 28: 'handbag', 29: 'tie',
            30: 'suitcase',
            
            # Food items
            46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
            54: 'donut', 55: 'cake'
        }
        
        # Priority objects that need immediate attention
        self.priority_objects = {
            0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 
            1: 'bicycle', 6: 'train', 16: 'bird', 17: 'cat', 18: 'dog',
            9: 'traffic light', 11: 'stop sign', 10: 'fire hydrant'
        }
        
        # Large furniture objects
        self.furniture_objects = {
            56: 'chair', 57: 'couch', 59: 'bed', 60: 'dining table', 
            62: 'tv', 69: 'oven', 72: 'refrigerator'
        }
        
        # Audio feedback management
        self.last_announcement_time = 0
        self.last_urgent_announcement = 0
        self.is_speaking = False
        
        # Timing intervals
        self.urgent_interval = 1.0
        self.important_interval = 3.0
        self.general_interval = 5.0
        self.clear_path_interval = 8.0
        
        # Detection history
        self.detection_history = deque(maxlen=5)
        self.last_detections = {}
        
        # Step conversion parameters
        self.meters_per_step = 0.6
        self.depth_scale_factor = 4.0
        self.close_threshold = 3.0
        self.very_close_threshold = 1.5
        
        # Motion detection
        self.last_frame = None
        self.motion_threshold = 1000
        self.stationary_time = 0
        self.last_motion_time = time.time()
        
        # Scanning control
        self.last_scan_time = 0
        self.scan_interval = 2.0
        self.stationary_scan_interval = 6.0
        
    def speak_async(self, text, priority="normal"):
        """Non-blocking speech synthesis"""
        if not self.engine:
            logger.warning("TTS engine not available")
            return
            
        def _speak_internal():
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.error(f"Error in speech synthesis: {e}")
            finally:
                self.is_speaking = False

        if priority == "urgent" or not self.is_speaking:
            self.is_speaking = True
            thread = threading.Thread(target=_speak_internal)
            thread.daemon = True
            thread.start()
    
    def detect_motion(self, current_frame):
        """Detect motion using frame difference"""
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
    
    def get_position_description(self, x_center, width):
        """Get position description for navigation"""
        relative_pos = x_center / width
        if relative_pos < 0.15:
            return "far left"
        elif relative_pos < 0.35:
            return "left"
        elif relative_pos < 0.65:
            return "directly ahead"
        elif relative_pos < 0.85:
            return "right"
        else:
            return "far right"
    
    def meters_to_steps(self, distance_meters):
        """Convert distance to steps"""
        steps = max(1, round(distance_meters / self.meters_per_step))
        return steps
    
    def estimate_distance_in_steps(self, depth_value, box_area, frame_area):
        """Estimate distance in steps"""
        normalized_depth = 1.0 - depth_value
        distance_meters = normalized_depth * self.depth_scale_factor
        size_factor = (box_area / frame_area) * 1.5
        adjusted_distance = max(0.3, distance_meters - size_factor)
        steps = self.meters_to_steps(adjusted_distance)
        return steps
    
    def classify_urgency(self, steps, object_class):
        """Classify detection urgency"""
        is_priority = object_class in self.priority_objects
        is_furniture = object_class in self.furniture_objects
        
        if steps <= 2:
            if is_priority:
                return "urgent"
            elif is_furniture:
                return "warning"
            else:
                return "caution"
        elif steps <= 5:
            if is_priority:
                return "caution"
            elif is_furniture:
                return "info"
            else:
                return "info"
        else:
            return "info"
    
    def filter_detections(self, detections, frame_shape):
        """Filter and prioritize detections"""
        height, width = frame_shape[:2]
        frame_area = height * width
        filtered = []
        
        for det in detections:
            confidence = det[4]
            class_id = int(det[5])
            
            if class_id in self.navigation_objects and confidence > 0.35:
                x1, y1, x2, y2 = map(int, det[:4])
                box_area = (x2 - x1) * (y2 - y1)
                
                if box_area > frame_area * 0.002:
                    filtered.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': self.navigation_objects[class_id],
                        'box_area': box_area
                    })
        
        return filtered
    
    def detect_wall_or_deadend(self, depth_map, width):
        """Detect walls and dead ends"""
        height = depth_map.shape[0]
        
        front_center = depth_map[height//3:, width//3:2*width//3]
        left_region = depth_map[height//3:, :width//3]
        right_region = depth_map[height//3:, 2*width//3:]
        
        wall_detected = False
        deadend_detected = False
        alternative_paths = []
        
        if front_center.size > 0:
            high_depth_pixels = np.sum(front_center > 0.8)
            total_front_pixels = front_center.size
            
            if high_depth_pixels / total_front_pixels > 0.4:
                upper_front = front_center[:front_center.shape[0]//2, :]
                if upper_front.size > 0:
                    consistent_rows = 0
                    for row in upper_front:
                        if np.sum(row > 0.8) / len(row) > 0.3:
                            consistent_rows += 1
                    
                    if consistent_rows / upper_front.shape[0] > 0.6:
                        wall_detected = True
                        avg_wall_depth = np.mean(front_center[front_center > 0.8])
                        wall_distance_steps = self.meters_to_steps((1.0 - avg_wall_depth) * self.depth_scale_factor)
        
        if wall_detected:
            if left_region.size > 0:
                left_clear_pixels = np.sum(left_region < 0.6)
                if left_clear_pixels / left_region.size > 0.3:
                    alternative_paths.append("left")
            
            if right_region.size > 0:
                right_clear_pixels = np.sum(right_region < 0.6)
                if right_clear_pixels / right_region.size > 0.3:
                    alternative_paths.append("right")
            
            if not alternative_paths:
                deadend_detected = True
        
        if deadend_detected:
            return "dead end reached. Turn around"
        elif wall_detected:
            if alternative_paths:
                if len(alternative_paths) == 2:
                    return f"wall ahead, {wall_distance_steps} steps. Turn left or right"
                else:
                    direction = alternative_paths[0]
                    return f"wall ahead, {wall_distance_steps} steps. Turn {direction}"
            else:
                return f"wall ahead, {wall_distance_steps} steps"
        
        return None
    
    def generate_navigation_guidance(self, frame, detections, depth_map):
        """Generate navigation guidance"""
        height, width = frame.shape[:2]
        frame_area = height * width
        
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        filtered_detections = self.filter_detections(detections, frame.shape)
        
        urgent_warnings = []
        important_obstacles = []
        general_info = []
        
        for det in filtered_detections:
            x1, y1, x2, y2 = det['box']
            x_center = (x1 + x2) / 2
            
            object_depth = depth_normalized[y1:y2, x1:x2]
            if object_depth.size > 0:
                avg_depth = np.mean(object_depth)
                steps = self.estimate_distance_in_steps(avg_depth, det['box_area'], frame_area)
                
                position = self.get_position_description(x_center, width)
                urgency = self.classify_urgency(steps, det['class_id'])
                
                step_str = "1 step" if steps == 1 else f"{steps} steps"
                description = f"{det['class_name']} {position}, {step_str} away"
                
                if urgency == "urgent":
                    urgent_warnings.append(f"Caution! {description}")
                elif urgency == "warning" or urgency == "caution":
                    important_obstacles.append(description)
                else:
                    general_info.append(description)
        
        # Check for walls/dead ends
        wall_status = self.detect_wall_or_deadend(depth_normalized, width)
        if wall_status:
            important_obstacles.append(wall_status)
        
        return urgent_warnings, important_obstacles, general_info
    
    def should_announce(self, urgent_warnings, important_obstacles, general_info, is_moving):
        """Determine if announcement should be made"""
        current_time = time.time()
        
        wall_messages = [msg for msg in important_obstacles if any(keyword in msg.lower() 
                        for keyword in ['wall', 'dead end', 'turn around'])]
        
        if urgent_warnings:
            if current_time - self.last_urgent_announcement > self.urgent_interval:
                self.last_urgent_announcement = current_time
                return "urgent", urgent_warnings[:1]
        
        if wall_messages:
            if current_time - self.last_announcement_time > self.urgent_interval * 1.5:
                return "wall", wall_messages[:1]
        
        if important_obstacles and not wall_messages:
            interval = self.important_interval if is_moving else self.important_interval * 1.5
            if current_time - self.last_announcement_time > interval:
                return "important", important_obstacles[:2]
        
        if general_info:
            interval = self.general_interval if is_moving else self.general_interval * 2
            if current_time - self.last_announcement_time > interval:
                return "general", general_info[:2]
        
        if not urgent_warnings and not important_obstacles and not general_info:
            if current_time - self.last_announcement_time > self.clear_path_interval:
                return "clear", ["Path appears clear"]
        
        return None, []
    
    def should_scan(self, is_moving):
        """Check if should perform scan"""
        current_time = time.time()
        
        if is_moving:
            return current_time - self.last_scan_time > self.scan_interval
        else:
            return current_time - self.last_scan_time > self.stationary_scan_interval
    
    def process_frame(self, frame_data):
        """Process frame from frontend"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(frame_data.split(',')[1])
            image = Image.open(io.BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect motion
            is_moving = self.detect_motion(frame)
            
            # Check if should scan
            if self.should_scan(is_moving):
                self.last_scan_time = time.time()
                
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Object detection
                results = self.yolo_model(frame_rgb)
                detections = results.xyxy[0].cpu().numpy()
                
                # Depth estimation
                input_tensor = self.midas_transform(frame_rgb).to(self.device)

                with torch.no_grad():
                    prediction = self.midas(input_tensor)

                    if prediction.dim() == 3:
                        prediction = prediction.unsqueeze(1)

                    prediction = torch.nn.functional.interpolate(
                        prediction,
                        size=frame.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    )

                depth_map = prediction.squeeze().cpu().numpy()
                
                # Generate guidance
                urgent_warnings, important_obstacles, general_info = self.generate_navigation_guidance(
                    frame, detections, depth_map
                )
                
                # Check announcements
                priority, announcement_list = self.should_announce(
                    urgent_warnings, important_obstacles, general_info, is_moving
                )
                
                if priority and announcement_list:
                    self.last_announcement_time = time.time()
                    announcement = ". ".join(announcement_list)
                    
                    if priority == "urgent":
                        logger.info(f"🚨 URGENT: {announcement}")
                        self.speak_async(announcement, priority="urgent")
                    else:
                        motion_status = "moving" if is_moving else "stationary"
                        logger.info(f"🧭 [{motion_status.upper()}] {announcement}")
                        self.speak_async(announcement)
                    
                    # Emit to all clients
                    socketio.emit('navigation_update', {
                        'type': 'navigation_update',
                        'priority': priority,
                        'message': announcement,
                        'is_moving': is_moving,
                        'stationary_time': self.stationary_time,
                        'urgent_warnings': urgent_warnings,
                        'important_obstacles': important_obstacles,
                        'general_info': general_info
                    })
                
                return {
                    'status': 'processed',
                    'is_moving': is_moving,
                    'stationary_time': self.stationary_time
                }
            else:
                return {
                    'status': 'skipped',
                    'is_moving': is_moving,
                    'stationary_time': self.stationary_time
                }
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {'status': 'error', 'message': str(e)}

# Initialize the navigation assistant
nav_assistant = AutoNavigationAssistant()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(nav_assistant.device),
        'running': nav_assistant.running
    })

# SocketIO event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    nav_assistant.connected_clients.add(request.sid)
    logger.info(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to NavPro backend'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    nav_assistant.connected_clients.discard(request.sid)
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_navigation')
def handle_start_navigation():
    """Start navigation assistance"""
    nav_assistant.running = True
    nav_assistant.speak_async("Navigation assistant ready. Start moving for guidance.")
    logger.info("Navigation started")
    emit('status', {'message': 'Navigation started'})

@socketio.on('stop_navigation')  
def handle_stop_navigation():
    """Stop navigation assistance"""
    nav_assistant.running = False
    nav_assistant.speak_async("Navigation assistant stopped. Stay safe!")
    logger.info("Navigation stopped")
    emit('status', {'message': 'Navigation stopped'})

@socketio.on('process_frame')
def handle_frame(data):
    """Process incoming video frame"""
    if nav_assistant.running:
        result = nav_assistant.process_frame(data['frame'])
        emit('frame_result', result)
    else:
        emit('frame_result', {'status': 'navigation_stopped'})

@socketio.on('force_scan')
def handle_force_scan():
    """Force immediate scan"""
    nav_assistant.last_scan_time = 0
    emit('status', {'message': 'Scan forced'})

if __name__ == '__main__':
    logger.info("🚀 Starting NavPro Flask-SocketIO server...")
    logger.info("📱 Open your browser to http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)