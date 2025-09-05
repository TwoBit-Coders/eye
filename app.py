# app.py (Final version, 100% compatible with index.html)

import base64
import io
import logging
import cv2
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from PIL import Image

# This will import the corrected version of your assistant class
from autonavigation import AutoNavigationAssistant

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Flask & SocketIO Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here!' 
socketio = SocketIO(app, async_mode='eventlet')

# --- Global AI Instance ---
assistant = None
try:
    logger.info("Initializing AutoNavigationAssistant...")
    assistant = AutoNavigationAssistant()
    logger.info("AutoNavigationAssistant initialized successfully.")
except Exception as e:
    logger.critical(f"FATAL: Could not initialize AutoNavigationAssistant. Server cannot start. Error: {e}")

# --- Route to Serve Frontend ---
@app.route('/')
def index():
    return render_template('index.html')

# --- SocketIO Event Handlers ---

@socketio.on('connect')
def handle_connect():
    if not assistant:
        emit('status', {'message': 'Server error: AI module failed to initialize.'})
        return
    logger.info(f"Client connected: {request.sid}")
    emit('status', {'message': 'System ready. You can start navigation.'})

@socketio.on('disconnect')
def handle_disconnect():
    if assistant and assistant.running:
        assistant.stop()
    logger.info(f"Client disconnected: {request.sid}. Navigation automatically stopped.")

def base64_to_cv2_image(base64_string):
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    img_bytes = base64.b64decode(base64_string)
    pil_image = Image.open(io.BytesIO(img_bytes))
    frame_rgb = np.array(pil_image)
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

# --- Navigation Handlers ---

@socketio.on('start_navigation')
def handle_start():
    if not assistant: return
    assistant.start()
    logger.info("Navigation started by client.")
    emit('status', {'message': 'Navigation started.'})

@socketio.on('stop_navigation')
def handle_stop():
    if not assistant: return
    assistant.stop()
    logger.info("Navigation stopped by client.")
    emit('status', {'message': 'Navigation stopped.'})

@socketio.on('process_frame')
def handle_frame(data):
    """
    Handles incoming video frames. This function now sends the
    full data payload required by the frontend.
    """
    if not assistant or not assistant.running: return

    try:
        frame_bgr = base64_to_cv2_image(data['frame'])
        result_dict = assistant.process_frame_for_navigation(frame_bgr)
        
        if not result_dict: return

        # --- THIS IS THE CORRECTED LOGIC ---
        
        # If there is a new announcement, send the full 'navigation_update' event
        if result_dict.get('status') == 'processed' and result_dict.get('message'):
            payload = {
                'message': result_dict.get('message'),
                'priority': result_dict.get('priority', 'normal'), # Default to 'normal' if not present
                'is_moving': result_dict.get('is_moving', False),
                'stationary_time': result_dict.get('stationary_time', 0)
            }
            emit('navigation_update', payload)
        
        # Otherwise, send a 'frame_result' to keep the motion status updated
        else:
            payload = {
                'status': result_dict.get('status'),
                'is_moving': result_dict.get('is_moving', False),
                'stationary_time': result_dict.get('stationary_time', 0)
            }
            emit('frame_result', payload)

    except Exception as e:
        logger.error(f"Critical error in handle_frame for client {request.sid}: {e}", exc_info=True)
        emit('status', {'message': 'An error occurred on the server.'})


# --- Main Execution ---
if __name__ == '__main__':
    if assistant:
        logger.info("Starting Flask-SocketIO server on http://127.0.0.1:5000")
        socketio.run(app, host='127.0.0.1', port=5000, debug=True)
    else:
        logger.critical("AutoNavigationAssistant module failed to initialize. Server will not run.")