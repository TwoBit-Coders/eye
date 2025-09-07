import base64
import io
import logging
import cv2
import numpy as np
import threading
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from PIL import Image

def _base64_to_cv2_image(base64_string):
    """
    Decodes a Base64 string (from a data URL) into an OpenCV image (NumPy array).
    """
    try:
        if "," in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_bytes = base64.b64decode(base64_string)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        
        return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Error decoding base64 string: {e}")
        return None

# Import the AI modules
from describe_scene import analyze_frame 
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
    """Process incoming video frames for navigation guidance."""
    if not assistant or not assistant.running: return

    try:
        frame_bgr = _base64_to_cv2_image(data['frame'])
        if frame_bgr is None: return

        result_dict = assistant.process_frame_for_navigation(frame_bgr)
        if not result_dict: return

        # If there is a new announcement, send the full 'navigation_update' event
        if result_dict.get('status') == 'processed' and result_dict.get('message'):
            payload = {
                'message': result_dict.get('message'),
                'priority': result_dict.get('priority', 'normal'),
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

# --- Scene Description Handlers ---

def _run_scene_analysis(sid, image_data_url):
    """
    This function performs the heavy AI work in a separate thread.
    """
    logger.info(f"Starting background scene analysis for client {sid}...")
    try:
        frame = _base64_to_cv2_image(image_data_url)
        
        if frame is None:
            # CHANGE #1: Use the 'response' event for errors
            socketio.emit('scene_description_response', 
                          {'error': 'Could not decode image.'},
                          to=sid)
            return

        description = analyze_frame(frame)

        logger.info(f"Analysis complete for {sid}. Sending result.")
        # CHANGE #2: Use the 'response' event for the final, successful result
        socketio.emit('scene_description_response', 
                      {'description': description},
                      to=sid)
                          
    except Exception as e:
        logger.error(f"Error in background task for {sid}: {e}", exc_info=True)
        # CHANGE #3: Use the 'response' event for exceptions
        socketio.emit('scene_description_response', 
                      {'error': 'A server error occurred during analysis.'},
                      to=sid)

@socketio.on('request_scene_description')
def handle_scene_request(data):
    """
    This handler is NON-BLOCKING and starts the background analysis task.
    """
    client_sid = request.sid
    image_data_url = data.get('frame')

    if not image_data_url:
        # CHANGE #4: Use the 'response' event for this error too
        emit('scene_description_response', {'error': 'No image frame received.'})
        return

    # CHANGE #5: Use a DEDICATED event for intermediate status updates
    emit('analysis_status_update', {'message': 'AI is analyzing your scene...'})

    # This part is already correct
    socketio.start_background_task(
        target=_run_scene_analysis,
        sid=client_sid,
        image_data_url=image_data_url
    )
    logger.info(f"Started background task for client {client_sid}.")


# --- AI Model Warm-up ---
def warm_up_ai_models():
    """
    Runs a dummy inference to warm up the AI models (especially for scene description).
    This prevents a long "cold start" delay on the first user request.
    """
    logger.info("Warming up AI models... This may take a moment.")
    try:
        # Create a dummy black image (640x480) to send to the model
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Call the analysis function. The result is discarded.
        # This forces the models in 'describe_scene.py' to be loaded into memory and initialized.
        _ = analyze_frame(dummy_frame)
        
        logger.info("AI models are warmed up and ready.")
    except Exception as e:
        logger.error(f"Error during AI model warm-up: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    if assistant:
        # Run the warm-up in a separate thread so it doesn't block the server from starting.
        warmup_thread = threading.Thread(target=warm_up_ai_models)
        warmup_thread.start()

        logger.info("Starting Flask-SocketIO server on http://127.0.0.1:5000")
        socketio.run(app, host='127.0.0.1', port=5000, debug=True)
    else:
        logger.critical("AutoNavigationAssistant module failed to initialize. Server will not run.")