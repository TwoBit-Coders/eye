from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from PIL import Image
import io
import logging

from autonavigation import AutoNavigationAssistant
from OCR import OCRProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize components
nav_assistant = AutoNavigationAssistant()
ocr_processor = OCRProcessor()

def decode_frame_data(frame_data):
    """Decode base64 image data to OpenCV frame"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(frame_data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return frame
    except Exception as e:
        logger.error(f"Error decoding frame data: {e}")
        return None

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
        'navigation_running': nav_assistant.running,
        'connected_clients': len(nav_assistant.connected_clients)
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
    """Process incoming video frame for navigation"""
    if nav_assistant.running:
        frame = decode_frame_data(data['frame'])
        if frame is not None:
            result = nav_assistant.process_frame_for_navigation(frame)
            
            # Emit navigation update if there's an announcement
            if result.get('status') == 'processed' and 'message' in result:
                socketio.emit('navigation_update', {
                    'type': 'navigation_update',
                    'priority': result['priority'],
                    'message': result['message'],
                    'is_moving': result['is_moving'],
                    'stationary_time': result['stationary_time'],
                    'urgent_warnings': result.get('urgent_warnings', []),
                    'important_obstacles': result.get('important_obstacles', []),
                    'general_info': result.get('general_info', [])
                })
            
            emit('frame_result', result)
        else:
            emit('frame_result', {'status': 'decode_error'})
    else:
        emit('frame_result', {'status': 'navigation_stopped'})

@socketio.on('ocr_capture')
def handle_ocr_capture(data):
    """Process frame for OCR text extraction"""
    try:
        logger.info("OCR capture request received")
        
        frame = decode_frame_data(data['frame'])
        if frame is not None:
            # Process frame with OCR
            result = ocr_processor.process_frame_for_ocr(frame)
            
            # Generate speech text if successful
            if result['status'] == 'success':
                speech_text = ocr_processor.speak_ocr_result(
                    result['text'], 
                    result.get('structured_data')
                )
                
                # Speak the result
                nav_assistant.speak_async(speech_text, priority="urgent")
                
                logger.info(f"OCR successful: {len(result['text'])} characters extracted")
            else:
                # Speak error message
                error_message = result.get('message', 'Could not read text from image')
                nav_assistant.speak_async(f"OCR failed: {error_message}")
                logger.warning(f"OCR failed: {result.get('message', 'Unknown error')}")
            
            # Emit result to client
            emit('ocr_result', result)
            
        else:
            error_result = {
                'status': 'error',
                'message': 'Failed to decode image data',
                'text': '',
                'confidence': 0
            }
            nav_assistant.speak_async("Failed to process image for text reading")
            emit('ocr_result', error_result)
            
    except Exception as e:
        logger.error(f"Error in OCR processing: {e}")
        error_result = {
            'status': 'error',
            'message': str(e),
            'text': '',
            'confidence': 0
        }
        nav_assistant.speak_async("Error occurred during text reading")
        emit('ocr_result', error_result)

@socketio.on('force_scan')
def handle_force_scan():
    """Force immediate navigation scan"""
    nav_assistant.last_scan_time = 0
    emit('status', {'message': 'Navigation scan forced'})

@socketio.on('test_tts')
def handle_test_tts(data):
    """Test text-to-speech functionality"""
    message = data.get('message', 'Testing text to speech')
    nav_assistant.speak_async(message)
    emit('status', {'message': f'TTS test: "{message}"'})

if __name__ == '__main__':
    logger.info("🚀 Starting NavPro Flask-SocketIO server...")
    logger.info("📱 Open your browser to http://localhost:5000")
    logger.info("📋 Features available:")
    logger.info("   - 🧭 AI Navigation Assistant")
    logger.info("   - 🔍 OCR Text Reading")
    logger.info("   - 🗣️ Text-to-Speech")
    logger.info("   - 📱 Real-time Video Processing")
    
    try:
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")