import cv2
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
from collections import Counter
import logging


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Model Loading ---
print("Loading AI models, this may take a moment...")
PROCESSOR = None
MODEL = None
OBJECT_MODEL = None


try:
    print("Loading BLIP image captioning model...")
    PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
    print("✅ Scene captioning model loaded successfully.")

    print("Loading YOLOv8 object detection model...")
    OBJECT_MODEL = YOLO('yolov8n.pt')
    print("✅ Object detection model loaded successfully.")
    
    print("🎉 All AI models loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    logger.error(f"Model loading failed: {e}")

# --- AI Helper Functions ---

def _generate_caption(pil_image):
    """(Internal) Generates a scene caption for a given PIL image."""
    try:
        if not PROCESSOR or not MODEL:
            return "Error: Captioning model not loaded."
        
        inputs = PROCESSOR(pil_image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = MODEL.generate(**inputs, max_length=50)
        
        caption = PROCESSOR.decode(out[0], skip_special_tokens=True)
        return caption
        
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        return "a scene with various elements"

def _pluralize_object(name, count):
    """(Internal) Converts a count and object name into a readable phrase."""
    try:
        num_to_word = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 
                      6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
        count_str = num_to_word.get(count, str(count))
        
        if count == 1:
            return f"{count_str} {name}"
        
        # Handle irregular plurals
        irregulars = {
            'person': 'people',
            'child': 'children',
            'mouse': 'mice',
            'goose': 'geese'
        }
        
        if name in irregulars:
            return f"{count_str} {irregulars[name]}"
        
        # Handle regular plurals
        if name.endswith(('s', 'sh', 'ch', 'x', 'z')):
            return f"{count_str} {name}es"
        elif name.endswith('y') and name[-2] not in 'aeiou':
            return f"{count_str} {name[:-1]}ies"
        elif name.endswith('f'):
            return f"{count_str} {name[:-1]}ves"
        elif name.endswith('fe'):
            return f"{count_str} {name[:-2]}ves"
        else:
            return f"{count_str} {name}s"
            
    except Exception as e:
        logger.error(f"Error pluralizing {name}: {e}")
        return f"{count} {name}"

def _detect_objects(frame):
    """(Internal) Detects, groups, and counts objects in a frame."""
    try:
        if not OBJECT_MODEL:
            return "some objects"

        results = OBJECT_MODEL(frame, verbose=False)
        detected_names = []
        
        if results and results[0].boxes:
            for box in results[0].boxes:
                confidence = float(box.conf[0])
                if confidence > 0.4:
                    class_id = int(box.cls[0])
                    object_name = OBJECT_MODEL.names[class_id]
                    detected_names.append(object_name)

        object_counts = Counter(detected_names)
        
        if not object_counts:
            return "various objects and elements"
        
        # Convert to readable phrases
        object_strings = [_pluralize_object(name, count) for name, count in object_counts.most_common()]
        
        # Format the list grammatically - Bug Fix Applied
        if len(object_strings) == 0:
            return "various objects"
        elif len(object_strings) == 1:
            return object_strings[0]
        elif len(object_strings) == 2:
            return f"{object_strings[0]} and {object_strings[1]}"
        else:
            # Define last_item before using it in the f-string
            last_item = object_strings[-1]
            return f"{', '.join(object_strings[:-1])}, and {last_item}"
            
    except Exception as e:
        logger.error(f"Error in object detection: {e}")
        return "some objects"

def _enhance_description(scene_caption, objects_phrase):
    """(Internal) Creates a more natural and detailed description."""
    try:
        # Remove redundant "a picture of" or "an image of" from BLIP output
        scene_caption = scene_caption.replace("a picture of ", "").replace("an image of ", "")
        
        # Create a more natural flowing description
        if "various objects" in objects_phrase or "some objects" in objects_phrase:
            return f"I can see {scene_caption}."
        else:
            return f"I can see {scene_caption}. In this scene, there are {objects_phrase}."
            
    except Exception as e:
        logger.error(f"Error enhancing description: {e}")
        return f"I can see {scene_caption} with {objects_phrase}."

# --- Main Public Function for Backend Integration ---

def analyze_frame(frame):
    """
    Receives a single video frame (as a NumPy array from OpenCV) and returns a 
    full, descriptive narration of the scene.

    This is the primary function to be called by the backend server.
    """
    try:
        # Check if models are loaded
        if not all([PROCESSOR, MODEL, OBJECT_MODEL]):
            return "I'm having trouble analyzing the scene right now. The AI models may not be properly loaded."

        # Validate input frame
        if frame is None or frame.size == 0:
            return "I couldn't process the image. Please try again."
        
        # 1. Convert frame to RGB and create PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # 2. Generate the overall scene caption
        scene_caption = _generate_caption(pil_img)
        
        # 3. Detect and list the specific objects
        objects_phrase = _detect_objects(frame)

        # 4. Create enhanced description
        final_description = _enhance_description(scene_caption, objects_phrase)
        
        return final_description
        
    except Exception as e:
        logger.error(f"Critical error in analyze_frame: {e}")
        return "I encountered an error while analyzing the scene. Please try again."

# --- Additional Utility Functions ---

def get_model_status():
    """Returns the loading status of all AI models."""
    return {
        'captioning_model': PROCESSOR is not None and MODEL is not None,
        'object_detection_model': OBJECT_MODEL is not None,
        'all_models_ready': all([PROCESSOR, MODEL, OBJECT_MODEL])
    }

def warm_up_models():
    """Warm up the models with a dummy image to reduce first-call latency."""
    try:
        import numpy as np
        # Create a small dummy image
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_frame.fill(128)  # Gray image
        
        # Run a quick analysis
        result = analyze_frame(dummy_frame)
        logger.info("Models warmed up successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error warming up models: {e}")
        return False

# --- Module Initialization ---
if __name__ == "__main__":
    # Test the module if run directly
    import numpy as np
    
    print("Testing describe_scene module...")
    
    # Check model status
    status = get_model_status()
    print(f"Model Status: {status}")
    
    if status['all_models_ready']:
        # Warm up models
        warm_up_models()
        print("✅ Module is ready for use!")
    else:
        print("❌ Some models failed to load. Check your internet connection and try again.")