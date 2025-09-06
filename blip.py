import cv2
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
from collections import Counter

# --- Global Model Loading ---
# This block loads the AI models into memory once when this module is imported.
# This is highly efficient as they won't be reloaded on every call.
print("Loading AI models, this may take a moment...")
PROCESSOR = None
MODEL = None
OBJECT_MODEL = None
try:
    PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("✅ Scene captioning model loaded successfully.")

    OBJECT_MODEL = YOLO('yolov8n.pt')
    print("✅ Object detection model loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

# --- AI Helper Functions ---

def _generate_caption(pil_image):
    """(Internal) Generates a scene caption for a given PIL image."""
    if not PROCESSOR or not MODEL:
        return "Error: Captioning model not loaded."
        
    inputs = PROCESSOR(pil_image, return_tensors="pt")
    with torch.no_grad():
        out = MODEL.generate(**inputs, max_length=50)
    return PROCESSOR.decode(out[0], skip_special_tokens=True)

def _pluralize_object(name, count):
    """(Internal) Converts a count and object name into a readable phrase."""
    num_to_word = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}
    count_str = num_to_word.get(count, str(count))
    if count == 1:
        return f"{count_str} {name}"
    
    irregulars = {'person': 'people'}
    if name in irregulars:
        return f"{count_str} {irregulars[name]}"
        
    return f"{count_str} {name}s"

def _detect_objects(frame):
    """(Internal) Detects, groups, and counts objects in a frame."""
    if not OBJECT_MODEL:
        return "Error: Object detection model not loaded."

    results = OBJECT_MODEL(frame)
    detected_names = []
    if results and results[0].boxes:
        for box in results[0].boxes:
            if float(box.conf[0]) > 0.5:
                class_id = int(box.cls[0])
                detected_names.append(OBJECT_MODEL.names[class_id])

    object_counts = Counter(detected_names)
    if not object_counts:
        return "no specific objects"
    
    object_strings = [_pluralize_object(name, count) for name, count in object_counts.items()]
    if len(object_strings) == 1:
        return object_strings[0]
    elif len(object_strings) == 2:
        return f"{object_strings[0]} and {object_strings[1]}"
    else:
        return f"{', '.join(object_strings[:-1])}, and {last_item}"

# --- Main Public Function for Backend Integration ---

def analyze_frame(frame):
    """
    Receives a single video frame (as a NumPy array from OpenCV) and returns a 
    full, descriptive narration of the scene.

    This is the primary function to be called by the backend server.
    """
    if not all([PROCESSOR, MODEL, OBJECT_MODEL]):
        return "An error occurred: One or more AI models failed to load."

    # 1. Generate the overall scene caption.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    scene_caption = _generate_caption(pil_img)
    
    # 2. Detect and list the specific objects.
    objects_phrase = _detect_objects(frame)

    # 3. Combine results into a single, cohesive narration string.
    final_narration = f"The overall scene is {scene_caption}. Looking closer, I can see {objects_phrase}."
    
    return final_narration