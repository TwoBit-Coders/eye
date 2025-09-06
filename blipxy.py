import cv2
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import sys
import pyttsx3
from collections import Counter

# --- Global Model and Engine Loading (for efficiency) ---
print("Loading models and engines, this may take a moment...")
try:
    # 1. Load Scene Captioning Model
    PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("✅ Scene captioning model loaded successfully.")

    # 2. Load Object Detection Model (YOLOv8)
    OBJECT_MODEL = YOLO('yolov8n.pt')  # 'n' is for the small, fast 'nano' version
    print("✅ Object detection model loaded successfully.")

    # 3. Initialize Text-to-Speech Engine
    TTS_ENGINE = pyttsx3.init()
    print("✅ Text-to-speech engine initialized successfully.")

except Exception as e:
    print(f"Error loading models or engines: {e}")
    PROCESSOR = None
    MODEL = None
    OBJECT_MODEL = None
    TTS_ENGINE = None

def generate_caption(pil_image):
    """
    Generates a scene caption for a given PIL image.
    """
    if not PROCESSOR or not MODEL:
        return "Error: Captioning model not loaded."
        
    inputs = PROCESSOR(pil_image, return_tensors="pt")
    with torch.no_grad():
        out = MODEL.generate(**inputs, max_length=50)
    caption = PROCESSOR.decode(out[0], skip_special_tokens=True)
    return caption

def pluralize_object(name, count):
    """
    Converts a count and object name into a readable phrase (e.g., "one person", "three cats").
    """
    # Number to word conversion for small numbers for more natural speech
    num_to_word = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}
    count_str = num_to_word.get(count, str(count))

    if count == 1:
        return f"{count_str} {name}"

    # Handle some common irregular plurals
    irregulars = {
        'person': 'people',
        'mouse': 'mice',
        'goose': 'geese'
    }
    if name in irregulars:
        plural_name = irregulars[name]
    # Handle words ending in s, x, z, ch, sh
    elif any(name.endswith(suffix) for suffix in ['s', 'x', 'z', 'ch', 'sh']):
        plural_name = f'{name}es'
    else:
        plural_name = f'{name}s'
        
    return f"{count_str} {plural_name}"


def detect_objects(frame):
    """
    Detects objects, groups and counts them, formats them into a descriptive phrase, 
    and returns the annotated frame.
    """
    if not OBJECT_MODEL:
        return "Error: Object detection model not loaded.", frame
    
    results = OBJECT_MODEL(frame)
    annotated_frame = results[0].plot() # Get the frame with boxes drawn on it

    # Store just the names of detected objects that meet the confidence threshold
    detected_names = []
    if results and results[0].boxes:
        for box in results[0].boxes:
            confidence = float(box.conf[0])
            if confidence > 0.5: # Example threshold: 50%
                class_id = int(box.cls[0])
                class_name = OBJECT_MODEL.names[class_id]
                detected_names.append(class_name)

    # Count the occurrences of each unique object
    object_counts = Counter(detected_names)

    # Format the counts into a grammatical phrase
    if not object_counts:
        description_phrase = "no specific objects"
    else:
        # Create a list of strings like "two people", "one laptop"
        object_strings = [pluralize_object(name, count) for name, count in object_counts.items()]
        
        if len(object_strings) == 1:
            description_phrase = object_strings[0]
        elif len(object_strings) == 2:
            description_phrase = f"{object_strings[0]} and {object_strings[1]}"
        else:
            most_items = ", ".join(object_strings[:-1])
            last_item = object_strings[-1]
            description_phrase = f"{most_items}, and {last_item}"

    return description_phrase, annotated_frame

if __name__ == "__main__":
    if not all([PROCESSOR, MODEL, OBJECT_MODEL, TTS_ENGINE]):
        sys.exit(1) # Exit if any model or engine failed to load

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
    else:
        print("\n📷 Webcam open. Press SPACE to capture. Press ESC to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Live Feed | Press SPACE to capture / ESC to exit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            print("\n" + "="*40)
            print("Processing frame...")
            
            # --- Generate Caption ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            scene_caption = generate_caption(pil_img)
            
            # --- Detect Objects ---
            objects_phrase, annotated_frame = detect_objects(frame)

            # --- Combine into a single narration for accessibility ---
            final_narration = f"Here is a description of the scene. The overall view is of {scene_caption}. Looking closer, I can identify {objects_phrase}."
            
            print("\n🎤 Audio Narration:")
            print(final_narration)
            print("="*40)

            # Use TTS to speak the description
            TTS_ENGINE.say(final_narration)
            TTS_ENGINE.runAndWait()

            # Show the frame with detections in a new window
            cv2.imshow("Detection Results", annotated_frame)

        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

