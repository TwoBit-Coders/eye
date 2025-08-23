import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self):
        """Initialize OCR processor with optimized settings for medicine, signs, and books"""
        # Configure Tesseract path if needed (uncomment and modify for Windows)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # OCR configurations for different content types
        self.general_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?@#$%&*()-+=[]{}:;"\'/<>|\\~` '
        self.medicine_config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()-+/% '
        self.sign_config = '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()-/: '
        self.book_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?:;"\'-&() '
        
        # Text processing patterns
        self.phone_pattern = r'[\+]?[(]?[\d\s\-\(\)]{10,}'
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        self.price_pattern = r'[$€£¥₹]\s?\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s?[$€£¥₹]'
        
        # Medicine-specific patterns
        self.medicine_dosage_pattern = r'\b\d+\s?(?:mg|g|ml|mcg|iu|units?)\b'
        self.medicine_strength_pattern = r'\b\d+(?:\.\d+)?\s?(?:mg|g|ml|mcg|iu|units?)/(?:tablet|capsule|ml|dose)\b'
        self.medicine_form_pattern = r'\b(?:tablet|capsule|syrup|injection|cream|ointment|drops|spray)s?\b'
        
        # Highway/Road sign patterns
        self.highway_pattern = r'\b(?:US|I|SR|CR|FM|RM|SH|route|highway|interstate)\s?\d+\b'
        self.exit_pattern = r'\bexit\s?\d+[a-z]?\b'
        self.mile_pattern = r'\b\d+\s?(?:mile|mi|km|miles)\b'
        
        # Common medicine names database (subset - can be expanded)
        self.common_medicines = [
            'acetaminophen', 'tylenol', 'ibuprofen', 'advil', 'motrin', 'aspirin', 'bayer',
            'naproxen', 'aleve', 'amoxicillin', 'penicillin', 'azithromycin', 'zpack',
            'ciprofloxacin', 'cipro', 'metformin', 'glucophage', 'lisinopril', 'prinivil',
            'atorvastatin', 'lipitor', 'omeprazole', 'prilosec', 'amlodipine', 'norvasc',
            'metoprolol', 'lopressor', 'hydrochlorothiazide', 'hctz', 'simvastatin', 'zocor',
            'levothyroxine', 'synthroid', 'azithromycin', 'zithromax', 'prednisone',
            'albuterol', 'ventolin', 'insulin', 'novolin', 'humalog', 'lantus',
            'warfarin', 'coumadin', 'clopidogrel', 'plavix', 'gabapentin', 'neurontin',
            'tramadol', 'ultram', 'oxycodone', 'percocet', 'hydrocodone', 'vicodin',
            'lorazepam', 'ativan', 'alprazolam', 'xanax', 'clonazepam', 'klonopin',
            'sertraline', 'zoloft', 'escitalopram', 'lexapro', 'fluoxetine', 'prozac',
            'trazodone', 'desyrel', 'zolpidem', 'ambien', 'cetirizine', 'zyrtec',
            'loratadine', 'claritin', 'diphenhydramine', 'benadryl', 'dextromethorphan'
        ]
        
        # Common highway/road terms
        self.road_terms = [
            'highway', 'interstate', 'route', 'exit', 'north', 'south', 'east', 'west',
            'street', 'avenue', 'boulevard', 'road', 'drive', 'lane', 'way', 'circle',
            'next', 'miles', 'km', 'junction', 'merge', 'split', 'downtown', 'airport',
            'hospital', 'university', 'mall', 'park', 'bridge', 'tunnel', 'toll'
        ]
    
    def detect_content_type(self, text):
        """Detect what type of content we're dealing with"""
        text_lower = text.lower()
        
        # Count indicators for each category
        medicine_score = 0
        highway_score = 0
        book_score = 0
        
        # Medicine indicators
        medicine_keywords = ['mg', 'ml', 'tablet', 'capsule', 'dosage', 'prescription', 
                           'rx', 'drug', 'medicine', 'pharmaceutical', 'generic', 'brand']
        for keyword in medicine_keywords:
            if keyword in text_lower:
                medicine_score += 2
        
        # Check for common medicine names
        for medicine in self.common_medicines:
            if medicine.lower() in text_lower:
                medicine_score += 3
        
        # Check for dosage patterns
        if re.search(self.medicine_dosage_pattern, text_lower):
            medicine_score += 2
            
        if re.search(self.medicine_form_pattern, text_lower):
            medicine_score += 2
        
        # Highway/Sign indicators
        highway_keywords = ['exit', 'mile', 'highway', 'interstate', 'route', 'north', 'south', 
                          'east', 'west', 'next', 'junction', 'merge']
        for keyword in highway_keywords:
            if keyword in text_lower:
                highway_score += 2
        
        if re.search(self.highway_pattern, text_lower):
            highway_score += 3
            
        if re.search(self.exit_pattern, text_lower):
            highway_score += 3
        
        # Book indicators
        book_keywords = ['chapter', 'page', 'edition', 'volume', 'author', 'publisher', 
                        'isbn', 'copyright', 'press', 'novel', 'story', 'book']
        for keyword in book_keywords:
            if keyword in text_lower:
                book_score += 2
        
        # Determine content type
        if medicine_score >= highway_score and medicine_score >= book_score and medicine_score > 0:
            return 'medicine'
        elif highway_score >= book_score and highway_score > 0:
            return 'highway'
        elif book_score > 0:
            return 'book'
        else:
            return 'general'
    
    def preprocess_for_content_type(self, frame, content_type):
        """Apply content-type specific preprocessing"""
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            processed_images = []
            
            if content_type == 'medicine':
                # Medicine-specific preprocessing
                
                # Original grayscale
                processed_images.append(gray)
                
                # High contrast for small text
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                processed_images.append(enhanced)
                
                # Adaptive threshold with smaller block size for fine text
                adaptive_thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2
                )
                processed_images.append(adaptive_thresh)
                
                # Morphological operations for cleaning small text
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
                processed_images.append(morph)
                
                # Sharpening for blurry medicine labels
                kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(gray, -1, kernel_sharp)
                processed_images.append(sharpened)
                
            elif content_type == 'highway':
                # Highway sign preprocessing
                
                # Original grayscale
                processed_images.append(gray)
                
                # High contrast for outdoor signs
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                processed_images.append(enhanced)
                
                # Binary threshold for high contrast signs
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                processed_images.append(thresh)
                
                # Inverted threshold for dark signs with white text
                processed_images.append(255 - thresh)
                
                # Gaussian blur to smooth edges
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                _, thresh_blur = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                processed_images.append(thresh_blur)
                
            elif content_type == 'book':
                # Book title preprocessing
                
                # Original grayscale
                processed_images.append(gray)
                
                # Gentle enhancement for printed text
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                processed_images.append(enhanced)
                
                # Adaptive threshold
                adaptive_thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                processed_images.append(adaptive_thresh)
                
                # Bilateral filter to preserve edges while removing noise
                bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
                processed_images.append(bilateral)
                
            else:
                # General preprocessing
                processed_images = self.preprocess_image(frame)
            
            return processed_images
            
        except Exception as e:
            logger.error(f"Error in content-specific preprocessing: {e}")
            return [frame]
    
    def preprocess_image(self, frame):
        """Original general preprocessing method"""
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Apply various preprocessing techniques
            processed_images = []
            
            # Original grayscale
            processed_images.append(gray)
            
            # Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            processed_images.append(blurred)
            
            # Apply threshold (binary)
            _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(thresh1)
            
            # Adaptive threshold
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            processed_images.append(adaptive_thresh)
            
            # Morphological operations to clean up the image
            kernel = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
            processed_images.append(morph)
            
            # Edge enhancement
            edges = cv2.Canny(gray, 50, 150)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            processed_images.append(255 - dilated_edges)
            
            return processed_images
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            return [frame]
    
    def extract_text_with_config(self, image, config):
        """Extract text using specified configuration"""
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Extract text using Tesseract with specific config
            text = pytesseract.image_to_string(pil_image, config=config)
            
            # Clean up the text
            text = text.strip()
            text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
            text = re.sub(r' +', ' ', text)    # Remove multiple spaces
            
            return text
            
        except Exception as e:
            logger.error(f"Error in text extraction: {e}")
            return ""
    
    def extract_text_from_image(self, image):
        """Extract text from preprocessed image (legacy method)"""
        return self.extract_text_with_config(image, self.general_config)
    
    def find_medicine_name(self, text):
        """Extract medicine name from text"""
        try:
            text_lower = text.lower()
            found_medicines = []
            
            # Direct matches with common medicines
            for medicine in self.common_medicines:
                if medicine.lower() in text_lower:
                    # Find the actual case-sensitive match in original text
                    for word in text.split():
                        if word.lower() == medicine.lower():
                            found_medicines.append(word)
                            break
                    else:
                        found_medicines.append(medicine.title())
            
            # Fuzzy matching for potential medicine names
            words = text.split()
            for word in words:
                word_clean = re.sub(r'[^a-zA-Z]', '', word).lower()
                if len(word_clean) > 4:  # Only check longer words
                    for medicine in self.common_medicines:
                        similarity = SequenceMatcher(None, word_clean, medicine).ratio()
                        if similarity > 0.8 and word not in found_medicines:
                            found_medicines.append(word)
                            break
            
            # Look for patterns that might be medicine names (capitalized words)
            medicine_pattern = r'\b[A-Z][a-z]+(?:in|ol|ex|il|um|ic|ate|ide)\b'
            potential_medicines = re.findall(medicine_pattern, text)
            found_medicines.extend(potential_medicines)
            
            return list(set(found_medicines))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error finding medicine name: {e}")
            return []
    
    def extract_structured_data(self, text, content_type='general'):
        """Extract structured information from text based on content type"""
        structured_data = {
            'phones': [],
            'emails': [],
            'urls': [],
            'prices': [],
            'full_text': text,
            'content_type': content_type
        }
        
        try:
            # General patterns
            phones = re.findall(self.phone_pattern, text)
            structured_data['phones'] = [phone.strip() for phone in phones if len(phone.strip()) >= 10]
            
            emails = re.findall(self.email_pattern, text)
            structured_data['emails'] = emails
            
            urls = re.findall(self.url_pattern, text)
            structured_data['urls'] = urls
            
            prices = re.findall(self.price_pattern, text)
            structured_data['prices'] = prices
            
            # Content-specific extractions
            if content_type == 'medicine':
                structured_data['medicine_names'] = self.find_medicine_name(text)
                structured_data['dosages'] = re.findall(self.medicine_dosage_pattern, text.lower())
                structured_data['strengths'] = re.findall(self.medicine_strength_pattern, text.lower())
                structured_data['forms'] = re.findall(self.medicine_form_pattern, text.lower())
                
            elif content_type == 'highway':
                structured_data['highways'] = re.findall(self.highway_pattern, text, re.IGNORECASE)
                structured_data['exits'] = re.findall(self.exit_pattern, text, re.IGNORECASE)
                structured_data['distances'] = re.findall(self.mile_pattern, text.lower())
                
                # Extract destination cities/places
                directions = ['north', 'south', 'east', 'west', 'to', 'toward']
                direction_pattern = r'\b(?:' + '|'.join(directions) + r')\s+([A-Z][a-zA-Z\s]+)'
                structured_data['destinations'] = re.findall(direction_pattern, text, re.IGNORECASE)
                
            elif content_type == 'book':
                # Extract potential book titles (capitalized phrases)
                title_pattern = r'\b(?:[A-Z][a-z]*\s*){2,6}\b'
                potential_titles = re.findall(title_pattern, text)
                structured_data['potential_titles'] = [title.strip() for title in potential_titles if len(title.strip()) > 5]
                
                # Extract author patterns
                author_pattern = r'\bby\s+([A-Z][a-zA-Z\s\.]+)'
                structured_data['authors'] = re.findall(author_pattern, text)
                
                # Extract edition/volume info
                edition_pattern = r'\b(?:(\d+)(?:st|nd|rd|th)?\s+edition|volume\s+(\d+))\b'
                structured_data['editions'] = re.findall(edition_pattern, text.lower())
        
        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")
        
        return structured_data
    
    def get_text_confidence(self, image):
        """Get OCR confidence scores"""
        try:
            pil_image = Image.fromarray(image)
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return avg_confidence, data
            
        except Exception as e:
            logger.error(f"Error getting text confidence: {e}")
            return 0, None
    
    def process_frame_for_ocr(self, frame):
        """Process frame and extract text with OCR, optimized for medicines, signs, and books"""
        try:
            logger.info("Starting enhanced OCR processing...")
            
            # First, do a quick OCR to detect content type
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            quick_text = self.extract_text_from_image(gray)
            content_type = self.detect_content_type(quick_text)
            
            logger.info(f"Detected content type: {content_type}")
            
            # Apply content-specific preprocessing
            processed_images = self.preprocess_for_content_type(frame, content_type)
            
            # Choose appropriate OCR configuration
            if content_type == 'medicine':
                ocr_config = self.medicine_config
            elif content_type == 'highway':
                ocr_config = self.sign_config
            elif content_type == 'book':
                ocr_config = self.book_config
            else:
                ocr_config = self.general_config
            
            best_text = ""
            best_confidence = 0
            best_structured_data = None
            all_results = []
            
            # Try OCR on each preprocessed image with appropriate config
            for i, processed_img in enumerate(processed_images):
                try:
                    # Extract text with content-specific config
                    text = self.extract_text_with_config(processed_img, ocr_config)
                    
                    if text.strip():  # Only process if text was found
                        # Get confidence score
                        confidence, _ = self.get_text_confidence(processed_img)
                        
                        # Extract structured data with content type
                        structured_data = self.extract_structured_data(text, content_type)
                        
                        # Store result
                        result = {
                            'method': f'{content_type}_preprocessing_method_{i+1}',
                            'text': text,
                            'confidence': confidence,
                            'structured_data': structured_data,
                            'content_type': content_type
                        }
                        all_results.append(result)
                        
                        # Keep track of best result
                        if confidence > best_confidence and len(text.strip()) > len(best_text.strip()):
                            best_text = text
                            best_confidence = confidence
                            best_structured_data = structured_data
                            
                except Exception as e:
                    logger.warning(f"Error processing image variant {i}: {e}")
                    continue
            
            # If no good result found, try with original image and general config
            if not best_text.strip():
                logger.info("Trying OCR on original image with general config...")
                try:
                    text = self.extract_text_from_image(frame)
                    if text.strip():
                        best_text = text
                        content_type = self.detect_content_type(text)
                        best_structured_data = self.extract_structured_data(text, content_type)
                        best_confidence = 50  # Default confidence
                except Exception as e:
                    logger.error(f"Error with original image OCR: {e}")
            
            # Prepare final result
            if best_text.strip():
                logger.info(f"OCR completed successfully. Content type: {content_type}, Confidence: {best_confidence:.1f}%")
                logger.info(f"Extracted text length: {len(best_text)} characters")
                
                return {
                    'status': 'success',
                    'text': best_text,
                    'confidence': best_confidence,
                    'content_type': content_type,
                    'structured_data': best_structured_data,
                    'all_results': all_results,
                    'summary': self.generate_text_summary(best_text, best_structured_data, content_type)
                }
            else:
                logger.warning("No text could be extracted from the image")
                return {
                    'status': 'no_text',
                    'text': '',
                    'confidence': 0,
                    'content_type': 'unknown',
                    'structured_data': None,
                    'message': 'No text could be detected in the image'
                }
                
        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'text': '',
                'confidence': 0,
                'content_type': 'unknown'
            }
    
    def generate_text_summary(self, text, structured_data, content_type='general'):
        """Generate a content-type specific summary of extracted text"""
        try:
            summary = []
            
            # Basic text info
            word_count = len(text.split())
            line_count = len(text.split('\n'))
            char_count = len(text)
            
            summary.append(f"Text contains {word_count} words, {line_count} lines, {char_count} characters")
            
            # Content-specific summaries
            if content_type == 'medicine' and structured_data:
                if structured_data.get('medicine_names'):
                    medicines = ', '.join(structured_data['medicine_names'][:3])
                    summary.append(f"Medicine(s) identified: {medicines}")
                
                if structured_data.get('dosages'):
                    dosages = ', '.join(structured_data['dosages'][:3])
                    summary.append(f"Dosage information: {dosages}")
                
                if structured_data.get('forms'):
                    forms = ', '.join(set(structured_data['forms']))
                    summary.append(f"Medicine form(s): {forms}")
            
            elif content_type == 'highway' and structured_data:
                if structured_data.get('highways'):
                    highways = ', '.join(structured_data['highways'][:3])
                    summary.append(f"Highway(s) identified: {highways}")
                
                if structured_data.get('exits'):
                    exits = ', '.join(structured_data['exits'][:3])
                    summary.append(f"Exit(s): {exits}")
                
                if structured_data.get('destinations'):
                    destinations = ', '.join(structured_data['destinations'][:3])
                    summary.append(f"Destination(s): {destinations}")
            
            elif content_type == 'book' and structured_data:
                if structured_data.get('potential_titles'):
                    titles = ', '.join(structured_data['potential_titles'][:2])
                    summary.append(f"Potential title(s): {titles}")
                
                if structured_data.get('authors'):
                    authors = ', '.join(structured_data['authors'][:2])
                    summary.append(f"Author(s): {authors}")
            
            # General structured data
            if structured_data:
                if structured_data.get('phones'):
                    summary.append(f"Found {len(structured_data['phones'])} phone number(s)")
                
                if structured_data.get('emails'):
                    summary.append(f"Found {len(structured_data['emails'])} email address(es)")
                
                if structured_data.get('urls'):
                    summary.append(f"Found {len(structured_data['urls'])} URL(s)")
                
                if structured_data.get('prices'):
                    summary.append(f"Found {len(structured_data['prices'])} price(s)")
            
            return ". ".join(summary) if summary else f"{content_type.title()} text extracted successfully"
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Text extracted"
    
    def speak_ocr_result(self, text, structured_data=None, content_type='general'):
        """Generate speech-friendly version of OCR result with content-specific focus"""
        try:
            if not text.strip():
                return "No text detected in the image"
            
            speech_parts = []
            
            # Content-specific announcements
            if content_type == 'medicine' and structured_data and structured_data.get('medicine_names'):
                medicines = structured_data['medicine_names'][:2]  # Limit to first 2
                if medicines:
                    medicine_text = "Medicine detected: " + ", ".join(medicines)
                    speech_parts.append(medicine_text)
                    
                    # Add dosage if available
                    if structured_data.get('dosages'):
                        dosage_text = "Dosage: " + ", ".join(structured_data['dosages'][:2])
                        speech_parts.append(dosage_text)
                    
                    # Add form if available
                    if structured_data.get('forms'):
                        form_text = "Form: " + ", ".join(set(structured_data['forms'][:2]))
                        speech_parts.append(form_text)
            
            elif content_type == 'highway' and structured_data:
                if structured_data.get('highways'):
                    highway_text = "Highway sign detected: " + ", ".join(structured_data['highways'][:2])
                    speech_parts.append(highway_text)
                
                if structured_data.get('exits'):
                    exit_text = "Exit information: " + ", ".join(structured_data['exits'][:2])
                    speech_parts.append(exit_text)
                
                if structured_data.get('destinations'):
                    dest_text = "Destinations: " + ", ".join(structured_data['destinations'][:2])
                    speech_parts.append(dest_text)
            
            elif content_type == 'book' and structured_data:
                if structured_data.get('potential_titles'):
                    title_text = "Book title detected: " + structured_data['potential_titles'][0]
                    speech_parts.append(title_text)
                
                if structured_data.get('authors'):
                    author_text = "Author: " + structured_data['authors'][0]
                    speech_parts.append(author_text)
            
            # If no specific content detected, provide general text
            if not speech_parts:
                # Limit text length for speech
                if len(text) > 300:
                    speech_text = text[:300] + "... text continues"
                else:
                    speech_text = text
                
                speech_parts.append(f"{content_type.title()} text detected: {speech_text}")
            
            # Add other structured data if available
            if structured_data:
                if structured_data.get('phones'):
                    phones_str = ", ".join(structured_data['phones'][:2])
                    speech_parts.append(f"Phone numbers found: {phones_str}")
                
                if structured_data.get('emails'):
                    emails_str = ", ".join(structured_data['emails'][:2])
                    speech_parts.append(f"Email addresses found: {emails_str}")
                
                if structured_data.get('prices'):
                    prices_str = ", ".join(structured_data['prices'][:3])
                    speech_parts.append(f"Prices found: {prices_str}")
            
            return ". ".join(speech_parts)
            
        except Exception as e:
            logger.error(f"Error generating speech text: {e}")
            return f"Error reading {content_type} text from image"
    
    def get_primary_content(self, text, structured_data, content_type):
        """Get the most important content to announce based on content type"""
        try:
            if content_type == 'medicine' and structured_data and structured_data.get('medicine_names'):
                # For medicines, prioritize medicine name
                primary = structured_data['medicine_names'][0]
                if structured_data.get('dosages'):
                    primary += f" {structured_data['dosages'][0]}"
                return primary
            
            elif content_type == 'highway' and structured_data:
                # For highway signs, prioritize highway numbers and exits
                if structured_data.get('highways'):
                    return structured_data['highways'][0]
                elif structured_data.get('exits'):
                    return structured_data['exits'][0]
                elif structured_data.get('destinations'):
                    return structured_data['destinations'][0]
            
            elif content_type == 'book' and structured_data and structured_data.get('potential_titles'):
                # For books, prioritize the title
                return structured_data['potential_titles'][0]
            
            # Default: return first significant line of text
            lines = text.strip().split('\n')
            for line in lines:
                if len(line.strip()) > 3:  # Skip very short lines
                    return line.strip()
            
            return text.strip()[:100] if text.strip() else "No primary content identified"
            
        except Exception as e:
            logger.error(f"Error getting primary content: {e}")
            return text.strip()[:100] if text.strip() else "Error identifying content"
    
    def enhance_medicine_recognition(self, text):
        """Additional processing specifically for medicine recognition"""
        try:
            # Common medicine name corrections
            corrections = {
                'tylen0l': 'tylenol',
                'advi1': 'advil',
                'motrin': 'motrin',
                'aspirn': 'aspirin',
                'ibuprofin': 'ibuprofen',
                'acetaminophn': 'acetaminophen',
                'amoxicillin': 'amoxicillin',
                'lipitor': 'lipitor',
                'metformin': 'metformin',
                'lisinopril': 'lisinopril'
            }
            
            corrected_text = text
            for wrong, right in corrections.items():
                corrected_text = corrected_text.replace(wrong, right)
                corrected_text = corrected_text.replace(wrong.upper(), right.upper())
                corrected_text = corrected_text.replace(wrong.title(), right.title())
            
            return corrected_text
            
        except Exception as e:
            logger.error(f"Error in medicine recognition enhancement: {e}")
            return text
    
    def enhance_highway_recognition(self, text):
        """Additional processing specifically for highway sign recognition"""
        try:
            # Common highway sign corrections
            corrections = {
                'INTER5TATE': 'INTERSTATE',
                'H1GHWAY': 'HIGHWAY',
                'EX1T': 'EXIT',
                'M1LE': 'MILE',
                'ROUTE': 'ROUTE',
                '0': 'O',  # Common OCR mistake
                'l': '1',  # Common OCR mistake for numbers
            }
            
            corrected_text = text
            for wrong, right in corrections.items():
                corrected_text = corrected_text.replace(wrong, right)
            
            return corrected_text
            
        except Exception as e:
            logger.error(f"Error in highway recognition enhancement: {e}")
            return text
    
    def enhance_book_recognition(self, text):
        """Additional processing specifically for book title recognition"""
        try:
            # Clean up common book title issues
            lines = text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if line:
                    # Remove excessive punctuation
                    line = re.sub(r'[^\w\s\-:\'".,!?]', '', line)
                    # Fix common spacing issues
                    line = re.sub(r'\s+', ' ', line)
                    cleaned_lines.append(line)
            
            return '\n'.join(cleaned_lines)
            
        except Exception as e:
            logger.error(f"Error in book recognition enhancement: {e}")
            return text