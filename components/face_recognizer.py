"""Face recognition component for identifying specific people from reference images."""

import os
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import cv2


class FaceRecognizer:
    """Recognizes specific people using reference images and face embeddings."""
    
    def __init__(self, reference_dir: str = "./reference_faces"):
        """Initialize face recognizer.
        
        Args:
            reference_dir: Directory containing reference face folders
        """
        self.reference_dir = reference_dir
        self.face_encodings = {}  # Category -> list of face encodings
        self.reference_loaded = False
        
        # Initialize face detection
        self.face_cascade = self._load_face_detector()
        
        # Try to load deep learning face recognizer if available
        try:
            import face_recognition
            self.use_deep_learning = True
            self.face_recognition = face_recognition
            print("✓ Using deep learning face recognition (face_recognition library)")
        except ImportError:
            self.use_deep_learning = False
            print("⚠ face_recognition library not available, using basic detection")
            print("  Install with: pip install face_recognition")
    
    def _load_face_detector(self):
        """Load OpenCV face detector."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        return cv2.CascadeClassifier(cascade_path)
    
    def create_reference_directories(self):
        """Create reference directory structure."""
        categories = [
            'bride-groom',
            'bride-parents',
            'bride-siblings',
            'groom-parents'
        ]
        
        os.makedirs(self.reference_dir, exist_ok=True)
        
        for category in categories:
            category_path = os.path.join(self.reference_dir, category)
            os.makedirs(category_path, exist_ok=True)
        
        # Create README
        readme_path = os.path.join(self.reference_dir, 'README.txt')
        with open(readme_path, 'w') as f:
            f.write("REFERENCE FACES FOR PERSON RECOGNITION\n")
            f.write("=" * 60 + "\n\n")
            f.write("Add reference photos to these folders:\n\n")
            f.write("1. bride-groom/\n")
            f.write("   - Add clear photos of the bride and/or groom\n")
            f.write("   - Can be individual or together\n")
            f.write("   - Face should be clearly visible\n\n")
            f.write("2. bride-parents/\n")
            f.write("   - Add photos of bride's mother and/or father\n")
            f.write("   - Clear face shots preferred\n\n")
            f.write("3. bride-siblings/\n")
            f.write("   - Add photos of bride's brothers/sisters\n")
            f.write("   - One photo per sibling\n\n")
            f.write("4. groom-parents/\n")
            f.write("   - Add photos of groom's mother and/or father\n")
            f.write("   - Clear face shots preferred\n\n")
            f.write("TIPS:\n")
            f.write("- Use clear, well-lit photos\n")
            f.write("- Face should be clearly visible (no sunglasses, side angles)\n")
            f.write("- One face per photo is ideal\n")
            f.write("- JPG, JPEG, PNG formats supported\n")
            f.write("- You can add multiple reference photos per category\n")
        
        print(f"\n✓ Created reference directory structure at: {self.reference_dir}")
        print("\nReference folders created:")
        for category in categories:
            print(f"  - {category}/")
        print(f"\nREADME created at: {readme_path}")
        print("\nNext steps:")
        print("1. Add reference photos to appropriate folders")
        print("2. Run the selection pipeline")
        print("3. System will automatically learn faces from reference images")
    
    def load_reference_faces(self) -> Dict[str, int]:
        """Load and encode all reference faces.
        
        Returns:
            Dict with category -> count of faces loaded
        """
        if not os.path.exists(self.reference_dir):
            print(f"⚠ Reference directory not found: {self.reference_dir}")
            return {}
        
        print(f"\nLoading reference faces from: {self.reference_dir}")
        
        stats = {}
        
        # Scan all subdirectories
        for category in os.listdir(self.reference_dir):
            category_path = os.path.join(self.reference_dir, category)
            
            if not os.path.isdir(category_path):
                continue
            
            # Load images from this category
            encodings = []
            image_files = [f for f in os.listdir(category_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files:
                img_path = os.path.join(category_path, img_file)
                
                try:
                    # Load and encode faces
                    faces = self._encode_faces_in_image(img_path)
                    
                    if faces:
                        encodings.extend(faces)
                        print(f"  ✓ Loaded {len(faces)} face(s) from {category}/{img_file}")
                    else:
                        print(f"  ⚠ No faces detected in {category}/{img_file}")
                
                except Exception as e:
                    print(f"  ✗ Error loading {category}/{img_file}: {e}")
            
            if encodings:
                self.face_encodings[category] = encodings
                stats[category] = len(encodings)
                print(f"✓ Loaded {len(encodings)} total face(s) for category: {category}")
            else:
                print(f"⚠ No faces loaded for category: {category}")
        
        self.reference_loaded = len(self.face_encodings) > 0
        
        if self.reference_loaded:
            print(f"\n✓ Reference faces loaded successfully!")
            print(f"  Total categories: {len(self.face_encodings)}")
            print(f"  Total faces: {sum(stats.values())}")
        else:
            print(f"\n⚠ No reference faces loaded. Add photos to {self.reference_dir}")
        
        return stats
    
    def _encode_faces_in_image(self, image_path: str) -> List[np.ndarray]:
        """Extract and encode faces from an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of face encodings
        """
        if self.use_deep_learning:
            return self._encode_faces_deep_learning(image_path)
        else:
            return self._encode_faces_basic(image_path)
    
    def _encode_faces_deep_learning(self, image_path: str) -> List[np.ndarray]:
        """Encode faces using face_recognition library (dlib-based)."""
        try:
            # Load image
            image = self.face_recognition.load_image_file(image_path)
            
            # Find faces and encode
            face_locations = self.face_recognition.face_locations(image)
            face_encodings = self.face_recognition.face_encodings(image, face_locations)
            
            return face_encodings
            
        except Exception as e:
            print(f"  ⚠ Error encoding faces: {e}")
            return []
    
    def _encode_faces_basic(self, image_path: str) -> List[np.ndarray]:
        """Basic face encoding using OpenCV (fallback method)."""
        try:
            # Load image
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # Create simple encodings (not as accurate as deep learning)
            encodings = []
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                # Resize to standard size and flatten as encoding
                face_resized = cv2.resize(face_roi, (128, 128))
                encoding = face_resized.flatten().astype(np.float32)
                encodings.append(encoding)
            
            return encodings
            
        except Exception as e:
            print(f"  ⚠ Error encoding faces: {e}")
            return []
    
    def recognize_faces_in_image(self, image_path: str, 
                                 threshold: float = 0.6) -> Dict[str, float]:
        """Recognize faces in an image against reference faces.
        
        Args:
            image_path: Path to image to analyze
            threshold: Recognition confidence threshold (0-1)
            
        Returns:
            Dict mapping category -> confidence score
        """
        if not self.reference_loaded:
            return {}
        
        # Extract faces from test image
        test_encodings = self._encode_faces_in_image(image_path)
        
        if not test_encodings:
            return {}
        
        # Compare against reference faces
        matches = {}
        
        for category, ref_encodings in self.face_encodings.items():
            max_confidence = 0.0
            
            # Compare each test face against all reference faces in this category
            for test_enc in test_encodings:
                for ref_enc in ref_encodings:
                    confidence = self._compare_faces(test_enc, ref_enc)
                    max_confidence = max(max_confidence, confidence)
            
            # Only include if above threshold
            if max_confidence >= threshold:
                matches[category] = max_confidence
        
        return matches
    
    def recognize_faces_in_pil_image(self, pil_image: Image.Image,
                                    threshold: float = 0.6) -> Dict[str, float]:
        """Recognize faces in a PIL Image.
        
        Args:
            pil_image: PIL Image object
            threshold: Recognition confidence threshold
            
        Returns:
            Dict mapping category -> confidence score
        """
        import tempfile
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            pil_image.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            result = self.recognize_faces_in_image(tmp_path, threshold)
        finally:
            os.unlink(tmp_path)
        
        return result
    
    def _compare_faces(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """Compare two face encodings and return confidence score.
        
        Args:
            encoding1: First face encoding
            encoding2: Second face encoding
            
        Returns:
            Confidence score (0-1), higher is better match
        """
        if self.use_deep_learning:
            # Use euclidean distance (face_recognition default)
            distance = np.linalg.norm(encoding1 - encoding2)
            # Convert distance to confidence (0-1 scale)
            # Typical threshold is 0.6, so we map it
            confidence = max(0.0, 1.0 - (distance / 1.2))
            return confidence
        else:
            # For basic method, use correlation
            correlation = np.corrcoef(encoding1, encoding2)[0, 1]
            # Map to 0-1 scale
            confidence = (correlation + 1.0) / 2.0
            return max(0.0, min(1.0, confidence))
    
    def batch_recognize(self, images: List[Dict], 
                       image_loader, 
                       image_processor,
                       threshold: float = 0.6) -> List[Dict]:
        """Recognize faces in a batch of images.
        
        Args:
            images: List of image metadata dicts
            image_loader: ImageLoader instance
            image_processor: ImageProcessor instance
            threshold: Recognition confidence threshold
            
        Returns:
            List of images with face recognition results added
        """
        from tqdm import tqdm
        
        if not self.reference_loaded:
            print("⚠ No reference faces loaded. Skipping face recognition.")
            return images
        
        print(f"\nRecognizing faces in {len(images)} images...")
        
        for img in tqdm(images, desc="Face recognition", ncols=80):
            try:
                img_bytes = image_loader.get_image_bytes(img['path'])
                pil_image = image_processor.safe_open_image(img_bytes)
                
                if pil_image is None:
                    continue
                
                # Recognize faces
                matches = self.recognize_faces_in_pil_image(pil_image, threshold)
                
                # Add to image metadata
                img['face_matches'] = matches
                img['recognized_categories'] = list(matches.keys())
                
                del pil_image, img_bytes
                
            except Exception as e:
                img['face_matches'] = {}
                img['recognized_categories'] = []
        
        return images
    
    def save_recognition_stats(self, images: List[Dict], output_path: str):
        """Save face recognition statistics.
        
        Args:
            images: List of images with face recognition results
            output_path: Path to save statistics
        """
        stats = {
            'total_images': len(images),
            'images_with_matches': 0,
            'category_counts': {},
            'images': []
        }
        
        for img in images:
            matches = img.get('face_matches', {})
            
            if matches:
                stats['images_with_matches'] += 1
            
            for category in matches.keys():
                stats['category_counts'][category] = stats['category_counts'].get(category, 0) + 1
            
            if matches:
                stats['images'].append({
                    'name': img['name'],
                    'path': img['path'],
                    'matches': matches
                })
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✓ Face recognition statistics saved: {output_path}")
        print(f"  Images processed: {stats['total_images']}")
        print(f"  Images with matches: {stats['images_with_matches']}")
        if stats['category_counts']:
            print("  Matches by category:")
            for category, count in stats['category_counts'].items():
                print(f"    {category}: {count} images")

