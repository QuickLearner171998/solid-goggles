"""Image processing and quality metrics component."""

import io
import numpy as np
import cv2
from PIL import Image, ImageOps, UnidentifiedImageError
from typing import Tuple, Optional
from config import Config


class ImageProcessor:
    """Handles image processing and quality assessment."""
    
    def __init__(self):
        """Initialize image processor with face detection cascade."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    @staticmethod
    def safe_open_image(image_bytes: bytes) -> Optional[Image.Image]:
        """Safely open an image from bytes.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            PIL Image object or None if failed
        """
        try:
            im = Image.open(io.BytesIO(image_bytes))
            im.load()
            return im
        except (UnidentifiedImageError, OSError):
            return None
    
    @staticmethod
    def create_thumbnail(image: Image.Image, max_side: int = None, quality: int = None) -> Tuple[bytes, Tuple[int, int]]:
        """Create a thumbnail from an image.
        
        Args:
            image: PIL Image object
            max_side: Maximum side length (defaults to Config.MAX_THUMB_SIDE)
            quality: JPEG quality (defaults to Config.THUMB_QUALITY)
            
        Returns:
            Tuple of (thumbnail_bytes, (width, height))
        """
        max_side = max_side or Config.MAX_THUMB_SIDE
        quality = quality or Config.THUMB_QUALITY
        
        img = ImageOps.exif_transpose(image)
        img = img.convert("RGB")
        w, h = img.size
        
        scale = max(w, h) / float(max_side) if max(w, h) > max_side else 1.0
        new_w, new_h = int(w / scale), int(h / scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue(), (new_w, new_h)
    
    @staticmethod
    def calculate_sharpness(gray: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance.
        
        Args:
            gray: Grayscale image as numpy array
            
        Returns:
            Sharpness score
        """
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    @staticmethod
    def calculate_exposure(gray: np.ndarray) -> float:
        """Calculate exposure quality score.
        
        Args:
            gray: Grayscale image as numpy array
            
        Returns:
            Exposure score (0-1)
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        total = hist.sum() + 1e-9
        dark = hist[:5].sum() / total
        bright = hist[-5:].sum() / total
        middle = hist[50:205].sum() / total
        return float(max(0.0, middle - (dark + bright)))
    
    @staticmethod
    def calculate_colorfulness(img_bgr: np.ndarray) -> float:
        """Calculate colorfulness score.
        
        Args:
            img_bgr: BGR image as numpy array
            
        Returns:
            Colorfulness score
        """
        img = img_bgr.astype("float")
        B, G, R = cv2.split(img)
        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        std_rg, mean_rg = np.std(rg), np.mean(rg)
        std_yb, mean_yb = np.std(yb), np.mean(yb)
        score = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
        return float(score)
    
    def detect_faces(self, gray: np.ndarray) -> Tuple[int, float]:
        """Detect faces and calculate face score.
        
        Args:
            gray: Grayscale image as numpy array
            
        Returns:
            Tuple of (num_faces, face_score)
        """
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30), 
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        num_faces = 0 if faces is None else len(faces)
        
        if num_faces == 0:
            face_score = 0.2
        elif num_faces <= 6:
            face_score = 0.5 + 0.1 * num_faces
        else:
            face_score = 0.6 - 0.02 * (num_faces - 6)
            face_score = max(0.2, face_score)
        
        return int(num_faces), float(face_score)
    
    def calculate_combined_score(self, img_bgr: np.ndarray) -> dict:
        """Calculate combined local quality score.
        
        Args:
            img_bgr: BGR image as numpy array
            
        Returns:
            Dict with keys: score, sharpness, exposure, colorfulness, faces
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        sharpness = self.calculate_sharpness(gray)
        exposure = self.calculate_exposure(gray)
        colorfulness = self.calculate_colorfulness(img_bgr)
        num_faces, face_score = self.detect_faces(gray)
        
        # Normalize scores
        sharpness_norm = min(1.0, sharpness / Config.SHARPNESS_NORMALIZE_FACTOR)
        colorfulness_norm = min(1.0, colorfulness / Config.COLORFULNESS_NORMALIZE_FACTOR)
        exposure_norm = exposure
        
        # Weighted combination
        combined_score = (
            Config.LOCAL_SCORE_SHARPNESS_WEIGHT * sharpness_norm +
            Config.LOCAL_SCORE_EXPOSURE_WEIGHT * exposure_norm +
            Config.LOCAL_SCORE_FACE_WEIGHT * face_score +
            Config.LOCAL_SCORE_COLOR_WEIGHT * colorfulness_norm
        )
        
        return {
            'score': float(combined_score),
            'sharpness': sharpness,
            'exposure': exposure,
            'colorfulness': colorfulness,
            'faces': num_faces
        }
