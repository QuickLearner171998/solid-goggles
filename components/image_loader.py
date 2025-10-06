"""Image loading component."""

import os
from typing import List, Dict, Generator
from pathlib import Path
from config import Config


class ImageLoader:
    """Handles loading images from local filesystem."""
    
    def __init__(self, source_folder: str = None):
        """Initialize image loader.
        
        Args:
            source_folder: Path to folder containing images. Defaults to Config.SOURCE_FOLDER
        """
        self.source_folder = source_folder or Config.SOURCE_FOLDER
        if not os.path.exists(self.source_folder):
            raise ValueError(f"Source folder does not exist: {self.source_folder}")
    
    def discover_images(self) -> List[Dict[str, str]]:
        """Discover all supported images in the source folder.
        
        Returns:
            List of dicts with keys: path, name, size
        """
        images = []
        path_obj = Path(self.source_folder)
        
        for ext in Config.IMG_EXTENSIONS:
            for img_path in path_obj.rglob(f"*{ext}"):
                if img_path.is_file():
                    images.append({
                        'path': str(img_path.absolute()),
                        'name': img_path.name,
                        'size': img_path.stat().st_size
                    })
        
        return images
    
    def load_batch(self, image_list: List[Dict[str, str]], start: int, batch_size: int) -> List[Dict[str, str]]:
        """Load a batch of images.
        
        Args:
            image_list: List of image metadata dicts
            start: Starting index
            batch_size: Number of images to load
            
        Returns:
            List of image metadata dicts for the batch
        """
        end = min(start + batch_size, len(image_list))
        return image_list[start:end]
    
    def get_image_bytes(self, image_path: str) -> bytes:
        """Read image file as bytes.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image bytes
        """
        with open(image_path, 'rb') as f:
            return f.read()
