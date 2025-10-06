"""Deduplication component using perceptual hashing."""

import imagehash
from PIL import Image
from typing import Dict, Optional
from config import Config


class Deduplicator:
    """Handles near-duplicate detection using perceptual hashing."""
    
    def __init__(self):
        """Initialize deduplicator."""
        self.hash_to_best = {}  # phash -> (score, image_data)
    
    @staticmethod
    def compute_hash(image: Image.Image) -> str:
        """Compute perceptual hash for an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Hash string
        """
        thumb = image.copy()
        thumb.thumbnail((Config.DEDUPE_HASH_SIZE, Config.DEDUPE_HASH_SIZE))
        return str(imagehash.phash(thumb))
    
    def add_or_update(self, image_hash: str, score: float, image_data: dict) -> bool:
        """Add image or update if better than existing with same hash.
        
        Args:
            image_hash: Perceptual hash of the image
            score: Quality score of the image
            image_data: Dict containing image metadata
            
        Returns:
            True if added/updated, False if rejected (duplicate with lower score)
        """
        existing = self.hash_to_best.get(image_hash)
        
        if existing is None or score > existing[0]:
            self.hash_to_best[image_hash] = (score, image_data)
            return True
        
        return False
    
    def get_unique_images(self) -> list:
        """Get all unique images (best per hash).
        
        Returns:
            List of image data dicts
        """
        return [data for score, data in self.hash_to_best.values()]
    
    def get_count(self) -> int:
        """Get count of unique images.
        
        Returns:
            Number of unique images
        """
        return len(self.hash_to_best)
    
    def clear(self):
        """Clear all stored hashes."""
        self.hash_to_best.clear()
