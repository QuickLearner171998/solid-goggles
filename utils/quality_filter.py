"""Concurrent image quality filtering component."""

import cv2
import numpy as np
import os
import shutil
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

from .data_models import ImageData, ImageMetrics


class QualityFilter:
    """Fast concurrent quality filtering for images."""
    
    def __init__(self, 
                 blur_threshold: float = 100.0,
                 min_sharpness: float = 50.0,
                 min_exposure: float = 0.2,
                 max_workers: int = 8):
        """Initialize quality filter.
        
        Args:
            blur_threshold: Laplacian variance threshold for blur
            min_sharpness: Minimum acceptable sharpness
            min_exposure: Minimum acceptable exposure (0-1)
            max_workers: Number of concurrent workers
        """
        self.blur_threshold = blur_threshold
        self.min_sharpness = min_sharpness
        self.min_exposure = min_exposure
        self.max_workers = max_workers
        self.rejected_images = []  # Track rejected images
    
    @staticmethod
    def calculate_sharpness(gray: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Sharpness score
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    
    @staticmethod
    def calculate_exposure(gray: np.ndarray) -> float:
        """Calculate exposure quality (0-1, higher is better).
        
        Args:
            gray: Grayscale image
            
        Returns:
            Exposure score (0-1)
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Check for clipping
        shadows = hist[:10].sum()
        highlights = hist[-10:].sum()
        clipping = shadows + highlights
        
        # Check for good distribution
        mid_tones = hist[50:200].sum()
        
        # Good exposure: minimal clipping, good mid-tone distribution
        exposure_score = mid_tones * (1 - clipping)
        return float(np.clip(exposure_score, 0, 1))
    
    @staticmethod
    def calculate_colorfulness(img_bgr: np.ndarray) -> float:
        """Calculate image colorfulness.
        
        Args:
            img_bgr: BGR color image
            
        Returns:
            Colorfulness score
        """
        (B, G, R) = cv2.split(img_bgr.astype("float"))
        
        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        
        rg_mean, rg_std = (np.mean(rg), np.std(rg))
        yb_mean, yb_std = (np.mean(yb), np.std(yb))
        
        std_root = np.sqrt((rg_std ** 2) + (yb_std ** 2))
        mean_root = np.sqrt((rg_mean ** 2) + (yb_mean ** 2))
        
        return float(std_root + (0.3 * mean_root))
    
    def assess_single_image(self, image_path: str) -> Tuple[Optional[ImageData], Optional[Dict]]:
        """Assess quality of a single image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Tuple of (ImageData if passes filter else None, rejection_info dict if rejected else None)
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None, {'path': image_path, 'reason': 'Failed to load image'}
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            sharpness = self.calculate_sharpness(gray)
            exposure = self.calculate_exposure(gray)
            colorfulness = self.calculate_colorfulness(img)
            
            # Apply filters - reject blurry or poorly exposed images
            if sharpness < self.blur_threshold:
                return None, {
                    'path': image_path,
                    'name': Path(image_path).name,
                    'reason': 'Too blurry',
                    'sharpness': round(sharpness, 2),
                    'exposure': round(exposure, 2),
                    'colorfulness': round(colorfulness, 2),
                    'threshold': self.blur_threshold
                }
            
            if exposure < self.min_exposure:
                return None, {
                    'path': image_path,
                    'name': Path(image_path).name,
                    'reason': 'Poor exposure',
                    'sharpness': round(sharpness, 2),
                    'exposure': round(exposure, 2),
                    'colorfulness': round(colorfulness, 2),
                    'min_exposure': self.min_exposure
                }
            
            # Calculate combined local score (for ranking)
            normalized_sharpness = min(sharpness / 300.0, 1.0)
            local_score = (
                normalized_sharpness * 0.5 +
                exposure * 0.35 +
                min(colorfulness / 50.0, 1.0) * 0.15
            ) * 100
            
            # Create ImageData object
            metrics = ImageMetrics(
                sharpness=sharpness,
                exposure=exposure,
                colorfulness=colorfulness
            )
            
            image_data = ImageData(
                path=image_path,
                name=Path(image_path).name,
                metrics=metrics,
                local_score=local_score
            )
            
            return image_data, None
            
        except Exception as e:
            return None, {
                'path': image_path,
                'reason': f'Error processing: {str(e)}'
            }
    
    def filter_images_concurrent(self, 
                                 image_paths: List[str],
                                 show_progress: bool = True) -> List[ImageData]:
        """Filter images concurrently.
        
        Args:
            image_paths: List of image paths
            show_progress: Show progress bar
            
        Returns:
            List of ImageData for images that pass filter
        """
        print(f"\n[Quality Filter] Processing {len(image_paths)} images (concurrent)...")
        
        filtered_images = []
        self.rejected_images = []  # Reset rejected list
        
        # Process images concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.assess_single_image, path): path 
                for path in image_paths
            }
            
            # Collect results with progress bar
            if show_progress:
                iterator = tqdm(
                    as_completed(future_to_path),
                    total=len(image_paths),
                    desc="Quality filtering",
                    ncols=80
                )
            else:
                iterator = as_completed(future_to_path)
            
            for future in iterator:
                try:
                    image_data, rejection_info = future.result()
                    if image_data is not None:
                        filtered_images.append(image_data)
                    elif rejection_info is not None:
                        self.rejected_images.append(rejection_info)
                except Exception:
                    pass
        
        print(f"✓ Quality filtering complete:")
        print(f"  Input: {len(image_paths)} images")
        print(f"  Passed: {len(filtered_images)} images")
        print(f"  Rejected: {len(self.rejected_images)} images")
        
        # Sort by local score
        filtered_images.sort(key=lambda x: x.local_score, reverse=True)
        
        return filtered_images
    
    def apply_cap(self, images: List[ImageData], max_count: int) -> List[ImageData]:
        """Cap number of images to maximum.
        
        Args:
            images: List of filtered images
            max_count: Maximum images to keep
            
        Returns:
            Top max_count images
        """
        if len(images) <= max_count:
            return images
        
        print(f"  Capping to top {max_count} images by local score")
        return images[:max_count]
    
    def save_rejected_images(self, output_dir: str, max_samples: int = 100):
        """Save rejected images with reasons to output directory.
        
        Args:
            output_dir: Directory to save rejected images
            max_samples: Maximum number of rejected images to save
        """
        if not self.rejected_images:
            print("  No rejected images to save")
            return
        
        import json
        import csv
        
        rejected_dir = os.path.join(output_dir, "rejected_images")
        os.makedirs(rejected_dir, exist_ok=True)
        
        # Group by rejection reason
        by_reason = {}
        for rejected in self.rejected_images:
            reason = rejected.get('reason', 'unknown')
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(rejected)
        
        # Save JSON report
        json_path = os.path.join(rejected_dir, "rejection_report.json")
        with open(json_path, 'w') as f:
            json.dump({
                'total_rejected': len(self.rejected_images),
                'by_reason': {k: len(v) for k, v in by_reason.items()},
                'rejected_images': self.rejected_images  # Limit JSON size
            }, f, indent=2)
        
        # Save CSV report
        csv_path = os.path.join(rejected_dir, "rejection_report.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'filename', 'reason', 'sharpness', 'exposure', 
                'colorfulness', 'threshold_info'
            ])
            
            for rejected in self.rejected_images:
                writer.writerow([
                    rejected.get('name', Path(rejected['path']).name),
                    rejected.get('reason', 'unknown'),
                    rejected.get('sharpness', ''),
                    rejected.get('exposure', ''),
                    rejected.get('colorfulness', ''),
                    rejected.get('threshold', rejected.get('min_exposure', ''))
                ])
        
        # Save sample images (organized by reason)
        samples_saved = 0
        for reason, rejected_list in by_reason.items():
            # Sanitize reason for directory name
            safe_reason = reason.replace(' ', '_').replace('/', '_').lower()
            reason_dir = os.path.join(rejected_dir, f"samples_{safe_reason}")
            os.makedirs(reason_dir, exist_ok=True)
            
            # Save up to 20 samples per reason
            samples_per_reason = min(20, max_samples // len(by_reason))
            for i, rejected in enumerate(rejected_list[:samples_per_reason], 1):
                if samples_saved >= max_samples:
                    break
                
                try:
                    src_path = rejected['path']
                    if os.path.exists(src_path):
                        filename = f"{i:03d}_{Path(src_path).name}"
                        dest_path = os.path.join(reason_dir, filename)
                        shutil.copy2(src_path, dest_path)
                        samples_saved += 1
                except Exception:
                    pass
        
        print(f"✓ Saved rejected images analysis:")
        print(f"  Total rejected: {len(self.rejected_images)}")
        print(f"  Rejection reasons:")
        for reason, count in by_reason.items():
            print(f"    {reason}: {count} images")
        print(f"  Saved {samples_saved} sample images to: {rejected_dir}")
        print(f"  Reports: rejection_report.json, rejection_report.csv")

