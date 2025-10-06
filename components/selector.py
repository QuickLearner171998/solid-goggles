"""Image selection and ranking component."""

import csv
import os
import shutil
from typing import List, Dict
from collections import defaultdict
from config import Config


class Selector:
    """Handles final selection and ranking of images."""
    
    def __init__(self, output_dir: str = None):
        """Initialize selector.
        
        Args:
            output_dir: Output directory path (defaults to Config.OUTPUT_DIR)
        """
        self.output_dir = output_dir or Config.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    @staticmethod
    def calculate_final_scores(results: List[Dict]) -> List[Dict]:
        """Calculate final scores combining local and LLM scores.
        
        Args:
            results: List of result dicts with local_score and overall_score
            
        Returns:
            Updated results with final_score added
        """
        if not results:
            return results
        
        # Normalize local scores to 0-100 scale
        max_local = max(r["local_score"] for r in results) or 1.0
        
        for r in results:
            local_normalized = 100.0 * (r["local_score"] / max_local)
            r["final_score"] = (
                Config.FINAL_SCORE_LLM_WEIGHT * r["overall_score"] +
                Config.FINAL_SCORE_LOCAL_WEIGHT * local_normalized
            )
        
        return results
    
    @staticmethod
    def rank_and_select(results: List[Dict], num_select: int = None) -> List[Dict]:
        """Rank results and select top N.
        
        Args:
            results: List of result dicts with final_score
            num_select: Number to select (defaults to Config.NUM_SELECT)
            
        Returns:
            Top N results sorted by final_score
        """
        num_select = num_select or Config.NUM_SELECT
        sorted_results = sorted(results, key=lambda r: r["final_score"], reverse=True)
        return sorted_results[:num_select]
    
    def save_csv_report(self, results: List[Dict], filename: str = "album_scores_log.csv"):
        """Save scoring results to CSV.
        
        Args:
            results: List of result dicts
            filename: Output CSV filename or full path
        """
        # Check if filename is a full path
        if os.path.isabs(filename):
            csv_path = filename
        else:
            csv_path = os.path.join(self.output_dir, filename)
        
        fieldnames = [
            "name", "path", "final_score",
            "overall_score", "technical_score", "composition_score", "moment_score",
            "local_score", "sharpness", "exposure", "colorfulness", "faces", 
            "tags", "reject_reason", "phash"
        ]
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in results:
                row = {k: r.get(k, "") for k in fieldnames}
                if isinstance(row.get("tags"), list):
                    row["tags"] = "|".join(str(t) for t in row["tags"])
                writer.writerow(row)
        
        return csv_path
    
    def copy_selected_images(self, selected: List[Dict]) -> int:
        """Copy selected images to output directory.
        
        Args:
            selected: List of selected image dicts with 'path' and 'name'
            
        Returns:
            Number of images successfully copied
        """
        name_counts = defaultdict(int)
        copied_count = 0
        
        for item in selected:
            source_path = item.get("path")
            if not source_path or not os.path.exists(source_path):
                continue
            
            name = item["name"]
            name_counts[name] += 1
            
            # Handle filename collisions
            if name_counts[name] > 1:
                root, ext = os.path.splitext(name)
                dest_name = f"{root}_{name_counts[name]}{ext}"
            else:
                dest_name = name
            
            dest_path = os.path.join(self.output_dir, dest_name)
            
            try:
                shutil.copy2(source_path, dest_path)
                copied_count += 1
            except Exception:
                continue
        
        return copied_count
