"""Verbose logging component for saving intermediate results."""

import os
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any


class VerboseLogger:
    """Logs and saves intermediate results at each pipeline stage."""
    
    def __init__(self, base_output_dir: str):
        """Initialize verbose logger.
        
        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_output_dir = base_output_dir
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create main output directories
        self.intermediate_dir = os.path.join(base_output_dir, "intermediate_results")
        self.logs_dir = os.path.join(base_output_dir, "logs")
        self.final_dir = os.path.join(base_output_dir, "final_selection")
        
        # Create person-specific directories
        self.person_dirs = {
            'bride': os.path.join(self.final_dir, "01_bride"),
            'groom': os.path.join(self.final_dir, "02_groom"),
            'couple': os.path.join(self.final_dir, "03_couple"),
            'bride_parents': os.path.join(self.final_dir, "04_bride_parents"),
            'bride_mom': os.path.join(self.final_dir, "04a_bride_mom"),
            'bride_dad': os.path.join(self.final_dir, "04b_bride_dad"),
            'bride_brother': os.path.join(self.final_dir, "05_bride_siblings"),
            'groom_parents': os.path.join(self.final_dir, "06_groom_parents"),
            'groom_mom': os.path.join(self.final_dir, "06a_groom_mom"),
            'groom_dad': os.path.join(self.final_dir, "06b_groom_dad"),
            'family': os.path.join(self.final_dir, "07_family"),
            'ceremony': os.path.join(self.final_dir, "08_ceremony"),
            'guests': os.path.join(self.final_dir, "09_guests"),
            'children': os.path.join(self.final_dir, "10_children"),
            'other': os.path.join(self.final_dir, "11_other")
        }
        
        # Create all directories
        self._create_directories()
        
        # Initialize log file
        self.log_file = os.path.join(self.logs_dir, f"pipeline_log_{self.run_timestamp}.txt")
        self._log(f"Initialized VerboseLogger at {datetime.now()}")
        self._log(f"Output directory: {self.base_output_dir}")
    
    def _create_directories(self):
        """Create all necessary directories."""
        dirs_to_create = [
            self.intermediate_dir,
            self.logs_dir,
            self.final_dir
        ] + list(self.person_dirs.values())
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
    
    def _log(self, message: str, also_print: bool = True):
        """Write message to log file.
        
        Args:
            message: Message to log
            also_print: Whether to also print to console
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
        
        if also_print:
            print(message)
    
    def log_phase(self, phase_number: int, phase_name: str, description: str = ""):
        """Log the start of a pipeline phase.
        
        Args:
            phase_number: Phase number
            phase_name: Name of the phase
            description: Optional description
        """
        separator = "=" * 60
        self._log(f"\n{separator}")
        self._log(f"PHASE {phase_number}: {phase_name}")
        if description:
            self._log(f"  {description}")
        self._log(separator)
    
    def save_phase_results(self, phase_name: str, data: Dict[str, Any], 
                          suffix: str = ""):
        """Save phase results as JSON.
        
        Args:
            phase_name: Name of the phase
            data: Data to save
            suffix: Optional suffix for filename
        """
        filename = f"{phase_name.lower().replace(' ', '_')}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".json"
        
        output_path = os.path.join(self.intermediate_dir, filename)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self._log(f"✓ Saved {phase_name} results: {output_path}", also_print=False)
            
        except Exception as e:
            self._log(f"⚠ Error saving {phase_name} results: {e}")
    
    def save_image_metadata(self, phase_name: str, images: List[Dict]):
        """Save image metadata for a phase.
        
        Args:
            phase_name: Name of the phase
            images: List of image metadata dicts
        """
        data = {
            'phase': phase_name,
            'timestamp': datetime.now().isoformat(),
            'count': len(images),
            'images': images
        }
        
        self.save_phase_results(phase_name, data)
    
    def save_statistics(self, phase_name: str, stats: Dict[str, Any]):
        """Save statistics for a phase.
        
        Args:
            phase_name: Name of the phase
            stats: Statistics dict
        """
        stats['timestamp'] = datetime.now().isoformat()
        self.save_phase_results(f"{phase_name}_statistics", stats)
    
    def copy_images_to_person_directory(self, images: List[Dict], 
                                       person_category: str,
                                       max_images: int = 100) -> int:
        """Copy images to person-specific directory.
        
        Args:
            images: List of image metadata dicts
            person_category: Person category
            max_images: Maximum images to copy
            
        Returns:
            Number of images copied
        """
        if person_category not in self.person_dirs:
            self._log(f"⚠ Unknown person category: {person_category}")
            return 0
        
        target_dir = self.person_dirs[person_category]
        
        copied = 0
        for i, img in enumerate(images[:max_images], 1):
            try:
                src_path = img['path']
                
                # Create filename with rank prefix
                filename = f"{i:03d}_{os.path.basename(src_path)}"
                dest_path = os.path.join(target_dir, filename)
                
                shutil.copy2(src_path, dest_path)
                copied += 1
                
            except Exception as e:
                self._log(f"⚠ Error copying {img.get('name', 'unknown')}: {e}", 
                         also_print=False)
        
        return copied
    
    def generate_summary_report(self, pipeline_stats: Dict[str, Any]):
        """Generate final summary report.
        
        Args:
            pipeline_stats: Statistics from all pipeline phases
        """
        report_path = os.path.join(self.final_dir, "SUMMARY_REPORT.txt")
        
        lines = []
        lines.append("=" * 70)
        lines.append("WEDDING ALBUM SELECTOR - FINAL REPORT")
        lines.append("=" * 70)
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Output Directory: {self.base_output_dir}")
        lines.append("\n" + "=" * 70)
        lines.append("PIPELINE STATISTICS")
        lines.append("=" * 70)
        
        for phase, stats in pipeline_stats.items():
            lines.append(f"\n{phase}:")
            for key, value in stats.items():
                if key != 'timestamp':
                    lines.append(f"  {key}: {value}")
        
        lines.append("\n" + "=" * 70)
        lines.append("PERSON-SPECIFIC SELECTIONS")
        lines.append("=" * 70)
        
        for category, dir_path in self.person_dirs.items():
            if os.path.exists(dir_path):
                count = len([f for f in os.listdir(dir_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                if count > 0:
                    lines.append(f"  {category.replace('_', ' ').title()}: {count} images")
        
        lines.append("\n" + "=" * 70)
        lines.append("DIRECTORY STRUCTURE")
        lines.append("=" * 70)
        lines.append(f"\nFinal Selection: {self.final_dir}")
        lines.append(f"Intermediate Results: {self.intermediate_dir}")
        lines.append(f"Logs: {self.logs_dir}")
        lines.append("\n" + "=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)
        
        # Write report
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        
        # Also print to console
        print("\n" + '\n'.join(lines))
        
        self._log(f"\n✓ Summary report saved: {report_path}")
    
    def log_completion(self, total_time: float):
        """Log pipeline completion.
        
        Args:
            total_time: Total execution time in seconds
        """
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        
        self._log(f"\n{'=' * 60}")
        self._log(f"Pipeline completed successfully!")
        self._log(f"Total execution time: {minutes}m {seconds}s")
        self._log(f"{'=' * 60}\n")

