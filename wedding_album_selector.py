"""Main orchestrator for wedding album selection."""

import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Dict

from config import Config
from components import ImageLoader, ImageProcessor, LLMScorer, Deduplicator, Selector, AlbumCurator


class WeddingAlbumSelector:
    """Main orchestrator that coordinates all components."""
    
    def __init__(self, config: Config = None):
        """Initialize the wedding album selector.
        
        Args:
            config: Configuration object (uses Config class if not provided)
        """
        self.config = config or Config
        
        # Initialize components
        self.loader = ImageLoader(self.config.SOURCE_FOLDER)
        self.processor = ImageProcessor()
        self.scorer = LLMScorer(self.config.OPENAI_API_KEY, self.config.OPENAI_VISION_MODEL)
        self.deduplicator = Deduplicator()
        self.selector = Selector(self.config.OUTPUT_DIR)
        self.curator = AlbumCurator()
    
    def run(self):
        """Execute the full album selection pipeline."""
        print("=" * 60)
        print("Wedding Album Auto-Selector")
        print("=" * 60)
        
        # Phase 1: Discover images
        print("\n[Phase 1] Discovering images...")
        all_images = self.loader.discover_images()
        
        if not all_images:
            print("No supported images found.")
            return
        
        print(f"Found {len(all_images)} images in {self.config.SOURCE_FOLDER}")
        
        # Phase 2: Local prefiltering and deduplication
        print("\n[Phase 2] Local quality assessment and deduplication...")
        candidates = self._prefilter_images(all_images)
        
        if not candidates:
            print("No candidates passed prefiltering.")
            return
        
        print(f"After prefiltering: {len(candidates)} unique candidates")
        
        # Phase 3: LLM scoring
        print("\n[Phase 3] LLM scoring...")
        scored_results = self._llm_score_images(candidates)
        
        if not scored_results:
            print("No results from LLM scoring.")
            return
        
        # Phase 4: Final selection
        print("\n[Phase 4] Final selection and ranking...")
        final_results = self._finalize_selection(scored_results)
        
        # Phase 5: Save results
        print("\n[Phase 5] Saving results...")
        self._save_results(final_results)
        
        print("\n" + "=" * 60)
        print("✅ Album selection complete!")
        print(f"Output directory: {self.config.OUTPUT_DIR}")
        print("=" * 60)
    
    def _prefilter_images(self, images: List[Dict]) -> List[Dict]:
        """Prefilter images using local quality metrics and deduplication.
        
        Args:
            images: List of image metadata dicts
            
        Returns:
            List of candidate image dicts with quality scores
        """
        pbar = tqdm(total=len(images), desc="Processing", ncols=80)
        
        for i in range(0, len(images), self.config.BATCH_SIZE):
            batch = self.loader.load_batch(images, i, self.config.BATCH_SIZE)
            
            for img_meta in batch:
                pbar.update(1)
                
                try:
                    # Load image
                    img_bytes = self.loader.get_image_bytes(img_meta['path'])
                    pil_image = self.processor.safe_open_image(img_bytes)
                    
                    if pil_image is None:
                        continue
                    
                    # Calculate perceptual hash
                    img_hash = self.deduplicator.compute_hash(pil_image)
                    
                    # Calculate local quality metrics
                    img_cv = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
                    metrics = self.processor.calculate_combined_score(img_cv)
                    
                    # Prepare candidate data
                    candidate = {
                        'path': img_meta['path'],
                        'name': img_meta['name'],
                        'phash': img_hash,
                        'local_score': metrics['score'],
                        'sharpness': metrics['sharpness'],
                        'exposure': metrics['exposure'],
                        'colorfulness': metrics['colorfulness'],
                        'faces': metrics['faces']
                    }
                    
                    # Add to deduplicator (keeps best per hash)
                    self.deduplicator.add_or_update(img_hash, metrics['score'], candidate)
                    
                    # Early cap if too many unique images
                    if self.deduplicator.get_count() > int(self.config.LLM_PREFILTER_TARGET * 1.5):
                        candidates = self.deduplicator.get_unique_images()
                        candidates.sort(key=lambda x: x['local_score'], reverse=True)
                        self.deduplicator.clear()
                        for cand in candidates[:self.config.LLM_PREFILTER_TARGET]:
                            self.deduplicator.add_or_update(cand['phash'], cand['local_score'], cand)
                    
                    del pil_image, img_cv, img_bytes
                    
                except Exception:
                    continue
        
        pbar.close()
        
        # Get unique candidates and sort by local score
        candidates = self.deduplicator.get_unique_images()
        candidates.sort(key=lambda x: x['local_score'], reverse=True)
        
        # Cap to target
        return candidates[:self.config.LLM_PREFILTER_TARGET]
    
    def _llm_score_images(self, candidates: List[Dict]) -> List[Dict]:
        """Score images using LLM vision model.
        
        Args:
            candidates: List of candidate image dicts
            
        Returns:
            List of candidates with LLM scores added
        """
        results = []
        pbar = tqdm(total=len(candidates), desc="LLM scoring", ncols=80)
        
        for i in range(0, len(candidates), self.config.IMAGES_PER_LLM_CALL):
            batch = candidates[i:i + self.config.IMAGES_PER_LLM_CALL]
            
            # Prepare batch for LLM
            llm_batch = []
            valid_candidates = []
            
            for candidate in batch:
                try:
                    img_bytes = self.loader.get_image_bytes(candidate['path'])
                    pil_image = self.processor.safe_open_image(img_bytes)
                    
                    if pil_image is None:
                        pbar.update(1)
                        continue
                    
                    thumb_bytes, _ = self.processor.create_thumbnail(pil_image)
                    llm_batch.append({'name': candidate['name'], 'thumb_bytes': thumb_bytes})
                    valid_candidates.append(candidate)
                    
                    del pil_image, img_bytes, thumb_bytes
                    
                except Exception:
                    pass
                
                pbar.update(1)
            
            if not llm_batch:
                continue
            
            # Score batch with LLM
            try:
                scores = self.scorer.score_batch(llm_batch)
                
                # Merge scores with candidates
                for candidate, score in zip(valid_candidates, scores):
                    result = {**candidate}
                    result['technical_score'] = float(score.get('technical_score', 0))
                    result['composition_score'] = float(score.get('composition_score', 0))
                    result['moment_score'] = float(score.get('moment_score', 0))
                    result['overall_score'] = float(score.get('overall_score', 0))
                    result['tags'] = score.get('tags', [])
                    result['reject_reason'] = score.get('reject_reason', '')
                    results.append(result)
                    
            except Exception:
                continue
        
        pbar.close()
        return results
    
    def _finalize_selection(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Calculate final scores and select top images with balanced distribution.
        
        Args:
            results: List of scored result dicts
            
        Returns:
            Dict with 'all' and 'selected' keys containing result lists
        """
        # Calculate final scores
        results = self.selector.calculate_final_scores(results)
        
        # Use curator for balanced selection across ceremony types
        print("\nApplying intelligent curation for ceremony balance...")
        selected = self.curator.curate_balanced_selection(results, self.config.NUM_SELECT)
        
        print(f"Selected {len(selected)} images from {len(results)} scored images")
        
        # Show distribution report
        print(self.curator.get_distribution_report(selected))
        
        return {
            'all': results,
            'selected': selected
        }
    
    def _save_results(self, final_results: Dict[str, List[Dict]]):
        """Save results to disk.
        
        Args:
            final_results: Dict with 'all' and 'selected' keys
        """
        # Save CSV report
        csv_path = self.selector.save_csv_report(final_results['all'])
        print(f"Saved scoring report: {csv_path}")
        
        # Copy selected images
        print(f"Copying {len(final_results['selected'])} selected images...")
        copied = self.selector.copy_selected_images(final_results['selected'])
        print(f"Successfully copied {copied} images to {self.config.OUTPUT_DIR}")
