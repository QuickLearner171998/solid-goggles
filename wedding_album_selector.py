"""Main orchestrator for wedding album selection with enhanced AI features."""

import gc
import time
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional

from config import Config
from components import (
    ImageLoader, ImageProcessor, LLMScorer,
    Selector, AlbumCurator, ImageEmbedder, ImageClusterer,
    VerboseLogger, LLMClusterSelector, PipelineStatistics
)


class WeddingAlbumSelector:
    """Enhanced orchestrator with embeddings, clustering, and person detection."""
    
    def __init__(self, config: Config = None):
        """Initialize the wedding album selector.
        
        Args:
            config: Configuration object (uses Config class if not provided)
        """
        self.config = config or Config
        
        # Initialize core components
        self.loader = ImageLoader(self.config.SOURCE_FOLDER)
        self.processor = ImageProcessor()
        self.scorer = LLMScorer(self.config.OPENAI_API_KEY, self.config.OPENAI_VISION_MODEL)
        self.selector = Selector(self.config.OUTPUT_DIR)
        self.curator = AlbumCurator()
        
        # Pipeline statistics
        self.stats = PipelineStatistics()
        
        # Initialize new components
        if self.config.VERBOSE_LOGGING:
            self.logger = VerboseLogger(self.config.OUTPUT_DIR)
        else:
            self.logger = None
        
        if self.config.USE_EMBEDDINGS:
            self.embedder = ImageEmbedder(
                model_name=self.config.EMBEDDING_MODEL,
                device=None if self.config.USE_GPU_IF_AVAILABLE else 'cpu'
            )
            self.clusterer = ImageClusterer(method=self.config.CLUSTERING_METHOD)
            
            if self.config.USE_LLM_FOR_CLUSTER_SELECTION:
                self.llm_cluster_selector = LLMClusterSelector(
                    self.config.OPENAI_API_KEY,
                    self.config.OPENAI_VISION_MODEL
                )
            else:
                self.llm_cluster_selector = None
        else:
            self.embedder = None
            self.clusterer = None
            self.llm_cluster_selector = None
        
    
    def run(self):
        """Execute the enhanced album selection pipeline."""
        start_time = time.time()
        
        print("=" * 60)
        print("Wedding Album Auto-Selector (Enhanced)")
        print("=" * 60)
        
        if self.logger:
            self.logger.log_phase(0, "Initialization", 
                                 "Pipeline: All images → Embeddings → Clustering → LLM Selection → Final")
        
        # Phase 1: Discover all images (no filtering)
        if self.logger:
            self.logger.log_phase(1, "Image Discovery")
        
        print("\n[Phase 1] Discovering all images...")
        all_images = self.loader.discover_images()
        
        if not all_images:
            print("No supported images found.")
            return
        
        print(f"Found {len(all_images)} images in {self.config.SOURCE_FOLDER}")
        self.stats.total_images = len(all_images)
        
        # Prepare image list for embeddings (no filtering)
        candidates = [{'path': img['path'], 'name': img['name']} for img in all_images]
        
        if self.logger:
            self.logger.save_image_metadata("01_all_images", candidates)
        
        # Phase 2: Generate embeddings
        embeddings = None
        if self.config.USE_EMBEDDINGS and self.embedder:
            if self.logger:
                self.logger.log_phase(2, "Image Embedding Generation")
            
            embeddings = self._generate_embeddings(candidates)
        
        # Phase 3: Cluster similar images
        cluster_labels = None
        if self.config.USE_EMBEDDINGS and embeddings is not None and self.clusterer:
            if self.logger:
                self.logger.log_phase(3, "Image Clustering")
            
            cluster_labels = self._cluster_images(embeddings, candidates)
            
            if cluster_labels is not None:
                self.stats.num_clusters = len(set(cluster_labels))
        
        # Phase 4: LLM scoring
        if self.logger:
            self.logger.log_phase(4, "LLM Quality Scoring")
        
        print("\n[Phase 4] LLM scoring...")
        scored_results = self._llm_score_images(candidates)
        
        if not scored_results:
            print("No results from LLM scoring.")
            return
        
        self.stats.llm_scored = len(scored_results)
        
        # Save LLM responses for debugging
        if self.config.SAVE_LLM_RESPONSES and self.logger:
            self.scorer.save_all_responses(self.logger.intermediate_dir)
        
        if self.logger:
            self.logger.save_image_metadata("04_llm_scored_results", scored_results)
        
        # Phase 5: Final selection and ranking
        if self.logger:
            self.logger.log_phase(5, "Final Selection & Ranking")
        
        print("\n[Phase 5] Final selection and ranking...")
        final_results = self._finalize_selection(scored_results, cluster_labels)
        
        # Phase 6: Save results
        if self.logger:
            self.logger.log_phase(6, "Saving Results")
        
        print("\n[Phase 6] Saving results...")
        self._save_results_simple(final_results)
        
        # Generate summary
        total_time = time.time() - start_time
        self.stats.total_time = total_time
        
        if self.logger:
            self.logger.generate_summary_report(self.stats.to_dict())
            self.logger.log_completion(total_time)
        
        print("\n" + "=" * 60)
        print("✅ Enhanced album selection complete!")
        print(f"Output directory: {self.config.OUTPUT_DIR}")
        print(f"Total time: {int(total_time // 60)}m {int(total_time % 60)}s")
        print("=" * 60)
    
    def _llm_score_images(self, candidates: List[Dict]) -> List[Dict]:
        """Score images using LLM vision model.
        
        Args:
            candidates: List of image dicts
            
        Returns:
            List of images with LLM scores added
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
                
                # Add scores to candidates
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
    
    def _generate_embeddings(self, candidates: List[Dict]) -> Optional[np.ndarray]:
        """Generate embeddings for candidate images.
        
        Args:
            candidates: List of candidate image dicts
            
        Returns:
            Numpy array of embeddings
        """
        print(f"\n[Phase 2] Generating embeddings for {len(candidates)} images...")
        
        images_for_embedding = []
        valid_candidates = []
        
        print("Loading images for embedding...")
        for candidate in tqdm(candidates, desc="Loading", ncols=80):
            try:
                img_bytes = self.loader.get_image_bytes(candidate['path'])
                pil_image = self.processor.safe_open_image(img_bytes)
                
                if pil_image is None:
                    continue
                
                # Resize for efficiency
                pil_image.thumbnail((512, 512))
                images_for_embedding.append(pil_image)
                valid_candidates.append(candidate)
                
            except Exception:
                continue
        
        print(f"Successfully loaded {len(images_for_embedding)} images")
        
        if not images_for_embedding:
            print("⚠ No images loaded for embedding")
            return None
        
        # Generate embeddings
        embeddings = self.embedder.generate_embeddings_batch(
            images_for_embedding,
            batch_size=self.config.EMBEDDING_BATCH_SIZE,
            show_progress=True
        )
        
        # Update candidates list to only include valid ones
        candidates.clear()
        candidates.extend(valid_candidates)
        
        # Save embeddings
        if self.config.SAVE_EMBEDDINGS and self.logger:
            embeddings_path = f"{self.logger.intermediate_dir}/02_embeddings.npz"
            self.embedder.save_embeddings(embeddings, candidates, embeddings_path)
        
        # Clean up
        if self.config.CLEAR_MEMORY_AFTER_EMBEDDING:
            del images_for_embedding
            gc.collect()
        
        return embeddings
    
    def _cluster_images(self, embeddings: np.ndarray, 
                       candidates: List[Dict]) -> Optional[np.ndarray]:
        """Cluster images based on embeddings.
        
        Args:
            embeddings: Image embeddings
            candidates: List of candidate image dicts
            
        Returns:
            Cluster labels array
        """
        print(f"\n[Phase 3] Clustering {len(embeddings)} images...")
        
        # Perform clustering
        cluster_labels = self.clusterer.cluster(
            embeddings,
            n_clusters=self.config.NUM_CLUSTERS,
            reduce_dims=self.config.REDUCE_DIMENSIONS
        )
        
        # Add cluster info to candidates
        for i, candidate in enumerate(candidates):
            if i < len(cluster_labels):
                candidate['cluster_id'] = int(cluster_labels[i])
        
        # Save clustering results
        if self.config.SAVE_CLUSTERING_RESULTS and self.logger:
            clustering_path = f"{self.logger.intermediate_dir}/03_clustering_results.json"
            self.clusterer.save_clustering_results(clustering_path, embeddings, candidates)
        
        # Group images by cluster
        from collections import defaultdict
        cluster_groups = defaultdict(list)
        for candidate in candidates:
            cluster_id = candidate.get('cluster_id', -1)
            cluster_groups[cluster_id].append(candidate)
        
        # Save sample images from each cluster for debugging
        if self.config.SAVE_CLUSTER_IMAGES and self.logger:
            print("\nSaving sample images from each cluster...")
            self._save_cluster_samples(cluster_groups)
        
        # Use LLM to select best from each cluster (heavy LLM processing)
        if self.config.USE_LLM_FOR_CLUSTER_SELECTION and self.llm_cluster_selector:
            print("\n[Phase 3b] Using LLM to select best images from each cluster...")
            
            llm_selections = self.llm_cluster_selector.select_from_all_clusters(
                cluster_groups,
                self.loader,
                self.processor,
                top_k_per_cluster=self.config.IMAGES_PER_CLUSTER
            )
            
            # Save LLM selections
            if self.config.SAVE_CLUSTER_SELECTIONS and self.logger:
                self.llm_cluster_selector.save_selections(
                    llm_selections,
                    self.logger.intermediate_dir
                )
            
            # Mark LLM-selected images as cluster representatives
            selected_paths = set()
            for cluster_id, selection in llm_selections.items():
                for img_data in selection.get('selected_images', []):
                    if 'original_metadata' in img_data:
                        selected_paths.add(img_data['original_metadata']['path'])
                        # Add LLM scores to metadata
                        img_data['original_metadata']['llm_cluster_rank'] = img_data['rank']
                        img_data['original_metadata']['llm_cluster_reason'] = img_data['reason']
            
            for candidate in candidates:
                candidate['is_cluster_representative'] = candidate['path'] in selected_paths
            
            print(f"✓ Marked {len(selected_paths)} LLM-selected cluster representatives")
            
        else:
            # Fallback: Use basic representative selection
            print("\nSelecting best representatives from each cluster (basic method)...")
            representatives = self.clusterer.get_cluster_representatives(
                embeddings,
                candidates,
                top_k=self.config.IMAGES_PER_CLUSTER
            )
            
            # Mark representatives
            representative_paths = set()
            for cluster_reps in representatives.values():
                for rep in cluster_reps:
                    representative_paths.add(rep['path'])
            
            for candidate in candidates:
                candidate['is_cluster_representative'] = candidate['path'] in representative_paths
            
            print(f"Marked {len(representative_paths)} cluster representatives")
        
        return cluster_labels
    
    def _save_cluster_samples(self, cluster_groups: Dict[int, List[Dict]]):
        """Save sample images from each cluster for debugging.
        
        Args:
            cluster_groups: Dict mapping cluster_id -> list of images
        """
        import os
        import shutil
        
        cluster_samples_dir = os.path.join(self.logger.intermediate_dir, "cluster_samples")
        os.makedirs(cluster_samples_dir, exist_ok=True)
        
        # Save up to 5 samples from each cluster
        for cluster_id, images in cluster_groups.items():
            if cluster_id == -1:  # Skip noise cluster
                continue
            
            cluster_dir = os.path.join(cluster_samples_dir, f"cluster_{cluster_id:03d}")
            os.makedirs(cluster_dir, exist_ok=True)
            
            # Sort by local score and take top 5
            sorted_images = sorted(images, key=lambda x: x.get('local_score', 0), reverse=True)
            samples = sorted_images[:5]
            
            for i, img in enumerate(samples, 1):
                try:
                    src_path = img['path']
                    filename = f"{i:02d}_{os.path.basename(src_path)}"
                    dest_path = os.path.join(cluster_dir, filename)
                    shutil.copy2(src_path, dest_path)
                except Exception:
                    pass
        
        print(f"✓ Saved cluster samples to: {cluster_samples_dir}")
        print(f"  Clusters: {len([c for c in cluster_groups.keys() if c != -1])}")
        print(f"  Sample images per cluster: up to 5")
    
    def _finalize_selection(self, results: List[Dict], 
                           cluster_labels: Optional[np.ndarray] = None) -> Dict[str, List[Dict]]:
        """Calculate final scores and select top images with balanced distribution.
        
        Args:
            results: List of scored result dicts
            cluster_labels: Optional cluster labels for bonus scoring
            
        Returns:
            Dict with 'all' and 'selected' keys containing result lists
        """
        # Calculate final scores
        for result in results:
            llm_score = result.get('overall_score', 0)
            
            # Final score is primarily LLM score
            final_score = llm_score
            
            # Add cluster representative bonus if applicable
            if result.get('is_cluster_representative', False):
                final_score += 5  # Small bonus for cluster representatives
            
            result['final_score'] = final_score
        
        # Sort by final score
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Use curator for balanced selection across ceremony types
        print("\nApplying intelligent curation for ceremony balance...")
        selected = self.curator.curate_balanced_selection(results, self.config.NUM_SELECT)
        
        print(f"Selected {len(selected)} images from {len(results)} scored images")
        self.stats.final_selected = len(selected)
        
        # Show distribution report
        print(self.curator.get_distribution_report(selected))
        
        return {
            'all': results,
            'selected': selected
        }
    
    def _save_results(self, final_results: Dict[str, List[Dict]]):
        """Save results to disk (legacy method, kept for compatibility).
        
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
    
    def _save_results_enhanced(self, final_results: Dict[str, List[Dict]], 
                              categorized_by_person: Dict[str, List[Dict]]):
        """Save results with person-specific organization.
        
        Args:
            final_results: Dict with 'all' and 'selected' keys
            categorized_by_person: Dict mapping person category to images
        """
        import os
        import shutil
        
        # Save CSV report with all metadata
        if self.logger:
            csv_path = os.path.join(self.logger.logs_dir, "full_scoring_report.csv")
        else:
            csv_path = os.path.join(self.config.OUTPUT_DIR, "scoring_report.csv")
        
        csv_path = self.selector.save_csv_report(final_results['all'], csv_path)
        print(f"✓ Saved scoring report: {csv_path}")
        
        selected_images = final_results['selected']
        
        # Organize by person if person detection was used
        if categorized_by_person and self.logger:
            print(f"\nOrganizing {len(selected_images)} selected images by person...")
            
            # Create a mapping of image path to person categories
            image_to_persons = {}
            for img in selected_images:
                persons = img.get('persons', [])
                if persons:
                    image_to_persons[img['path']] = persons
            
            # Organize images by person with quotas
            organized_counts = {}
            
            for person_category, quota in self.config.PERSON_QUOTAS.items():
                if person_category not in categorized_by_person:
                    continue
                
                # Get images for this person category that are in selected set
                selected_paths = {img['path'] for img in selected_images}
                person_images = [
                    img for img in categorized_by_person[person_category]
                    if img['path'] in selected_paths
                ]
                
                # Sort by score and take top quota
                person_images.sort(key=lambda x: x.get('final_score', 0), reverse=True)
                top_images = person_images[:quota]
                
                # Copy to person directory
                if top_images and self.logger:
                    copied = self.logger.copy_images_to_person_directory(
                        top_images,
                        person_category,
                        max_images=quota
                    )
                    organized_counts[person_category] = copied
            
            # Print organization summary
            print("\n✓ Images organized by person:")
            for category, count in organized_counts.items():
                if count > 0:
                    print(f"  {category.replace('_', ' ').title()}: {count} images")
        
        else:
            # Fall back to simple copy
            print(f"\nCopying {len(selected_images)} selected images...")
            if self.logger:
                output_dir = self.logger.final_dir
            else:
                output_dir = self.config.OUTPUT_DIR
            
            os.makedirs(output_dir, exist_ok=True)
            
            copied = 0
            for i, img in enumerate(selected_images, 1):
                try:
                    src_path = img['path']
                    filename = f"{i:04d}_{os.path.basename(src_path)}"
                    dest_path = os.path.join(output_dir, filename)
                    shutil.copy2(src_path, dest_path)
                    copied += 1
                except Exception as e:
                    print(f"⚠ Error copying {img.get('name', 'unknown')}: {e}")
            
            print(f"✓ Copied {copied} images to {output_dir}")
    
    def _save_results_simple(self, final_results: Dict[str, List]):
        """Save results (simple, ranked by score).
        
        Args:
            final_results: Dict with 'all' and 'selected' keys
        """
        import os
        import shutil
        
        # Save CSV report
        if self.logger:
            csv_path = os.path.join(self.logger.logs_dir, "full_scoring_report.csv")
            output_dir = self.logger.final_dir
        else:
            csv_path = os.path.join(self.config.OUTPUT_DIR, "scoring_report.csv")
            output_dir = self.config.OUTPUT_DIR
        
        csv_path = self.selector.save_csv_report(final_results['all'], csv_path)
        print(f"✓ Saved scoring report: {csv_path}")
        
        # Copy selected images
        selected_images = final_results['selected']
        
        print(f"\nCopying {len(selected_images)} selected images...")
        os.makedirs(output_dir, exist_ok=True)
        
        copied = 0
        for i, img in enumerate(selected_images, 1):
            try:
                src_path = img['path']
                img_name = img.get('name', os.path.basename(src_path))
                
                # Create filename with rank prefix
                filename = f"{i:04d}_{img_name}"
                dest_path = os.path.join(output_dir, filename)
                
                shutil.copy2(src_path, dest_path)
                copied += 1
                
            except Exception as e:
                print(f"⚠ Error copying image {i}: {e}")
        
        print(f"✓ Copied {copied} images to {output_dir}")
        print(f"  Images ranked 1-{copied} by LLM score")
