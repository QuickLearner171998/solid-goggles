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
                    self.config.OPENAI_VISION_MODEL,
                    max_workers=3  # 3 concurrent LLM calls for optimal speed
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
        
        # Save cluster visualization (copy ALL images to cluster directories)
        if self.config.SAVE_CLUSTER_IMAGES and self.logger:
            self.clusterer.save_cluster_visualization(
                self.logger.intermediate_dir, 
                candidates,
                save_all=True
            )
        
        # Group images by cluster
        from collections import defaultdict
        cluster_groups = defaultdict(list)
        for candidate in candidates:
            cluster_id = candidate.get('cluster_id', -1)
            cluster_groups[cluster_id].append(candidate)
        
        # Use LLM to select best from each cluster (heavy LLM processing)
        if self.config.USE_LLM_FOR_CLUSTER_SELECTION and self.llm_cluster_selector:
            print("\n[Phase 3b] Using LLM to select best images from each cluster...")
            
            llm_selections = self.llm_cluster_selector.select_from_all_clusters(
                cluster_groups,
                self.loader,
                self.processor,
                top_k_per_cluster=None,  # No hard limit - let LLM decide
                output_dir=self.logger.intermediate_dir if self.logger else None  # Real-time saving
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
        """Calculate final scores and select with CLUSTER DIVERSITY optimization.
        
        Uses clustering intelligently:
        1. Prioritizes high-importance clusters
        2. Ensures balanced representation across clusters
        3. Avoids over-selecting from single clusters
        4. Maintains quality threshold
        
        Args:
            results: List of scored result dicts
            cluster_labels: Optional cluster labels for bonus scoring
            
        Returns:
            Dict with 'all' and 'selected' keys containing result lists
        """
        print("\n[Final Selection] Applying cluster-aware intelligent selection...")
        
        # Calculate final scores
        for result in results:
            llm_score = result.get('overall_score', 0)
            
            # Final score is primarily LLM score
            final_score = llm_score
            
            # Add cluster representative bonus if applicable
            if result.get('is_cluster_representative', False):
                final_score += 5  # Small bonus for cluster representatives
            
            result['final_score'] = final_score
        
        # Group by cluster for diversity-aware selection
        from collections import defaultdict
        cluster_images = defaultdict(list)
        for result in results:
            cluster_id = result.get('cluster_id', -1)
            cluster_images[cluster_id].append(result)
        
        # Sort images within each cluster by score
        for cluster_id in cluster_images:
            cluster_images[cluster_id].sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Determine cluster importance (if available from LLM)
        cluster_importance = {}
        for result in results:
            cluster_id = result.get('cluster_id', -1)
            importance = result.get('cluster_importance', 'medium')
            if cluster_id not in cluster_importance:
                cluster_importance[cluster_id] = importance
        
        # CLUSTER-AWARE SELECTION
        selected = self._select_with_cluster_diversity(
            cluster_images, 
            cluster_importance,
            target_count=self.config.NUM_SELECT
        )
        
        print(f"✓ Selected {len(selected)} images from {len(results)} scored images")
        print(f"  Represented clusters: {len(set(img.get('cluster_id') for img in selected))}/{len(cluster_images)}")
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
    
    def _select_with_cluster_diversity(self, cluster_images: Dict[int, List[Dict]],
                                      cluster_importance: Dict[int, str],
                                      target_count: int = 1000) -> List[Dict]:
        """SMART cluster-based selection - skip weak clusters, prioritize quality.
        
        Strategy:
        1. Evaluate cluster quality from LLM selections and scores
        2. SKIP clusters with no good images (low quality + low importance)
        3. Allocate slots dynamically based on cluster importance AND quality
        4. Global ranking: pick absolute best images across all clusters
        5. Ensure diversity while maintaining quality threshold
        
        Args:
            cluster_images: Dict of cluster_id -> list of images (sorted by score)
            cluster_importance: Dict of cluster_id -> importance level
            target_count: Target number of images
            
        Returns:
            List of best images with smart cluster awareness
        """
        print("\n[Smart Cluster Selection]")
        
        # STEP 1: Evaluate each cluster's quality
        cluster_quality = {}
        for cluster_id, images in cluster_images.items():
            if not images:
                continue
            
            # Calculate cluster quality metrics
            avg_score = sum(img.get('final_score', 0) for img in images) / len(images)
            max_score = max((img.get('final_score', 0) for img in images), default=0)
            num_images = len(images)
            importance = cluster_importance.get(cluster_id, 'medium')
            
            cluster_quality[cluster_id] = {
                'avg_score': avg_score,
                'max_score': max_score,
                'num_images': num_images,
                'importance': importance,
                'images': images
            }
        
        # STEP 2: Filter out weak clusters (skip if no good images)
        QUALITY_THRESHOLD = 60  # Minimum average score
        SKIP_LOW_IMPORTANCE_THRESHOLD = 70  # Low importance needs higher scores
        
        eligible_clusters = {}
        skipped_clusters = []
        
        for cluster_id, quality in cluster_quality.items():
            importance = quality['importance']
            avg_score = quality['avg_score']
            max_score = quality['max_score']
            
            # Skip criteria
            should_skip = False
            
            if importance == 'low' and avg_score < SKIP_LOW_IMPORTANCE_THRESHOLD:
                should_skip = True
                skip_reason = f"low importance + low avg score ({avg_score:.1f})"
            elif importance == 'none':
                should_skip = True
                skip_reason = "no importance"
            elif max_score < QUALITY_THRESHOLD:
                should_skip = True
                skip_reason = f"no images above quality threshold (max={max_score:.1f})"
            
            if should_skip:
                skipped_clusters.append((cluster_id, skip_reason))
            else:
                eligible_clusters[cluster_id] = quality
        
        print(f"  Eligible clusters: {len(eligible_clusters)} (skipped {len(skipped_clusters)} weak clusters)")
        if skipped_clusters[:5]:  # Show first 5 skipped
            for cid, reason in skipped_clusters[:5]:
                print(f"    Skipped cluster {cid}: {reason}")
        
        # STEP 3: Smart allocation based on importance AND quality
        importance_base_weights = {
            'high': 20,      # High importance = critical moments
            'medium': 10,    # Medium = good moments worth including  
            'low': 4,        # Low = only if high quality
        }
        
        cluster_allocations = {}
        for cluster_id, quality in eligible_clusters.items():
            importance = quality['importance']
            avg_score = quality['avg_score']
            num_images = quality['num_images']
            
            # Base weight from importance
            base_weight = importance_base_weights.get(importance, 10)
            
            # Quality multiplier (0.6 to 1.4 range)
            quality_multiplier = 0.6 + (avg_score / 100) * 0.8
            
            # Size factor - larger clusters get more (but capped)
            size_factor = min(1.0 + (num_images / 50) * 0.3, 1.5)
            
            # Final allocation
            allocation = int(base_weight * quality_multiplier * size_factor)
            allocation = max(2, min(allocation, num_images))  # At least 2, max available
            
            cluster_allocations[cluster_id] = allocation
        
        # STEP 4: Normalize allocations to target count
        total_allocated = sum(cluster_allocations.values())
        
        if total_allocated > target_count:
            scale_factor = target_count / total_allocated
            for cluster_id in cluster_allocations:
                cluster_allocations[cluster_id] = max(1, int(cluster_allocations[cluster_id] * scale_factor))
        
        print(f"  Allocation strategy:")
        print(f"    Target images: {target_count}")
        print(f"    Total allocated: {sum(cluster_allocations.values())}")
        
        # Show allocation breakdown
        by_importance = {'high': 0, 'medium': 0, 'low': 0}
        for cluster_id, allocation in cluster_allocations.items():
            importance = eligible_clusters[cluster_id]['importance']
            by_importance[importance] += allocation
        print(f"    High-importance: {by_importance['high']} images")
        print(f"    Medium-importance: {by_importance['medium']} images")
        print(f"    Low-importance: {by_importance['low']} images")
        
        # STEP 5: Select top images from each eligible cluster
        selected = []
        for cluster_id, allocation in sorted(cluster_allocations.items(), 
                                            key=lambda x: eligible_clusters[x[0]]['avg_score'], 
                                            reverse=True):
            images = eligible_clusters[cluster_id]['images'][:allocation]
            selected.extend(images)
        
        # STEP 6: Global ranking pass - ensure we have the ABSOLUTE BEST images
        # Sort all selected by score and take top N
        selected.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # If we have more than target, trim to exactly target
        if len(selected) > target_count:
            selected = selected[:target_count]
            print(f"  Trimmed to top {target_count} images (highest scores across all clusters)")
        
        # STEP 7: Fill remaining slots if needed (from unused high-quality images)
        elif len(selected) < target_count:
            remaining_needed = target_count - len(selected)
            print(f"  Need {remaining_needed} more images to reach target")
            
            # Collect all images not yet selected (only from eligible clusters)
            selected_paths = set(img['path'] for img in selected)
            remaining_images = []
            
            for cluster_id, quality in eligible_clusters.items():
                for img in quality['images']:
                    if img['path'] not in selected_paths and img.get('final_score', 0) >= QUALITY_THRESHOLD:
                        remaining_images.append(img)
            
            # Sort by score and add best remaining
            remaining_images.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            additional = remaining_images[:remaining_needed]
            selected.extend(additional)
            
            print(f"  Added {len(additional)} additional high-quality images from unused pool")
        
        # Final global ranking by score
        selected.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        print(f"\n✓ Final selection: {len(selected)} images")
        print(f"  Score range: {selected[-1].get('final_score', 0):.1f} - {selected[0].get('final_score', 0):.1f}")
        print(f"  Represented clusters: {len(set(img.get('cluster_id', -1) for img in selected))}")
        
        return selected
