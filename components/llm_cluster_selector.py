"""LLM-based cluster selection component with parallel processing."""

import json
import time
import base64
import os
import shutil
from typing import List, Dict
from openai import OpenAI
from config import Config
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class LLMClusterSelector:
    """Uses LLM to intelligently select best images from each cluster."""
    
    CLUSTER_SELECTION_PROMPT = """You are an expert wedding photographer with 20+ years of experience selecting images for award-winning albums.

You are reviewing a CLUSTER of similar wedding photos (similar moment/scene/composition). Your task is to select THE BEST images from this cluster that deserve to be in the final album, following professional album curation principles.

**CRITICAL: NO HARD LIMITS** - Select as many or as few images as truly deserve to be in the album. Some clusters may have 15 great shots, others may have only 2-3 worthy images.

**PROFESSIONAL SELECTION CRITERIA:**

**1. Technical Excellence (35%)**:
   - **CRITICAL**: Tack-sharp focus on subject's eyes/face
   - Proper exposure with detail in highlights and shadows
   - Natural, accurate skin tones and colors
   - Clean image with minimal noise
   - No motion blur on important subjects

**2. Composition & Artistry (35%)**:
   - Strong visual impact and "wow factor"
   - Pleasing framing and balance
   - Intentional use of depth of field
   - Leading lines or visual flow
   - No distracting elements in frame
   - Creative angles or perspectives

**3. Emotional Impact & Storytelling (30%)**:
   - **MOST IMPORTANT**: Genuine emotions and authentic moments
   - Connection between subjects
   - Peak moment capture (perfect timing)
   - Tells a story or conveys feeling
   - Would make viewer emotional

**SELECTION STRATEGY (Professional Photographer Approach):**

1. **Eliminate First**:
   - ✗ Soft focus or blur on faces
   - ✗ Closed eyes during important moments
   - ✗ Awkward expressions (mid-blink, talking, grimacing)
   - ✗ Poor lighting (too dark/bright, washed out)
   - ✗ Distracting backgrounds
   - ✗ Technically flawed (wrong focus, motion blur)

2. **Compare Remaining**:
   - Which has SHARPEST focus?
   - Which has BEST expression?
   - Which has STRONGEST composition?
   - Which tells the BEST story?
   - Which captures peak emotion/moment?

3. **Smart Redundancy Management**:
   - If multiple shots show DIFFERENT emotions/angles of the SAME moment: KEEP multiple shots
   - If shots are true duplicates (same pose/expression): Choose ONLY the best one
   - Prefer diverse moments and angles within cluster
   - **KEY**: If a cluster captures an important ceremony moment (e.g., vows, ring exchange), keep MORE images to tell the complete story

4. **Quality AND Quantity Balance**:
   - **IMPORTANT CLUSTERS** (ceremony moments, emotional peaks): Select 10-20 images if they're all album-worthy
   - **STANDARD CLUSTERS** (portraits, group shots): Select 3-8 of the best
   - **WEAK CLUSTERS** (less important, repetitive): Select 1-3 only if excellent
   - Every selected image must be album-worthy
   - **DON'T lose good images** - better to include an extra good shot than exclude it

**RESPONSE FORMAT (STRICT JSON):**
Return a JSON object:
{
  "selected_images": [
    {
      "filename": "exact_filename",
      "rank": 1-N,
      "reason": "brief reason for selection",
      "technical_score": 0-100,
      "composition_score": 0-100,
      "moment_score": 0-100
    }
  ],
  "cluster_summary": "brief description of what this cluster contains",
  "cluster_importance": "low/medium/high - how critical is this cluster for album storytelling"
}

Order by rank (1 = best). Select AS MANY images as truly deserve to be in the album - NO artificial limits!"""
    
    def __init__(self, api_key: str = None, model: str = None, max_workers: int = 3):
        """Initialize LLM cluster selector with parallel processing support.
        
        Args:
            api_key: OpenAI API key
            model: Model name
            max_workers: Maximum concurrent LLM API calls (default: 3)
        """
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model = model or Config.OPENAI_VISION_MODEL
        self.max_workers = max_workers
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=self.api_key)
        self.rate_limit_lock = threading.Lock()
        self.rate_limit_delay = 0  # Dynamic delay based on rate limits
        
        # For real-time saving
        self.output_dir = None
        self.save_lock = threading.Lock()  # Thread-safe file operations
    
    @staticmethod
    def _encode_image_base64(image_bytes: bytes) -> str:
        """Encode image bytes to base64."""
        return base64.b64encode(image_bytes).decode("utf-8")
    
    @staticmethod
    def _resize_image_for_llm(pil_image: Image.Image, 
                             max_size: int = None) -> bytes:
        """Resize image for optimal LLM processing.
        
        Args:
            pil_image: PIL Image
            max_size: Maximum dimension (uses Config if None)
            
        Returns:
            Resized image as JPEG bytes
        """
        max_size = max_size or Config.LLM_IMAGE_MAX_SIZE
        
        # Resize if needed
        if max(pil_image.size) > max_size:
            pil_image = pil_image.copy()
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if pil_image.mode not in ('RGB', 'L'):
            pil_image = pil_image.convert('RGB')
        
        # Save to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=Config.LLM_IMAGE_QUALITY)
        return buffer.getvalue()
    
    def select_from_cluster(self, 
                           cluster_images: List[Dict],
                           image_loader,
                           image_processor,
                           top_k: int = None) -> Dict:
        """Select best images from a cluster using INTELLIGENT MULTI-PASS LLM evaluation.
        
        Strategy:
        - For clusters ≤20 images: Single LLM call evaluates all
        - For clusters >20 images: Multiple passes, each evaluating 20 images
        - All images are evaluated, none are lost
        - Results aggregated and ranked across all passes
        
        Args:
            cluster_images: List of image metadata dicts from this cluster
            image_loader: ImageLoader instance
            image_processor: ImageProcessor instance
            top_k: Soft suggestion (None = let LLM decide based on quality)
            
        Returns:
            Dict with selection results and cluster summary
        """
        if not cluster_images:
            return {
                'selected_images': [],
                'cluster_summary': 'Empty cluster',
                'cluster_importance': 'none'
            }
        
        # INTELLIGENT BATCHING: Process ALL images in multiple passes
        batch_size = 20  # Optimal for LLM vision models
        num_images = len(cluster_images)
        
        if num_images <= batch_size:
            # Small cluster: single pass evaluates all
            return self._evaluate_batch(cluster_images, image_loader, image_processor, top_k)
        else:
            # Large cluster: multi-pass evaluation of ALL images
            print(f"    Cluster has {num_images} images - using multi-pass evaluation")
            return self._evaluate_large_cluster(cluster_images, image_loader, image_processor, 
                                               batch_size, top_k)
    
    def _evaluate_batch(self, batch_images: List[Dict], 
                       image_loader, image_processor, top_k: int = None) -> Dict:
        """Evaluate a single batch of images with LLM.
        
        Args:
            batch_images: List of images to evaluate (≤20 recommended)
            image_loader: ImageLoader instance
            image_processor: ImageProcessor instance
            top_k: Optional limit
            
        Returns:
            Dict with evaluation results
        """
        # Prepare images for LLM
        llm_content = []
        selection_instruction = f"You have {len(batch_images)} images from this cluster. Select AS MANY as truly deserve to be in the album (could be 2, could be 15, could be all of them). Quality over arbitrary limits!"
        llm_content.append({
            "type": "text",
            "text": selection_instruction
        })
        
        valid_images = []
        for img in batch_images:
            try:
                img_bytes = image_loader.get_image_bytes(img['path'])
                pil_image = image_processor.safe_open_image(img_bytes)
                
                if pil_image is None:
                    continue
                
                # Resize for LLM
                resized_bytes = self._resize_image_for_llm(pil_image)
                data_uri = f"data:image/jpeg;base64,{self._encode_image_base64(resized_bytes)}"
                
                llm_content.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri, "detail": "high"}
                })
                llm_content.append({
                    "type": "text",
                    "text": f'filename="{img["name"]}"'
                })
                
                valid_images.append(img)
                
                del pil_image, img_bytes, resized_bytes
                
            except Exception as e:
                print(f"  ⚠ Error loading image for LLM: {e}")
                continue
        
        if not valid_images:
            return {
                'selected_images': [],
                'cluster_summary': 'No valid images in cluster'
            }
        
        # Call LLM with rate limit handling
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Apply dynamic rate limit delay
                with self.rate_limit_lock:
                    if self.rate_limit_delay > 0:
                        time.sleep(self.rate_limit_delay)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.CLUSTER_SELECTION_PROMPT},
                        {"role": "user", "content": llm_content},
                    ],
                    temperature=0.1,
                    max_tokens=Config.LLM_MAX_TOKENS,
                )
                
                # Success - reduce rate limit delay
                with self.rate_limit_lock:
                    self.rate_limit_delay = max(0, self.rate_limit_delay - 0.5)
                
                result_text = response.choices[0].message.content.strip()
                
                # Clean up markdown code blocks
                if result_text.startswith("```"):
                    result_text = result_text.strip("`")
                    if result_text.startswith("json"):
                        result_text = result_text[4:].strip()
                
                result = json.loads(result_text)
                
                # Add original metadata to selected images
                for selected in result.get('selected_images', []):
                    filename = selected['filename']
                    # Find original image metadata
                    for img in valid_images:
                        if img['name'] == filename:
                            selected['original_metadata'] = img
                            break
                
                # Save LLM result immediately (real-time)
                if self.output_dir:
                    self._save_llm_result_realtime(result, valid_images)
                
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Handle rate limits
                if 'rate' in error_str or 'limit' in error_str or '429' in error_str:
                    retry_delay = min(2 ** attempt, 32)  # Exponential backoff, max 32s
                    
                    with self.rate_limit_lock:
                        self.rate_limit_delay = max(self.rate_limit_delay, retry_delay / 2)
                    
                    print(f"  ⚠ Rate limit hit, waiting {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                
                # Handle other errors
                if attempt == max_retries - 1:
                    print(f"  ✗ LLM cluster selection failed: {e}")
                    # Return top images by local score as fallback
                    sorted_imgs = sorted(valid_images, 
                                       key=lambda x: x.get('local_score', 0), 
                                       reverse=True)
                    return {
                        'selected_images': [
                            {
                                'filename': img['name'],
                                'rank': i+1,
                                'reason': 'Fallback: selected by local score',
                                'technical_score': 0,
                                'composition_score': 0,
                                'moment_score': 0,
                                'original_metadata': img
                            }
                            for i, img in enumerate(sorted_imgs[:top_k] if top_k else sorted_imgs[:5])
                        ],
                        'cluster_summary': 'LLM selection failed, using local scores'
                    }
                
                time.sleep(1.0 * (2 ** attempt))
        
        return {
            'selected_images': [],
            'cluster_summary': 'Selection failed'
        }
    
    def _evaluate_large_cluster(self, cluster_images: List[Dict],
                               image_loader, image_processor,
                               batch_size: int = 20, top_k: int = None) -> Dict:
        """Evaluate large cluster using multi-pass approach - NO IMAGES LOST.
        
        Strategy:
        1. Split cluster into batches of `batch_size` images
        2. Evaluate each batch with LLM
        3. Collect ALL selected images from all batches
        4. Rank and aggregate results
        
        Args:
            cluster_images: All images in the cluster
            image_loader: ImageLoader instance
            image_processor: ImageProcessor instance
            batch_size: Images per LLM call
            top_k: Optional soft limit
            
        Returns:
            Aggregated results from all batches
        """
        all_selected = []
        cluster_summaries = []
        importance_levels = []
        
        # Process cluster in batches - evaluate ALL images
        num_batches = (len(cluster_images) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(cluster_images))
            batch = cluster_images[start_idx:end_idx]
            
            print(f"      Pass {batch_idx+1}/{num_batches}: Evaluating images {start_idx+1}-{end_idx}")
            
            # Evaluate this batch
            batch_result = self._evaluate_batch(batch, image_loader, image_processor, top_k)
            
            # Collect results
            if batch_result.get('selected_images'):
                all_selected.extend(batch_result['selected_images'])
            
            if batch_result.get('cluster_summary'):
                cluster_summaries.append(batch_result['cluster_summary'])
            
            if batch_result.get('cluster_importance'):
                importance_levels.append(batch_result['cluster_importance'])
        
        # Aggregate results
        if not all_selected:
            return {
                'selected_images': [],
                'cluster_summary': 'No images selected from any batch',
                'cluster_importance': 'low'
            }
        
        # Sort by overall score (from LLM)
        all_selected.sort(key=lambda x: (
            x.get('technical_score', 0) + 
            x.get('composition_score', 0) + 
            x.get('moment_score', 0)
        ) / 3, reverse=True)
        
        # Re-rank
        for idx, img in enumerate(all_selected, 1):
            img['rank'] = idx
        
        # Determine overall importance
        importance_map = {'high': 3, 'medium': 2, 'low': 1, 'none': 0}
        avg_importance = sum(importance_map.get(imp, 1) for imp in importance_levels) / max(1, len(importance_levels))
        
        if avg_importance >= 2.5:
            overall_importance = 'high'
        elif avg_importance >= 1.5:
            overall_importance = 'medium'
        else:
            overall_importance = 'low'
        
        # Combine summaries
        combined_summary = f"Multi-pass evaluation of {len(cluster_images)} images. " + \
                          " | ".join(set(cluster_summaries))
        
        print(f"      ✓ Selected {len(all_selected)} images from {len(cluster_images)} total (multi-pass)")
        
        result = {
            'selected_images': all_selected,
            'cluster_summary': combined_summary,
            'cluster_importance': overall_importance
        }
        
        # Save selected images from this cluster (real-time)
        if self.output_dir and all_selected:
            self._save_selected_images_realtime(all_selected, cluster_images[0].get('cluster_id', 'unknown'))
        
        return result
    
    def select_from_all_clusters(self,
                                cluster_groups: Dict[int, List[Dict]],
                                image_loader,
                                image_processor,
                                top_k_per_cluster: int = None,
                                use_parallel: bool = True,
                                output_dir: str = None) -> Dict[int, Dict]:
        """Select best images from all clusters using PARALLEL LLM calls with rate limit handling.
        
        Benefits of parallelism:
        - 3x faster processing with 3 concurrent workers
        - Intelligent rate limit handling with exponential backoff
        - Progress tracking across all parallel operations
        - Automatic retry with adaptive delays
        
        Args:
            cluster_groups: Dict mapping cluster_id -> list of images
            image_loader: ImageLoader instance
            image_processor: ImageProcessor instance
            top_k_per_cluster: Soft suggestion (None = let LLM decide based on quality)
            use_parallel: Use parallel processing (default: True)
            output_dir: Directory for real-time saving of results and images
            
        Returns:
            Dict mapping cluster_id -> selection results
        """
        print(f"\n[LLM Cluster Selection] Processing {len(cluster_groups)} clusters...")
        print(f"  Strategy: Parallel LLM evaluation with {self.max_workers} workers")
        print("  Rate limit handling: Enabled with exponential backoff")
        print("  No hard limits - prioritizing quality over arbitrary quotas")
        
        # Setup real-time saving
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(os.path.join(output_dir, 'llm_selections'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'selected_images'), exist_ok=True)
            print(f"  Real-time saving: Enabled → {output_dir}")
        
        if not use_parallel or len(cluster_groups) <= 1:
            # Sequential processing for small workloads
            return self._select_sequential(cluster_groups, image_loader, image_processor, top_k_per_cluster)
        else:
            # Parallel processing with rate limit handling
            return self._select_parallel(cluster_groups, image_loader, image_processor, top_k_per_cluster)
    
    def _select_sequential(self, cluster_groups: Dict[int, List[Dict]],
                          image_loader, image_processor,
                          top_k_per_cluster: int = None) -> Dict[int, Dict]:
        """Sequential processing (fallback)."""
        from tqdm import tqdm
        
        selections = {}
        
        for cluster_id, images in tqdm(cluster_groups.items(), 
                                       desc="Sequential LLM",
                                       ncols=80):
            if cluster_id == -1:
                continue
            
            selection = self.select_from_cluster(
                images, image_loader, image_processor, top_k=top_k_per_cluster
            )
            selections[cluster_id] = selection
        
        self._print_summary(selections, cluster_groups)
        return selections
    
    def _select_parallel(self, cluster_groups: Dict[int, List[Dict]],
                        image_loader, image_processor,
                        top_k_per_cluster: int = None) -> Dict[int, Dict]:
        """Parallel processing with rate limit handling."""
        from tqdm import tqdm
        
        selections = {}
        
        # Filter out noise cluster and prepare tasks
        valid_clusters = [(cid, images) for cid, images in cluster_groups.items() if cid != -1]
        
        # Process clusters in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_cluster = {
                executor.submit(
                    self.select_from_cluster,
                    images,
                    image_loader,
                    image_processor,
                    top_k_per_cluster
                ): cluster_id
                for cluster_id, images in valid_clusters
            }
            
            # Process results as they complete
            with tqdm(total=len(future_to_cluster), desc="Parallel LLM", ncols=80) as pbar:
                for future in as_completed(future_to_cluster):
                    cluster_id = future_to_cluster[future]
                    try:
                        selection = future.result()
                        selections[cluster_id] = selection
                    except Exception as e:
                        print(f"\n  ✗ Cluster {cluster_id} failed: {e}")
                        selections[cluster_id] = {
                            'selected_images': [],
                            'cluster_summary': f'Processing failed: {e}',
                            'cluster_importance': 'low'
                        }
                    pbar.update(1)
        
        self._print_summary(selections, cluster_groups)
        return selections
    
    def _print_summary(self, selections: Dict[int, Dict], 
                      cluster_groups: Dict[int, List[Dict]]):
        """Print processing summary."""
        total_selected = sum(len(s['selected_images']) for s in selections.values())
        total_available = sum(len(images) for cid, images in cluster_groups.items() if cid != -1)
        high_importance = sum(1 for s in selections.values() if s.get('cluster_importance') == 'high')
        
        print(f"\n✓ LLM cluster selection complete:")
        print(f"  Clusters processed: {len(selections)}")
        print(f"  Total images selected: {total_selected} out of {total_available} ({100*total_selected/max(1,total_available):.1f}%)")
        print(f"  High-importance clusters: {high_importance}")
        print(f"  Avg images per cluster: {total_selected // max(1, len(selections))}")
        
        if self.output_dir:
            print(f"  Real-time saves: {self.output_dir}/llm_selections/ and /selected_images/")
    
    def _save_llm_result_realtime(self, result: Dict, valid_images: List[Dict]):
        """Save LLM result immediately after processing (real-time).
        
        Args:
            result: LLM selection result
            valid_images: Original image metadata
        """
        if not self.output_dir:
            return
        
        try:
            with self.save_lock:
                # Create unique filename based on first image in batch
                first_img = valid_images[0]['name'] if valid_images else 'unknown'
                timestamp = time.time()
                filename = f"llm_result_{timestamp}_{first_img[:20]}.json"
                filepath = os.path.join(self.output_dir, 'llm_selections', filename)
                
                # Save result
                with open(filepath, 'w') as f:
                    json.dump({
                        'timestamp': timestamp,
                        'num_images_evaluated': len(valid_images),
                        'num_selected': len(result.get('selected_images', [])),
                        'cluster_summary': result.get('cluster_summary', ''),
                        'cluster_importance': result.get('cluster_importance', ''),
                        'selected_images': result.get('selected_images', []),
                        'evaluated_images': [img['name'] for img in valid_images]
                    }, f, indent=2)
                    
        except Exception as e:
            print(f"  ⚠ Error saving LLM result: {e}")
    
    def _save_selected_images_realtime(self, selected_images: List[Dict], cluster_id: int):
        """Copy selected images to output directory immediately (real-time).
        
        Args:
            selected_images: List of selected image dicts
            cluster_id: Cluster identifier
        """
        if not self.output_dir:
            return
        
        try:
            with self.save_lock:
                # Create cluster subdirectory
                cluster_dir = os.path.join(self.output_dir, 'selected_images', f'cluster_{cluster_id:03d}')
                os.makedirs(cluster_dir, exist_ok=True)
                
                # Copy images
                for img_data in selected_images:
                    if 'original_metadata' in img_data:
                        metadata = img_data['original_metadata']
                        src_path = metadata.get('path')
                        
                        if src_path and os.path.exists(src_path):
                            rank = img_data.get('rank', 0)
                            filename = f"rank{rank:02d}_{metadata['name']}"
                            dest_path = os.path.join(cluster_dir, filename)
                            
                            shutil.copy2(src_path, dest_path)
                            
        except Exception as e:
            print(f"  ⚠ Error saving selected images: {e}")
    
    def save_selections(self, 
                       selections: Dict[int, Dict],
                       output_dir: str):
        """Save cluster selections to disk (final summary).
        
        Args:
            selections: Selection results from select_from_all_clusters
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        json_path = os.path.join(output_dir, 'llm_cluster_selections.json')
        
        # Prepare serializable data
        serializable_selections = {}
        for cluster_id, result in selections.items():
            serializable_selections[str(cluster_id)] = {
                'cluster_summary': result.get('cluster_summary', ''),
                'cluster_importance': result.get('cluster_importance', ''),
                'num_selected': len(result.get('selected_images', [])),
                'selected_images': [
                    {
                        'filename': img.get('filename', ''),
                        'rank': img.get('rank', 0),
                        'reason': img.get('reason', ''),
                        'technical_score': img.get('technical_score', 0),
                        'composition_score': img.get('composition_score', 0),
                        'moment_score': img.get('moment_score', 0)
                    }
                    for img in result['selected_images']
                ]
            }
        
        with open(json_path, 'w') as f:
            json.dump(serializable_selections, f, indent=2)
        
        print(f"✓ Saved LLM cluster selections: {json_path}")
        
        # Save as CSV
        csv_path = os.path.join(output_dir, 'llm_cluster_selections.csv')
        
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'cluster_id', 'rank', 'filename', 'path',
                'technical_score', 'composition_score', 'moment_score',
                'reason', 'cluster_summary'
            ])
            
            csv_rows = []
            for cluster_id, result in selections.items():
                for img in result.get('selected_images', []):
                    metadata = img.get('original_metadata', {})
                    csv_rows.append([
                        cluster_id,
                        img.get('rank', ''),
                        img.get('filename', ''),
                        metadata.get('path', ''),
                        img.get('technical_score', ''),
                        img.get('composition_score', ''),
                        img.get('moment_score', ''),
                        img.get('reason', ''),
                        result.get('cluster_summary', '')
                    ])
            
            writer.writerows(csv_rows)
        
        print(f"\n✓ Saved {len(csv_rows)} LLM selections to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")
