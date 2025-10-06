"""LLM-based cluster selection component."""

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
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize LLM cluster selector.
        
        Args:
            api_key: OpenAI API key
            model: Model name
        """
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model = model or Config.OPENAI_VISION_MODEL
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=self.api_key)
    
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
        """Select best images from a cluster using LLM (NO HARD LIMITS).
        
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
        
        # Process ALL images from cluster (or up to 25 for very large clusters)
        max_batch = min(len(cluster_images), 25)
        batch_images = cluster_images[:max_batch]
        
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
        
        # Call LLM with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.CLUSTER_SELECTION_PROMPT},
                        {"role": "user", "content": llm_content},
                    ],
                    temperature=0.1,
                    max_tokens=Config.LLM_MAX_TOKENS,
                )
                
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
                
                return result
                
            except Exception as e:
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
                                'rank': i + 1,
                                'reason': 'Fallback: top local score',
                                'technical_score': 0,
                                'composition_score': 0,
                                'moment_score': 0,
                                'original_metadata': img
                            }
                            for i, img in enumerate(sorted_imgs[:top_k])
                        ],
                        'cluster_summary': 'LLM selection failed, using local scores'
                    }
                
                time.sleep(1.0 * (2 ** attempt))
        
        return {
            'selected_images': [],
            'cluster_summary': 'Selection failed'
        }
    
    def select_from_all_clusters(self,
                                cluster_groups: Dict[int, List[Dict]],
                                image_loader,
                                image_processor,
                                top_k_per_cluster: int = None) -> Dict[int, Dict]:
        """Select best images from all clusters using LLM (SMART, NO HARD LIMITS).
        
        Args:
            cluster_groups: Dict mapping cluster_id -> list of images
            image_loader: ImageLoader instance
            image_processor: ImageProcessor instance
            top_k_per_cluster: Soft suggestion (None = let LLM decide based on quality)
            
        Returns:
            Dict mapping cluster_id -> selection results
        """
        from tqdm import tqdm
        
        print(f"\n[LLM Cluster Selection] Processing {len(cluster_groups)} clusters...")
        print("  Strategy: LLM will intelligently select best images from each cluster")
        print("  No hard limits - prioritizing quality over arbitrary quotas")
        
        selections = {}
        
        for cluster_id, images in tqdm(cluster_groups.items(), 
                                       desc="LLM smart selection",
                                       ncols=80):
            if cluster_id == -1:  # Skip noise cluster
                continue
            
            selection = self.select_from_cluster(
                images,
                image_loader,
                image_processor,
                top_k=top_k_per_cluster
            )
            
            selections[cluster_id] = selection
        
        # Print detailed summary
        total_selected = sum(len(s['selected_images']) for s in selections.values())
        total_available = sum(len(images) for cid, images in cluster_groups.items() if cid != -1)
        high_importance = sum(1 for s in selections.values() if s.get('cluster_importance') == 'high')
        
        print(f"\n✓ LLM smart cluster selection complete:")
        print(f"  Clusters processed: {len(selections)}")
        print(f"  Total images selected: {total_selected} out of {total_available} ({100*total_selected/max(1,total_available):.1f}%)")
        print(f"  High-importance clusters: {high_importance}")
        print(f"  Avg images per cluster: {total_selected // max(1, len(selections))}")
        
        return selections
    
    def save_selections(self, 
                       selections: Dict[int, Dict],
                       output_dir: str):
        """Save cluster selections to disk.
        
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
                'cluster_summary': result['cluster_summary'],
                'selected_count': len(result['selected_images']),
                'selected_images': [
                    {
                        'filename': img['filename'],
                        'rank': img['rank'],
                        'reason': img['reason'],
                        'technical_score': img['technical_score'],
                        'composition_score': img['composition_score'],
                        'moment_score': img['moment_score'],
                        'path': img['original_metadata']['path']
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
            
            for cluster_id, result in selections.items():
                for img in result['selected_images']:
                    writer.writerow([
                        cluster_id,
                        img['rank'],
                        img['filename'],
                        img['original_metadata']['path'],
                        img['technical_score'],
                        img['composition_score'],
                        img['moment_score'],
                        img['reason'],
                        result['cluster_summary']
                    ])
        
        print(f"✓ Saved LLM cluster selections CSV: {csv_path}")

