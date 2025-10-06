"""Person detection and categorization component using LLM."""

import json
import time
import base64
from typing import List, Dict, Set
from openai import OpenAI
from config import Config


class PersonDetector:
    """Detects and categorizes people in wedding photos using LLM."""
    
    PERSON_CATEGORIES = {
        'bride': 'Bride',
        'groom': 'Groom',
        'couple': 'Bride and Groom Together',
        'bride_mom': "Bride's Mother",
        'bride_dad': "Bride's Father",
        'bride_parents': "Bride's Parents",
        'bride_brother': "Bride's Brother/Sibling",
        'groom_mom': "Groom's Mother",
        'groom_dad': "Groom's Father",
        'groom_parents': "Groom's Parents",
        'family': 'Family Group',
        'guests': 'Guests',
        'children': 'Children',
        'ceremony': 'Ceremony/Ritual Focus',
        'unknown': 'Unknown/Multiple People'
    }
    
    DETECTION_PROMPT = """You are an expert at identifying people in Indian wedding photographs.

Analyze each image and identify which of the following people/groups are the PRIMARY subjects:

PERSON CATEGORIES:
- "bride": The bride (woman in wedding attire, bridal lehenga, heavy jewelry, mehendi)
- "groom": The groom (man in wedding attire, sherwani, sehra, turban)
- "couple": Both bride and groom together as the main focus
- "bride_mom": Bride's mother (older woman close to bride, helping her, emotional during vidaai)
- "bride_dad": Bride's father (older man close to bride, performing kanyadaan)
- "bride_parents": Both bride's parents together
- "bride_brother": Bride's brother or siblings (younger relatives, protective of bride)
- "groom_mom": Groom's mother (older woman close to groom, welcoming bride)
- "groom_dad": Groom's father (older man close to groom)
- "groom_parents": Both groom's parents together
- "family": Extended family group photo
- "guests": Guests or attendees (not immediate family)
- "children": Children as main subjects
- "ceremony": Ritual or ceremony focus (mandap, fire, decorations) without clear person focus
- "unknown": Multiple people or unclear who are main subjects

INSTRUCTIONS:
1. Identify the PRIMARY subjects in each photo (people who are the main focus)
2. You can assign MULTIPLE categories if applicable (e.g., both "bride" and "groom" for couple shots)
3. Look for contextual clues: clothing, jewelry, positioning, interactions, emotions
4. For family members, use context like who's helping/interacting with bride/groom
5. Be conservative - use "unknown" if you're not confident

RESPONSE FORMAT (STRICT JSON):
Return a JSON array with one object per image in the same order received:
{
  "filename": "exact_filename_provided",
  "persons": ["category1", "category2"],
  "confidence": 0.0-1.0,
  "description": "brief description of who is in the photo"
}

Example:
{"filename": "IMG_001.jpg", "persons": ["bride", "bride_mom"], "confidence": 0.9, "description": "Bride getting ready with mother helping with jewelry"}"""
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize person detector.
        
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
    def _create_user_prompt(batch_items: List[Dict]) -> str:
        """Create user prompt for batch person detection."""
        lines = ["Identify the people in the following images. Respond with a single JSON array of objects in the same order."]
        for idx, item in enumerate(batch_items, 1):
            lines.append(f'{idx}. filename="{item["name"]}"')
        return "\n".join(lines)
    
    def detect_persons_batch(self, batch_items: List[Dict[str, any]], 
                            max_retries: int = 3) -> List[Dict]:
        """Detect persons in a batch of images.
        
        Args:
            batch_items: List of dicts with keys: name, thumb_bytes
            max_retries: Maximum retry attempts
            
        Returns:
            List of detection results
        """
        content = []
        content.append({"type": "text", "text": self._create_user_prompt(batch_items)})
        
        for item in batch_items:
            data_uri = f"data:image/jpeg;base64,{self._encode_image_base64(item['thumb_bytes'])}"
            content.append({"type": "image_url", "image_url": {"url": data_uri}})
        
        # Retry with exponential backoff
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.DETECTION_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.1,
                    max_tokens=1000,
                )
                
                result_text = response.choices[0].message.content.strip()
                
                # Clean up markdown code blocks
                if result_text.startswith("```"):
                    result_text = result_text.strip("`")
                    if result_text.startswith("json"):
                        result_text = result_text[4:].strip()
                
                data = json.loads(result_text)
                
                if not isinstance(data, list):
                    raise ValueError("Model did not return a JSON array.")
                
                return data
                
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt failed, return error placeholders
                    return [{
                        "persons": ["unknown"],
                        "confidence": 0.0,
                        "description": f"Detection failed: {str(e)}"
                    } for _ in batch_items]
                
                # Wait before retry
                time.sleep(1.0 * (2 ** attempt))
        
        return []
    
    def categorize_images(self, images: List[Dict],
                         image_loader,
                         image_processor,
                         batch_size: int = 5) -> Dict[str, List[Dict]]:
        """Categorize images by person/group.
        
        Args:
            images: List of image metadata dicts
            image_loader: ImageLoader instance
            image_processor: ImageProcessor instance
            batch_size: Number of images per LLM call
            
        Returns:
            Dict mapping category to list of images
        """
        from tqdm import tqdm
        
        print(f"\n[Person Detection] Categorizing {len(images)} images...")
        
        categorized = {category: [] for category in self.PERSON_CATEGORIES.keys()}
        
        pbar = tqdm(total=len(images), desc="Detecting persons", ncols=80)
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Prepare batch for LLM
            llm_batch = []
            valid_images = []
            
            for img in batch:
                try:
                    img_bytes = image_loader.get_image_bytes(img['path'])
                    pil_image = image_processor.safe_open_image(img_bytes)
                    
                    if pil_image is None:
                        pbar.update(1)
                        continue
                    
                    thumb_bytes, _ = image_processor.create_thumbnail(pil_image)
                    llm_batch.append({'name': img['name'], 'thumb_bytes': thumb_bytes})
                    valid_images.append(img)
                    
                    del pil_image, img_bytes, thumb_bytes
                    
                except Exception:
                    pass
                
                pbar.update(1)
            
            if not llm_batch:
                continue
            
            # Detect persons in batch
            try:
                detections = self.detect_persons_batch(llm_batch)
                
                # Categorize based on detections
                for img, detection in zip(valid_images, detections):
                    persons = detection.get('persons', ['unknown'])
                    confidence = detection.get('confidence', 0.0)
                    description = detection.get('description', '')
                    
                    # Add person info to image metadata
                    img['persons'] = persons
                    img['person_confidence'] = confidence
                    img['person_description'] = description
                    
                    # Add to each relevant category
                    for person in persons:
                        if person in categorized:
                            categorized[person].append(img)
                    
            except Exception as e:
                print(f"\n⚠ Error in person detection batch: {e}")
                continue
        
        pbar.close()
        
        # Print distribution report
        print("\n✓ Person Detection Complete:")
        for category, imgs in categorized.items():
            if len(imgs) > 0:
                print(f"  {self.PERSON_CATEGORIES[category]}: {len(imgs)} images")
        
        return categorized
    
    def get_best_from_category(self, category_images: List[Dict],
                               top_k: int = 50,
                               score_key: str = 'final_score') -> List[Dict]:
        """Get best images from a category.
        
        Args:
            category_images: List of images in a category
            top_k: Number of top images to return
            score_key: Key to use for sorting
            
        Returns:
            List of top images
        """
        if not category_images:
            return []
        
        # Sort by score
        sorted_images = sorted(category_images, 
                              key=lambda x: x.get(score_key, 0), 
                              reverse=True)
        
        return sorted_images[:top_k]
    
    def save_categorization(self, categorized: Dict[str, List[Dict]], 
                           output_dir: str):
        """Save categorization results.
        
        Args:
            categorized: Dict mapping category to images
            output_dir: Output directory
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'person_categorization.json')
        
        # Prepare data (without full metadata to reduce size)
        results = {}
        for category, images in categorized.items():
            results[category] = {
                'count': len(images),
                'category_name': self.PERSON_CATEGORIES[category],
                'images': [
                    {
                        'name': img['name'],
                        'path': img['path'],
                        'confidence': img.get('person_confidence', 0),
                        'description': img.get('person_description', ''),
                        'score': img.get('final_score', img.get('overall_score', 0))
                    }
                    for img in images
                ]
            }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Saved person categorization to: {output_path}")

