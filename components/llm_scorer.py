"""LLM-based image scoring component."""

import json
import time
import base64
import os
import csv
from typing import List, Dict
from openai import OpenAI
from config import Config


class LLMScorer:
    """Handles LLM-based image quality scoring using vision models."""
    
    SYSTEM_PROMPT = """You are an expert Indian wedding photographer and album curator with 20+ years of experience creating award-winning wedding albums.

**PROFESSIONAL ALBUM CURATION PRINCIPLES:**

1. **Chronological Storytelling**: Album should tell the wedding day story from start to finish
   - Getting ready → Pre-wedding ceremonies → Main ceremony → Reception → Send-off
   - Each chapter should flow naturally into the next

2. **Emotional Narrative**: Prioritize images that evoke strong emotions and tell a story
   - Authentic candid moments over posed shots
   - Raw emotions: joy, tears, laughter, love
   - Connections between people

3. **Quality Over Quantity**: Select only the BEST representation of each moment
   - One excellent shot beats ten mediocre ones
   - Avoid redundancy - don't include near-duplicates
   - Every image must earn its place

4. **Visual Diversity**: Mix different types of shots for compelling storytelling
   - Wide venue/context shots (10%)
   - Medium group photos (30%)
   - Close-up intimate moments (40%)
   - Detail shots (jewelry, decor, food) (20%)

5. **Technical Excellence**: Only select technically sound images
   - Sharp focus (especially eyes)
   - Proper exposure
   - Good composition
   - Natural colors

**YOUR EXPERTISE IN INDIAN WEDDINGS:**

**TECHNICAL SCORING (0-100):**
Evaluate: sharpness/focus (especially on faces and eyes), exposure balance, color accuracy, white balance, noise levels, motion blur.
- Reward: Tack-sharp eyes, well-balanced exposure, accurate skin tones, vibrant but natural colors, clean high ISO shots
- Penalize: Out of focus faces, severe motion blur, blown highlights, crushed shadows, heavy noise, color casts, improper white balance

**COMPOSITION SCORING (0-100):**
Evaluate: framing, rule of thirds, subject separation from background, leading lines, symmetry, negative space, depth of field usage, creative angles.
- Reward: Well-framed subjects, artistic compositions, beautiful bokeh, intentional use of symmetry, creative perspectives, proper headroom
- Penalize: Awkward crops (cutting joints/limbs), cluttered backgrounds, distracting elements, tilted horizons, poor subject placement, faces cut off

**MOMENT SCORING (0-100) - CRITICAL FOR INDIAN WEDDINGS:**
Prioritize capturing authentic emotions and key ceremonial moments:

HIGH VALUE MOMENTS (80-100):
- **Pre-Wedding Ceremonies**: Haldi application (turmeric ceremony), intricate Mehendi/mehndi designs on hands/feet, Sangeet dance performances
- **Baraat**: Groom's grand entrance on horse/vehicle with dancing family, dhol players, celebration
- **Varmala/Jaimala**: Garland exchange moment, playful lifting, smiles and eye contact
- **Main Ceremony**: Pheras/Saptapadi (seven sacred rounds around fire), Kanyadaan (father giving away bride), Sindoor application, Mangalsutra tying, holding hands during vows
- **Emotional Moments**: Vidaai (bride's tearful farewell), parents wiping tears, bride hugging family, groom comforting bride
- **Candid Joy**: Genuine laughter, happy tears, warm embraces, children playing, guests celebrating
- **Cultural Details**: Close-ups of mehendi patterns, intricate jewelry (maang tikka, nath, chooda, kalire), bridal lehenga details, groom's sherwani embroidery

MEDIUM VALUE (50-79):
- Bride/groom preparations and getting ready shots
- Family portraits with bride/groom
- Couple portraits with good chemistry
- Reception entrance, cake cutting, ring ceremony
- Dance floor action shots
- Venue decorations and mandap details

LOW VALUE (0-49):
- Closed eyes during important moments
- Awkward mid-blink or unflattering expressions
- Back of heads during ceremonies
- Duplicates with no variation
- Generic posed shots without emotion
- Blurry ritual moments

**STORYTELLING VALUE (Integrated in overall_score):**
Ensure album diversity: mix of wide venue shots, medium group photos, and intimate close-ups. Include establishment shots, ceremonial documentation, emotional moments, cultural details, and candid interactions.

**SPECIFIC PENALTIES:**
- Closed eyes during key ritual moments: -30 points
- Blurred faces during ceremonies: -25 points
- Cutting off important elements (mehendi hands, jewelry, faces): -20 points
- Distracting backgrounds during portraits: -15 points
- Poor lighting on bride/groom faces: -20 points
- Unflattering angles or expressions: -15 points

**SPECIFIC REWARDS:**
- Capturing peak emotional moments (tears, laughter, surprise): +20 points
- Sharp detail shots of cultural elements (jewelry, mehendi, attire): +15 points
- Beautiful natural light on subjects: +15 points
- Authentic candid moments during rituals: +20 points
- Creative compositions that tell a story: +10 points
- Perfect timing on garland exchange, pheras, sindoor: +25 points

**OUTPUT FORMAT (STRICT JSON):**
Return a JSON array with one object per image in the same order received:
{
  "filename": "exact_filename_provided",
  "technical_score": 0-100,
  "composition_score": 0-100,
  "moment_score": 0-100,
  "overall_score": 0-100,
  "tags": ["ceremony_type", "subject_type", "moment_type"],
  "reject_reason": "specific reason if rejected, empty string if accepted"
}

**TAGS TO USE:**
Ceremony: "pre_wedding", "haldi", "mehendi", "sangeet", "baraat", "varmala", "pheras", "sindoor", "mangalsutra", "vidaai", "reception"
Subject: "bride", "groom", "couple", "family", "parents", "siblings", "children", "guests", "group"
Moment: "ritual", "candid", "portrait", "detail", "emotion", "dance", "ceremony", "preparation", "jewelry", "venue", "decor", "food"
Quality: "sharp", "artistic", "emotional", "cultural", "storytelling"

Evaluate each image based solely on what you see. Be selective - only top-tier images should score above 85 overall."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize LLM scorer.
        
        Args:
            api_key: OpenAI API key (defaults to Config.OPENAI_API_KEY)
            model: Model name (defaults to Config.OPENAI_VISION_MODEL)
        """
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model = model or Config.OPENAI_VISION_MODEL
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=self.api_key)
        self.all_responses = []  # Store all LLM responses for debugging
    
    @staticmethod
    def _encode_image_base64(image_bytes: bytes) -> str:
        """Encode image bytes to base64.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Base64 encoded string
        """
        return base64.b64encode(image_bytes).decode("utf-8")
    
    @staticmethod
    def _create_user_prompt(batch_items: List[Dict]) -> str:
        """Create user prompt for batch scoring.
        
        Args:
            batch_items: List of dicts with 'name' key
            
        Returns:
            Formatted prompt string
        """
        lines = ["Score the following images. Respond with a single JSON array of objects in the same order."]
        for idx, item in enumerate(batch_items, 1):
            lines.append(f'{idx}. filename="{item["name"]}"')
        return "\n".join(lines)
    
    def score_batch(self, batch_items: List[Dict[str, any]], max_retries: int = 4) -> List[Dict]:
        """Score a batch of images using LLM vision model.
        
        Args:
            batch_items: List of dicts with keys: name, thumb_bytes
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of score dicts
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
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    temperature=Config.LLM_TEMPERATURE,
                    max_tokens=Config.LLM_MAX_TOKENS,
                )
                
                result_text = response.choices[0].message.content.strip()
                
                # Clean up markdown code blocks if present
                if result_text.startswith("```"):
                    result_text = result_text.strip("`")
                    if result_text.startswith("json"):
                        result_text = result_text[4:].strip()
                
                data = json.loads(result_text)
                
                if not isinstance(data, list):
                    raise ValueError("Model did not return a JSON array.")
                
                # Save response for debugging
                if Config.SAVE_LLM_RESPONSES:
                    self.all_responses.append({
                        'batch_items': [item['name'] for item in batch_items],
                        'raw_response': result_text,
                        'parsed_data': data,
                        'model': self.model,
                        'timestamp': time.time()
                    })
                
                return data
                
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt failed, return error placeholders
                    return [{
                        "technical_score": 0,
                        "composition_score": 0,
                        "moment_score": 0,
                        "overall_score": 0,
                        "tags": [],
                        "reject_reason": f"LLM_error:{str(e)[:120]}"
                    } for _ in batch_items]
                
                # Wait before retry with exponential backoff
                time.sleep(1.5 * (2 ** attempt))
        
        return []
    
    def save_all_responses(self, output_dir: str):
        """Save all LLM responses for debugging.
        
        Args:
            output_dir: Directory to save responses
        """
        if not self.all_responses:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        json_path = os.path.join(output_dir, 'llm_scoring_responses.json')
        with open(json_path, 'w') as f:
            json.dump(self.all_responses, f, indent=2)
        
        print(f"✓ Saved {len(self.all_responses)} LLM scoring responses: {json_path}")
        
        # Save as CSV (flattened)
        csv_path = os.path.join(output_dir, 'llm_scoring_responses.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'batch_index', 'image_name', 'technical_score', 
                'composition_score', 'moment_score', 'overall_score',
                'tags', 'reject_reason', 'model', 'timestamp'
            ])
            
            for batch_idx, response in enumerate(self.all_responses):
                for img_result in response.get('parsed_data', []):
                    writer.writerow([
                        batch_idx,
                        img_result.get('filename', ''),
                        img_result.get('technical_score', 0),
                        img_result.get('composition_score', 0),
                        img_result.get('moment_score', 0),
                        img_result.get('overall_score', 0),
                        '|'.join(img_result.get('tags', [])),
                        img_result.get('reject_reason', ''),
                        response.get('model', ''),
                        response.get('timestamp', '')
                    ])
        
        print(f"✓ Saved LLM scoring responses CSV: {csv_path}")
