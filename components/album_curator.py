"""Advanced album curation for Indian weddings ensuring ceremony diversity."""

from typing import List, Dict
from collections import defaultdict


class AlbumCurator:
    """Ensures balanced representation of different wedding ceremonies and moments."""
    
    # Ideal distribution for Indian wedding albums (percentages)
    IDEAL_DISTRIBUTION = {
        'key_rituals': 0.35,      # Pheras, varmala, sindoor, mangalsutra - most important
        'emotional_moments': 0.20, # Vidaai, tears, hugs, candid joy
        'pre_wedding': 0.15,       # Haldi, mehendi, sangeet
        'details': 0.10,           # Jewelry, decor, venue, attire details
        'portraits': 0.12,         # Couple and family portraits
        'other': 0.08             # Reception, baraat, misc
    }
    
    # Tag to category mapping
    CATEGORY_MAP = {
        'key_rituals': ['pheras', 'varmala', 'sindoor', 'mangalsutra', 'ceremony'],
        'emotional_moments': ['vidaai', 'emotion', 'candid'],
        'pre_wedding': ['haldi', 'mehendi', 'sangeet', 'pre_wedding'],
        'details': ['detail', 'jewelry', 'decor', 'venue'],
        'portraits': ['portrait', 'couple', 'family', 'bride', 'groom'],
        'other': ['baraat', 'reception', 'dance', 'food', 'preparation']
    }
    
    def __init__(self):
        """Initialize album curator."""
        pass
    
    def categorize_image(self, tags: List[str]) -> str:
        """Categorize an image based on its tags.
        
        Args:
            tags: List of tags from LLM scoring
            
        Returns:
            Category name
        """
        if not tags:
            return 'other'
        
        # Check each category for matching tags
        for category, category_tags in self.CATEGORY_MAP.items():
            for tag in tags:
                if tag.lower() in category_tags:
                    return category
        
        return 'other'
    
    def analyze_distribution(self, images: List[Dict]) -> Dict[str, int]:
        """Analyze current distribution of images across categories.
        
        Args:
            images: List of image dicts with 'tags' field
            
        Returns:
            Dict mapping category to count
        """
        distribution = defaultdict(int)
        
        for img in images:
            category = self.categorize_image(img.get('tags', []))
            distribution[category] += 1
        
        return dict(distribution)
    
    def curate_balanced_selection(self, images: List[Dict], target_count: int) -> List[Dict]:
        """Select images ensuring balanced distribution across ceremony types.
        
        Args:
            images: List of scored image dicts (must have 'tags' and 'final_score')
            target_count: Target number of images to select
            
        Returns:
            Curated list of images with balanced distribution
        """
        # Categorize all images
        categorized = defaultdict(list)
        for img in images:
            category = self.categorize_image(img.get('tags', []))
            categorized[category].append(img)
        
        # Sort each category by score
        for category in categorized:
            categorized[category].sort(key=lambda x: x['final_score'], reverse=True)
        
        # Calculate target counts per category
        target_counts = {}
        for category, percentage in self.IDEAL_DISTRIBUTION.items():
            target_counts[category] = int(target_count * percentage)
        
        # Select images from each category
        selected = []
        remaining_slots = target_count
        
        # First pass: take up to target from each category
        for category, target in target_counts.items():
            available = categorized.get(category, [])
            take = min(target, len(available))
            selected.extend(available[:take])
            remaining_slots -= take
        
        # Second pass: fill remaining slots with highest scoring images not yet selected
        if remaining_slots > 0:
            selected_ids = {id(img) for img in selected}
            remaining = [img for img in images if id(img) not in selected_ids]
            remaining.sort(key=lambda x: x['final_score'], reverse=True)
            selected.extend(remaining[:remaining_slots])
        
        # Final sort by score
        selected.sort(key=lambda x: x['final_score'], reverse=True)
        
        return selected[:target_count]
    
    def get_distribution_report(self, images: List[Dict]) -> str:
        """Generate a human-readable distribution report.
        
        Args:
            images: List of image dicts with 'tags' field
            
        Returns:
            Formatted distribution report string
        """
        distribution = self.analyze_distribution(images)
        total = len(images)
        
        report = ["\nAlbum Distribution Analysis:"]
        report.append("=" * 60)
        
        for category in self.IDEAL_DISTRIBUTION.keys():
            count = distribution.get(category, 0)
            percentage = (count / total * 100) if total > 0 else 0
            ideal = self.IDEAL_DISTRIBUTION[category] * 100
            
            status = "✓" if abs(percentage - ideal) < 5 else "⚠"
            report.append(
                f"{status} {category.replace('_', ' ').title():20s}: "
                f"{count:4d} ({percentage:5.1f}%) [Ideal: {ideal:4.1f}%]"
            )
        
        report.append("=" * 60)
        report.append(f"Total Images: {total}")
        
        return "\n".join(report)
    
    def suggest_additional_images(self, selected: List[Dict], candidates: List[Dict], 
                                  target_count: int) -> Dict[str, int]:
        """Suggest which ceremony types need more representation.
        
        Args:
            selected: Currently selected images
            candidates: Pool of candidate images
            target_count: Target number of images
            
        Returns:
            Dict mapping category to number of additional images needed
        """
        current_dist = self.analyze_distribution(selected)
        suggestions = {}
        
        for category, ideal_pct in self.IDEAL_DISTRIBUTION.items():
            target = int(target_count * ideal_pct)
            current = current_dist.get(category, 0)
            
            if current < target:
                # Check if we have candidates in this category
                category_candidates = [
                    img for img in candidates 
                    if self.categorize_image(img.get('tags', [])) == category
                ]
                
                available = len(category_candidates)
                needed = target - current
                suggestions[category] = min(needed, available)
        
        return suggestions
