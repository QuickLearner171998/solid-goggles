"""Quick tests to verify component functionality."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components import ImageLoader, ImageProcessor, Deduplicator, Selector
from config import Config


def test_image_loader():
    """Test ImageLoader component."""
    print("Testing ImageLoader...")
    try:
        loader = ImageLoader(Config.SOURCE_FOLDER)
        images = loader.discover_images()
        print(f"  ✓ Found {len(images)} images")
        
        if images:
            # Test batch loading
            batch = loader.load_batch(images, 0, 2)
            print(f"  ✓ Batch loading works: {len(batch)} images in batch")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_image_processor():
    """Test ImageProcessor component."""
    print("Testing ImageProcessor...")
    try:
        processor = ImageProcessor()
        print("  ✓ ImageProcessor initialized")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_deduplicator():
    """Test Deduplicator component."""
    print("Testing Deduplicator...")
    try:
        dedup = Deduplicator()
        
        # Add some test data
        dedup.add_or_update("hash1", 0.8, {"name": "image1.jpg"})
        dedup.add_or_update("hash2", 0.9, {"name": "image2.jpg"})
        dedup.add_or_update("hash1", 0.7, {"name": "image1_dup.jpg"})  # Should not replace
        
        unique = dedup.get_unique_images()
        assert len(unique) == 2, f"Expected 2 unique, got {len(unique)}"
        print(f"  ✓ Deduplication works: {len(unique)} unique images")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_selector():
    """Test Selector component."""
    print("Testing Selector...")
    try:
        selector = Selector()
        
        # Test score calculation
        test_results = [
            {"local_score": 0.8, "overall_score": 75},
            {"local_score": 0.6, "overall_score": 85},
        ]
        
        scored = selector.calculate_final_scores(test_results)
        assert all("final_score" in r for r in scored)
        print(f"  ✓ Score calculation works")
        
        # Test ranking
        selected = selector.rank_and_select(scored, 1)
        assert len(selected) == 1
        print(f"  ✓ Ranking and selection works")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Component Tests")
    print("=" * 60 + "\n")
    
    results = []
    results.append(("ImageLoader", test_image_loader()))
    results.append(("ImageProcessor", test_image_processor()))
    results.append(("Deduplicator", test_deduplicator()))
    results.append(("Selector", test_selector()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("✅ All tests passed!" if all_passed else "❌ Some tests failed"))
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
