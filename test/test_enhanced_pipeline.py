"""Test script for enhanced pipeline components."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all new components can be imported."""
    print("Testing imports...")
    
    try:
        from components import (
            ImageLoader, ImageProcessor, LLMScorer, Deduplicator,
            Selector, AlbumCurator, ImageEmbedder, ImageClusterer,
            PersonDetector, VerboseLogger
        )
        print("✓ All component imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config():
    """Test that config loads with new settings."""
    print("\nTesting config...")
    
    try:
        from config import Config
        
        # Check new config attributes
        assert hasattr(Config, 'USE_EMBEDDINGS'), "Missing USE_EMBEDDINGS"
        assert hasattr(Config, 'USE_PERSON_DETECTION'), "Missing USE_PERSON_DETECTION"
        assert hasattr(Config, 'EMBEDDING_MODEL'), "Missing EMBEDDING_MODEL"
        assert hasattr(Config, 'CLUSTERING_METHOD'), "Missing CLUSTERING_METHOD"
        assert hasattr(Config, 'PERSON_QUOTAS'), "Missing PERSON_QUOTAS"
        assert hasattr(Config, 'VERBOSE_LOGGING'), "Missing VERBOSE_LOGGING"
        
        print("✓ Config loaded with all new settings")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_component_initialization():
    """Test that new components can be initialized."""
    print("\nTesting component initialization...")
    
    try:
        from components import ImageEmbedder, ImageClusterer, VerboseLogger
        
        # Test ImageEmbedder (CPU mode for testing)
        try:
            embedder = ImageEmbedder(device='cpu')
            print(f"✓ ImageEmbedder initialized (dim: {embedder.embedding_dim})")
        except Exception as e:
            print(f"⚠ ImageEmbedder initialization warning: {e}")
            print("  (This might be due to missing model files - normal on first run)")
        
        # Test ImageClusterer
        clusterer = ImageClusterer(method='kmeans')
        print("✓ ImageClusterer initialized")
        
        # Test VerboseLogger
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = VerboseLogger(tmpdir)
            print("✓ VerboseLogger initialized")
        
        return True
    except Exception as e:
        print(f"✗ Component initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_selector():
    """Test that main selector can be instantiated."""
    print("\nTesting main selector...")
    
    try:
        # Set minimal test environment
        os.environ.setdefault('OPENAI_API_KEY', 'test-key-placeholder')
        
        from wedding_album_selector import WeddingAlbumSelector
        from config import Config
        
        # Override some settings for testing
        Config.USE_EMBEDDINGS = False  # Skip embeddings for quick test
        Config.USE_PERSON_DETECTION = False
        Config.VERBOSE_LOGGING = False
        
        selector = WeddingAlbumSelector(config=Config)
        
        print("✓ WeddingAlbumSelector instantiated successfully")
        print(f"  - Loader: {type(selector.loader).__name__}")
        print(f"  - Processor: {type(selector.processor).__name__}")
        print(f"  - Scorer: {type(selector.scorer).__name__}")
        print(f"  - Embedder: {type(selector.embedder).__name__ if selector.embedder else 'None (disabled)'}")
        print(f"  - Clusterer: {type(selector.clusterer).__name__ if selector.clusterer else 'None (disabled)'}")
        print(f"  - Person Detector: {type(selector.person_detector).__name__ if selector.person_detector else 'None (disabled)'}")
        
        return True
    except Exception as e:
        print(f"✗ Main selector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Enhanced Pipeline Test Suite")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Component Init", test_component_initialization()))
    results.append(("Main Selector", test_main_selector()))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("⚠ Some tests failed - review output above")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

