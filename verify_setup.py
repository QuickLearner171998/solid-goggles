#!/usr/bin/env python3
"""Verify that all required dependencies are installed."""

import sys

def check_dependency(module_name, package_name=None):
    """Check if a module can be imported."""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} (not installed)")
        return False

def main():
    """Check all dependencies."""
    print("=" * 60)
    print("Dependency Check")
    print("=" * 60)
    
    print("\nCore Dependencies:")
    core_dependencies = [
        ("PIL", "pillow"),
        ("cv2", "opencv-python-headless"),
        ("imagehash", "imagehash"),
        ("openai", "openai"),
        ("tqdm", "tqdm"),
        ("dotenv", "python-dotenv"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("sklearn", "scikit-learn"),
        ("sentence_transformers", "sentence-transformers"),
        ("umap", "umap-learn"),
        ("hdbscan", "hdbscan"),
    ]
    
    core_results = [check_dependency(mod, pkg) for mod, pkg in core_dependencies]
    
    print("\nOptional Dependencies (for enhanced face recognition):")
    optional_dependencies = [
        ("face_recognition", "face_recognition"),
        ("dlib", "dlib"),
    ]
    
    optional_results = [check_dependency(mod, pkg) for mod, pkg in optional_dependencies]
    
    print("=" * 60)
    
    if all(core_results):
        print("✓ All core dependencies installed!")
        
        if not all(optional_results):
            print("\n⚠ Optional face recognition libraries not installed:")
            print("  - System will use basic OpenCV face detection (fallback)")
            print("  - For better accuracy, install: pip install face_recognition")
            print("  - Note: face_recognition requires dlib (may need compilation)")
        else:
            print("✓ Optional face recognition libraries also installed!")
        
        print("\n✅ You can now run: python run.py")
        return 0
    else:
        print("⚠ Some core dependencies are missing.")
        print("\nPlease run: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())

