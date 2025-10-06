#!/usr/bin/env python3
"""Setup script to create reference face directories."""

import sys
from components import FaceRecognizer
from config import Config

def main():
    """Create reference face directory structure."""
    print("=" * 60)
    print("Wedding Album Selector - Reference Face Setup")
    print("=" * 60)
    
    # Initialize face recognizer
    recognizer = FaceRecognizer(reference_dir=Config.FACE_RECOGNITION_DIR)
    
    # Create directories
    recognizer.create_reference_directories()
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\n📸 Next Steps:")
    print("\n1. Add reference photos to the folders:")
    print(f"   {Config.FACE_RECOGNITION_DIR}/")
    print("\n2. Folder Structure:")
    print("   - bride-groom/       → Add photos of bride and/or groom")
    print("   - bride-parents/     → Add photos of bride's parents")
    print("   - bride-siblings/    → Add photos of bride's siblings")
    print("   - groom-parents/     → Add photos of groom's parents")
    print("\n3. Photo Guidelines:")
    print("   ✓ Clear, well-lit photos")
    print("   ✓ Face clearly visible (no sunglasses, hats)")
    print("   ✓ Front-facing preferred")
    print("   ✓ One or more faces per photo is fine")
    print("   ✓ Multiple reference photos per category recommended")
    print("\n4. After adding photos, run:")
    print("   python run.py")
    print("\n5. The system will:")
    print("   • Load reference faces automatically")
    print("   • Match faces in wedding photos")
    print("   • Organize photos by recognized people")
    print("   • Combine face recognition with AI detection")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    sys.exit(main())

