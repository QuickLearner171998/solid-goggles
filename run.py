#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wedding Album Auto-Selector
Selects best wedding photos using local quality metrics and LLM vision scoring.

Install dependencies:
    pip install openai pillow imagehash opencv-python-headless tqdm python-dotenv
"""

import sys
from config import Config
from wedding_album_selector import WeddingAlbumSelector


def main():
    """Main entry point."""
    try:
        # Check API key
        if not Config.OPENAI_API_KEY:
            print("\n❌ Configuration Error: OPENAI_API_KEY not found in environment")
            try:
                api_key = input("\nEnter your OpenAI API key: ").strip()
                if api_key:
                    Config.OPENAI_API_KEY = api_key
                else:
                    print("No API key provided. Exiting.")
                    sys.exit(1)
            except KeyboardInterrupt:
                print("\nCancelled.")
                sys.exit(1)
        
        # Create and run selector
        selector = WeddingAlbumSelector()
        selector.run()
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()