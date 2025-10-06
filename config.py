"""Configuration settings for the wedding album selector."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration for the application."""
    
    # Input/Output
    SOURCE_FOLDER = os.path.abspath("./album_pics")
    OUTPUT_DIR = os.path.abspath("./album_selection_top1000")
    
    # Selection parameters
    NUM_SELECT = 1000
    BATCH_SIZE = 5
    LLM_PREFILTER_TARGET = 1800  # How many to send to LLM after local prefilter
    
    # Image processing
    MAX_THUMB_SIDE = 512
    THUMB_QUALITY = 70
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    
    # LLM settings
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_VISION_MODEL = "gpt-4.1"
    IMAGES_PER_LLM_CALL = 5
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 800
    
    # Scoring weights (tuned for Indian weddings)
    FINAL_SCORE_LLM_WEIGHT = 0.75  # LLM better at identifying cultural moments
    FINAL_SCORE_LOCAL_WEIGHT = 0.25
    
    LOCAL_SCORE_SHARPNESS_WEIGHT = 0.40
    LOCAL_SCORE_EXPOSURE_WEIGHT = 0.25
    LOCAL_SCORE_FACE_WEIGHT = 0.25
    LOCAL_SCORE_COLOR_WEIGHT = 0.10  # Color vibrancy matters for Indian weddings
    
    # Album curation
    USE_BALANCED_CURATION = True  # Ensure diverse representation of ceremonies
    
    # Processing parameters
    SHARPNESS_NORMALIZE_FACTOR = 300.0
    COLORFULNESS_NORMALIZE_FACTOR = 50.0
    DEDUPE_HASH_SIZE = 256
    
    @classmethod
    def validate(cls):
        """Validate configuration."""
        if not os.path.exists(cls.SOURCE_FOLDER):
            raise ValueError(f"Source folder does not exist: {cls.SOURCE_FOLDER}")
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment or .env file")
        return True
