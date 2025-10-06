"""Configuration settings for the wedding album selector."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration for the application."""
    
    # Input/Output
    SOURCE_FOLDER = os.path.abspath("./album_pics")
    OUTPUT_DIR = os.path.abspath("./album_selection_enhanced")
    
    # Selection parameters
    NUM_SELECT = 1000
    BATCH_SIZE = 5
    LLM_PREFILTER_TARGET = 2000  # How many to send to LLM after quality filter
    
    # Quality filtering (concurrent)
    QUALITY_FILTER_WORKERS = 8  # Concurrent workers for quality filtering
    BLUR_THRESHOLD = 100.0  # Laplacian variance threshold for blur detection
    MIN_SHARPNESS = 50.0  # Minimum sharpness to pass filter
    MIN_EXPOSURE = 0.2  # Minimum exposure score (0-1)
    
    # Image processing
    MAX_THUMB_SIDE = 512
    THUMB_QUALITY = 70
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    
    # LLM settings
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_VISION_MODEL = "chatgpt-4o-latest"  # GPT-4o latest (GPT-5 when available)
    IMAGES_PER_LLM_CALL = 5
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 1000
    
    # LLM image resizing (before sending to vision model)
    LLM_IMAGE_MAX_SIZE = 512  # Max dimension for images sent to LLM
    LLM_IMAGE_QUALITY = 85  # JPEG quality for LLM images
    
    # Scoring weights (tuned for Indian weddings)
    FINAL_SCORE_LLM_WEIGHT = 0.70  # LLM better at identifying cultural moments
    FINAL_SCORE_LOCAL_WEIGHT = 0.20
    FINAL_SCORE_CLUSTER_WEIGHT = 0.10  # Bonus for representative images
    
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
    
    # ===== NEW FEATURES =====
    
    # Embedding & Clustering
    USE_EMBEDDINGS = True  # Use CLIP embeddings for clustering
    EMBEDDING_MODEL = "clip-ViT-B-32"  # CLIP model for embeddings
    EMBEDDING_BATCH_SIZE = 32  # Batch size for embedding generation
    REDUCE_DIMENSIONS = True  # Use UMAP for dimensionality reduction
    DIMENSION_TARGET = 50  # Target dimensions after reduction
    
    # Clustering parameters
    CLUSTERING_METHOD = "kmeans"  # 'kmeans', 'dbscan', or 'auto'
    NUM_CLUSTERS = None  # Auto-determine if None
    MIN_CLUSTERS = 20  # Minimum clusters
    MAX_CLUSTERS = 100  # Maximum clusters
    IMAGES_PER_CLUSTER = 10  # Best images to select from each cluster
    USE_LLM_FOR_CLUSTER_SELECTION = True  # Use LLM to select best from each cluster
    
    # Person detection
    USE_PERSON_DETECTION = False  # DISABLED - Skip person categorization of final images
    PERSON_DETECTION_BATCH_SIZE = 3  # Images per person detection call
    
    # Face recognition (reference-based)
    USE_FACE_RECOGNITION = False  # DISABLED by default (enable if you add reference photos)
    FACE_RECOGNITION_DIR = os.path.abspath("./reference_faces")  # Reference images directory
    FACE_RECOGNITION_THRESHOLD = 0.6  # Confidence threshold (0-1)
    COMBINE_FACE_AND_LLM = True  # Combine face recognition with LLM detection
    
    # Person-specific quotas (DISABLED - not used unless USE_PERSON_DETECTION = True)
    PERSON_QUOTAS = {
        'bride': 150,
        'groom': 150,
        'couple': 200,
        'bride_parents': 50,
        'bride_mom': 30,
        'bride_dad': 30,
        'bride_brother': 40,
        'groom_parents': 50,
        'groom_mom': 30,
        'groom_dad': 30,
        'family': 100,
        'ceremony': 150,
        'other': 100
    }
    
    # Final selection organization
    ORGANIZE_BY_PERSON = False  # Organize final images by person (requires USE_PERSON_DETECTION = True)
    
    # Verbose logging & debugging
    VERBOSE_LOGGING = True  # Save intermediate results
    SAVE_EMBEDDINGS = True  # Save embeddings to disk
    SAVE_CLUSTERING_RESULTS = True  # Save cluster assignments
    SAVE_PERSON_DETECTION_RESULTS = True  # Save person categorization
    SAVE_CLUSTER_IMAGES = True  # Save sample images from each cluster
    SAVE_LLM_RESPONSES = True  # Save all LLM responses (JSON + CSV)
    SAVE_CLUSTER_SELECTIONS = True  # Save images selected from each cluster
    SAVE_REJECTED_IMAGES = True  # Save rejected images with reasons
    MAX_REJECTED_SAMPLES = 30000  # Maximum rejected image samples to save
    DEBUG_MODE = True  # Enable comprehensive debugging
    
    # Memory optimization
    CLEAR_MEMORY_AFTER_EMBEDDING = True  # Clear images from memory after embedding
    USE_GPU_IF_AVAILABLE = True  # Use GPU for embeddings if available
    
    @classmethod
    def validate(cls):
        """Validate configuration."""
        if not os.path.exists(cls.SOURCE_FOLDER):
            raise ValueError(f"Source folder does not exist: {cls.SOURCE_FOLDER}")
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment or .env file")
        return True
