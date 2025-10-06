"""Configuration settings for wedding album selector."""

import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Central configuration for the wedding album selection pipeline."""
    
    # Input/Output
    SOURCE_FOLDER = os.path.abspath("./album_pics")
    
    # Output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.abspath(f"./album_selection_{timestamp}")
    
    # Selection parameters
    NUM_SELECT = 1000
    BATCH_SIZE = 5
    
    # Image processing
    MAX_THUMB_SIDE = 512
    THUMB_QUALITY = 70
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    DUPLICATE_THRESHOLD = 5
    
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
    FINAL_SCORE_LLM_WEIGHT = 0.70  # LLM vision score
    FINAL_SCORE_LOCAL_WEIGHT = 0.20  # Local image quality metrics
    FINAL_SCORE_CLUSTER_WEIGHT = 0.10  # Cluster representative bonus
    
    # Image processing thresholds (not used for filtering, kept for reference)
    MIN_SHARPNESS = 100.0
    GOOD_SHARPNESS = 300.0
    MIN_COLORFULNESS = 20.0
    GOOD_COLORFULNESS = 40.0
    MIN_EXPOSURE = 0.15
    MAX_EXPOSURE = 0.85
    
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
    
    # Verbose logging & debugging
    VERBOSE_LOGGING = True  # Save intermediate results
    SAVE_EMBEDDINGS = True  # Save embeddings to disk
    SAVE_CLUSTERING_RESULTS = True  # Save cluster assignments
    SAVE_CLUSTER_IMAGES = True  # Save sample images from each cluster
    SAVE_LLM_RESPONSES = True  # Save all LLM responses (JSON + CSV)
    SAVE_CLUSTER_SELECTIONS = True  # Save images selected from each cluster
    DEBUG_MODE = True  # Enable comprehensive debugging
    
    # Memory optimization
    CLEAR_MEMORY_AFTER_EMBEDDING = True  # Clear images from memory after embedding
    USE_GPU_IF_AVAILABLE = True  # Use GPU for embeddings if available
