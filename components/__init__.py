"""Modular components for wedding album selection."""

from .data_models import ImageData, ImageMetrics, LLMScores, ClusterInfo, PipelineStatistics
from .image_loader import ImageLoader
from .image_processor import ImageProcessor
from .quality_filter import QualityFilter
from .llm_scorer import LLMScorer
from .deduplicator import Deduplicator
from .selector import Selector
from .album_curator import AlbumCurator
from .image_embedder import ImageEmbedder
from .image_clusterer import ImageClusterer
from .person_detector import PersonDetector
from .verbose_logger import VerboseLogger
from .face_recognizer import FaceRecognizer
from .llm_cluster_selector import LLMClusterSelector

__all__ = [
    # Data models
    'ImageData',
    'ImageMetrics',
    'LLMScores',
    'ClusterInfo',
    'PipelineStatistics',
    # Components
    'ImageLoader', 
    'ImageProcessor',
    'QualityFilter',
    'LLMScorer', 
    'Deduplicator', 
    'Selector', 
    'AlbumCurator',
    'ImageEmbedder',
    'ImageClusterer',
    'PersonDetector',
    'VerboseLogger',
    'FaceRecognizer',
    'LLMClusterSelector'
]
