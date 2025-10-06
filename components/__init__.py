"""Components package for wedding album selector."""

from .image_loader import ImageLoader
from .image_processor import ImageProcessor
from .llm_scorer import LLMScorer
from .selector import Selector
from .album_curator import AlbumCurator
from .image_embedder import ImageEmbedder
from .image_clusterer import ImageClusterer
from .verbose_logger import VerboseLogger
from .llm_cluster_selector import LLMClusterSelector

# Import simple dataclass for statistics
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PipelineStatistics:
    """Track pipeline statistics."""
    total_images: int = 0
    prefiltered: int = 0
    llm_scored: int = 0
    num_clusters: int = 0
    final_selected: int = 0
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for reporting."""
        return {
            'total_images': self.total_images,
            'prefiltered': self.prefiltered,
            'llm_scored': self.llm_scored,
            'num_clusters': self.num_clusters,
            'final_selected': self.final_selected,
            'total_time': self.total_time
        }

__all__ = [
    'ImageLoader',
    'ImageProcessor',
    'LLMScorer',
    'Selector',
    'AlbumCurator',
    'ImageEmbedder',
    'ImageClusterer',
    'VerboseLogger',
    'LLMClusterSelector',
    'PipelineStatistics'
]
