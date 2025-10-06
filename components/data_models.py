"""Data models for wedding album selection pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class ImageMetrics:
    """Quality metrics for an image."""
    sharpness: float = 0.0
    exposure: float = 0.0
    colorfulness: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'sharpness': self.sharpness,
            'exposure': self.exposure,
            'colorfulness': self.colorfulness
        }


@dataclass
class LLMScores:
    """LLM scoring results for an image."""
    technical_score: float = 0.0
    composition_score: float = 0.0
    moment_score: float = 0.0
    overall_score: float = 0.0
    tags: List[str] = field(default_factory=list)
    reject_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'technical_score': self.technical_score,
            'composition_score': self.composition_score,
            'moment_score': self.moment_score,
            'overall_score': self.overall_score,
            'tags': self.tags,
            'reject_reason': self.reject_reason
        }


@dataclass
class ImageData:
    """Complete data for a wedding photo."""
    path: str
    name: str
    
    # Quality metrics
    metrics: Optional[ImageMetrics] = None
    local_score: float = 0.0
    
    # Clustering
    cluster_id: int = -1
    is_cluster_representative: bool = False
    llm_cluster_rank: Optional[int] = None
    llm_cluster_reason: str = ""
    
    # LLM scoring
    llm_scores: Optional[LLMScores] = None
    
    # Person detection
    persons: List[str] = field(default_factory=list)
    person_confidence: float = 0.0
    person_description: str = ""
    
    # Face recognition
    face_matches: Dict[str, float] = field(default_factory=dict)
    face_confidence: Dict[str, float] = field(default_factory=dict)
    
    # Final scoring
    final_score: float = 0.0
    
    # Deduplication
    phash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'path': self.path,
            'name': self.name,
            'local_score': self.local_score,
            'cluster_id': self.cluster_id,
            'is_cluster_representative': self.is_cluster_representative,
            'persons': self.persons,
            'person_confidence': self.person_confidence,
            'person_description': self.person_description,
            'face_matches': self.face_matches,
            'face_confidence': self.face_confidence,
            'final_score': self.final_score,
            'phash': self.phash
        }
        
        if self.metrics:
            result.update(self.metrics.to_dict())
        
        if self.llm_scores:
            result.update(self.llm_scores.to_dict())
        
        if self.llm_cluster_rank is not None:
            result['llm_cluster_rank'] = self.llm_cluster_rank
            result['llm_cluster_reason'] = self.llm_cluster_reason
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImageData':
        """Create from dictionary."""
        # Extract metrics
        metrics = None
        if 'sharpness' in data:
            metrics = ImageMetrics(
                sharpness=data.get('sharpness', 0.0),
                exposure=data.get('exposure', 0.0),
                colorfulness=data.get('colorfulness', 0.0)
            )
        
        # Extract LLM scores
        llm_scores = None
        if 'technical_score' in data:
            llm_scores = LLMScores(
                technical_score=data.get('technical_score', 0.0),
                composition_score=data.get('composition_score', 0.0),
                moment_score=data.get('moment_score', 0.0),
                overall_score=data.get('overall_score', 0.0),
                tags=data.get('tags', []),
                reject_reason=data.get('reject_reason', '')
            )
        
        return cls(
            path=data['path'],
            name=data['name'],
            metrics=metrics,
            local_score=data.get('local_score', 0.0),
            cluster_id=data.get('cluster_id', -1),
            is_cluster_representative=data.get('is_cluster_representative', False),
            llm_cluster_rank=data.get('llm_cluster_rank'),
            llm_cluster_reason=data.get('llm_cluster_reason', ''),
            llm_scores=llm_scores,
            persons=data.get('persons', []),
            person_confidence=data.get('person_confidence', 0.0),
            person_description=data.get('person_description', ''),
            face_matches=data.get('face_matches', {}),
            face_confidence=data.get('face_confidence', {}),
            final_score=data.get('final_score', 0.0),
            phash=data.get('phash')
        )


@dataclass
class ClusterInfo:
    """Information about an image cluster."""
    cluster_id: int
    images: List[ImageData]
    summary: str = ""
    selected_count: int = 0
    
    def get_top_images(self, k: int = 10) -> List[ImageData]:
        """Get top k images from cluster."""
        sorted_images = sorted(
            self.images, 
            key=lambda x: x.local_score, 
            reverse=True
        )
        return sorted_images[:k]


@dataclass
class PipelineStatistics:
    """Statistics from pipeline execution."""
    total_images: int = 0
    prefiltered: int = 0
    embedded: int = 0
    clustered: int = 0
    llm_scored: int = 0
    person_detected: int = 0
    final_selected: int = 0
    
    # Timing
    total_time: float = 0.0
    phase_times: Dict[str, float] = field(default_factory=dict)
    
    # Clustering
    n_clusters: int = 0
    cluster_distribution: Dict[int, int] = field(default_factory=dict)
    
    # Person detection
    person_counts: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_images': self.total_images,
            'prefiltered': self.prefiltered,
            'embedded': self.embedded,
            'clustered': self.clustered,
            'llm_scored': self.llm_scored,
            'person_detected': self.person_detected,
            'final_selected': self.final_selected,
            'total_time': self.total_time,
            'phase_times': self.phase_times,
            'n_clusters': self.n_clusters,
            'cluster_distribution': self.cluster_distribution,
            'person_counts': self.person_counts
        }

