"""Image clustering component for grouping similar images."""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap
from collections import Counter


class ImageClusterer:
    """Clusters images based on their embeddings."""
    
    def __init__(self, method: str = 'kmeans'):
        """Initialize the image clusterer.
        
        Args:
            method: Clustering method ('kmeans', 'dbscan', 'auto')
        """
        self.method = method
        self.cluster_labels = None
        self.cluster_centers = None
        self.scaler = StandardScaler()
        self.reducer = None
    
    def reduce_dimensions(self, embeddings: np.ndarray, 
                         n_components: int = 50,
                         use_umap: bool = True) -> np.ndarray:
        """Reduce dimensionality of embeddings for better clustering.
        
        Args:
            embeddings: High-dimensional embedding vectors
            n_components: Target number of dimensions
            use_umap: Whether to use UMAP (True) or skip reduction (False)
            
        Returns:
            Reduced embeddings
        """
        if not use_umap or embeddings.shape[1] <= n_components:
            print(f"Skipping dimensionality reduction (current dim: {embeddings.shape[1]})")
            return embeddings
        
        print(f"Reducing dimensions from {embeddings.shape[1]} to {n_components} using UMAP...")
        
        try:
            self.reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine',
                random_state=42,
                verbose=False
            )
            
            reduced = self.reducer.fit_transform(embeddings)
            print(f"✓ Dimensionality reduction complete: {reduced.shape}")
            return reduced
            
        except Exception as e:
            print(f"⚠ UMAP failed: {e}. Using original embeddings.")
            return embeddings
    
    def determine_optimal_clusters(self, embeddings: np.ndarray,
                                   min_clusters: int = 10,
                                   max_clusters: int = 100,
                                   step: int = 10) -> int:
        """Determine optimal number of clusters using elbow method and silhouette score.
        
        Args:
            embeddings: Embedding vectors
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            step: Step size for trying different cluster counts
            
        Returns:
            Optimal number of clusters
        """
        print("Determining optimal number of clusters...")
        
        # Adjust max_clusters based on data size
        n_samples = embeddings.shape[0]
        max_clusters = min(max_clusters, n_samples // 10)
        min_clusters = min(min_clusters, max_clusters)
        
        if max_clusters <= min_clusters:
            print(f"Using {max_clusters} clusters based on dataset size")
            return max_clusters
        
        best_score = -1
        best_k = min_clusters
        
        cluster_range = range(min_clusters, max_clusters + 1, step)
        
        for k in cluster_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
                labels = kmeans.fit_predict(embeddings)
                
                # Calculate silhouette score
                score = silhouette_score(embeddings, labels, sample_size=min(1000, n_samples))
                
                print(f"  k={k}: silhouette={score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    
            except Exception as e:
                print(f"  k={k}: failed ({e})")
                continue
        
        print(f"✓ Optimal clusters: {best_k} (silhouette score: {best_score:.3f})")
        return best_k
    
    def cluster_kmeans(self, embeddings: np.ndarray, 
                      n_clusters: Optional[int] = None,
                      auto_optimize: bool = True) -> np.ndarray:
        """Cluster images using K-Means.
        
        Args:
            embeddings: Embedding vectors
            n_clusters: Number of clusters (auto-determined if None)
            auto_optimize: Whether to automatically find optimal k
            
        Returns:
            Cluster labels for each image
        """
        print(f"\nClustering {embeddings.shape[0]} images using K-Means...")
        
        # Normalize embeddings
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Determine number of clusters
        if n_clusters is None and auto_optimize:
            n_clusters = self.determine_optimal_clusters(embeddings_scaled)
        elif n_clusters is None:
            # Default heuristic: sqrt(n/2)
            n_clusters = max(10, int(np.sqrt(embeddings.shape[0] / 2)))
            print(f"Using default cluster count: {n_clusters}")
        
        # Perform clustering
        print(f"Running K-Means with {n_clusters} clusters...")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,
            max_iter=300,
            verbose=0
        )
        
        self.cluster_labels = kmeans.fit_predict(embeddings_scaled)
        self.cluster_centers = kmeans.cluster_centers_
        
        # Print cluster distribution
        cluster_counts = Counter(self.cluster_labels)
        print(f"\n✓ Clustering complete:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Cluster sizes: min={min(cluster_counts.values())}, "
              f"max={max(cluster_counts.values())}, "
              f"avg={np.mean(list(cluster_counts.values())):.1f}")
        
        return self.cluster_labels
    
    def cluster_dbscan(self, embeddings: np.ndarray,
                      eps: float = 0.5,
                      min_samples: int = 5) -> np.ndarray:
        """Cluster images using DBSCAN (density-based clustering).
        
        Args:
            embeddings: Embedding vectors
            eps: Maximum distance between samples in same neighborhood
            min_samples: Minimum samples in neighborhood for core point
            
        Returns:
            Cluster labels for each image (-1 for noise points)
        """
        print(f"\nClustering {embeddings.shape[0]} images using DBSCAN...")
        
        # Normalize embeddings
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Perform clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
        self.cluster_labels = dbscan.fit_predict(embeddings_scaled)
        
        # Print cluster distribution
        cluster_counts = Counter(self.cluster_labels)
        n_clusters = len([k for k in cluster_counts.keys() if k != -1])
        n_noise = cluster_counts.get(-1, 0)
        
        print(f"\n✓ Clustering complete:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Noise points: {n_noise}")
        if n_clusters > 0:
            sizes = [v for k, v in cluster_counts.items() if k != -1]
            print(f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
        
        return self.cluster_labels
    
    def cluster(self, embeddings: np.ndarray,
               n_clusters: Optional[int] = None,
               reduce_dims: bool = True,
               **kwargs) -> np.ndarray:
        """Cluster images using the configured method.
        
        Args:
            embeddings: Embedding vectors
            n_clusters: Number of clusters (for kmeans)
            reduce_dims: Whether to reduce dimensions before clustering
            **kwargs: Additional parameters for specific clustering methods
            
        Returns:
            Cluster labels for each image
        """
        # Reduce dimensions if requested
        if reduce_dims:
            embeddings = self.reduce_dimensions(embeddings)
        
        # Perform clustering based on method
        if self.method == 'kmeans':
            return self.cluster_kmeans(embeddings, n_clusters, **kwargs)
        elif self.method == 'dbscan':
            return self.cluster_dbscan(embeddings, **kwargs)
        elif self.method == 'auto':
            # Try K-Means with auto-optimization
            return self.cluster_kmeans(embeddings, n_clusters, auto_optimize=True)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
    
    def get_cluster_representatives(self, embeddings: np.ndarray,
                                   metadata: List[Dict],
                                   top_k: int = 10,
                                   diversity_weight: float = 0.3) -> Dict[int, List[Dict]]:
        """Get representative images from each cluster.
        
        Args:
            embeddings: Original embedding vectors
            metadata: List of metadata dicts for each image
            top_k: Number of representatives per cluster
            diversity_weight: Weight for diversity vs. centrality (0-1)
            
        Returns:
            Dict mapping cluster_id to list of representative image metadata
        """
        if self.cluster_labels is None:
            raise ValueError("Must call cluster() first")
        
        print(f"\nSelecting top {top_k} representatives from each cluster...")
        
        representatives = {}
        
        unique_clusters = sorted(set(self.cluster_labels))
        # Remove noise cluster if present
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        
        for cluster_id in unique_clusters:
            # Get indices of images in this cluster
            cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Get embeddings for this cluster
            cluster_embeddings = embeddings[cluster_indices]
            
            # Calculate cluster center
            center = np.mean(cluster_embeddings, axis=0)
            
            # Score each image based on distance to center and diversity
            scores = []
            for idx, emb in zip(cluster_indices, cluster_embeddings):
                # Centrality score (inverse distance to center)
                dist_to_center = np.linalg.norm(emb - center)
                centrality = 1.0 / (1.0 + dist_to_center)
                
                # Diversity score (avg distance to other selected images)
                diversity = 1.0  # Start with max diversity
                
                # Combined score
                score = (1 - diversity_weight) * centrality + diversity_weight * diversity
                
                scores.append((idx, score))
            
            # Sort by score and select top k
            scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in scores[:top_k]]
            
            # Get metadata for representatives
            representatives[cluster_id] = [metadata[idx] for idx in top_indices]
        
        print(f"✓ Selected representatives from {len(representatives)} clusters")
        
        return representatives
    
    def save_clustering_results(self, output_path: str,
                               embeddings: np.ndarray,
                               metadata: List[Dict]):
        """Save clustering results to disk.
        
        Args:
            output_path: Path to save results
            embeddings: Embedding vectors
            metadata: Image metadata
        """
        if self.cluster_labels is None:
            raise ValueError("Must call cluster() first")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare data
        results = {
            'n_clusters': len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0),
            'cluster_distribution': dict(Counter(self.cluster_labels)),
            'images': []
        }
        
        # Add image data
        for i, (meta, label) in enumerate(zip(metadata, self.cluster_labels)):
            results['images'].append({
                'index': i,
                'path': meta.get('path', ''),
                'name': meta.get('name', ''),
                'cluster_id': int(label),
                'local_score': meta.get('local_score', 0)
            })
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Saved clustering results to: {output_path}")
        
        # Also save embeddings with cluster labels
        embeddings_path = output_path.replace('.json', '_embeddings.npz')
        np.savez_compressed(
            embeddings_path,
            embeddings=embeddings,
            cluster_labels=self.cluster_labels,
            metadata=np.array(metadata, dtype=object)
        )
        
        print(f"✓ Saved embeddings with labels to: {embeddings_path}")

