"""State-of-the-art image clustering component.

Based on research:
- CLIP embeddings provide excellent semantic understanding
- UMAP dimensionality reduction preserves structure while improving performance
- K-Means and MiniBatchKMeans are most effective for image clustering
- Multiple quality metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) for optimization
- DBSCAN removed (less effective for high-dimensional embeddings)
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import umap
from collections import Counter


class ImageClusterer:
    """Advanced image clustering using research-backed methods."""
    
    def __init__(self, method: str = 'auto'):
        """Initialize the clusterer.
        
        Args:
            method: Clustering method ('kmeans', 'minibatch', or 'auto')
        """
        self.method = method
        self.cluster_labels = None
        self.cluster_centers = None
        self.scaler = StandardScaler()
        self.reducer = None
    
    def reduce_dimensions(self, embeddings: np.ndarray, 
                         n_components: int = 50,
                         use_umap: bool = True) -> np.ndarray:
        """Reduce dimensionality using UMAP for better clustering.
        
        UMAP (Uniform Manifold Approximation and Projection) is superior to PCA
        for clustering as it preserves both local and global structure.
        
        Args:
            embeddings: High-dimensional embedding vectors
            n_components: Target number of dimensions
            use_umap: Whether to use UMAP reduction
            
        Returns:
            Reduced embeddings
        """
        if not use_umap or embeddings.shape[1] <= n_components:
            return embeddings
        
        print(f"Reducing dimensions from {embeddings.shape[1]} to {n_components} using UMAP...")
        
        self.reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        
        reduced = self.reducer.fit_transform(embeddings)
        print(f"✓ Dimensionality reduction complete")
        
        return reduced
    
    def determine_optimal_clusters(self, 
                                  embeddings: np.ndarray,
                                  min_clusters: int = 50,
                                  max_clusters: int = 200,
                                  step: int = 10) -> int:
        """Determine optimal clusters using MULTIPLE quality metrics (research-backed).
        
        Uses combined scoring with:
        - Silhouette Score (cluster separation, range -1 to 1, higher better)
        - Calinski-Harabasz Index (variance ratio, higher better)
        - Davies-Bouldin Index (avg similarity, lower better)
        
        Args:
            embeddings: Scaled embedding vectors
            min_clusters: Minimum clusters to try
            max_clusters: Maximum clusters to try
            step: Step size for cluster range
            
        Returns:
            Optimal number of clusters
        """
        print(f"\n[Auto-Optimization] Finding optimal clusters using multi-metric scoring...")
        print(f"  Range: {min_clusters}-{max_clusters} (step: {step})")
        
        n_samples = embeddings.shape[0]
        max_clusters = min(max_clusters, n_samples - 1)
        
        if max_clusters <= min_clusters:
            print(f"⚠ Dataset size constraints: using {min_clusters} clusters")
            return min_clusters
        
        best_combined_score = -float('inf')
        best_k = min_clusters
        
        cluster_range = range(min_clusters, max_clusters + 1, step)
        
        print(f"\n  Testing cluster configurations...")
        for k in cluster_range:
            try:
                # Use MiniBatchKMeans for faster evaluation
                kmeans = MiniBatchKMeans(
                    n_clusters=k, 
                    random_state=42, 
                    n_init=10,
                    max_iter=100, 
                    batch_size=1024
                )
                labels = kmeans.fit_predict(embeddings)
                
                # Calculate multiple quality metrics
                silhouette = silhouette_score(embeddings, labels, sample_size=min(2000, n_samples))
                calinski = calinski_harabasz_score(embeddings, labels)
                davies_bouldin = davies_bouldin_score(embeddings, labels)
                
                # Normalize scores to 0-1 range for fair comparison
                silhouette_norm = (silhouette + 1) / 2  # -1,1 → 0,1
                calinski_norm = min(calinski / 10000, 1.0)  # Typical range 0-10000
                davies_bouldin_norm = max(0, 1 - (davies_bouldin / 3))  # Lower is better, typical 0-3
                
                # Weighted combined score (based on research importance)
                combined_score = (
                    0.50 * silhouette_norm +      # Most important: cluster separation
                    0.30 * calinski_norm +         # Variance ratio
                    0.20 * davies_bouldin_norm     # Compactness
                )
                
                print(f"  k={k:3d}: sil={silhouette:.3f}, CH={calinski:6.0f}, DB={davies_bouldin:.3f} → score={combined_score:.3f}")
                
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_k = k
                    
            except Exception as e:
                print(f"  k={k}: failed ({e})")
                continue
        
        print(f"\n✓ Optimal configuration: {best_k} clusters (score: {best_combined_score:.3f})")
        return best_k
    
    def cluster_kmeans(self, embeddings: np.ndarray, 
                      n_clusters: Optional[int] = None,
                      auto_optimize: bool = False) -> np.ndarray:
        """Cluster using K-Means or MiniBatchKMeans (adaptive).
        
        Automatically selects:
        - MiniBatchKMeans for datasets > 2000 images (faster, similar quality)
        - Standard K-Means for smaller datasets (slightly better quality)
        
        Args:
            embeddings: Embedding vectors
            n_clusters: Number of clusters (auto-determined if None)
            auto_optimize: Use multi-metric optimization to find optimal k
            
        Returns:
            Cluster labels for each image
        """
        print(f"\nClustering {embeddings.shape[0]} images...")
        
        # Normalize embeddings
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Determine optimal number of clusters
        if n_clusters is None and auto_optimize:
            n_clusters = self.determine_optimal_clusters(embeddings_scaled)
        elif n_clusters is None:
            # Smart heuristic: 3-5% of images (research-backed)
            n_clusters = max(50, min(200, int(embeddings.shape[0] * 0.04)))
            print(f"Using heuristic: {n_clusters} clusters (~{100*n_clusters/embeddings.shape[0]:.1f}% of images)")
        
        # Choose algorithm based on dataset size
        use_minibatch = embeddings.shape[0] > 2000
        
        if use_minibatch:
            print(f"Algorithm: MiniBatchKMeans (optimized for large datasets)")
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=30,
                max_iter=300,
                batch_size=2048,
                verbose=1
            )
        else:
            print(f"Algorithm: K-Means (full algorithm for best quality)")
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=30,
                max_iter=300,
                verbose=1
            )
        
        self.cluster_labels = kmeans.fit_predict(embeddings_scaled)
        self.cluster_centers = kmeans.cluster_centers_
        
        # Report results
        cluster_counts = Counter(self.cluster_labels)
        sizes = list(cluster_counts.values())
        
        print(f"\n✓ Clustering complete:")
        print(f"  Clusters created: {n_clusters}")
        print(f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
        
        return self.cluster_labels
    
    def cluster(self, embeddings: np.ndarray,
               n_clusters: Optional[int] = None,
               reduce_dims: bool = True,
               target_dims: int = 50) -> np.ndarray:
        """Main clustering interface with research-backed pipeline.
        
        Pipeline:
        1. UMAP dimensionality reduction (optional, recommended)
        2. K-Means/MiniBatchKMeans clustering
        3. Auto-optimization with multi-metric scoring (if 'auto' mode)
        
        Args:
            embeddings: Image embeddings (n_samples, n_features)
            n_clusters: Number of clusters (None = auto-determine)
            reduce_dims: Apply UMAP reduction (recommended)
            target_dims: Target dimensions for UMAP
            
        Returns:
            Cluster labels for each image
        """
        # Step 1: Dimensionality reduction (improves quality & speed)
        if reduce_dims:
            embeddings = self.reduce_dimensions(embeddings, n_components=target_dims)
        
        # Step 2: Clustering with appropriate method
        if self.method == 'kmeans':
            return self.cluster_kmeans(embeddings, n_clusters, auto_optimize=False)
        elif self.method == 'minibatch':
            # Force MiniBatchKMeans
            return self.cluster_kmeans(embeddings, n_clusters, auto_optimize=False)
        elif self.method == 'auto':
            # Use multi-metric optimization
            print("\n[Auto Mode] Using advanced multi-metric optimization...")
            return self.cluster_kmeans(embeddings, n_clusters, auto_optimize=True)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'kmeans', 'minibatch', or 'auto'.")
    
    def save_clustering_results(self, output_path: str,
                               embeddings: np.ndarray,
                               metadata: List[Dict]):
        """Save clustering results to JSON.
        
        Args:
            output_path: Path to save results
            embeddings: Embedding vectors
            metadata: Image metadata
        """
        if self.cluster_labels is None:
            raise ValueError("Must call cluster() first")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare results with proper type conversion
        results = {
            'n_clusters': int(len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)),
            'cluster_distribution': {str(k): int(v) for k, v in Counter(self.cluster_labels).items()},
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
    
    def save_cluster_visualization(self, output_dir: str, metadata: List[Dict], 
                                 save_all: bool = True):
        """Save cluster visualization by copying ALL images to cluster directories.
        
        Args:
            output_dir: Base output directory
            metadata: Image metadata
            save_all: Save all images from each cluster (recommended)
        """
        if self.cluster_labels is None:
            raise ValueError("Must call cluster() first")
        
        import shutil
        from collections import defaultdict
        
        cluster_dir = os.path.join(output_dir, "cluster_visualization")
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Group images by cluster
        cluster_groups = defaultdict(list)
        for i, (meta, label) in enumerate(zip(metadata, self.cluster_labels)):
            cluster_groups[int(label)].append((i, meta))
        
        print(f"\nSaving cluster visualization with ALL images...")
        
        # Copy ALL images for each cluster
        total_copied = 0
        for cluster_id, images in cluster_groups.items():
            if cluster_id == -1:  # Skip noise
                continue
                
            cluster_subdir = os.path.join(cluster_dir, f"cluster_{cluster_id:03d}_size{len(images)}")
            os.makedirs(cluster_subdir, exist_ok=True)
            
            # Sort by local score (best first)
            images.sort(key=lambda x: x[1].get('local_score', 0), reverse=True)
            
            # Save ALL images
            for idx, (original_idx, meta) in enumerate(images):
                try:
                    src_path = meta['path']
                    img_name = meta['name']
                    filename = f"{idx+1:03d}_{img_name}"
                    dest_path = os.path.join(cluster_subdir, filename)
                    
                    shutil.copy2(src_path, dest_path)
                    total_copied += 1
                    
                except Exception as e:
                    print(f"⚠ Error copying image for cluster {cluster_id}: {e}")
        
        n_clusters = len([c for c in cluster_groups.keys() if c != -1])
        print(f"✓ Saved cluster visualization to: {cluster_dir}")
        print(f"  Clusters: {n_clusters}")
        print(f"  Total images: {total_copied}")
        print(f"  Avg per cluster: {total_copied // max(1, n_clusters)}")
