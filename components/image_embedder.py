"""Image embedding component using CLIP for feature extraction."""

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class ImageEmbedder:
    """Generates embeddings for images using CLIP model."""
    
    def __init__(self, model_name: str = 'clip-ViT-B-32', device: Optional[str] = None):
        """Initialize the image embedder.
        
        Args:
            model_name: Name of the CLIP model to use
            device: Device to run the model on ('cuda', 'mps', or 'cpu')
        """
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        print(f"Initializing ImageEmbedder with model '{model_name}' on device '{self.device}'")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✓ Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            print("Falling back to basic CLIP model...")
            self.model = SentenceTransformer('clip-ViT-B-32', device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def generate_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate embedding for a single image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Numpy array of embeddings
        """
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate embedding
            embedding = self.model.encode(image, convert_to_numpy=True)
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector on error
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def generate_embeddings_batch(self, images: List[Image.Image], 
                                  batch_size: int = 32,
                                  show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a batch of images.
        
        Args:
            images: List of PIL Image objects
            batch_size: Number of images to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of shape (num_images, embedding_dim)
        """
        embeddings = []
        
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings", ncols=80)
        
        for i in iterator:
            batch = images[i:i + batch_size]
            
            # Convert all images to RGB
            batch_rgb = []
            for img in batch:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                batch_rgb.append(img)
            
            try:
                # Generate embeddings for batch
                batch_embeddings = self.model.encode(batch_rgb, 
                                                    convert_to_numpy=True,
                                                    batch_size=len(batch_rgb))
                embeddings.append(batch_embeddings)
            except Exception as e:
                print(f"\n⚠ Error processing batch at index {i}: {e}")
                # Add zero vectors for failed batch
                embeddings.append(np.zeros((len(batch), self.embedding_dim), dtype=np.float32))
        
        # Concatenate all batches
        all_embeddings = np.vstack(embeddings)
        return all_embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, 
                       metadata: List[Dict],
                       output_path: str):
        """Save embeddings and metadata to disk.
        
        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata dicts (same order as embeddings)
            output_path: Path to save the embeddings
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as compressed numpy file
        np.savez_compressed(
            output_path,
            embeddings=embeddings,
            metadata=np.array(metadata, dtype=object)
        )
        
        print(f"✓ Saved embeddings to: {output_path}")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    @staticmethod
    def load_embeddings(input_path: str) -> tuple:
        """Load embeddings and metadata from disk.
        
        Args:
            input_path: Path to the saved embeddings file
            
        Returns:
            Tuple of (embeddings, metadata)
        """
        data = np.load(input_path, allow_pickle=True)
        embeddings = data['embeddings']
        metadata = data['metadata'].tolist()
        
        print(f"✓ Loaded embeddings from: {input_path}")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Number of images: {len(metadata)}")
        
        return embeddings, metadata
    
    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0 to 1)
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Clip to [0, 1] range
        return max(0.0, min(1.0, (similarity + 1.0) / 2.0))
    
    def find_similar_images(self, query_embedding: np.ndarray,
                           all_embeddings: np.ndarray,
                           top_k: int = 10) -> List[int]:
        """Find most similar images to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            all_embeddings: Array of all embeddings to search
            top_k: Number of top similar images to return
            
        Returns:
            List of indices of most similar images
        """
        # Compute similarities
        similarities = []
        for i, emb in enumerate(all_embeddings):
            sim = self.compute_similarity(query_embedding, emb)
            similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k indices
        return [idx for idx, _ in similarities[:top_k]]

