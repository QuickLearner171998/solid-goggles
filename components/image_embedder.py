"""Image embedding component supporting CLIP and DINOv2.

Supported models:
- CLIP: Excellent for semantic understanding (vision + text)
- DINOv2: Meta's self-supervised model, excellent for pure visual features
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class ImageEmbedder:
    """Generates embeddings for images using CLIP or DINOv2."""
    
    def __init__(self, model_name: str = 'dinov2-base', device: Optional[str] = None):
        """Initialize the image embedder.
        
        Args:
            model_name: Model to use:
                - 'clip-ViT-B-32': CLIP (vision + text, 512D)
                - 'dinov2-small': DINOv2 small (384D, fast)
                - 'dinov2-base': DINOv2 base (768D, balanced) [RECOMMENDED]
                - 'dinov2-large': DINOv2 large (1024D, best quality)
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
        self.model_name = model_name
        self.model_type = 'dinov2' if 'dinov2' in model_name else 'clip'
        
        print(f"Initializing ImageEmbedder with {self.model_type.upper()} model '{model_name}' on device '{self.device}'")
        
        try:
            if self.model_type == 'dinov2':
                self._init_dinov2(model_name)
            else:
                self._init_clip(model_name)
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            print("Falling back to CLIP ViT-B-32...")
            self._init_clip('clip-ViT-B-32')
    
    def _init_clip(self, model_name: str):
        """Initialize CLIP model."""
        self.model = SentenceTransformer(model_name, device=self.device, use_fast=True)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.model_type = 'clip'
        print(f"✓ CLIP model loaded. Embedding dimension: {self.embedding_dim}")
    
    def _init_dinov2(self, model_name: str):
        """Initialize DINOv2 model."""
        from transformers import AutoImageProcessor, AutoModel
        import os
        
        # DINOv2 has MPS compatibility issues - use CPU fallback
        if self.device == 'mps':
            print("⚠ DINOv2 has MPS compatibility issues (upsample_bicubic2d operator)")
            print("  Enabling CPU fallback for MPS (PYTORCH_ENABLE_MPS_FALLBACK=1)")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            # Note: Model stays on MPS, but unsupported ops will fall back to CPU
        
        # Map short names to full Hugging Face model names
        model_map = {
            'dinov2-small': 'facebook/dinov2-small',
            'dinov2-base': 'facebook/dinov2-base',
            'dinov2-large': 'facebook/dinov2-large',
            'dinov2-giant': 'facebook/dinov2-giant'
        }
        
        full_model_name = model_map.get(model_name, model_name)
        
        print(f"Loading DINOv2 model: {full_model_name}")
        # Use fast processor (future default behavior)
        self.processor = AutoImageProcessor.from_pretrained(full_model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(full_model_name).to(self.device)
        self.model.eval()
        
        # DINOv2 embedding dimensions
        dim_map = {
            'dinov2-small': 384,
            'dinov2-base': 768,
            'dinov2-large': 1024,
            'dinov2-giant': 1536
        }
        self.embedding_dim = dim_map.get(model_name, 768)
        
        print(f"✓ DINOv2 model loaded. Embedding dimension: {self.embedding_dim}")
        print(f"  Note: DINOv2 excels at pure visual features (self-supervised learning)")
    
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
            
            if self.model_type == 'dinov2':
                return self._generate_dinov2_embedding(image)
            else:
                return self._generate_clip_embedding(image)
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector on error
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _generate_clip_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate CLIP embedding."""
        embedding = self.model.encode(image, convert_to_numpy=True)
        return embedding
    
    def _generate_dinov2_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate DINOv2 embedding."""
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Extract CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding
    
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
        if self.model_type == 'dinov2':
            return self._generate_dinov2_embeddings_batch(images, batch_size, show_progress)
        else:
            return self._generate_clip_embeddings_batch(images, batch_size, show_progress)
    
    def _generate_clip_embeddings_batch(self, images: List[Image.Image], 
                                        batch_size: int = 32,
                                        show_progress: bool = True) -> np.ndarray:
        """Generate CLIP embeddings for batch."""
        embeddings = []
        
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="CLIP embeddings", ncols=80)
        
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
        
        return np.vstack(embeddings)
    
    def _generate_dinov2_embeddings_batch(self, images: List[Image.Image], 
                                         batch_size: int = 16,
                                         show_progress: bool = True) -> np.ndarray:
        """Generate DINOv2 embeddings for batch."""
        embeddings = []
        
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="DINOv2 embeddings", ncols=80)
        
        for i in iterator:
            batch = images[i:i + batch_size]
            
            # Convert all images to RGB
            batch_rgb = []
            for img in batch:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                batch_rgb.append(img)
            
            try:
                # Process batch with DINOv2
                inputs = self.processor(images=batch_rgb, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Extract CLS token embeddings (first token from each image)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
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

