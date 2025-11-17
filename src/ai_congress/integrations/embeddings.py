"""
Embeddings Module
Provides text embedding generation using sentence-transformers
"""
import logging
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using sentence-transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cuda' or 'cpu'). Auto-detect if None
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Loading embedding model: {model_name} on {self.device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        try:
            # Convert single string to list
            if isinstance(texts, str):
                texts = [texts]
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization for cosine similarity
            )
            
            logger.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            numpy array of embedding (shape: [embedding_dim])
        """
        embeddings = self.generate_embeddings(text)
        return embeddings[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.embedding_dimension
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0 to 1)
        """
        embeddings = self.generate_embeddings([text1, text2])
        
        # Cosine similarity (embeddings are already normalized)
        similarity = np.dot(embeddings[0], embeddings[1])
        
        return float(similarity)
    
    def batch_similarities(self, query: str, texts: List[str]) -> List[float]:
        """
        Calculate similarities between a query and multiple texts
        
        Args:
            query: Query text
            texts: List of texts to compare
            
        Returns:
            List of similarity scores
        """
        # Generate embeddings
        query_embedding = self.generate_embedding(query)
        text_embeddings = self.generate_embeddings(texts)
        
        # Calculate cosine similarities
        similarities = np.dot(text_embeddings, query_embedding)
        
        return similarities.tolist()


# Global singleton instance for reuse
_embedding_generator = None


def get_embedding_generator(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingGenerator:
    """
    Get or create singleton embedding generator
    
    Args:
        model_name: Name of the sentence-transformers model
        
    Returns:
        EmbeddingGenerator instance
    """
    global _embedding_generator
    
    if _embedding_generator is None or _embedding_generator.model_name != model_name:
        _embedding_generator = EmbeddingGenerator(model_name)
    
    return _embedding_generator

