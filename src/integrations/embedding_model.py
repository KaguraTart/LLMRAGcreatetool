"""
Embedding Model Wrapper
Supports: HuggingFace sentence-transformers / MiniMax API
"""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Unified Embedding Model Wrapper
    
    Supports:
    - HuggingFace sentence-transformers (local inference, recommended)
    - MiniMax API (cloud, backup)
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = "cuda",
        batch_size: int = 32,
        normalize: bool = True,
        minimax_client=None,  # Optional: MiniMax API client
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.minimax_client = minimax_client
        
        self._model = None
        self._dimension = None
    
    def _load_model(self):
        """Lazy load HuggingFace model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Loaded Embedding model: {self.model_name}, "
                           f"dimension: {self._dimension}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed: pip install sentence-transformers"
                )
            except Exception as e:
                logger.warning(f"Model loading failed: {e}, falling back to MiniMax API")
                self._model = None
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        if self._dimension is None:
            self._load_model()
        if self._dimension is None and self.minimax_client:
            # MiniMax API dynamic fetch
            return 1024  # MiniMax embedding dimension
        return self._dimension or 1024
    
    def encode(
        self,
        texts: str | list[str],
        batch_size: int = None,
        show_progress: bool = False,
        normalize: bool = None,
    ) -> np.ndarray:
        """
        Encode text into embedding vectors
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size (defaults to self.batch_size)
            show_progress: Whether to show progress bar
            normalize: Whether to L2 normalize (defaults to self.normalize)
            
        Returns:
            numpy.ndarray, shape = (n, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        normalize = normalize if normalize is not None else self.normalize
        batch_size = batch_size or self.batch_size
        
        # Prefer local model
        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
            return embeddings
        
        # Fallback: MiniMax API
        if self.minimax_client:
            return self._encode_via_api(texts)
        
        raise RuntimeError("No local model and no MiniMax API client available")
    
    def _encode_via_api(self, texts: list[str]) -> np.ndarray:
        """Get embedding via MiniMax API"""
        if not self.minimax_client:
            raise RuntimeError("MiniMax client not configured")
        
        embeddings = []
        
        for text in texts:
            try:
                import requests
                
                resp = requests.post(
                    f"{self.minimax_client.base_url}/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.minimax_client.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.minimax_client.embedding_model,
                        "input": text,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                
                emb = data["data"][0]["embedding"]
                emb = np.array(emb)
                
                if self.normalize:
                    emb = emb / (np.linalg.norm(emb) + 1e-8)
                
                embeddings.append(emb)
            
            except Exception as e:
                logger.warning(f"MiniMax embedding failed: {e}")
                # Return zero vector
                dim = 1024  # MiniMax default
                embeddings.append(np.zeros(dim))
        
        return np.array(embeddings)
    
    def similarity(
        self,
        text1: str | list[str],
        text2: str | list[str],
    ) -> np.ndarray:
        """
        Calculate cosine similarity between text pairs
        
        Returns:
            Similarity matrix (dot product after normalization)
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        # Normalize
        n1 = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
        n2 = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
        
        return np.dot(n1, n2.T)
    
    def batch_encode(self, chunks: list[dict], text_key: str = "content") -> np.ndarray:
        """
        Batch encode list of Chunk objects
        
        Args:
            chunks: List of Chunk objects (dataclass)
            text_key: Field name to extract text from chunk
        """
        texts = []
        for chunk in chunks:
            if hasattr(chunk, text_key):
                texts.append(getattr(chunk, text_key))
            elif isinstance(chunk, dict):
                texts.append(chunk.get(text_key, ""))
            else:
                texts.append(str(chunk))
        
        return self.encode(texts)
