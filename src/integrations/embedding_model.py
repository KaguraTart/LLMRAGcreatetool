"""
Embedding Model Wrapper
Supports: HuggingFace sentence-transformers / provider embedding APIs
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Unified Embedding Model Wrapper

    Supports:
    - HuggingFace sentence-transformers (local inference)
    - Provider embedding API via active provider / fallback chain
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = "cuda",
        batch_size: int = 32,
        normalize: bool = True,
        minimax_client=None,  # legacy alias
        provider=None,
        provider_registry=None,
        fallback_chain: list[str] | None = None,
        use_local_model: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize

        self.provider = provider or minimax_client
        self.provider_registry = provider_registry
        self.fallback_chain = fallback_chain or []
        self.use_local_model = use_local_model

        self._model = None
        self._dimension = None

    def _load_model(self):
        """Lazy load HuggingFace model"""
        if self._model is None and self.use_local_model:
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
                logger.warning("sentence-transformers not installed")
                self._model = None
            except Exception as e:
                logger.warning(f"Model loading failed: {e}, falling back to provider embedding")
                self._model = None

    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        if self._dimension is None:
            self._load_model()
        if self._dimension is None:
            return 1024
        return self._dimension

    def encode(
        self,
        texts: str | list[str],
        batch_size: int = None,
        show_progress: bool = False,
        normalize: bool = None,
    ) -> np.ndarray:
        """
        Encode text into embedding vectors.

        Return shape:
        - input str -> (dim,)
        - input list[str] -> (n, dim)
        """
        is_single = isinstance(texts, str)
        if is_single:
            batch = [texts]
        else:
            batch = texts

        normalize = normalize if normalize is not None else self.normalize
        batch_size = batch_size or self.batch_size

        self._load_model()

        # Prefer local model
        if self._model is not None:
            embeddings = self._model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
            embeddings = np.asarray(embeddings)
            if is_single:
                return embeddings[0]
            return embeddings

        # Fallback: provider embedding API (active provider)
        provider_embeddings = self._encode_via_provider(batch)
        if provider_embeddings is not None:
            provider_embeddings = np.asarray(provider_embeddings)
            if is_single:
                return provider_embeddings[0]
            return provider_embeddings

        raise RuntimeError("No local embedding model and no provider embedding backend available")

    def _encode_via_provider(self, texts: list[str]) -> np.ndarray | None:
        provider = self.provider

        if provider is not None:
            try:
                if getattr(provider, "supports_embedding", False):
                    emb = provider.embed(texts)
                    return self._normalize_if_needed(np.asarray(emb))
            except Exception as e:
                logger.warning(f"Provider embedding failed: {e}")

        if self.provider_registry and self.fallback_chain:
            for name in self.fallback_chain:
                try:
                    p = self.provider_registry.get(name)
                    if not getattr(p, "supports_embedding", False):
                        continue
                    emb = p.embed(texts)
                    return self._normalize_if_needed(np.asarray(emb))
                except Exception as e:
                    logger.warning(f"Fallback embedding failed for {name}: {e}")

        return None

    def _normalize_if_needed(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return embeddings
        if embeddings.ndim == 1:
            denom = np.linalg.norm(embeddings) + 1e-8
            return embeddings / denom
        denom = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        return embeddings / denom

    def similarity(
        self,
        text1: str | list[str],
        text2: str | list[str],
    ) -> np.ndarray:
        """Calculate cosine similarity between text pairs."""
        emb1 = np.array(self.encode(text1))
        emb2 = np.array(self.encode(text2))

        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)

        n1 = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
        n2 = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)

        return np.dot(n1, n2.T)

    def batch_encode(self, chunks: list[dict], text_key: str = "content") -> np.ndarray:
        """Batch encode list of Chunk objects."""
        texts = []
        for chunk in chunks:
            if hasattr(chunk, text_key):
                texts.append(getattr(chunk, text_key))
            elif isinstance(chunk, dict):
                texts.append(chunk.get(text_key, ""))
            else:
                texts.append(str(chunk))

        return self.encode(texts)
