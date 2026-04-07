"""
Vector Database Wrapper
Supports: Qdrant (recommended) / Milvus / Chroma
"""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class VectorStoreBase:
    """Vector database base class"""
    
    def insert(self, chunk_id: str, content: str, embedding: np.ndarray, metadata: dict = None):
        raise NotImplementedError
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_expr: dict = None,
    ) -> list[dict]:
        raise NotImplementedError
    
    def delete(self, chunk_id: str):
        raise NotImplementedError


class QdrantStore(VectorStoreBase):
    """
    Qdrant Vector Database Wrapper
    
    Features:
    - High-performance HNSW index
    - Supports metadata filtering
    - Supports score rescore
    - Lightweight deployment (standalone / Docker)
    """
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "rag_knowledge_base",
        vector_size: int = 1024,
        distance: str = "Cosine",
        recreate: bool = False,
    ):
        self.url = url
        self.collection = collection
        self.vector_size = vector_size
        self.distance = distance
        self._client = None
        self._collection_obj = None
    
    def _get_client(self):
        """Lazy connection"""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(url=self.url)
                self._ensure_collection()
                logger.info(f"Connected to Qdrant: {self.url}, collection: {self.collection}")
            except ImportError:
                raise ImportError(
                    "qdrant-client not installed: pip install qdrant-client"
                )
        return self._client
    
    def _ensure_collection(self):
        """Ensure collection exists"""
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams, OptimizersConfig2_0
        )
        
        client = self._client
        
        # Check if it exists
        try:
            collections = client.get_collections().collections
            exists = any(c.name == self.collection for c in collections)
        except:
            exists = False
        
        if not exists:
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Dot": Distance.DOT,
            }
            
            client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=distance_map.get(self.distance, Distance.COSINE),
                ),
                optimizer_config=OptimizersConfig2_0(
                    indexing_threshold=20000,  # Start indexing after 20000 vectors
                )
            )
            logger.info(f"Created collection: {self.collection}")
    
    @property
    def collection_obj(self):
        if self._collection_obj is None:
            from qdrant_client.models import Collection
            client = self._get_client()
            self._collection_obj = Collection(
                client=client,
                collection_name=self.collection
            )
        return self._collection_obj
    
    def insert(
        self,
        chunk_id: str,
        content: str,
        embedding: np.ndarray,
        metadata: dict = None,
    ):
        """Insert single vector"""
        client = self._get_client()
        
        from qdrant_client.models import PointStruct, Payload
        
        point = PointStruct(
            id=chunk_id,
            vector=embedding.tolist(),
            payload={
                "content": content,
                **(metadata or {})
            }
        )
        
        client.upsert(
            collection_name=self.collection,
            points=[point]
        )
    
    def insert_batch(
        self,
        items: list[dict],
    ):
        """
        Batch insert
        
        items: [{"chunk_id": str, "content": str, "embedding": np.ndarray, "metadata": dict}]
        """
        client = self._get_client()
        
        from qdrant_client.models import PointStruct
        
        points = []
        for item in items:
            points.append(PointStruct(
                id=item["chunk_id"],
                vector=item["embedding"].tolist(),
                payload={
                    "content": item["content"],
                    **(item.get("metadata") or {})
                }
            ))
        
        client.upsert(
            collection_name=self.collection,
            points=points
        )
        
        logger.info(f"Batch inserted {len(points)} vectors")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_expr: dict = None,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Vector search
        
        Returns:
            [{"id": str, "content": str, "score": float, "metadata": dict}, ...]
        """
        client = self._get_client()
        
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
        
        # Build filter conditions
        q_filter = None
        if filter_expr:
            must_clauses = []
            for key, value in filter_expr.items():
                if isinstance(value, (int, float)):
                    must_clauses.append(
                        FieldCondition(
                            key=f"payload.{key}",
                            range=Range(gte=value, lte=value)
                        )
                    )
                elif isinstance(value, str):
                    must_clauses.append(
                        FieldCondition(
                            key=f"payload.{key}",
                            match=MatchValue(value=value)
                        )
                    )
            if must_clauses:
                q_filter = Filter(must=must_clauses)
        
        results = client.search(
            collection_name=self.collection,
            query_vector=query_embedding.tolist(),
            limit=k,
            query_filter=q_filter,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        
        return [
            {
                "id": r.id,
                "content": r.payload.get("content", ""),
                "score": r.score,
                "metadata": {k: v for k, v in r.payload.items() if k != "content"}
            }
            for r in results
        ]
    
    def delete(self, chunk_id: str):
        """Delete vector"""
        client = self._get_client()
        
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(
                    key="id",
                    match=MatchValue(value=chunk_id)
                )]
            )
        )
    
    def count(self) -> int:
        """Return total vector count"""
        client = self._get_client()
        info = client.get_collection(self.collection)
        return info.vectors_count


class ChromaStore(VectorStoreBase):
    """
    Chroma Vector Database Wrapper (lightweight, standalone)
    
    Suitable for: local development / small-scale knowledge base (< 100k vectors)
    """
    
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection: str = "rag",
        embedding_model=None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection
        self.embedding_model = embedding_model
        self._client = None
        self._collection = None
    
    def _get_collection(self):
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                self._client = chromadb.PersistentClient(
                    path=self.persist_dir,
                    settings=Settings(anonymized_telemetry=False)
                )
                
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "RAG Knowledge Base"}
                )
                
                logger.info(f"Chroma collection: {self.collection_name}")
            
            except ImportError:
                raise ImportError("chroma not installed: pip install chromadb")
        
        return self._collection
    
    def insert(
        self,
        chunk_id: str,
        content: str,
        embedding: np.ndarray,
        metadata: dict = None,
    ):
        collection = self._get_collection()
        
        collection.add(
            ids=[chunk_id],
            embeddings=[embedding.tolist()],
            documents=[content],
            metadatas=[metadata or {}]
        )
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_expr: dict = None,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        collection = self._get_collection()

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=filter_expr,
            include=["documents", "metadatas", "distances"]
        )
        
        output = []
        for i in range(len(results["ids"][0])):
            score = 1.0 - results["distances"][0][i]  # Chroma stores distance
            if score < score_threshold:
                continue
            output.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "score": score,
                "metadata": results["metadatas"][0][i]
            })

        return output
    
    def insert_batch(self, items: list[dict]):
        collection = self._get_collection()
        
        ids = [item["chunk_id"] for item in items]
        embeddings = [item["embedding"].tolist() for item in items]
        documents = [item["content"] for item in items]
        metadatas = [item.get("metadata", {}) for item in items]
        
        collection.add(ids=ids, embeddings=embeddings,
                     documents=documents, metadatas=metadatas)
    
    def count(self) -> int:
        return self._get_collection().count()
