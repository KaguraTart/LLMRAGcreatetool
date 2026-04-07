"""
Cascade Classifier
Layer 1: Rules (keyword matching, millisecond-level)
Layer 2: Embedding similarity (fast, no LLM needed)
Layer 3: LLM inference (most accurate, but slowest)
"""

import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    category: str
    confidence: float
    method: str  # rule / embedding / llm


class CascadeClassifier:
    """
    Cascade Classifier
    
    Try from fast to accurate layer by layer:
    Rules → Embedding → LLM
    """
    
    def __init__(
        self,
        taxonomy: dict[str, list[str]],  # category -> keywords
        taxonomy_embeddings: Optional[dict] = None,  # category -> embeddings
        rule_threshold: float = 0.9,
        embedding_threshold: float = 0.7,
    ):
        """
        Args:
            taxonomy: Classification taxonomy, format: {category_name: [keyword_list]}
            taxonomy_embeddings: Embeddings for each category (optional)
            rule_threshold: Confidence threshold for rule-based classification
            embedding_threshold: Confidence threshold for embedding-based classification
        """
        self.taxonomy = taxonomy
        self.taxonomy_embeddings = taxonomy_embeddings
        self.rule_threshold = rule_threshold
        self.embedding_threshold = embedding_threshold
    
    def classify(self, text: str, strategy: str = "cascade") -> ClassificationResult:
        """
        Classify text
        
        Args:
            text: Text to classify
            strategy: cascade / rule / embedding / llm
            
        Returns:
            ClassificationResult
        """
        if strategy == "rule":
            return self._classify_rule(text)
        elif strategy == "embedding":
            return self._classify_embedding(text)
        elif strategy == "llm":
            return self._classify_llm(text)
        else:
            return self._classify_cascade(text)
    
    def _classify_rule(self, text: str) -> ClassificationResult:
        """Layer 1: Keyword rules"""
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.taxonomy.items():
            hits = sum(1 for kw in keywords if kw.lower() in text_lower)
            if hits > 0:
                scores[category] = hits / len(keywords)
        
        if scores:
            best_cat = max(scores, key=scores.get)
            return ClassificationResult(
                category=best_cat,
                confidence=scores[best_cat],
                method="rule"
            )
        
        return ClassificationResult(
            category="unknown",
            confidence=0.0,
            method="rule"
        )
    
    def _classify_embedding(self, text: str) -> ClassificationResult:
        """Layer 2: Embedding similarity"""
        if not self.taxonomy_embeddings:
            return ClassificationResult(
                category="unknown",
                confidence=0.0,
                method="embedding"
            )
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return ClassificationResult(
                category="unknown",
                confidence=0.0,
                method="embedding"
            )
        
        # Get text embedding
        model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        text_emb = model.encode([text], normalize=True)[0]
        
        # Calculate similarity with category keyword embeddings
        best_cat = None
        best_score = -1
        
        for category, cat_embs in self.taxonomy_embeddings.items():
            if not isinstance(cat_embs, np.ndarray):
                continue
            sims = np.dot(text_emb, cat_embs)  # Already normalized
            max_sim = float(sims.max())
            
            if max_sim > best_score:
                best_score = max_sim
                best_cat = category
        
        return ClassificationResult(
            category=best_cat or "unknown",
            confidence=best_score if best_score > 0 else 0.0,
            method="embedding"
        )
    
    def _classify_llm(self, text: str) -> ClassificationResult:
        """Layer 3: LLM inference"""
        # This method requires an external LLM, call it in the pipeline
        return ClassificationResult(
            category="unknown",
            confidence=0.0,
            method="llm"
        )
    
    def _classify_cascade(self, text: str) -> ClassificationResult:
        """Cascade classification"""
        # Layer 1: Rules
        result = self._classify_rule(text)
        if result.confidence >= self.rule_threshold:
            return result
        
        # Layer 2: Embedding
        result = self._classify_embedding(text)
        if result.confidence >= self.embedding_threshold:
            return result
        
        # Layer 3: Return Embedding result (closest match)
        return result
    
    def build_taxonomy_embeddings(self):
        """Precompute taxonomy embeddings (for Embedding layer speedup)"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers not installed")
            return
        
        model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        embeddings = {}
        
        for category, keywords in self.taxonomy.items():
            embs = model.encode(keywords, normalize=True)
            # Take mean as category center
            embeddings[category] = embs.mean(axis=0)
        
        self.taxonomy_embeddings = embeddings
        logger.info(f"Precomputed embeddings for {len(embeddings)} categories")
    
    def batch_classify(
        self, 
        texts: list[str],
        strategy: str = "cascade"
    ) -> list[ClassificationResult]:
        """Batch classification"""
        return [self.classify(t, strategy=strategy) for t in texts]
