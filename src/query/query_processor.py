"""
Query Intelligence Module

Features:
- Query rewriting (3 LLM variants) → expanded retrieval recall
- Query decomposition (compound → atomic sub-queries)
- HyDE (Hypothetical Document Embedding) — generate a fake answer, embed it
- Intent classification (factual / comparative / procedural / summarization)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    SUMMARIZATION = "summarization"
    UNKNOWN = "unknown"


@dataclass
class ProcessedQuery:
    original: str
    intent: QueryIntent = QueryIntent.UNKNOWN
    rewrites: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    hyde_text: str = ""
    hyde_embedding: Optional[np.ndarray] = None


class QueryProcessor:
    """
    Transforms a raw user query into an enriched ProcessedQuery
    ready for hybrid retrieval.
    """

    def __init__(self, provider, embedding_model=None):
        """
        Args:
            provider: LLMProvider instance for rewriting / HyDE
            embedding_model: EmbeddingModel instance for HyDE embedding
        """
        self.provider = provider
        self.embedding_model = embedding_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        query: str,
        rewrite: bool = True,
        decompose: bool = True,
        hyde: bool = True,
    ) -> ProcessedQuery:
        """
        Full query processing pipeline.

        Returns a ProcessedQuery with intent, rewrites, sub-queries, and HyDE.
        """
        result = ProcessedQuery(original=query)

        result.intent = self._classify_intent(query)

        if rewrite:
            result.rewrites = self._rewrite(query)

        if decompose:
            result.sub_queries = self._decompose(query)

        if hyde:
            result.hyde_text = self._hyde_generate(query)
            if self.embedding_model and result.hyde_text:
                try:
                    result.hyde_embedding = self.embedding_model.encode(result.hyde_text)
                except Exception as e:
                    logger.warning(f"HyDE embedding failed: {e}")

        return result

    def all_queries(self, processed: ProcessedQuery) -> list[str]:
        """Return all query variants (original + rewrites + sub-queries)."""
        seen = set()
        queries = []
        for q in [processed.original] + processed.rewrites + processed.sub_queries:
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                queries.append(q)
        return queries

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent using simple heuristics + LLM fallback."""
        q_lower = query.lower()

        # Heuristic rules (fast, no LLM call)
        if any(w in q_lower for w in ["how to", "steps to", "procedure", "process",
                                       "how do i", "how can i"]):
            return QueryIntent.PROCEDURAL
        if any(w in q_lower for w in ["compare", "difference", "vs", "versus",
                                       "better", "pros and cons"]):
            return QueryIntent.COMPARATIVE
        if any(w in q_lower for w in ["summarize", "summary", "overview", "explain"]):
            return QueryIntent.SUMMARIZATION

        # LLM classification
        try:
            prompt = (
                f"Classify the intent of this query into one of: "
                f"factual, comparative, procedural, summarization.\n"
                f"Query: {query}\n"
                f"Output only the intent word."
            )
            response = self.provider.generate(prompt, temperature=0, max_tokens=10)
            intent_str = response.strip().lower()
            return QueryIntent(intent_str) if intent_str in QueryIntent._value2member_map_ else QueryIntent.FACTUAL
        except Exception:
            return QueryIntent.FACTUAL

    # ------------------------------------------------------------------
    # Query rewriting
    # ------------------------------------------------------------------

    def _rewrite(self, query: str) -> list[str]:
        """Generate 3 alternative phrasings of the query."""
        prompt = (
            f"Generate 3 alternative phrasings of this search query to improve retrieval.\n"
            f"Original query: {query}\n\n"
            f"Output exactly 3 variants, one per line, no numbering or bullets."
        )
        try:
            response = self.provider.generate(prompt, temperature=0.7, max_tokens=200)
            lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
            # Remove leading numbers/bullets
            lines = [re.sub(r"^[\d\.\-\*\)]+\s*", "", l) for l in lines]
            return lines[:3]
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Query decomposition
    # ------------------------------------------------------------------

    def _decompose(self, query: str) -> list[str]:
        """Split a compound query into atomic sub-queries."""
        prompt = (
            f"Decompose this question into simpler atomic sub-questions "
            f"(return empty list if already simple).\n"
            f"Question: {query}\n\n"
            f"Output one sub-question per line. If the question is already simple, "
            f"output only the original question."
        )
        try:
            response = self.provider.generate(prompt, temperature=0.3, max_tokens=300)
            lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
            lines = [re.sub(r"^[\d\.\-\*\)]+\s*", "", l) for l in lines]
            # Filter out lines that are essentially the original
            sub = [l for l in lines if l.lower() != query.lower()]
            return sub[:5]  # Max 5 sub-queries
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return []

    # ------------------------------------------------------------------
    # HyDE — Hypothetical Document Embedding
    # ------------------------------------------------------------------

    def _hyde_generate(self, query: str) -> str:
        """
        Generate a hypothetical answer document for HyDE.
        The embedding of this fake answer is used alongside the query embedding
        for retrieval, significantly improving recall for complex questions.
        """
        prompt = (
            f"Write a short, factual passage (2-4 sentences) that would perfectly "
            f"answer the following question. Write as if it is an excerpt from a "
            f"knowledge base document.\n\nQuestion: {query}\n\nPassage:"
        )
        try:
            return self.provider.generate(prompt, temperature=0.5, max_tokens=200)
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return ""
