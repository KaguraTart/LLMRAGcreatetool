"""Lightweight ColBERT-style lexical reranker extension."""

from __future__ import annotations

import re
from collections import Counter

from src.extensions.base import ExtensionManifest, RerankerExtension


class ColBERTRerankerExtension(RerankerExtension):
    manifest = ExtensionManifest(
        name="colbert-reranker",
        version="1.0.0",
        entry_point="src.extensions.builtin.colbert_reranker:ColBERTRerankerExtension",
        capabilities=["reranker"],
        enabled=True,
    )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in re.split(r"\W+", (text or "").lower()) if t]

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        q = Counter(self._tokenize(query))
        if not q:
            return candidates

        rescored = []
        for item in candidates:
            content = str(item.get("content", ""))
            d = Counter(self._tokenize(content))
            overlap = sum(min(q[t], d[t]) for t in q)
            normalizer = max(sum(q.values()), 1)
            lexical = overlap / normalizer

            score = 0.5 * float(item.get("score", 0.0)) + 0.5 * lexical
            updated = dict(item)
            updated["colbert_score"] = lexical
            updated["score"] = score
            rescored.append(updated)

        rescored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return rescored[: max(top_n, 1)]
