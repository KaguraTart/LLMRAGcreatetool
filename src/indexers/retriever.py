"""Retriever: vector / BM25 / hybrid + optional rerank."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict

import numpy as np

logger = logging.getLogger(__name__)

RERANK_MAX_PASSAGE_LENGTH = 600  # Keeps rerank prompts bounded while preserving enough passage context.


class Retriever:
    def __init__(
        self,
        vector_store,
        embedding_model,
        provider=None,
        extension_registry=None,
        retrieval_mode: str = "vector",
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rerank_enabled: bool = False,
        rerank_top_n: int = 20,
        colbert_rerank_enabled: bool = False,
        colbert_rerank_top_n: int = 20,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.provider = provider
        self.extension_registry = extension_registry
        self.retrieval_mode = retrieval_mode
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rerank_enabled = rerank_enabled
        self.rerank_top_n = rerank_top_n
        self.colbert_rerank_enabled = colbert_rerank_enabled
        self.colbert_rerank_top_n = colbert_rerank_top_n
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        self._docs_by_id: dict[str, dict] = {}
        self._doc_terms: dict[str, list[str]] = {}
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def add_documents(self, items: list[dict]):
        changed = False
        for item in items:
            cid = str(item.get("chunk_id", ""))
            if not cid:
                continue
            self._docs_by_id[cid] = {
                "id": cid,
                "content": item.get("content", ""),
                "metadata": item.get("metadata", {}),
            }
            self._doc_terms[cid] = self._tokenize(item.get("content", ""))
            changed = True

        if changed:
            self._rebuild_bm25_stats()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        mode: str | None = None,
        category_filter: str | None = None,
        query_embedding=None,
        hyde_embedding=None,
        rerank: bool | None = None,
    ) -> list[dict]:
        mode = (mode or self.retrieval_mode or "vector").lower()
        do_rerank = self.rerank_enabled if rerank is None else rerank

        filter_expr = {"categories": category_filter} if category_filter else None

        vector_results = []
        bm25_results = []

        if mode in ("vector", "hybrid"):
            vector_results = self._vector_search(
                query=query,
                k=max(k, self.rerank_top_n),
                filter_expr=filter_expr,
                query_embedding=query_embedding,
                hyde_embedding=hyde_embedding,
            )

        if mode in ("bm25", "hybrid"):
            bm25_results = self._bm25_search(query=query, k=max(k, self.rerank_top_n))
            if filter_expr:
                allowed = set()
                for doc in bm25_results:
                    cats = doc.get("metadata", {}).get("categories", [])
                    if isinstance(cats, list) and category_filter in cats:
                        allowed.add(doc.get("id"))
                bm25_results = [d for d in bm25_results if d.get("id") in allowed]

        if mode == "vector":
            merged = vector_results
        elif mode == "bm25":
            merged = bm25_results
        else:
            merged = self._hybrid_fuse(vector_results, bm25_results)

        if do_rerank and merged:
            merged = self._rerank(query, merged)
        if self.colbert_rerank_enabled and merged:
            merged = self._colbert_rerank(query, merged)

        return merged[:k]

    def _vector_search(self, query: str, k: int, filter_expr: dict | None, query_embedding=None, hyde_embedding=None):
        emb = self._ensure_vector(query_embedding, query)
        results = self.vector_store.search(
            query_embedding=emb,
            k=k,
            filter_expr=filter_expr,
            score_threshold=0.0,
        )

        if hyde_embedding is not None:
            try:
                hyde_vec = self._ensure_vector(hyde_embedding)
                hyde_results = self.vector_store.search(
                    query_embedding=hyde_vec,
                    k=k,
                    filter_expr=filter_expr,
                    score_threshold=0.0,
                )
                by_id = {str(r.get("id")): r for r in results}
                for item in hyde_results:
                    rid = str(item.get("id"))
                    if rid in by_id:
                        by_id[rid]["score"] = max(
                            float(by_id[rid].get("score", 0.0)),
                            float(item.get("score", 0.0)),
                        )
                    else:
                        by_id[rid] = item
                results = list(by_id.values())
                results.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
            except Exception as e:
                logger.warning(f"HyDE vector search merge failed: {e}")

        return [
            {
                "id": str(r.get("id", "")),
                "content": r.get("content", ""),
                "score": float(r.get("score", 0.0)),
                "metadata": r.get("metadata", {}),
                "retrieval_source": "vector",
            }
            for r in results
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in re.split(r"\W+", (text or "").lower()) if t]

    def _rebuild_bm25_stats(self):
        N = len(self._doc_terms)
        if N == 0:
            self._idf = {}
            self._avgdl = 0.0
            return

        df = Counter()
        total_len = 0
        for terms in self._doc_terms.values():
            total_len += len(terms)
            for t in set(terms):
                df[t] += 1

        self._avgdl = total_len / max(N, 1)
        self._idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def _bm25_search(self, query: str, k: int = 5) -> list[dict]:
        if not self._docs_by_id:
            return []

        q_terms = self._tokenize(query)
        if not q_terms:
            return []

        k1 = self.bm25_k1
        b = self.bm25_b

        scores = {}
        for doc_id, terms in self._doc_terms.items():
            if not terms:
                continue
            tf = Counter(terms)
            dl = len(terms)
            score = 0.0
            for qt in q_terms:
                if qt not in tf:
                    continue
                idf = self._idf.get(qt, 0.0)
                f = tf[qt]
                denom = f + k1 * (1 - b + b * dl / max(self._avgdl, 1e-6))
                score += idf * (f * (k1 + 1) / max(denom, 1e-9))
            if score > 0:
                scores[doc_id] = score

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        output = []
        for doc_id, s in ordered:
            doc = self._docs_by_id.get(doc_id, {})
            output.append({
                "id": doc_id,
                "content": doc.get("content", ""),
                "score": float(s),
                "metadata": doc.get("metadata", {}),
                "retrieval_source": "bm25",
            })
        return output

    def _hybrid_fuse(self, vector_results: list[dict], bm25_results: list[dict]) -> list[dict]:
        by_id = defaultdict(dict)

        v_scores = [float(r.get("score", 0.0)) for r in vector_results]
        b_scores = [float(r.get("score", 0.0)) for r in bm25_results]
        v_min, v_max = (min(v_scores), max(v_scores)) if v_scores else (0.0, 1.0)
        b_min, b_max = (min(b_scores), max(b_scores)) if b_scores else (0.0, 1.0)

        def norm(s, s_min, s_max):
            if s_max <= s_min:
                return 1.0 if s > 0 else 0.0
            return (s - s_min) / (s_max - s_min)

        for item in vector_results:
            rid = str(item.get("id", ""))
            by_id[rid].update(item)
            by_id[rid]["_v"] = norm(float(item.get("score", 0.0)), v_min, v_max)
            by_id[rid].setdefault("_b", 0.0)

        for item in bm25_results:
            rid = str(item.get("id", ""))
            if rid not in by_id:
                by_id[rid].update(item)
            by_id[rid]["_b"] = norm(float(item.get("score", 0.0)), b_min, b_max)
            by_id[rid].setdefault("_v", 0.0)

        output = []
        for rid, item in by_id.items():
            fused = self.vector_weight * float(item.get("_v", 0.0)) + self.bm25_weight * float(item.get("_b", 0.0))
            output.append({
                "id": rid,
                "content": item.get("content", ""),
                "score": fused,
                "metadata": item.get("metadata", {}),
                "retrieval_source": "hybrid",
                "vector_score": float(item.get("_v", 0.0)),
                "bm25_score": float(item.get("_b", 0.0)),
            })

        output.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return output

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not self.provider:
            return candidates

        top = candidates[: self.rerank_top_n]
        prompt = [
            "Given the user query and candidate passages, score each candidate for relevance (0.0-1.0).",
            "Return JSON only: {'scores': [{'id': '...', 'score': 0.0}, ...]}.",
            f"Query: {query}",
            "Candidates:",
        ]
        for c in top:
            prompt.append(f"- id={c.get('id')}: {c.get('content','')[:RERANK_MAX_PASSAGE_LENGTH]}")

        try:
            resp = self.provider.generate("\n".join(prompt), temperature=0.0, max_tokens=800, json_mode=True)
            m = re.search(r"\{[\s\S]*\}", resp)
            if not m:
                return candidates
            data = json.loads(m.group())
            id2score = {str(x.get("id")): float(x.get("score", 0.0)) for x in data.get("scores", [])}

            reranked = []
            for c in candidates:
                rid = str(c.get("id", ""))
                rr = dict(c)
                if rid in id2score:
                    rr["rerank_score"] = id2score[rid]
                    rr["score"] = 0.5 * float(rr.get("score", 0.0)) + 0.5 * id2score[rid]
                reranked.append(rr)
            reranked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
            return reranked
        except Exception as e:
            logger.warning(f"Rerank failed: {e}")
            return candidates

    def _ensure_vector(self, embedding, text: str | None = None):
        if embedding is None:
            if text is None:
                raise ValueError("Either embedding or text must be provided")
            embedding = self.embedding_model.encode(text)

        arr = np.array(embedding)
        if arr.ndim == 2:
            if arr.shape[0] == 0:
                raise ValueError("Empty embedding")
            arr = arr[0]
        return arr

    def _colbert_rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        if not self.extension_registry:
            return candidates
        rerankers = self.extension_registry.by_capability("reranker")
        if not rerankers:
            return candidates
        reranker = rerankers[0]
        rerank_fn = getattr(reranker, "rerank", None)
        if not callable(rerank_fn):
            return candidates
        try:
            return rerank_fn(query=query, candidates=candidates, top_n=self.colbert_rerank_top_n)
        except Exception as e:
            logger.warning(f"ColBERT rerank failed: {e}")
            return candidates
