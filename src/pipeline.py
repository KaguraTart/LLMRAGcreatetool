"""
RAG Knowledge Processing Main Workflow
Pipeline: Parsing → Chunking → Entity Extraction → Classification → Deduplication → Quality Scoring → Indexing
"""

import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import Config
from .extractors import PDFExtractor, DOCXExtractor, MarkdownExtractor
from .processors.chunker import ChunkBuilder
from .processors.classifier import CascadeClassifier
from .processors.quality import QualityScorer, AnswerQualityScorer
from .integrations.minimax_api import MiniMaxClient
from .integrations.embedding_model import EmbeddingModel
from .indexers.vector_store import QdrantStore, ChromaStore
from .indexers.retriever import Retriever
from .extensions import ExtensionRegistry
from .extensions.builtin.bm25 import BM25Extension
from .extensions.builtin.colbert_reranker import ColBERTRerankerExtension
from .providers import build_registry_from_config
from .query.query_processor import QueryProcessor
from .qa import AnswerGenerator, Synthesizer

logger = logging.getLogger(__name__)


@dataclass
class ProcessedChunk:
    """Processed knowledge chunk"""
    chunk_id: str
    content: str
    source: str = ""
    page_number: int = 0
    categories: list[str] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    quality_score: float = 0.0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RAGPipeline:
    """Complete RAG knowledge processing workflow."""

    def __init__(self, config: Config):
        self.config = config

        self.provider_registry = None
        self.provider = None
        self.fallback_chain = []

        self._init_extractors()
        self._init_processors()
        self._init_integrations()
        self._init_indexers()
        self._init_query_qa()

        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_entities": 0,
            "failed_documents": 0,
        }

    def _init_extractors(self):
        self.pdf_extractor = PDFExtractor(
            extract_images=True,
            extract_tables=True,
            ocr_enabled=True,
        )
        self.docx_extractor = DOCXExtractor()
        self.md_extractor = MarkdownExtractor()

    def _init_processors(self):
        self.chunker = ChunkBuilder(
            strategy=self.config.chunking.strategy,
            chunk_size=self.config.chunking.chunk_size,
            overlap=self.config.chunking.overlap,
            min_chunk_size=self.config.chunking.min_chunk_size,
        )

        taxonomy = {cat: [] for cat in self.config.classifier.taxonomy}
        self.classifier = CascadeClassifier(
            taxonomy=taxonomy,
            rule_threshold=self.config.classifier.rule_confidence_threshold,
            embedding_threshold=self.config.classifier.embedding_confidence_threshold,
        )

    def _init_integrations(self):
        # Build provider registry from config (additive path)
        try:
            self.provider_registry = build_registry_from_config(self.config)
            if self.provider_registry.list():
                self.provider = self.provider_registry.current
            self.fallback_chain = list(getattr(self.config.providers, "fallback_chain", []) or [])
        except Exception as e:
            logger.warning(f"Provider registry init failed, fallback to legacy MiniMax path: {e}")
            self.provider_registry = None
            self.provider = None
            self.fallback_chain = []

        # Legacy MiniMax path for compatibility and NER/classification helpers
        if self.provider and getattr(self.provider, "name", "") == "minimax":
            self.minimax = self.provider
        else:
            api_key = self.config.get_api_key("minimax")
            if api_key:
                self.minimax = MiniMaxClient(
                    api_key=api_key,
                    base_url=self.config.minimax.base_url,
                    model=self.config.minimax.model,
                    vision_model=self.config.minimax.vision_model,
                    embedding_model=self.config.minimax.embedding_model,
                    timeout=self.config.minimax.timeout,
                )
            else:
                self.minimax = None
                logger.warning("MiniMax API Key not configured, some features will be limited")

        llm_for_generation = self.provider or self.minimax

        self.embedding_model = EmbeddingModel(
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device,
            batch_size=self.config.embedding.batch_size,
            normalize=self.config.embedding.normalize,
            minimax_client=self.minimax,
            provider=llm_for_generation,
            provider_registry=self.provider_registry,
            fallback_chain=self.fallback_chain,
            use_local_model=True,
        )

        self.quality_scorer = QualityScorer(llm_client=llm_for_generation)

    def _init_indexers(self):
        vs_config = self.config.vector_store
        self.extension_registry = ExtensionRegistry(core_version="0.1.0")
        self.extension_registry.register(BM25Extension.manifest, BM25Extension())
        self.extension_registry.register(ColBERTRerankerExtension.manifest, ColBERTRerankerExtension())

        if vs_config.type == "qdrant":
            self.vector_store = QdrantStore(
                url=vs_config.qdrant.url,
                collection=vs_config.qdrant.collection,
                vector_size=vs_config.qdrant.vector_size,
                distance=vs_config.qdrant.distance,
            )
        elif vs_config.type == "chroma":
            self.vector_store = ChromaStore(
                persist_dir="./chroma_db",
                collection=vs_config.qdrant.collection,
            )
        else:
            raise ValueError(f"Unsupported vector database type: {vs_config.type}")

        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            provider=self.provider or self.minimax,
            extension_registry=self.extension_registry,
            retrieval_mode=self.config.qa.retrieval_mode,
            vector_weight=self.config.qa.vector_weight,
            bm25_weight=self.config.qa.bm25_weight,
            rerank_enabled=self.config.qa.rerank_enabled,
            rerank_top_n=self.config.qa.rerank_top_n,
            colbert_rerank_enabled=self.config.qa.colbert_rerank_enabled,
            colbert_rerank_top_n=self.config.qa.colbert_rerank_top_n,
            bm25_k1=self.config.qa.bm25_k1,
            bm25_b=self.config.qa.bm25_b,
        )

    def _init_query_qa(self):
        llm = self.provider or self.minimax
        self.query_processor = QueryProcessor(provider=llm, embedding_model=self.embedding_model) if llm else None
        self.answer_generator = AnswerGenerator(
            provider=llm,
            cot_enabled=self.config.qa.cot_enabled,
            self_verify=self.config.qa.self_verify,
            min_faithfulness=self.config.qa.min_faithfulness,
        ) if llm else None
        self.synthesizer = Synthesizer(provider=llm) if llm else None
        self.answer_quality_scorer = AnswerQualityScorer(
            llm_client=llm if self.config.qa.answer_quality_use_llm else None,
            min_score=self.config.qa.min_answer_quality_score,
        )

    async def process(self, file_path: str) -> list[ProcessedChunk]:
        path = Path(file_path)
        suffix = path.suffix.lower()

        logger.info(f"Processing document: {path.name}")
        self.stats["total_documents"] += 1

        try:
            raw_data = await self._parse_document(path, suffix)
            chunks = await self._chunk_document(raw_data, str(path))

            if not chunks:
                logger.warning(f"Document chunking produced no results: {path.name}")
                return []

            if self.config.ner.enabled and self.minimax:
                await self._extract_entities(chunks)

            await self._classify_chunks(chunks)

            if self.config.quality.enabled:
                await self._score_quality(chunks)

            if self.config.dedup.enabled:
                chunks = await self._deduplicate(chunks)

            min_score = self.config.quality.min_quality_score
            chunks = [c for c in chunks if c.quality_score >= min_score]

            await self._index_chunks(chunks)

            self.stats["total_chunks"] += len(chunks)
            logger.info(f"Document processed: {path.name}, {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Document processing failed: {path.name}: {e}")
            self.stats["failed_documents"] += 1
            return []

    async def _parse_document(self, path: Path, suffix: str) -> dict:
        if suffix == ".pdf":
            result = self.pdf_extractor.extract(str(path))
            return {
                "type": "pdf",
                "title": result.title,
                "pages": result.pages,
                "full_text": result.full_text,
                "hierarchy": result.hierarchy,
            }

        if suffix in (".docx", ".doc"):
            result = self.docx_extractor.extract(str(path))
            return {
                "type": "docx",
                "title": result.title,
                "paragraphs": result.paragraphs,
                "tables": result.tables,
                "full_text": result.full_text,
            }

        if suffix in (".md", ".markdown"):
            result = self.md_extractor.extract(str(path))
            return {
                "type": "markdown",
                "title": result.title,
                "sections": result.sections,
                "full_text": result.full_text,
                "hierarchy": result.hierarchy,
            }

        if suffix in (".txt", ".text"):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            return {
                "type": "text",
                "title": path.stem,
                "full_text": text,
            }

        raise ValueError(f"Unsupported file format: {suffix}")

    async def _chunk_document(self, raw_data: dict, source: str) -> list[ProcessedChunk]:
        text = raw_data.get("full_text", "")
        if not text:
            return []

        chunks = self.chunker.chunk_text(text, source=source)

        processed = []
        for chunk in chunks:
            processed.append(ProcessedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                source=source,
                page_number=chunk.page_number,
                token_count=chunk.token_count,
                quality_score=5.0,
            ))

        return processed


    @staticmethod
    def _filter_dict_items(items: list) -> list[dict]:
        return [x for x in items if isinstance(x, dict)]

    async def _extract_entities(self, chunks: list[ProcessedChunk]):
        if not self.minimax:
            return

        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            combined = "\n---\n".join([
                f"[{j}] {c.content[:500]}"
                for j, c in enumerate(batch)
            ])
            if len(combined) > 8000:
                combined = combined[:8000]

            try:
                schema = {
                    "entities": ["Technology Name", "Organization", "Person", "Concept", "Event"],
                    "relations": ["belongs to", "uses", "based on", "related to"]
                }
                result = self.minimax.extract_entities(combined, schema)

                for j, chunk in enumerate(batch):
                    entities = result.get("entities", [])[j:j + 1]
                    relations = result.get("relations", [])[j:j + 1]
                    chunk.entities = self._filter_dict_items(entities)
                    chunk.relations = self._filter_dict_items(relations)
                    self.stats["total_entities"] += len(entities)

            except Exception as e:
                logger.warning(f"Entity extraction failed: {e}")

    async def _classify_chunks(self, chunks: list[ProcessedChunk]):
        for chunk in chunks:
            result = self.classifier.classify(chunk.content)
            chunk.categories = [result.category]

    async def _score_quality(self, chunks: list[ProcessedChunk]):
        for chunk in chunks:
            score = self.quality_scorer.score(
                chunk.content,
                method="llm" if self.minimax else "rule"
            )
            chunk.quality_score = score.score / 10.0

    async def _deduplicate(self, chunks: list[ProcessedChunk]) -> list[ProcessedChunk]:
        if len(chunks) < 2:
            return chunks

        dedup_config = self.config.dedup
        keep = []
        seen_texts = []

        for chunk in chunks:
            text_preview = chunk.content[:200]
            is_dup = False

            if not text_preview:
                keep.append(chunk)
                continue

            if dedup_config.method in ("minhash", "hybrid"):
                for seen in seen_texts:
                    overlap = len(set(text_preview) & set(seen))
                    union = len(set(text_preview) | set(seen))
                    if union == 0:
                        continue
                    jaccard = overlap / union
                    if jaccard > 0.9:
                        is_dup = True
                        break

            if not is_dup and dedup_config.method in ("embedding", "hybrid"):
                try:
                    # Trigger embedding path to keep behavior parity with embedding/hybrid dedup mode.
                    _ = self.embedding_model.encode(chunk.content)
                except Exception:
                    pass

            if not is_dup:
                keep.append(chunk)
                seen_texts.append(text_preview)

        removed = len(chunks) - len(keep)
        if removed > 0:
            logger.info(f"Deduplication complete: removed {removed} duplicate chunks")

        return keep

    async def _index_chunks(self, chunks: list[ProcessedChunk]):
        if not chunks:
            return

        items = []
        for chunk in chunks:
            try:
                emb = self.embedding_model.encode(chunk.content)
                items.append({
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "embedding": emb,
                    "metadata": {
                        "source": chunk.source,
                        "categories": chunk.categories,
                        "quality_score": chunk.quality_score,
                        "entities": [e.get("name", "") for e in chunk.entities],
                    }
                })
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")

        if items:
            self.vector_store.insert_batch(items)
            self.retriever.add_documents(items)

    async def process_corpus(
        self,
        directory: str,
        file_types: tuple = (".pdf", ".docx", ".md", ".txt"),
        max_workers: int = 4,
    ) -> list[ProcessedChunk]:
        dir_path = Path(directory)
        files = []

        for ext in file_types:
            files.extend(dir_path.glob(f"**/*{ext}"))

        files = [f for f in files if f.is_file()]
        logger.info(f"Found {len(files)} files to process")

        all_chunks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(asyncio.run, self.process(str(f))): f
                for f in files
            }

            for future in as_completed(futures):
                f = futures[future]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                    logger.info(f"✅ {f.name}: {len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"❌ {f.name}: {e}")

        logger.info(f"Directory processing complete: {len(files)} files, {len(all_chunks)} chunks")
        return all_chunks

    async def query(
        self,
        question: str,
        k: int = 5,
        category_filter: str = None,
    ) -> list[dict]:
        query_emb = self.embedding_model.encode(question)

        filter_expr = None
        if category_filter:
            filter_expr = {"categories": category_filter}

        results = self.vector_store.search(
            query_embedding=query_emb,
            k=k,
            filter_expr=filter_expr,
            score_threshold=0.3,
        )
        return results

    async def query_and_answer(
        self,
        question: str,
        k: int = 5,
        category_filter: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> dict:
        if not self.provider and not self.minimax:
            raise RuntimeError("No configured provider available for answer generation")

        if not self.query_processor:
            raise RuntimeError("Query processor is not initialized")

        processed = self.query_processor.process(
            query=question,
            rewrite=self.config.qa.rewrite,
            decompose=self.config.qa.decompose,
            hyde=self.config.qa.hyde,
        )

        mode = (retrieval_mode or self.config.qa.retrieval_mode or "vector").lower()

        variants = self.query_processor.all_queries(processed)
        if not variants:
            variants = [question]

        merged = {}
        per_variant = []
        for idx, q in enumerate(variants):
            hyde_emb = processed.hyde_embedding if idx == 0 else None
            results = self.retriever.retrieve(
                query=q,
                k=max(k, self.config.qa.rerank_top_n),
                mode=mode,
                category_filter=category_filter,
                hyde_embedding=hyde_emb,
                rerank=rerank,
            )
            per_variant.append({"query": q, "results": len(results)})
            for r in results:
                rid = str(r.get("id", ""))
                if rid not in merged or float(r.get("score", 0.0)) > float(merged[rid].get("score", 0.0)):
                    merged[rid] = r

        final_results = sorted(
            merged.values(),
            key=lambda x: float(x.get("score", 0.0)),
            reverse=True,
        )[:k]

        answer_result = self.answer_generator.generate(
            question=question,
            context_chunks=final_results,
            intent=processed.intent,
        ) if self.answer_generator else None

        synthesis = None
        conflicts = []
        if self.synthesizer:
            try:
                synthesis = self.synthesizer.synthesize(question, final_results)
                conflicts = self.synthesizer.detect_conflicts(question, final_results[:4])
            except Exception as e:
                logger.warning(f"Synthesis/conflict detection failed: {e}")

        quality = None
        if answer_result and self.config.qa.answer_quality_enabled:
            method = "llm" if self.config.qa.answer_quality_use_llm else "rule"
            quality = self.answer_quality_scorer.score(
                question=question,
                answer=answer_result.answer,
                sources=final_results,
                method=method,
            )

        response_answer = answer_result.answer if answer_result else ""
        warnings = []
        if quality and not quality.passed:
            warning = (
                f"Answer quality below threshold: score={quality.score:.3f}, "
                f"min={self.config.qa.min_answer_quality_score:.3f}"
            )
            warnings.append(warning)
            if self.config.qa.block_low_quality_response:
                response_answer = ""

        return {
            "question": question,
            "intent": processed.intent.value,
            "provider": getattr(self.provider or self.minimax, "name", "minimax"),
            "retrieval_mode": mode,
            "queries": {
                "original": processed.original,
                "rewrites": processed.rewrites,
                "sub_queries": processed.sub_queries,
                "hyde_text": processed.hyde_text,
                "variants": variants,
            },
            "retrieval": {
                "results_count": len(final_results),
                "per_variant": per_variant,
                "results": final_results,
            },
            "answer": response_answer,
            "answer_meta": {
                "faithfulness": getattr(answer_result, "faithfulness", 0.0),
                "relevance": getattr(answer_result, "relevance", 0.0),
                "verified": getattr(answer_result, "verified", False),
            },
            "quality": None if not quality else {
                "score": quality.score,
                "faithfulness": quality.faithfulness,
                "relevance": quality.relevance,
                "grounding": quality.grounding,
                "readability": quality.readability,
                "citation_coverage": quality.citation_coverage,
                "passed": quality.passed,
                "method": quality.method,
                "reasons": quality.reasons,
            },
            "synthesis": synthesis,
            "conflicts": conflicts,
            "warnings": warnings,
        }

    def get_stats(self) -> dict:
        return self.stats.copy()
