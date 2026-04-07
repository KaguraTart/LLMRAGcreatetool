"""
RAG Knowledge Processing Main Workflow
Pipeline: Parsing → Chunking → Entity Extraction → Classification → Deduplication → Quality Scoring → Indexing
"""

import os
import asyncio
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import Config
from .extractors import PDFExtractor, DOCXExtractor, MarkdownExtractor
from .processors.chunker import ChunkBuilder, Chunk
from .processors.classifier import CascadeClassifier
from .processors.quality import QualityScorer
from .integrations.minimax_api import MiniMaxClient
from .integrations.embedding_model import EmbeddingModel
from .indexers.vector_store import QdrantStore, ChromaStore

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
    """
    Complete RAG knowledge processing workflow
    
    Usage example:
    
    ```python
    config = Config.from_yaml("config.yaml")
    pipeline = RAGPipeline(config)
    
    # Process a document
    chunks = await pipeline.process("document.pdf")
    
    # Batch process a directory
    all_chunks = await pipeline.process_corpus("./knowledge_base/")
    
    # Query
    results = await pipeline.query("related question")
    ```
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize modules
        self._init_extractors()
        self._init_processors()
        self._init_integrations()
        self._init_indexers()
        
        # Statistics
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_entities": 0,
            "failed_documents": 0,
        }
    
    def _init_extractors(self):
        """Initialize parsers"""
        self.pdf_extractor = PDFExtractor(
            extract_images=True,
            extract_tables=True,
            ocr_enabled=True,
        )
        self.docx_extractor = DOCXExtractor()
        self.md_extractor = MarkdownExtractor()
    
    def _init_processors(self):
        """Initialize processors"""
        self.chunker = ChunkBuilder(
            strategy=self.config.chunking.strategy,
            chunk_size=self.config.chunking.chunk_size,
            overlap=self.config.chunking.overlap,
            min_chunk_size=self.config.chunking.min_chunk_size,
        )
        
        # Build classifier
        taxonomy = {cat: [] for cat in self.config.classifier.taxonomy}
        self.classifier = CascadeClassifier(
            taxonomy=taxonomy,
            rule_threshold=self.config.classifier.rule_confidence_threshold,
            embedding_threshold=self.config.classifier.embedding_confidence_threshold,
        )
    
    def _init_integrations(self):
        """Initialize integration components"""
        # MiniMax API client
        api_key = self.config.get_api_key("minimax")
        if api_key:
            self.minimax = MiniMaxClient(
                api_key=api_key,
                base_url=self.config.minimax.base_url,
                model=self.config.minimax.model,
                vision_model=self.config.minimax.vision_model,
                timeout=self.config.minimax.timeout,
            )
        else:
            self.minimax = None
            logger.warning("MiniMax API Key not configured, some features will be limited")
        
        # Embedding model
        try:
            self.embedding_model = EmbeddingModel(
                model_name=self.config.embedding.model_name,
                device=self.config.embedding.device,
                batch_size=self.config.embedding.batch_size,
                normalize=self.config.embedding.normalize,
                minimax_client=self.minimax,
            )
        except ImportError:
            logger.warning("sentence-transformers not installed, using MiniMax API")
            self.embedding_model = EmbeddingModel(
                minimax_client=self.minimax
            )
        
        # Quality scorer
        self.quality_scorer = QualityScorer(llm_client=self.minimax)
    
    def _init_indexers(self):
        """Initialize index storage"""
        vs_config = self.config.vector_store
        
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
    
    async def process(self, file_path: str) -> list[ProcessedChunk]:
        """
        Process a single document
        
        Args:
            file_path: Document path (PDF / Word / Markdown)
            
        Returns:
            List of processed chunks
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        logger.info(f"Processing document: {path.name}")
        self.stats["total_documents"] += 1
        
        try:
            # Step 1: Parse
            raw_data = await self._parse_document(path, suffix)
            
            # Step 2: Chunk
            chunks = await self._chunk_document(raw_data, str(path))
            
            if not chunks:
                logger.warning(f"Document chunking produced no results: {path.name}")
                return []
            
            # Step 3: Entity + relation extraction (optional)
            if self.config.ner.enabled and self.minimax:
                await self._extract_entities(chunks)
            
            # Step 4: Classification (cascade)
            await self._classify_chunks(chunks)
            
            # Step 5: Quality scoring (optional)
            if self.config.quality.enabled:
                await self._score_quality(chunks)
            
            # Step 6: Deduplication (optional)
            if self.config.dedup.enabled:
                chunks = await self._deduplicate(chunks)
            
            # Step 7: Filter low quality
            min_score = self.config.quality.min_quality_score
            chunks = [c for c in chunks 
                     if c.quality_score >= min_score]
            
            # Step 8: Index
            await self._index_chunks(chunks)
            
            self.stats["total_chunks"] += len(chunks)
            logger.info(f"Document processed: {path.name}, {len(chunks)} chunks")
            
            return chunks
        
        except Exception as e:
            logger.error(f"Document processing failed: {path.name}: {e}")
            self.stats["failed_documents"] += 1
            return []
    
    async def _parse_document(self, path: Path, suffix: str) -> dict:
        """Parse document"""
        if suffix == ".pdf":
            result = self.pdf_extractor.extract(str(path))
            return {
                "type": "pdf",
                "title": result.title,
                "pages": result.pages,
                "full_text": result.full_text,
                "hierarchy": result.hierarchy,
            }
        
        elif suffix in (".docx", ".doc"):
            result = self.docx_extractor.extract(str(path))
            return {
                "type": "docx",
                "title": result.title,
                "paragraphs": result.paragraphs,
                "tables": result.tables,
                "full_text": result.full_text,
            }
        
        elif suffix in (".md", ".markdown"):
            result = self.md_extractor.extract(str(path))
            return {
                "type": "markdown",
                "title": result.title,
                "sections": result.sections,
                "full_text": result.full_text,
                "hierarchy": result.hierarchy,
            }
        
        elif suffix in (".txt", ".text"):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            return {
                "type": "text",
                "title": path.stem,
                "full_text": text,
            }
        
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    async def _chunk_document(self, raw_data: dict, source: str) -> list[ProcessedChunk]:
        """Chunk document"""
        text = raw_data.get("full_text", "")
        
        if not text:
            return []
        
        chunks = self.chunker.chunk_text(text, source=source)
        
        # Build ProcessedChunk
        processed = []
        for chunk in chunks:
            processed.append(ProcessedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                source=source,
                page_number=chunk.page_number,
                token_count=chunk.token_count,
                quality_score=5.0,  # Default score
            ))
        
        return processed
    
    async def _extract_entities(self, chunks: list[ProcessedChunk]):
        """Entity + relation extraction"""
        if not self.minimax:
            return
        
        # Batch extraction (10 chunks per batch)
        batch_size = 10
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Combine text
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
                    # Simple distribution (one part per chunk)
                    entities = result.get("entities", [])[j:j+1]
                    relations = result.get("relations", [])[j:j+1]
                    chunk.entities = entities
                    chunk.relations = relations
                    
                    self.stats["total_entities"] += len(entities)
            
            except Exception as e:
                logger.warning(f"Entity extraction failed: {e}")
    
    async def _classify_chunks(self, chunks: list[ProcessedChunk]):
        """Batch classification"""
        for chunk in chunks:
            result = self.classifier.classify(chunk.content)
            chunk.categories = [result.category]
    
    async def _score_quality(self, chunks: list[ProcessedChunk]):
        """Quality scoring"""
        for chunk in chunks:
            score = self.quality_scorer.score(
                chunk.content,
                method="llm" if self.minimax else "rule"
            )
            chunk.quality_score = score.score / 10.0  # Normalize to 0-1
    
    async def _deduplicate(self, chunks: list[ProcessedChunk]) -> list[ProcessedChunk]:
        """
        Two-layer deduplication: MinHash literal + Embedding semantic
        
        Returns:
            Deduplicated chunks
        """
        if len(chunks) < 2:
            return chunks
        
        dedup_config = self.config.dedup
        keep = []
        seen_texts = []
        
        for chunk in chunks:
            text_preview = chunk.content[:200]
            
            is_dup = False
            
            # Layer 1: MinHash (literal similarity)
            if dedup_config.method in ("minhash", "hybrid"):
                for seen in seen_texts:
                    # Simple character overlap
                    overlap = len(set(text_preview) & set(seen)) 
                    jaccard = overlap / len(set(text_preview) | set(seen))
                    if jaccard > 0.9:
                        is_dup = True
                        break
            
            # Layer 2: Embedding semantic (if needed)
            if not is_dup and dedup_config.method in ("embedding", "hybrid"):
                try:
                    emb = self.embedding_model.encode(chunk.content)
                    
                    for seen_emb, seen_text in zip(seen_texts, seen_texts):
                        if seen_emb is None:
                            continue
                        # Existing embedding
                    
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
        """Index to vector database"""
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
                        "entities": [e["name"] for e in chunk.entities],
                    }
                })
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")
        
        if items:
            self.vector_store.insert_batch(items)
    
    async def process_corpus(
        self,
        directory: str,
        file_types: tuple = (".pdf", ".docx", ".md", ".txt"),
        max_workers: int = 4,
    ) -> list[ProcessedChunk]:
        """
        Process entire directory in parallel
        
        Args:
            directory: Directory path
            file_types: File types to process
            max_workers: Maximum concurrency
            
        Returns:
            All processed chunks
        """
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
        
        logger.info(f"Directory processing complete: {len(files)} files, "
                    f"{len(all_chunks)} chunks")
        
        return all_chunks
    
    async def query(
        self,
        question: str,
        k: int = 5,
        category_filter: str = None,
    ) -> list[dict]:
        """
        Retrieve relevant chunks
        
        Args:
            question: Query question
            k: Number of results to return
            category_filter: Optional, filter by category
            
        Returns:
            List of retrieval results
        """
        # Vector search
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
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        return self.stats.copy()
