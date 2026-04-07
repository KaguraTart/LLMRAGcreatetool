"""
Intelligent Text Chunker
Supports: fixed-length / recursive character / semantic / heading-aware four strategies
"""

import logging
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Optional tiktoken
try:
    import tiktoken
    _ENCODER = None  # Lazy load
except ImportError:
    _ENCODER = None


@dataclass
class Chunk:
    """Knowledge chunk"""
    chunk_id: str
    content: str
    source: str = ""
    page_number: int = 0
    chunk_type: str = "text"      # text / heading / table / code
    section_title: str = ""
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


class ChunkBuilder:
    """
    Intelligent text chunking builder
    
    Four strategies:
    - fixed:      Fixed token count chunking (fastest, simplest)
    - recursive:  Recursive character chunking (split at paragraph/sentence boundaries)
    - semantic:   Semantic chunking (split by embedding similarity)
    - heading:    Heading-aware chunking (split by Markdown/PDF headings)
    """
    
    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: int = 500,
        overlap: int = 50,
        min_chunk_size: int = 30,
        respect_headings: bool = True,
        encoding_name: str = "cl100k_base",
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.respect_headings = respect_headings
        
        # Lazy load tokenizer
        self._encoder = None
        self._encoding_name = encoding_name
    
    @property
    def encoder(self):
        """Lazy load tokenizer"""
        if self._encoder is None and _ENCODER is not None:
            try:
                self._encoder = tiktoken.get_encoding(self._encoding_name)
            except Exception:
                logger.warning(f"Failed to load tokenizer {self._encoding_name}, "
                             "using character count instead")
        return self._encoder
    
    def count_tokens(self, text: str) -> int:
        """Calculate token count for text"""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Fallback: rough estimate (~4 chars = 1 token)
            return len(text) // 4
    
    def chunk_text(
        self,
        text: str,
        source: str = "",
        page_number: int = 0,
        section_title: str = "",
        metadata: dict = None,
    ) -> list[Chunk]:
        """
        Chunk text
        
        Args:
            text: Text to chunk
            source: Source file
            page_number: Starting page number
            section_title: Section heading
            metadata: Other metadata
            
        Returns:
            List of Chunks
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        if self.strategy == "fixed":
            chunks = self._chunk_fixed(text)
        elif self.strategy == "recursive":
            chunks = self._chunk_recursive(text)
        elif self.strategy == "semantic":
            chunks = self._chunk_semantic(text)
        elif self.strategy == "heading":
            chunks = self._chunk_by_headings(text)
        else:
            chunks = self._chunk_recursive(text)
        
        # Populate metadata
        for chunk in chunks:
            chunk.source = source
            chunk.page_number = page_number
            chunk.section_title = section_title
            chunk.metadata = metadata or {}
            chunk.token_count = self.count_tokens(chunk.content)
            chunk.chunk_id = self._generate_id(chunk)
        
        # Filter out too-short chunks
        chunks = [c for c in chunks 
                  if len(c.content.strip()) >= self.min_chunk_size]
        
        logger.debug(f"Chunking complete: {len(chunks)} chunks, "
                    f"strategy={self.strategy}")
        
        return chunks
    
    def _chunk_fixed(self, text: str) -> list[Chunk]:
        """Fixed-length chunking"""
        tokens = self.encoder.encode(text) if self.encoder else list(text)
        
        chunks = []
        for start in range(0, len(tokens), self.chunk_size - self.overlap):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            
            if self.encoder:
                chunk_text = self.encoder.decode(chunk_tokens)
            else:
                chunk_text = "".join(chunk_tokens)
            
            chunks.append(Chunk(
                chunk_id="",
                content=chunk_text,
            ))
        
        return chunks
    
    def _chunk_recursive(self, text: str) -> list[Chunk]:
        """
        Recursive character chunking
        
        Prioritize splitting at large separators:
        Double newline → newline → period + space → space
        """
        separators = ['\n\n', '\n', '. ', ' ', '']
        
        def split_text(text: str, sep_idx: int = 0) -> list[str]:
            """Recursive splitting"""
            if sep_idx >= len(separators):
                return [text] if text else []
            
            sep = separators[sep_idx]
            parts = text.split(sep)
            
            if len(parts) == 1:
                return split_text(text, sep_idx + 1)
            
            merged = []
            current = ""
            
            for part in parts:
                test = current + sep + part if current else part
                
                if self.count_tokens(test) <= self.chunk_size:
                    current = test
                else:
                    if current:
                        merged.append(current)
                    # If single part exceeds limit, split further
                    if sep_idx + 1 < len(separators):
                        sub_parts = split_text(part, sep_idx + 1)
                        merged.extend(sub_parts[:-1] if sub_parts else [])
                        current = sub_parts[-1] if sub_parts else ""
                    else:
                        current = part
            
            if current:
                merged.append(current)
            
            return merged
        
        raw_chunks = split_text(text)
        
        # Handle overlap
        if self.overlap > 0 and len(raw_chunks) > 1:
            raw_chunks = self._add_overlap(raw_chunks)
        
        return [Chunk(chunk_id="", content=c.strip()) 
                for c in raw_chunks if c.strip()]
    
    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between adjacent chunks"""
        if len(chunks) <= 1:
            return chunks
        
        result = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev = result[-1]
            curr = chunks[i]
            
            # Calculate overlap portion
            prev_tokens = self.count_tokens(prev)
            overlap_tokens = min(
                self.overlap,
                prev_tokens
            )
            
            if overlap_tokens > 0:
                # Take overlap_tokens from the end of the previous chunk
                if self.encoder:
                    prev_tok_list = self.encoder.encode(prev)
                    overlap_toks = prev_tok_list[-overlap_tokens:]
                    overlap_text = self.encoder.decode(overlap_toks)
                else:
                    overlap_chars = overlap_tokens * 4
                    overlap_text = prev[-overlap_chars:]
                
                curr = overlap_text + "\n" + curr
            
            result.append(curr)
        
        return result
    
    def _chunk_semantic(self, text: str) -> list[Chunk]:
        """
        Semantic chunking (split by embedding similarity)
        
        Principle: sudden drop in embedding similarity between adjacent sentences = topic shift point
        """
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
        except ImportError:
            logger.warning("Semantic chunking requires sentence-transformers, "
                         "falling back to recursive strategy")
            return self._chunk_recursive(text)
        
        # Split by sentence
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        if not sentences:
            return self._chunk_recursive(text)
        
        # Add period back (removed during splitting)
        sentences = [s + "。" if not s.endswith('。') else s for s in sentences]
        
        # Embedding
        model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        embeddings = model.encode(sentences, normalize=True)
        
        # Calculate similarity between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                [embeddings[i]], [embeddings[i + 1]]
            )[0, 0]
            similarities.append(sim)
        
        # Split at points where similarity < threshold
        threshold = 0.7
        boundaries = [0]
        for i, sim in enumerate(similarities):
            if sim < threshold:
                boundaries.append(i + 1)
        boundaries.append(len(sentences))
        
        # Build chunks
        chunks = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            chunk_text = "".join(sentences[start:end])
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(chunk_id="", content=chunk_text))
            elif chunks:
                # Merge too-short chunks into the previous one
                chunks[-1].content += chunk_text
        
        return chunks
    
    def _chunk_by_headings(self, text: str) -> list[Chunk]:
        """Chunk by Markdown/PDF heading hierarchy"""
        import re
        HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        
        headings = list(HEADING_RE.finditer(text))
        if not headings:
            return self._chunk_recursive(text)
        
        chunks = []
        
        for i, match in enumerate(headings):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            
            section_text = text[start:end].strip()
            
            if len(section_text) < self.min_chunk_size:
                continue
            
            # If section is too long, further split recursively
            if self.count_tokens(section_text) > self.chunk_size * 1.5:
                sub_chunks = self._chunk_recursive(section_text)
                for sc in sub_chunks:
                    sc.section_title = title
                    chunks.append(sc)
            else:
                chunks.append(Chunk(
                    chunk_id="",
                    content=section_text,
                    chunk_type="heading",
                ))
        
        return chunks
    
    def _generate_id(self, chunk: Chunk) -> str:
        """Generate unique chunk ID"""
        raw = f"{chunk.content[:100]}{chunk.source}{chunk.page_number}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


def chunk_text(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = 500,
    **kwargs
) -> list[Chunk]:
    """Convenience function: quick chunking"""
    builder = ChunkBuilder(
        strategy=strategy,
        chunk_size=chunk_size,
        **kwargs
    )
    return builder.chunk_text(text)
