"""
智能分块器
支持：固定长度 / 递归字符 / 语义 / 层级感知 四种策略
"""

import logging
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# 可选 tiktoken
try:
    import tiktoken
    _ENCODER = None  # 延迟加载
except ImportError:
    _ENCODER = None


@dataclass
class Chunk:
    """知识片段"""
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
    智能分块构建器
    
    四种策略：
    - fixed:      固定 token 数分块（最快，最简单）
    - recursive:  递归字符分块（按段落/句子边界切分）
    - semantic:   语义分块（按 embedding 相似度切分）
    - heading:    层级感知分块（按 Markdown/PDF 标题切分）
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
        
        # 延迟加载 tokenizer
        self._encoder = None
        self._encoding_name = encoding_name
    
    @property
    def encoder(self):
        """延迟加载 tokenizer"""
        if self._encoder is None and _ENCODER is not None:
            try:
                self._encoder = tiktoken.get_encoding(self._encoding_name)
            except Exception:
                logger.warning(f"无法加载 tokenizer {self._encoding_name}，"
                             "使用字符数代替")
        return self._encoder
    
    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数"""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Fallback: 粗略估算（约 4 字符 = 1 token）
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
        将文本分块
        
        Args:
            text: 待分块文本
            source: 来源文件
            page_number: 起始页码
            section_title: 所属章节标题
            metadata: 其他元数据
            
        Returns:
            Chunk 列表
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
        
        # 填充元数据
        for chunk in chunks:
            chunk.source = source
            chunk.page_number = page_number
            chunk.section_title = section_title
            chunk.metadata = metadata or {}
            chunk.token_count = self.count_tokens(chunk.content)
            chunk.chunk_id = self._generate_id(chunk)
        
        # 过滤过短的 chunk
        chunks = [c for c in chunks 
                  if len(c.content.strip()) >= self.min_chunk_size]
        
        logger.debug(f"分块完成: {len(chunks)} chunks, "
                    f"策略={self.strategy}")
        
        return chunks
    
    def _chunk_fixed(self, text: str) -> list[Chunk]:
        """固定长度分块"""
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
        递归字符分块
        
        优先在大分隔符处切分：
        双重换行 → 换行 → 句号+空格 → 空格
        """
        separators = ['\n\n', '\n', '. ', ' ', '']
        
        def split_text(text: str, sep_idx: int = 0) -> list[str]:
            """递归切分"""
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
                    # 如果单个 part 就超出限制，继续细分
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
        
        # 处理重叠
        if self.overlap > 0 and len(raw_chunks) > 1:
            raw_chunks = self._add_overlap(raw_chunks)
        
        return [Chunk(chunk_id="", content=c.strip()) 
                for c in raw_chunks if c.strip()]
    
    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """为相邻 chunks 添加重叠"""
        if len(chunks) <= 1:
            return chunks
        
        result = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev = result[-1]
            curr = chunks[i]
            
            # 计算重叠部分
            prev_tokens = self.count_tokens(prev)
            overlap_tokens = min(
                self.overlap,
                prev_tokens
            )
            
            if overlap_tokens > 0:
                # 从前一个 chunk 末尾取 overlap_tokens 个 token
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
        语义分块（按 embedding 相似度切分）
        
        原理：相邻句子的 embedding 相似度突然下降 = 话题转换点
        """
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
        except ImportError:
            logger.warning("semantic chunking 需要 sentence-transformers，"
                         "降级为 recursive 策略")
            return self._chunk_recursive(text)
        
        # 按句子切分
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        if not sentences:
            return self._chunk_recursive(text)
        
        # 添加句号（切分时去掉了）
        sentences = [s + "。" if not s.endswith('。') else s for s in sentences]
        
        # Embedding
        model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        embeddings = model.encode(sentences, normalize=True)
        
        # 计算相邻句子的相似度
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                [embeddings[i]], [embeddings[i + 1]]
            )[0, 0]
            similarities.append(sim)
        
        # 相似度 < 阈值处切分
        threshold = 0.7
        boundaries = [0]
        for i, sim in enumerate(similarities):
            if sim < threshold:
                boundaries.append(i + 1)
        boundaries.append(len(sentences))
        
        # 构建 chunks
        chunks = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            chunk_text = "".join(sentences[start:end])
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(chunk_id="", content=chunk_text))
            elif chunks:
                # 合并过短的 chunk 到前一个
                chunks[-1].content += chunk_text
        
        return chunks
    
    def _chunk_by_headings(self, text: str) -> list[Chunk]:
        """按 Markdown/PDF 标题层级分块"""
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
            
            # 如果 section 过长，进一步递归分块
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
        """生成 chunk 唯一 ID"""
        raw = f"{chunk.content[:100]}{chunk.source}{chunk.page_number}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


def chunk_text(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = 500,
    **kwargs
) -> list[Chunk]:
    """便捷函数：快速分块"""
    builder = ChunkBuilder(
        strategy=strategy,
        chunk_size=chunk_size,
        **kwargs
    )
    return builder.chunk_text(text)
