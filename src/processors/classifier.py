"""
级联分类器
Layer 1: 规则（关键词匹配，毫秒级）
Layer 2: Embedding 相似度（快速，无需 LLM）
Layer 3: LLM 推理（最准，但最慢）
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
    级联分类器
    
    从快到准逐层尝试：
    规则 → Embedding → LLM
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
            taxonomy: 分类体系，格式：{类别名: [关键词列表]}
            taxonomy_embeddings: 各类的 embedding（可选）
            rule_threshold: 规则分类的置信度阈值
            embedding_threshold: Embedding 分类的置信度阈值
        """
        self.taxonomy = taxonomy
        self.taxonomy_embeddings = taxonomy_embeddings
        self.rule_threshold = rule_threshold
        self.embedding_threshold = embedding_threshold
    
    def classify(self, text: str, strategy: str = "cascade") -> ClassificationResult:
        """
        分类文本
        
        Args:
            text: 待分类文本
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
        """Layer 1: 关键词规则"""
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
        """Layer 2: Embedding 相似度"""
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
        
        # 获取文本 embedding
        model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        text_emb = model.encode([text], normalize=True)[0]
        
        # 与各类别关键词的 embedding 计算相似度
        best_cat = None
        best_score = -1
        
        for category, cat_embs in self.taxonomy_embeddings.items():
            if not isinstance(cat_embs, np.ndarray):
                continue
            sims = np.dot(text_emb, cat_embs)  # 已归一化
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
        """Layer 3: LLM 推理"""
        # 此方法需要外部 LLM，可在 pipeline 中调用
        return ClassificationResult(
            category="unknown",
            confidence=0.0,
            method="llm"
        )
    
    def _classify_cascade(self, text: str) -> ClassificationResult:
        """级联分类"""
        # Layer 1: 规则
        result = self._classify_rule(text)
        if result.confidence >= self.rule_threshold:
            return result
        
        # Layer 2: Embedding
        result = self._classify_embedding(text)
        if result.confidence >= self.embedding_threshold:
            return result
        
        # Layer 3: 返回 Embedding 结果（最接近的）
        return result
    
    def build_taxonomy_embeddings(self):
        """预计算分类体系的 embedding（用于 Embedding 层加速）"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers 未安装")
            return
        
        model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        embeddings = {}
        
        for category, keywords in self.taxonomy.items():
            embs = model.encode(keywords, normalize=True)
            # 取平均作为类别中心
            embeddings[category] = embs.mean(axis=0)
        
        self.taxonomy_embeddings = embeddings
        logger.info(f"预计算了 {len(embeddings)} 个类别的 embedding")
    
    def batch_classify(
        self, 
        texts: list[str],
        strategy: str = "cascade"
    ) -> list[ClassificationResult]:
        """批量分类"""
        return [self.classify(t, strategy=strategy) for t in texts]
