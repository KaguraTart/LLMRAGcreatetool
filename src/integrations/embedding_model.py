"""
Embedding 模型封装
支持：HuggingFace sentence-transformers / MiniMax API
"""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Embedding 模型统一封装
    
    支持：
    - HuggingFace sentence-transformers（本地推理，推荐）
    - MiniMax API（云端，备用）
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = "cuda",
        batch_size: int = 32,
        normalize: bool = True,
        minimax_client=None,  # 可选：MiniMax API 客户端
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.minimax_client = minimax_client
        
        self._model = None
        self._dimension = None
    
    def _load_model(self):
        """延迟加载 HuggingFace 模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"加载 Embedding 模型: {self.model_name}, "
                           f"维度: {self._dimension}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers 未安装: pip install sentence-transformers"
                )
            except Exception as e:
                logger.warning(f"加载模型失败: {e}，降级为 MiniMax API")
                self._model = None
    
    @property
    def dimension(self) -> int:
        """获取 embedding 维度"""
        if self._dimension is None:
            self._load_model()
        if self._dimension is None and self.minimax_client:
            # MiniMax API 动态获取
            return 1024  # MiniMax embedding 维度
        return self._dimension or 1024
    
    def encode(
        self,
        texts: str | list[str],
        batch_size: int = None,
        show_progress: bool = False,
        normalize: bool = None,
    ) -> np.ndarray:
        """
        将文本编码为 embedding 向量
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批大小（默认使用 self.batch_size）
            show_progress: 是否显示进度条
            normalize: 是否 L2 归一化（默认使用 self.normalize）
            
        Returns:
            numpy.ndarray，shape = (n, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        normalize = normalize if normalize is not None else self.normalize
        batch_size = batch_size or self.batch_size
        
        # 优先使用本地模型
        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
            return embeddings
        
        # Fallback: MiniMax API
        if self.minimax_client:
            return self._encode_via_api(texts)
        
        raise RuntimeError("既没有本地模型也没有 MiniMax API 客户端")
    
    def _encode_via_api(self, texts: list[str]) -> np.ndarray:
        """通过 MiniMax API 获取 embedding"""
        if not self.minimax_client:
            raise RuntimeError("MiniMax 客户端未配置")
        
        embeddings = []
        
        for text in texts:
            try:
                import requests
                
                resp = requests.post(
                    f"{self.minimax_client.base_url}/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.minimax_client.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.minimax_client.embedding_model,
                        "input": text,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                
                emb = data["data"][0]["embedding"]
                emb = np.array(emb)
                
                if self.normalize:
                    emb = emb / (np.linalg.norm(emb) + 1e-8)
                
                embeddings.append(emb)
            
            except Exception as e:
                logger.warning(f"MiniMax embedding 失败: {e}")
                # 返回零向量
                dim = 1024  # MiniMax default
                embeddings.append(np.zeros(dim))
        
        return np.array(embeddings)
    
    def similarity(
        self,
        text1: str | list[str],
        text2: str | list[str],
    ) -> np.ndarray:
        """
        计算文本对之间的余弦相似度
        
        Returns:
            相似度矩阵（已归一化后为点积）
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        # 归一化
        n1 = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
        n2 = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
        
        return np.dot(n1, n2.T)
    
    def batch_encode(self, chunks: list[dict], text_key: str = "content") -> np.ndarray:
        """
        批量编码 Chunk 对象列表
        
        Args:
            chunks: Chunk 对象列表（dataclass）
            text_key: 从 chunk 中提取文本的字段名
        """
        texts = []
        for chunk in chunks:
            if hasattr(chunk, text_key):
                texts.append(getattr(chunk, text_key))
            elif isinstance(chunk, dict):
                texts.append(chunk.get(text_key, ""))
            else:
                texts.append(str(chunk))
        
        return self.encode(texts)
