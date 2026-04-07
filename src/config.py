"""
Configuration Management Module
Supports YAML config file + environment variable override
"""

import os
import re
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class MiniMaxConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.minimaxi.com"
    model: str = "MiniMax-Text-01"
    vision_model: str = "MiniMax-Hailuo-VL-01"
    embedding_model: str = "embo-01"
    max_tokens: int = 8192
    timeout: int = 60


class ClaudeCodeConfig(BaseModel):
    enabled: bool = False
    model: str = "sonnet-4"
    timeout: int = 120


class GeminiCLIConfig(BaseModel):
    enabled: bool = False
    model: str = "gemini-2.0-flash"
    api_key: str = ""


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "rag_knowledge_base"
    vector_size: int = 1024
    distance: str = "Cosine"


class VectorStoreConfig(BaseModel):
    type: str = "qdrant"
    qdrant: QdrantConfig = QdrantConfig()


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-large-zh-v1.5"
    device: str = "cuda"
    batch_size: int = 32
    normalize: bool = True


class ChunkingConfig(BaseModel):
    strategy: str = "semantic"
    chunk_size: int = 500
    overlap: int = 50
    min_chunk_size: int = 30
    respect_headings: bool = True
    semantic_threshold: float = 0.7


class NERConfig(BaseModel):
    enabled: bool = True
    use_llm: bool = True
    extraction_schema: str = "technical"
    confidence_threshold: float = 0.6


class ClassifierConfig(BaseModel):
    cascade: bool = True
    taxonomy: list[str] = Field(default_factory=lambda: [
        "Technical Documentation", "Product Specification", "Policy & Regulation",
        "Research Report", "Operation Manual", "FAQ"
    ])
    rule_confidence_threshold: float = 0.9
    embedding_confidence_threshold: float = 0.7


class QualityConfig(BaseModel):
    enabled: bool = True
    use_llm: bool = True
    min_quality_score: float = 0.3


class DedupConfig(BaseModel):
    enabled: bool = True
    method: str = "hybrid"
    minhash_threshold: float = 0.8
    embedding_threshold: float = 0.92


class Config(BaseModel):
    """Global configuration"""
    minimax: MiniMaxConfig = MiniMaxConfig()
    claude_code: ClaudeCodeConfig = ClaudeCodeConfig()
    gemini_cli: GeminiCLIConfig = GeminiCLIConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    ner: NERConfig = NERConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    quality: QualityConfig = QualityConfig()
    dedup: DedupConfig = DedupConfig()

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        
        # Handle environment variable substitution: ${VAR_NAME}
        raw = cls._resolve_env_vars(raw)
        
        return cls(**raw)
    
    @classmethod
    def _resolve_env_vars(cls, obj):
        """Recursively resolve ${ENV_VAR} style environment variable references"""
        if isinstance(obj, str):
            # Match ${VAR_NAME} or ${VAR_NAME:default}
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
            
            def replacer(match):
                var_name = match.group(1)
                default = match.group(2) or ""
                return os.environ.get(var_name, default)
            
            return re.sub(pattern, replacer, obj)
        
        elif isinstance(obj, dict):
            return {k: cls._resolve_env_vars(v) for k, v in obj.items()}
        
        elif isinstance(obj, list):
            return [cls._resolve_env_vars(item) for item in obj]
        
        else:
            return obj
    
    def get_api_key(self, provider: str = "minimax") -> str:
        """Get API Key"""
        if provider == "minimax":
            return os.environ.get("MINIMAX_API_KEY", self.minimax.api_key)
        elif provider == "gemini":
            return os.environ.get("GOOGLE_API_KEY", self.gemini_cli.api_key)
        return ""
    
    def set_api_key(self, provider: str, key: str):
        """Set API Key"""
        if provider == "minimax":
            self.minimax.api_key = key
        elif provider == "gemini":
            self.gemini_cli.api_key = key
