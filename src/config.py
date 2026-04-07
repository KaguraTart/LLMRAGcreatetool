"""
Configuration Management Module
Supports YAML config file + environment variable override
"""

import os
import re
import yaml
from typing import Optional
from pydantic import BaseModel, Field


class MiniMaxConfig(BaseModel):
    api_key: str = ""
    base_url: str = "https://api.minimaxi.com"
    model: str = "MiniMax-Text-01"
    vision_model: str = "MiniMax-Hailuo-VL-01"
    embedding_model: str = "embo-01"
    max_tokens: int = 8192
    timeout: int = 60


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


class ProviderEntryConfig(BaseModel):
    enabled: bool = True
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    vision_model: str = ""
    embedding_model: str = ""
    timeout: int = 60
    api_version: str = ""


class ProvidersConfig(BaseModel):
    active: str = "minimax"
    enabled: list[str] = Field(default_factory=lambda: ["minimax"])
    fallback_chain: list[str] = Field(default_factory=lambda: ["minimax"])
    load_balancer_pool: list[str] = Field(default_factory=list)

    minimax: ProviderEntryConfig = ProviderEntryConfig(
        enabled=True,
        base_url="https://api.minimaxi.com",
        model="MiniMax-Text-01",
        vision_model="MiniMax-Hailuo-VL-01",
        embedding_model="embo-01",
        timeout=60,
    )
    openai: ProviderEntryConfig = ProviderEntryConfig(
        enabled=False,
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        embedding_model="text-embedding-3-large",
        timeout=60,
    )
    anthropic: ProviderEntryConfig = ProviderEntryConfig(
        enabled=False,
        model="claude-3-5-sonnet-20241022",
        timeout=60,
    )
    gemini: ProviderEntryConfig = ProviderEntryConfig(
        enabled=False,
        model="gemini-1.5-pro",
        embedding_model="models/text-embedding-004",
        timeout=60,
    )
    ollama: ProviderEntryConfig = ProviderEntryConfig(
        enabled=False,
        base_url="http://localhost:11434",
        model="llama3.1",
        embedding_model="nomic-embed-text",
        timeout=120,
    )
    zhipu: ProviderEntryConfig = ProviderEntryConfig(
        enabled=False,
        model="glm-4",
        embedding_model="embedding-2",
        timeout=60,
    )
    qwen: ProviderEntryConfig = ProviderEntryConfig(
        enabled=False,
        model="qwen-max",
        embedding_model="text-embedding-v3",
        timeout=60,
    )


class QAConfig(BaseModel):
    rewrite: bool = True
    decompose: bool = True
    hyde: bool = True

    retrieval_mode: str = "vector"   # vector / bm25 / hybrid
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    rerank_enabled: bool = False
    rerank_top_n: int = 20

    cot_enabled: bool = True
    self_verify: bool = True
    min_faithfulness: float = 0.6

    answer_quality_enabled: bool = True
    answer_quality_use_llm: bool = True
    min_answer_quality_score: float = 0.5
    block_low_quality_response: bool = False


class Config(BaseModel):
    """Global configuration"""
    minimax: MiniMaxConfig = MiniMaxConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    ner: NERConfig = NERConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    quality: QualityConfig = QualityConfig()
    dedup: DedupConfig = DedupConfig()

    providers: ProvidersConfig = ProvidersConfig()
    qa: QAConfig = QAConfig()

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}

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
        env_map = {
            "minimax": "MINIMAX_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "zhipu": "ZHIPU_API_KEY",
            "qwen": "QWEN_API_KEY",
        }

        # Prefer providers section if present
        providers_obj = getattr(self, "providers", None)
        if providers_obj and hasattr(providers_obj, provider):
            entry = getattr(providers_obj, provider)
            env_name = env_map.get(provider)
            if env_name:
                return os.environ.get(env_name, getattr(entry, "api_key", ""))
            return getattr(entry, "api_key", "")

        if provider == "minimax":
            return os.environ.get("MINIMAX_API_KEY", self.minimax.api_key)
        return ""

    def set_api_key(self, provider: str, key: str):
        """Set API Key"""
        if hasattr(self.providers, provider):
            getattr(self.providers, provider).api_key = key
        if provider == "minimax":
            self.minimax.api_key = key
