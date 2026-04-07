"""
providers package — multi-provider LLM abstraction layer.

Usage::

    from src.providers import build_registry_from_config
    registry = build_registry_from_config(config)
    provider = registry.current
    text = provider.generate("What is RAG?")
"""

from .base import (
    LLMProvider, ToolCallResult,
    ProviderError, AuthenticationError, RateLimitError,
    ModelNotFoundError, ContextLengthError, ProviderUnavailableError,
)
from .registry import ProviderRegistry, FallbackChain, LoadBalancer
from .minimax_provider import MiniMaxProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider
from .ollama_provider import OllamaProvider
from .zhipu_provider import ZhipuProvider
from .qwen_provider import QwenProvider

__all__ = [
    "LLMProvider", "ToolCallResult",
    "ProviderError", "AuthenticationError", "RateLimitError",
    "ModelNotFoundError", "ContextLengthError", "ProviderUnavailableError",
    "ProviderRegistry", "FallbackChain", "LoadBalancer",
    "MiniMaxProvider", "OpenAIProvider", "AnthropicProvider",
    "GeminiProvider", "OllamaProvider", "ZhipuProvider", "QwenProvider",
    "build_registry_from_config",
]


def build_registry_from_config(config) -> ProviderRegistry:
    """
    Build a ProviderRegistry from a Config object.

    Reads the providers: section of config, instantiates each configured
    provider, registers them all, and sets the active provider.
    """
    registry = ProviderRegistry()

    # Helper: get providers config section safely
    pc = getattr(config, "providers", None)
    if pc is None:
        # Legacy mode: only MiniMax configured
        api_key = getattr(getattr(config, "minimax", None), "api_key", "")
        mm_cfg = getattr(config, "minimax", None)
        if mm_cfg:
            registry.register("minimax", MiniMaxProvider(
                api_key=api_key,
                base_url=getattr(mm_cfg, "base_url", "https://api.minimaxi.com"),
                model=getattr(mm_cfg, "model", "MiniMax-Text-01"),
                vision_model=getattr(mm_cfg, "vision_model", "MiniMax-Hailuo-VL-01"),
                embedding_model=getattr(mm_cfg, "embedding_model", "embo-01"),
                timeout=getattr(mm_cfg, "timeout", 60),
            ), make_active=True)
        return registry

    _PROVIDER_MAP = {
        "minimax": _build_minimax,
        "openai": _build_openai,
        "anthropic": _build_anthropic,
        "gemini": _build_gemini,
        "ollama": _build_ollama,
        "zhipu": _build_zhipu,
        "qwen": _build_qwen,
    }

    active = getattr(pc, "active", "minimax")
    enabled = set(getattr(pc, "enabled", []) or [])

    for provider_name, builder in _PROVIDER_MAP.items():
        cfg = getattr(pc, provider_name, None)
        if cfg is None:
            continue

        cfg_enabled = getattr(cfg, "enabled", True)
        if enabled and provider_name not in enabled:
            continue
        if not cfg_enabled:
            continue

        try:
            provider = builder(cfg)
            is_active = (provider_name == active)
            registry.register(provider_name, provider, make_active=is_active)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"[providers] Failed to init {provider_name}: {e}")

    # Ensure active provider exists
    if registry.list() and registry.active_name not in registry.list():
        registry.switch(registry.list()[0])

    return registry


def _build_minimax(cfg) -> MiniMaxProvider:
    return MiniMaxProvider(
        api_key=getattr(cfg, "api_key", ""),
        base_url=getattr(cfg, "base_url", "https://api.minimaxi.com"),
        model=getattr(cfg, "model", "MiniMax-Text-01"),
        vision_model=getattr(cfg, "vision_model", "MiniMax-Hailuo-VL-01"),
        embedding_model=getattr(cfg, "embedding_model", "embo-01"),
        timeout=getattr(cfg, "timeout", 60),
    )

def _build_openai(cfg) -> OpenAIProvider:
    return OpenAIProvider(
        api_key=getattr(cfg, "api_key", ""),
        base_url=getattr(cfg, "base_url", "https://api.openai.com/v1"),
        model=getattr(cfg, "model", "gpt-4o"),
        embedding_model=getattr(cfg, "embedding_model", "text-embedding-3-large"),
        timeout=getattr(cfg, "timeout", 60),
        api_version=getattr(cfg, "api_version", ""),
    )

def _build_anthropic(cfg) -> AnthropicProvider:
    return AnthropicProvider(
        api_key=getattr(cfg, "api_key", ""),
        model=getattr(cfg, "model", "claude-3-5-sonnet-20241022"),
        timeout=getattr(cfg, "timeout", 60),
    )

def _build_gemini(cfg) -> GeminiProvider:
    return GeminiProvider(
        api_key=getattr(cfg, "api_key", ""),
        model=getattr(cfg, "model", "gemini-1.5-pro"),
        embedding_model=getattr(cfg, "embedding_model", "models/text-embedding-004"),
        timeout=getattr(cfg, "timeout", 60),
    )

def _build_ollama(cfg) -> OllamaProvider:
    return OllamaProvider(
        base_url=getattr(cfg, "base_url", "http://localhost:11434"),
        model=getattr(cfg, "model", "llama3.1"),
        embedding_model=getattr(cfg, "embedding_model", "nomic-embed-text"),
        timeout=getattr(cfg, "timeout", 120),
    )

def _build_zhipu(cfg) -> ZhipuProvider:
    return ZhipuProvider(
        api_key=getattr(cfg, "api_key", ""),
        model=getattr(cfg, "model", "glm-4"),
        embedding_model=getattr(cfg, "embedding_model", "embedding-2"),
        timeout=getattr(cfg, "timeout", 60),
    )

def _build_qwen(cfg) -> QwenProvider:
    return QwenProvider(
        api_key=getattr(cfg, "api_key", ""),
        model=getattr(cfg, "model", "qwen-max"),
        embedding_model=getattr(cfg, "embedding_model", "text-embedding-v3"),
        timeout=getattr(cfg, "timeout", 60),
    )
