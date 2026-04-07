"""
Provider Abstraction Layer — Base Protocol

All LLM / embedding providers must implement this interface.
The rest of the codebase ONLY depends on LLMProvider, never on a concrete class.

Design inspired by:
- cc switch  : runtime provider/model switching
- opencode   : provider-agnostic LLM tooling
- openclaw   : pluggable provider registry + fallback chain
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """Base class for all provider errors."""
    def __init__(self, message: str, provider: str = "", retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class AuthenticationError(ProviderError):
    """Invalid or missing API key."""


class RateLimitError(ProviderError):
    """Rate limit exceeded — always retryable."""
    def __init__(self, message: str, provider: str = ""):
        super().__init__(message, provider=provider, retryable=True)


class ModelNotFoundError(ProviderError):
    """Requested model does not exist."""


class ContextLengthError(ProviderError):
    """Prompt exceeds maximum context length."""


class ProviderUnavailableError(ProviderError):
    """Provider service is down or unreachable — retryable."""
    def __init__(self, message: str, provider: str = ""):
        super().__init__(message, provider=provider, retryable=True)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ToolCallResult:
    """Result of a generate_with_tools() call."""
    function_name: str
    arguments: dict
    provider: str = ""
    raw_content: str = ""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class LLMProvider(abc.ABC):
    """
    Unified interface for all LLM / embedding providers.

    Subclasses implement provider-specific API calls.
    All retry logic, auth, and payload shaping live inside the subclass.
    """

    # --- Identity & capability flags ----------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Provider identifier, e.g. 'openai', 'anthropic', 'gemini'."""

    @property
    def supports_vision(self) -> bool:
        return False

    @property
    def supports_function_calling(self) -> bool:
        return False

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_embedding(self) -> bool:
        return False

    # --- Core generation methods -------------------------------------------

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """
        Text generation (blocking).
        Returns the generated text string.
        Raises ProviderError subclasses on failure.
        """

    def generate_stream(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Iterator[str]:
        """
        Streaming text generation.
        Default: falls back to non-streaming generate().
        """
        yield self.generate(prompt, system=system,
                            temperature=temperature, max_tokens=max_tokens)

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        system: str = "",
        temperature: float = 0.3,
    ) -> ToolCallResult:
        """
        Function / tool calling.
        Default raises NotImplementedError — override when supported.
        """
        raise NotImplementedError(
            f"Provider '{self.name}' does not support function calling."
        )

    # --- Embedding -----------------------------------------------------------

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts into embedding vectors.
        Returns ndarray of shape (n, dim).
        Default raises NotImplementedError — override when supported.
        """
        raise NotImplementedError(
            f"Provider '{self.name}' does not support embeddings."
        )

    # --- Vision --------------------------------------------------------------

    def understand_image(
        self,
        image_bytes: bytes,
        prompt: str = "Please describe this image in detail.",
    ) -> str:
        """
        Multimodal image understanding.
        Default raises NotImplementedError — override when supported.
        """
        raise NotImplementedError(
            f"Provider '{self.name}' does not support vision."
        )

    # --- Health check -------------------------------------------------------

    def health_check(self) -> bool:
        """
        Probe whether the provider is reachable and the API key is valid.
        Returns True on success, False on failure.
        """
        try:
            result = self.generate("ping", max_tokens=4, temperature=0)
            return isinstance(result, str)
        except Exception as e:
            logger.debug(f"[{self.name}] health_check failed: {e}")
            return False

    # --- Convenience helpers -----------------------------------------------

    def extract_entities(self, text: str, schema: dict) -> dict:
        """
        Entity + relation extraction.
        Uses function calling if supported, else plain JSON generation.
        Returns {"entities": [...], "relations": [...]}.
        """
        import json, re

        entity_types = schema.get("entities", [])
        relation_types = schema.get("relations", [])

        entity_def = "\n".join(f"- {e}" for e in entity_types)
        relation_def = "\n".join(f"- {r}" for r in relation_types)

        if self.supports_function_calling:
            functions = [
                {
                    "name": "extract_knowledge",
                    "description": "Extract entities and relations from text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entities": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "type": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                },
                            },
                            "relations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source": {"type": "string"},
                                        "target": {"type": "string"},
                                        "relation": {"type": "string"},
                                    },
                                },
                            },
                        },
                        "required": ["entities", "relations"],
                    },
                }
            ]
            prompt = (
                f"Extract entities and relations from the text.\n"
                f"Entity types:\n{entity_def}\n"
                f"Relation types:\n{relation_def}\n\n"
                f"Text:\n---\n{text[:4000]}\n---\n"
                f"Call extract_knowledge with the results."
            )
            system = "You are a knowledge extraction system. Only output JSON."
            try:
                result = self.generate_with_tools(prompt, functions, system=system)
                if result.function_name == "extract_knowledge":
                    return result.arguments
            except Exception as e:
                logger.warning(f"[{self.name}] function calling failed, falling back: {e}")

        # Fallback: plain JSON generation
        prompt = (
            f"Extract entities and relations from the text.\n"
            f"Entity types:\n{entity_def}\n"
            f"Relation types:\n{relation_def}\n\n"
            f"Text:\n---\n{text[:4000]}\n---\n\n"
            f'Output JSON: {{"entities":[{{"name":"...","type":"...","description":"..."}}],'
            f'"relations":[{{"source":"...","target":"...","relation":"..."}}]}}'
        )
        system = "You are a knowledge extraction system. Only output JSON."
        try:
            response = self.generate(prompt, system=system, temperature=0.2, json_mode=True)
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.warning(f"[{self.name}] entity extraction failed: {e}")

        return {"entities": [], "relations": []}

    def classify(self, text: str, categories: list[str]) -> dict:
        """
        Text classification.
        Returns {"category": str, "confidence": float}.
        """
        import json, re

        prompt = (
            f"Classify the following text into one category.\n"
            f"Categories: {', '.join(categories)}\n\n"
            f"Text:\n---\n{text[:2000]}\n---\n\n"
            f'Output JSON: {{"category":"category name","confidence":0.0}}'
        )
        system = "You are a text classification expert. Only output JSON."
        try:
            response = self.generate(prompt, system=system, temperature=0.3, json_mode=True)
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.warning(f"[{self.name}] classify failed: {e}")

        return {"category": categories[0] if categories else "unknown", "confidence": 0.0}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
