"""
Provider Registry + Runtime Switch + Fallback Chain + Load Balancer

Design inspired by:
- cc switch  : runtime model/provider toggle (switch active at any time)
- opencode   : dynamic provider factory pattern
- openclaw   : pluggable registry + automatic fallback chain on failure
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from .base import LLMProvider, ProviderError

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Central registry for all LLM providers.

    Usage::

        registry = ProviderRegistry()
        registry.register("openai", openai_provider)
        registry.register("anthropic", anthropic_provider)
        registry.switch("anthropic")          # cc-switch style
        provider = registry.current           # always returns active provider
    """

    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}
        self._active: Optional[str] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, provider: LLMProvider, *, make_active: bool = False):
        """Add a provider to the registry."""
        with self._lock:
            self._providers[name] = provider
            if make_active or self._active is None:
                self._active = name
            logger.debug(f"[registry] registered provider: {name}")

    def unregister(self, name: str):
        """Remove a provider from the registry."""
        with self._lock:
            self._providers.pop(name, None)
            if self._active == name:
                self._active = next(iter(self._providers), None)
            logger.debug(f"[registry] unregistered provider: {name}")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list(self) -> list[str]:
        """Return names of all registered providers."""
        with self._lock:
            return list(self._providers.keys())

    def get(self, name: str) -> LLMProvider:
        """Get a provider by name."""
        with self._lock:
            if name not in self._providers:
                raise KeyError(f"Provider '{name}' is not registered. "
                               f"Available: {list(self._providers.keys())}")
            return self._providers[name]

    @property
    def current(self) -> LLMProvider:
        """Return the currently active provider."""
        with self._lock:
            if not self._active or self._active not in self._providers:
                raise RuntimeError(
                    "No active provider. Call registry.register() first."
                )
            return self._providers[self._active]

    @property
    def active_name(self) -> Optional[str]:
        """Name of the currently active provider."""
        return self._active

    # ------------------------------------------------------------------
    # cc-switch: runtime provider switching
    # ------------------------------------------------------------------

    def switch(self, name: str):
        """
        Switch the active provider at runtime (cc-switch pattern).

        Can be called from Python code or the CLI:
            python -m src provider switch anthropic
        """
        with self._lock:
            if name not in self._providers:
                raise KeyError(f"Cannot switch to unknown provider '{name}'. "
                               f"Registered: {list(self._providers.keys())}")
            old = self._active
            self._active = name
            logger.info(f"[registry] provider switched: {old} → {name}")

    def check(self, name: str) -> bool:
        """Health-check a named provider. Returns True if healthy."""
        provider = self.get(name)
        healthy = provider.health_check()
        status = "✅ healthy" if healthy else "❌ unavailable"
        logger.info(f"[registry] check {name}: {status}")
        return healthy

    def check_all(self) -> dict[str, bool]:
        """Health-check every registered provider."""
        results = {}
        for name in self.list():
            results[name] = self.check(name)
        return results

    def __repr__(self) -> str:
        return (f"<ProviderRegistry providers={self.list()} "
                f"active={self._active!r}>")


# ---------------------------------------------------------------------------
# Fallback Chain (openclaw pattern)
# ---------------------------------------------------------------------------

class FallbackChain:
    """
    Try providers in order; fall through to the next on ProviderError.

    Supports retryable errors (rate limits, unavailability) with optional
    delay between attempts.

    Usage::

        chain = FallbackChain(registry, ["openai", "anthropic", "ollama"])
        text = chain.generate("What is RAG?")
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        chain: list[str],
        retry_delay: float = 1.0,
    ):
        self.registry = registry
        self.chain = chain
        self.retry_delay = retry_delay

    def generate(self, *args, **kwargs) -> str:
        """Call generate() on each provider in chain until one succeeds."""
        last_error: Optional[Exception] = None
        for name in self.chain:
            try:
                provider = self.registry.get(name)
                result = provider.generate(*args, **kwargs)
                if name != self.chain[0]:
                    logger.info(f"[fallback] answered by fallback provider: {name}")
                return result
            except ProviderError as e:
                logger.warning(f"[fallback] {name} failed: {e} — trying next")
                last_error = e
                if e.retryable:
                    time.sleep(self.retry_delay)
            except Exception as e:
                logger.warning(f"[fallback] {name} unexpected error: {e} — trying next")
                last_error = e
        raise ProviderError(
            f"All providers in fallback chain failed: {self.chain}",
            retryable=False,
        ) from last_error

    def embed(self, texts: list[str]):
        """Call embed() on each provider in chain until one succeeds."""
        last_error: Optional[Exception] = None
        for name in self.chain:
            try:
                provider = self.registry.get(name)
                if not provider.supports_embedding:
                    continue
                return provider.embed(texts)
            except Exception as e:
                logger.warning(f"[fallback] embed {name} failed: {e} — trying next")
                last_error = e
        raise ProviderError(
            f"All providers in fallback chain failed for embed: {self.chain}",
        ) from last_error


# ---------------------------------------------------------------------------
# Load Balancer (round-robin across multiple providers / keys)
# ---------------------------------------------------------------------------

class LoadBalancer:
    """
    Distributes requests across multiple providers in round-robin order.

    Useful when you have multiple API keys for the same provider (or
    want to spread load across different providers).

    Usage::

        lb = LoadBalancer(registry, ["openai_key1", "openai_key2"])
        text = lb.generate("What is RAG?")
    """

    def __init__(self, registry: ProviderRegistry, providers: list[str]):
        self.registry = registry
        self.providers = providers
        self._index = 0
        self._lock = threading.Lock()

    def _next(self) -> LLMProvider:
        with self._lock:
            name = self.providers[self._index % len(self.providers)]
            self._index += 1
        return self.registry.get(name)

    def generate(self, *args, **kwargs) -> str:
        return self._next().generate(*args, **kwargs)

    def embed(self, texts: list[str]):
        return self._next().embed(texts)
