"""
Ollama Provider — local LLM inference via REST API
Supports any model installed in Ollama (llama3, mistral, qwen, etc.)
"""
from __future__ import annotations
import logging, os
from typing import Iterator
import numpy as np
from .base import (LLMProvider, ToolCallResult,
    ProviderError, ProviderUnavailableError)

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider (http://localhost:11434)."""

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "llama3.1", embedding_model: str = "nomic-embed-text",
                 timeout: int = 120):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._embedding_model = embedding_model
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_embedding(self) -> bool:
        return True

    @staticmethod
    def _handle_http_error(e, provider: str):
        import requests
        if isinstance(e, requests.exceptions.ConnectionError):
            raise ProviderUnavailableError(
                f"Cannot connect to Ollama at the configured URL. Is Ollama running?",
                provider=provider) from e
        raise ProviderError(str(e), provider=provider) from e

    def generate(self, prompt: str, system: str = "", temperature: float = 0.7,
                 max_tokens: int = 4096, json_mode: bool = False) -> str:
        import requests
        payload: dict = {
            "model": self._model, "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"
        try:
            resp = requests.post(f"{self._base_url}/api/generate",
                json=payload, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()["response"]
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)

    def generate_stream(self, prompt: str, system: str = "", temperature: float = 0.7,
                        max_tokens: int = 4096) -> Iterator[str]:
        import json as _json, requests
        payload: dict = {
            "model": self._model, "prompt": prompt, "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        try:
            resp = requests.post(f"{self._base_url}/api/generate",
                json=payload, timeout=self._timeout, stream=True)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    try:
                        ev = _json.loads(line.decode("utf-8"))
                        text = ev.get("response", "")
                        if text:
                            yield text
                        if ev.get("done"):
                            break
                    except Exception:
                        pass
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)

    def embed(self, texts: list[str]) -> np.ndarray:
        import requests
        embeddings = []
        for text in texts:
            try:
                resp = requests.post(f"{self._base_url}/api/embeddings",
                    json={"model": self._embedding_model, "prompt": text},
                    timeout=self._timeout)
                resp.raise_for_status()
                emb = np.array(resp.json()["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb)
                embeddings.append(emb / norm if norm > 0 else emb)
            except Exception as e:
                logger.warning(f"[ollama] embed failed: {e}")
                embeddings.append(np.zeros(768, dtype=np.float32))
        return np.array(embeddings)

    def health_check(self) -> bool:
        import requests
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
