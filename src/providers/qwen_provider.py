"""
Alibaba Qwen / DashScope Provider
Supports: qwen-max, qwen-plus, qwen-turbo, text-embedding-v3
"""
from __future__ import annotations
import logging, os
from typing import Iterator
import numpy as np
from .base import (LLMProvider, ToolCallResult,
    AuthenticationError, ProviderError, RateLimitError, ProviderUnavailableError)

logger = logging.getLogger(__name__)


class QwenProvider(LLMProvider):
    """Alibaba Qwen via DashScope API (OpenAI-compatible endpoint)."""

    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, api_key: str = "", model: str = "qwen-max",
                 embedding_model: str = "text-embedding-v3", timeout: int = 60):
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self._model = model
        self._embedding_model = embedding_model
        self._timeout = timeout
        if not self._api_key:
            logger.warning("[qwen] DASHSCOPE_API_KEY not set")

    @property
    def name(self) -> str:
        return "qwen"

    @property
    def supports_function_calling(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_embedding(self) -> bool:
        return True

    def _headers(self) -> dict:
        if not self._api_key:
            raise AuthenticationError("DASHSCOPE_API_KEY is not set", provider=self.name)
        return {"Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json"}

    @staticmethod
    def _handle_http_error(e, provider: str):
        import requests
        if isinstance(e, requests.exceptions.HTTPError):
            code = e.response.status_code if e.response is not None else 0
            if code == 401: raise AuthenticationError(str(e), provider=provider) from e
            if code == 429: raise RateLimitError(str(e), provider=provider) from e
            if code >= 500: raise ProviderUnavailableError(str(e), provider=provider) from e
        raise ProviderError(str(e), provider=provider) from e

    def generate(self, prompt: str, system: str = "", temperature: float = 0.7,
                 max_tokens: int = 4096, json_mode: bool = False) -> str:
        import requests
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict = {"model": self._model, "messages": messages,
                         "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = requests.post(f"{self.BASE_URL}/chat/completions",
                headers=self._headers(), json=payload, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)

    def generate_stream(self, prompt: str, system: str = "", temperature: float = 0.7,
                        max_tokens: int = 4096) -> Iterator[str]:
        import json as _json, requests
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self._model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens, "stream": True}
        try:
            resp = requests.post(f"{self.BASE_URL}/chat/completions",
                headers=self._headers(), json=payload, timeout=self._timeout, stream=True)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk = _json.loads(data)
                            content = chunk["choices"][0].get("delta",{}).get("content","")
                            if content:
                                yield content
                        except Exception:
                            pass
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)

    def generate_with_tools(self, prompt: str, tools: list[dict], system: str = "",
                            temperature: float = 0.3) -> ToolCallResult:
        import json as _json, requests
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self._model, "messages": messages,
                   "tools": [{"type": "function", "function": t} for t in tools],
                   "tool_choice": "auto", "temperature": temperature}
        try:
            resp = requests.post(f"{self.BASE_URL}/chat/completions",
                headers=self._headers(), json=payload, timeout=self._timeout)
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            if "tool_calls" in msg and msg["tool_calls"]:
                tc = msg["tool_calls"][0]
                return ToolCallResult(
                    function_name=tc["function"]["name"],
                    arguments=_json.loads(tc["function"]["arguments"]),
                    provider=self.name, raw_content=msg.get("content","") or "")
            return ToolCallResult(function_name="", arguments={},
                                  provider=self.name, raw_content=msg.get("content","") or "")
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)

    def embed(self, texts: list[str]) -> np.ndarray:
        import requests
        # DashScope text-embedding-v3 supports batch
        try:
            resp = requests.post(f"{self.BASE_URL}/embeddings",
                headers=self._headers(),
                json={"model": self._embedding_model, "input": texts,
                      "encoding_format": "float"},
                timeout=self._timeout)
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda x: x["index"])
            embeddings = np.array([d["embedding"] for d in data], dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return embeddings / norms
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)
