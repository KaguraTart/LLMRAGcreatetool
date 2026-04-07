"""
OpenAI / Azure OpenAI Provider
Supports: gpt-4o, gpt-4-turbo, gpt-3.5-turbo, text-embedding-3-large/small
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Iterator

import numpy as np

from .base import (
    LLMProvider, ToolCallResult,
    AuthenticationError, ProviderError, RateLimitError, ProviderUnavailableError,
    ContextLengthError,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI API Provider.
    Set base_url to an Azure endpoint to use Azure OpenAI.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-large",
        timeout: int = 60,
        api_version: str = "",
    ):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._embedding_model = embedding_model
        self._timeout = timeout
        self._api_version = api_version

        if not self._api_key:
            logger.warning("[openai] OPENAI_API_KEY not set")

    @property
    def name(self) -> str:
        return "openai"

    @property
    def supports_vision(self) -> bool:
        return True

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
            raise AuthenticationError("OPENAI_API_KEY is not set", provider=self.name)
        h = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._api_version:
            h["api-version"] = self._api_version
        return h

    @staticmethod
    def _handle_http_error(e, provider: str):
        import requests
        if isinstance(e, requests.exceptions.HTTPError):
            code = e.response.status_code if e.response is not None else 0
            if code == 401:
                raise AuthenticationError(str(e), provider=provider) from e
            if code == 429:
                raise RateLimitError(str(e), provider=provider) from e
            if code == 400:
                body = e.response.text if e.response is not None else ""
                if "context_length" in body or "maximum context" in body:
                    raise ContextLengthError(str(e), provider=provider) from e
            if code >= 500:
                raise ProviderUnavailableError(str(e), provider=provider) from e
        raise ProviderError(str(e), provider=provider) from e

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        import requests

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)

    def generate_stream(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Iterator[str]:
        import json as _json
        import requests

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            resp = requests.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self._timeout,
                stream=True,
            )
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
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except Exception:
                            pass
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        system: str = "",
        temperature: float = 0.3,
    ) -> ToolCallResult:
        import json as _json
        import requests

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "tools": [{"type": "function", "function": t} for t in tools],
            "tool_choice": "auto",
            "temperature": temperature,
        }

        try:
            resp = requests.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            if "tool_calls" in msg and msg["tool_calls"]:
                tc = msg["tool_calls"][0]
                return ToolCallResult(
                    function_name=tc["function"]["name"],
                    arguments=_json.loads(tc["function"]["arguments"]),
                    provider=self.name,
                    raw_content=msg.get("content", "") or "",
                )
            return ToolCallResult(
                function_name="",
                arguments={},
                provider=self.name,
                raw_content=msg.get("content", "") or "",
            )
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)

    def embed(self, texts: list[str]) -> np.ndarray:
        import requests

        try:
            resp = requests.post(
                f"{self._base_url}/embeddings",
                headers=self._headers(),
                json={"model": self._embedding_model, "input": texts},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            data.sort(key=lambda x: x["index"])
            embeddings = np.array([d["embedding"] for d in data], dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return embeddings / norms
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)

    def understand_image(
        self,
        image_bytes: bytes,
        prompt: str = "Please describe this image in detail.",
    ) -> str:
        import requests

        b64 = base64.b64encode(image_bytes).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 4096,
        }
        try:
            resp = requests.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)
