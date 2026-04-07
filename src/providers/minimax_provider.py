"""
MiniMax Provider
Refactored from integrations/minimax_api.py to implement LLMProvider.
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
)

logger = logging.getLogger(__name__)


class MiniMaxProvider(LLMProvider):
    """
    MiniMax API Provider (MiniMax-Text-01, MiniMax-Hailuo-VL-01, embo-01).
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.minimaxi.com",
        model: str = "MiniMax-Text-01",
        vision_model: str = "MiniMax-Hailuo-VL-01",
        embedding_model: str = "embo-01",
        timeout: int = 60,
    ):
        self._api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._vision_model = vision_model
        self._embedding_model = embedding_model
        self._timeout = timeout

        if not self._api_key:
            logger.warning("[minimax] MINIMAX_API_KEY not set")

    # --- Identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        return "minimax"

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

    # --- Helpers ------------------------------------------------------------

    def _headers(self) -> dict:
        if not self._api_key:
            raise AuthenticationError("MINIMAX_API_KEY is not set", provider=self.name)
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _handle_http_error(e, provider: str):
        import requests
        if isinstance(e, requests.exceptions.HTTPError):
            code = e.response.status_code if e.response is not None else 0
            if code == 401:
                raise AuthenticationError(str(e), provider=provider) from e
            if code == 429:
                raise RateLimitError(str(e), provider=provider) from e
            if code >= 500:
                raise ProviderUnavailableError(str(e), provider=provider) from e
        raise ProviderError(str(e), provider=provider) from e

    # --- generate -----------------------------------------------------------

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
                f"{self._base_url}/v1/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)

    # --- generate_stream ----------------------------------------------------

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
                f"{self._base_url}/v1/chat/completions",
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

    # --- generate_with_tools ------------------------------------------------

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
                f"{self._base_url}/v1/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            if "tool_calls" in msg:
                tc = msg["tool_calls"][0]
                return ToolCallResult(
                    function_name=tc["function"]["name"],
                    arguments=_json.loads(tc["function"]["arguments"]),
                    provider=self.name,
                    raw_content=msg.get("content", ""),
                )
            return ToolCallResult(
                function_name="",
                arguments={},
                provider=self.name,
                raw_content=msg.get("content", ""),
            )
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)

    # --- embed --------------------------------------------------------------

    def embed(self, texts: list[str]) -> np.ndarray:
        import requests

        embeddings = []
        for text in texts:
            try:
                resp = requests.post(
                    f"{self._base_url}/v1/embeddings",
                    headers=self._headers(),
                    json={"model": self._embedding_model, "input": text},
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                emb = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"[minimax] embed failed for text: {e}")
                embeddings.append(np.zeros(1024, dtype=np.float32))
        return np.array(embeddings)

    # --- vision -------------------------------------------------------------

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
            "model": self._vision_model,
            "messages": messages,
            "max_tokens": 4096,
        }
        try:
            resp = requests.post(
                f"{self._base_url}/v1/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)
