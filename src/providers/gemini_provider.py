"""
Google Gemini Provider
Supports: gemini-1.5-pro, gemini-1.5-flash, models/text-embedding-004
"""
from __future__ import annotations
import base64, logging, os
from typing import Iterator
import numpy as np
from .base import (LLMProvider, ToolCallResult,
    AuthenticationError, ProviderError, RateLimitError, ProviderUnavailableError)

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini REST API provider."""

    def __init__(self, api_key: str = "", model: str = "gemini-1.5-pro",
                 embedding_model: str = "models/text-embedding-004", timeout: int = 60):
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._model = model
        self._embedding_model = embedding_model
        self._timeout = timeout
        self._base = "https://generativelanguage.googleapis.com/v1beta"
        if not self._api_key:
            logger.warning("[gemini] GOOGLE_API_KEY not set")

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_embedding(self) -> bool:
        return True

    @staticmethod
    def _handle_http_error(e, provider: str):
        import requests
        if isinstance(e, requests.exceptions.HTTPError):
            code = e.response.status_code if e.response is not None else 0
            if code == 401 or code == 403:
                raise AuthenticationError(str(e), provider=provider) from e
            if code == 429: raise RateLimitError(str(e), provider=provider) from e
            if code >= 500: raise ProviderUnavailableError(str(e), provider=provider) from e
        raise ProviderError(str(e), provider=provider) from e

    def generate(self, prompt: str, system: str = "", temperature: float = 0.7,
                 max_tokens: int = 4096, json_mode: bool = False) -> str:
        import requests
        url = f"{self._base}/models/{self._model}:generateContent?key={self._api_key}"
        contents = []
        if system:
            contents.append({"role": "user", "parts": [{"text": f"[System]: {system}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        payload: dict = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        try:
            resp = requests.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)

    def generate_stream(self, prompt: str, system: str = "", temperature: float = 0.7,
                        max_tokens: int = 4096) -> Iterator[str]:
        import json as _json, requests
        url = (f"{self._base}/models/{self._model}:streamGenerateContent"
               f"?key={self._api_key}&alt=sse")
        contents = []
        if system:
            contents.append({"role": "user", "parts": [{"text": f"[System]: {system}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        payload = {"contents": contents,
                   "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
        try:
            resp = requests.post(url, json=payload, timeout=self._timeout, stream=True)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        try:
                            ev = _json.loads(line[6:])
                            text = ev["candidates"][0]["content"]["parts"][0].get("text","")
                            if text:
                                yield text
                        except Exception:
                            pass
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)

    def embed(self, texts: list[str]) -> np.ndarray:
        import requests
        url = f"{self._base}/{self._embedding_model}:batchEmbedContents?key={self._api_key}"
        requests_list = [{"content": {"parts": [{"text": t}]}} for t in texts]
        try:
            resp = requests.post(url, json={"requests": requests_list}, timeout=self._timeout)
            resp.raise_for_status()
            embeddings = np.array(
                [e["values"] for e in resp.json()["embeddings"]], dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return embeddings / norms
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)

    def understand_image(self, image_bytes: bytes,
                         prompt: str = "Please describe this image in detail.") -> str:
        import requests
        b64 = base64.b64encode(image_bytes).decode()
        url = f"{self._base}/models/{self._model}:generateContent?key={self._api_key}"
        payload = {"contents": [{"parts": [
            {"inlineData": {"mimeType": "image/png", "data": b64}},
            {"text": prompt},
        ]}]}
        try:
            resp = requests.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)
