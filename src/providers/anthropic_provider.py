"""
Anthropic Claude Provider
Supports: claude-3-5-sonnet, claude-3-opus, claude-3-haiku
"""
from __future__ import annotations
import base64, logging, os
from typing import Iterator
import numpy as np
from .base import (LLMProvider, ToolCallResult,
    AuthenticationError, ProviderError, RateLimitError, ProviderUnavailableError)

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API provider."""

    def __init__(self, api_key: str = "", model: str = "claude-3-5-sonnet-20241022",
                 timeout: int = 60):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model
        self._timeout = timeout
        if not self._api_key:
            logger.warning("[anthropic] ANTHROPIC_API_KEY not set")

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def supports_function_calling(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    def _headers(self) -> dict:
        if not self._api_key:
            raise AuthenticationError("ANTHROPIC_API_KEY is not set", provider=self.name)
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

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
        payload: dict = {
            "model": self._model, "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        try:
            resp = requests.post("https://api.anthropic.com/v1/messages",
                headers=self._headers(), json=payload, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)

    def generate_stream(self, prompt: str, system: str = "", temperature: float = 0.7,
                        max_tokens: int = 4096) -> Iterator[str]:
        import json as _json, requests
        payload: dict = {
            "model": self._model, "max_tokens": max_tokens,
            "temperature": temperature, "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        try:
            resp = requests.post("https://api.anthropic.com/v1/messages",
                headers=self._headers(), json=payload, timeout=self._timeout, stream=True)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        try:
                            ev = _json.loads(line[6:])
                            if ev.get("type") == "content_block_delta":
                                yield ev["delta"].get("text", "")
                        except Exception:
                            pass
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)

    def generate_with_tools(self, prompt: str, tools: list[dict], system: str = "",
                            temperature: float = 0.3) -> ToolCallResult:
        import json as _json, requests
        # Convert OpenAI-style function schema to Anthropic tool schema
        anthropic_tools = []
        for t in tools:
            anthropic_tools.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
            })
        payload: dict = {
            "model": self._model, "max_tokens": 4096,
            "temperature": temperature, "tools": anthropic_tools,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        try:
            resp = requests.post("https://api.anthropic.com/v1/messages",
                headers=self._headers(), json=payload, timeout=self._timeout)
            resp.raise_for_status()
            content = resp.json().get("content", [])
            for block in content:
                if block.get("type") == "tool_use":
                    return ToolCallResult(
                        function_name=block["name"],
                        arguments=block.get("input", {}),
                        provider=self.name,
                    )
            text = " ".join(b.get("text","") for b in content if b.get("type")=="text")
            return ToolCallResult(function_name="", arguments={},
                                  provider=self.name, raw_content=text)
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)

    def understand_image(self, image_bytes: bytes,
                         prompt: str = "Please describe this image in detail.") -> str:
        import requests
        b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self._model, "max_tokens": 4096,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                              "data": b64}},
                {"type": "text", "text": prompt},
            ]}],
        }
        try:
            resp = requests.post("https://api.anthropic.com/v1/messages",
                headers=self._headers(), json=payload, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]
        except requests.exceptions.RequestException as e:
            self._handle_http_error(e, self.name)
