"""
Deprecated MiniMax API wrapper.

This module is kept for backward compatibility and forwards calls to
`src.providers.minimax_provider.MiniMaxProvider`.
"""

from __future__ import annotations

import logging
import warnings

from ..providers.minimax_provider import MiniMaxProvider

logger = logging.getLogger(__name__)


class MiniMaxClient:
    """Compatibility shim over MiniMaxProvider."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.minimaxi.com",
        model: str = "MiniMax-Text-01",
        vision_model: str = "MiniMax-Hailuo-VL-01",
        embedding_model: str = "embo-01",
        timeout: int = 60,
    ):
        warnings.warn(
            "MiniMaxClient is deprecated; use src.providers.MiniMaxProvider instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning(
            "MiniMaxClient is deprecated; forwarding to MiniMaxProvider. "
            "Please migrate imports from src.integrations.minimax_api to src.providers.minimax_provider."
        )

        self._provider = MiniMaxProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            vision_model=vision_model,
            embedding_model=embedding_model,
            timeout=timeout,
        )

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.vision_model = vision_model
        self.embedding_model = embedding_model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        return self._provider.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

    def generate_with_functions(
        self,
        prompt: str,
        functions: list[dict],
        system: str = "",
        temperature: float = 0.3,
    ) -> dict:
        result = self._provider.generate_with_tools(
            prompt=prompt,
            tools=functions,
            system=system,
            temperature=temperature,
        )
        if result.function_name:
            return {"function": result.function_name, "arguments": result.arguments}
        return {"content": result.raw_content}

    def understand_image(
        self,
        image_bytes: bytes,
        prompt: str = "Please describe this image in detail",
        detail: str = "high",
    ) -> str:
        # `detail` is retained for backward compatibility with the deprecated API.
        _ = detail
        return self._provider.understand_image(image_bytes=image_bytes, prompt=prompt)

    def extract_entities(self, text: str, schema: dict) -> dict:
        return self._provider.extract_entities(text=text, schema=schema)

    def classify(self, text: str, categories: list[str]) -> dict:
        return self._provider.classify(text=text, categories=categories)
