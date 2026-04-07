"""Shared token counting helpers with tiktoken fallback."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None


class TokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base", chars_per_token: int = 4):
        self.encoding_name = encoding_name
        self.chars_per_token = max(chars_per_token, 1)
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None and tiktoken is not None:
            try:
                self._encoder = tiktoken.get_encoding(self.encoding_name)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer {self.encoding_name}: {e}")
        return self._encoder

    def count(self, text: str) -> int:
        if not text:
            return 0
        enc = self.encoder
        if enc is not None:
            try:
                return len(enc.encode(text))
            except Exception:
                pass
        return max(1, len(text) // self.chars_per_token)

    def count_batch(self, texts: list[str]) -> list[int]:
        return [self.count(t) for t in texts]


_default_counter = TokenCounter()


def count_tokens(text: str) -> int:
    return _default_counter.count(text)


def count_tokens_batch(texts: list[str]) -> list[int]:
    return _default_counter.count_batch(texts)
