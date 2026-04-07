"""
Answer Generator

Features:
- Chain-of-thought generation with <think> prefix
- [SOURCE_N] inline citations
- Answer templates by intent type
- Self-verification loop (faithfulness + relevance)
- Streaming output via answer_stream()
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Iterator

from ..query.query_processor import QueryIntent
from .prompts import (
    ANSWER_SYSTEM, ANSWER_WITH_COT, ANSWER_FACTUAL, ANSWER_PROCEDURAL,
    ANSWER_COMPARATIVE, ANSWER_SUMMARIZATION,
    SELF_VERIFY_SYSTEM, SELF_VERIFY_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    intent: QueryIntent = QueryIntent.UNKNOWN
    faithfulness: float = 1.0
    relevance: float = 1.0
    verified: bool = False
    provider: str = ""


class AnswerGenerator:
    """
    Generates grounded, cited answers from retrieved context passages.
    """

    def __init__(
        self,
        provider,
        cot_enabled: bool = True,
        self_verify: bool = True,
        min_faithfulness: float = 0.6,
    ):
        self.provider = provider
        self.cot_enabled = cot_enabled
        self.self_verify = self_verify
        self.min_faithfulness = min_faithfulness

    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        context_chunks: list[dict],
        intent: QueryIntent = QueryIntent.FACTUAL,
    ) -> GeneratedAnswer:
        """Generate a grounded answer with inline citations."""
        context_str, _ = self._build_context(context_chunks)
        prompt = self._build_prompt(question, context_str, intent)
        answer = self.provider.generate(prompt, system=ANSWER_SYSTEM, temperature=0.3)
        answer = self._clean_cot(answer)

        result = GeneratedAnswer(
            question=question,
            answer=answer,
            sources=context_chunks,
            intent=intent,
            provider=getattr(self.provider, "name", ""),
        )

        if self.self_verify:
            self._verify(result, context_str)

        return result

    def answer_stream(
        self,
        question: str,
        context_chunks: list[dict],
        intent: QueryIntent = QueryIntent.FACTUAL,
    ) -> Iterator[str]:
        """Stream the answer token-by-token."""
        context_str, _ = self._build_context(context_chunks)
        prompt = self._build_prompt(question, context_str, intent)

        if self.provider.supports_streaming:
            for chunk in self.provider.generate_stream(
                    prompt, system=ANSWER_SYSTEM, temperature=0.3):
                yield chunk
        else:
            yield self.provider.generate(prompt, system=ANSWER_SYSTEM, temperature=0.3)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_context(self, chunks: list[dict]) -> tuple[str, dict]:
        lines = []
        source_map = {}
        for i, chunk in enumerate(chunks, 1):
            label = f"SOURCE_{i}"
            content = chunk.get("content", "")
            source = chunk.get("metadata", {}).get("source", "unknown")
            lines.append(f"[{label}] (from: {source})\n{content}")
            source_map[label] = chunk
        return "\n\n".join(lines), source_map

    def _build_prompt(self, question: str, context: str, intent: QueryIntent) -> str:
        template_map = {
            QueryIntent.PROCEDURAL: ANSWER_PROCEDURAL,
            QueryIntent.COMPARATIVE: ANSWER_COMPARATIVE,
            QueryIntent.SUMMARIZATION: ANSWER_SUMMARIZATION,
        }
        if self.cot_enabled:
            base = ANSWER_WITH_COT
        else:
            base = template_map.get(intent, ANSWER_FACTUAL)
        return base.format(context=context, question=question)

    @staticmethod
    def _clean_cot(text: str) -> str:
        """Remove <think>...</think> block from the final answer."""
        return re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.IGNORECASE).strip()

    def _verify(self, result: GeneratedAnswer, context_str: str):
        """Self-verification: check faithfulness and relevance."""
        prompt = SELF_VERIFY_PROMPT.format(
            context=context_str[:3000],
            question=result.question,
            answer=result.answer[:2000],
        )
        try:
            response = self.provider.generate(
                prompt, system=SELF_VERIFY_SYSTEM, temperature=0, json_mode=True)
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                data = json.loads(match.group())
                result.faithfulness = float(data.get("faithfulness", 1.0))
                result.relevance = float(data.get("relevance", 1.0))
                result.verified = True
                if result.faithfulness < self.min_faithfulness:
                    logger.warning(
                        f"Low faithfulness ({result.faithfulness:.2f}) for: "
                        f"{result.question[:60]}")
        except Exception as e:
            logger.warning(f"Self-verification failed: {e}")
