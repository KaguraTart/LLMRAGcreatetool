"""Multi-turn conversation session with context compaction."""
from __future__ import annotations
import json, logging
from dataclasses import dataclass, field
from typing import Optional
from .prompts import FOLLOWUP_RESOLVE_PROMPT, COMPACT_HISTORY_PROMPT

logger = logging.getLogger(__name__)

@dataclass
class Turn:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)

class ConversationSession:
    def __init__(self, provider, max_turns: int = 10, context_budget_tokens: int = 6000):
        self.provider = provider
        self.max_turns = max_turns
        self.context_budget_tokens = context_budget_tokens
        self.history: list[Turn] = []
        self._compacted_summary: str = ""

    def add_turn(self, question: str, answer: str, sources: list[dict] = None):
        self.history.append(Turn(question=question, answer=answer, sources=sources or []))
        if len(self.history) > self.max_turns:
            self._compact()

    def resolve_followup(self, question: str) -> str:
        if not self.history:
            return question
        hist_str = "\n".join(f"Q: {t.question}\nA: {t.answer[:200]}" for t in self.history[-3:])
        prompt = FOLLOWUP_RESOLVE_PROMPT.format(history=hist_str, question=question)
        try:
            return self.provider.generate(prompt, temperature=0.2, max_tokens=200).strip()
        except Exception as e:
            logger.warning(f"Follow-up resolve failed: {e}")
            return question

    def _compact(self):
        hist_str = "\n".join(f"Q: {t.question}\nA: {t.answer[:300]}" for t in self.history)
        prompt = COMPACT_HISTORY_PROMPT.format(history=hist_str)
        try:
            self._compacted_summary = self.provider.generate(prompt, temperature=0.3, max_tokens=200)
            self.history = self.history[-3:]
        except Exception as e:
            logger.warning(f"History compaction failed: {e}")
            self.history = self.history[-3:]

    def to_dict(self) -> dict:
        return {"history": [{"q": t.question, "a": t.answer} for t in self.history],
                "summary": self._compacted_summary}

    @classmethod
    def from_dict(cls, data: dict, provider) -> "ConversationSession":
        session = cls(provider)
        session._compacted_summary = data.get("summary", "")
        for item in data.get("history", []):
            session.history.append(Turn(question=item["q"], answer=item["a"]))
        return session
