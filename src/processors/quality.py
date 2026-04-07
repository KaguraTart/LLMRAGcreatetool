"""
Quality Scorer
LLM self-assessment of text quality (completeness / clarity / independence / value)
"""

import json
import logging
import re
from dataclasses import dataclass

from ..utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

DEFAULT_EMPTY_QUERY_RELEVANCE = 0.7
DEFAULT_NO_SOURCE_GROUNDING = 0.5
DEFAULT_NO_SOURCE_CITATION_COVERAGE = 0.5


@dataclass
class QualityScore:
    score: float           # 0.0 - 10.0
    completeness: float    # information completeness
    clarity: float         # clarity
    independence: float    # independent readability
    value: float           # information value
    method: str            # llm / rule


@dataclass
class AnswerQualityScore:
    score: float                    # 0.0 - 1.0
    faithfulness: float             # grounded in context
    relevance: float                # answers the question
    grounding: float                # citation/context alignment
    readability: float              # readability/structure
    citation_coverage: float        # citations presence/coverage
    method: str                     # llm / rule
    passed: bool                    # pass threshold or not
    reasons: list[str]


class QualityScorer:
    """
    Text quality scorer

    Default: LLM self-assessment
    Fallback: Rule-based scoring (when LLM is unavailable)
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def score(self, text: str, method: str = "auto") -> QualityScore:
        if method == "rule" or (method == "auto" and not self.llm):
            return self._score_rule(text)
        return self._score_llm(text)

    def _score_rule(self, text: str) -> QualityScore:
        if not text:
            return QualityScore(
                score=0.0, completeness=0, clarity=0,
                independence=0, value=0, method="rule"
            )

        length = len(text)
        length_score = min(length / 500, 1.0) * 8

        punct_count = sum(1 for c in text if c in '。,，;；')
        punct_ratio = punct_count / length if length > 0 else 0
        punct_score = min(punct_ratio * 200, 2)

        unique_ratio = len(set(text)) / length if length > 0 else 0
        diversity_score = unique_ratio * 5

        total = length_score + punct_score + diversity_score

        return QualityScore(
            score=total,
            completeness=length_score / 8 * 10,
            clarity=punct_score / 2 * 10,
            independence=unique_ratio * 10,
            value=diversity_score / 5 * 10,
            method="rule"
        )

    def _score_llm(self, text: str) -> QualityScore:
        if not self.llm:
            return self._score_rule(text)

        truncated = text[:3000]

        prompt = f"""
Please evaluate the quality of the following text snippet, scoring it across four dimensions (0-10 for each):

Text to evaluate:
---
{truncated}
---

Scoring dimensions:
1. completeness: Is the information complete? Incomplete endings or taking things out of context = low score
2. clarity: Is the meaning clear and coherent? Ambiguity or fragmentation = low score
3. independence: Is it understandable without surrounding context? Over-reliance on context = low score
4. value: Does it contain substantial content? Empty or repetitive = low score

Output strictly in the following JSON format (no other content):
{{
  "completeness": float between 0-10,
  "clarity": float between 0-10,
  "independence": float between 0-10,
  "value": float between 0-10,
  "reasoning": "one-sentence scoring rationale"
}}
"""

        try:
            response = self.llm.generate(prompt)
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                data = json.loads(match.group())

                completeness = float(data.get('completeness', 5))
                clarity = float(data.get('clarity', 5))
                independence = float(data.get('independence', 5))
                value = float(data.get('value', 5))

                total = (completeness + clarity + independence + value) / 4

                return QualityScore(
                    score=total,
                    completeness=completeness,
                    clarity=clarity,
                    independence=independence,
                    value=value,
                    method="llm"
                )
        except Exception as e:
            logger.warning(f"LLM scoring failed: {e}")

        return self._score_rule(text)

    def batch_score(
        self,
        texts: list[str],
        min_score: float = 0.0,
        method: str = "auto"
    ) -> list[tuple[QualityScore, str]]:
        scored = []
        for text in texts:
            score = self.score(text, method=method)
            if score.score >= min_score * 10:
                scored.append((score, text))

        return scored


class AnswerQualityScorer:
    """Quality scorer for final generated answers."""

    def __init__(self, llm_client=None, min_score: float = 0.5):
        self.llm = llm_client
        self.min_score = min_score

    def score(
        self,
        question: str,
        answer: str,
        sources: list[dict] | None = None,
        method: str = "auto",
    ) -> AnswerQualityScore:
        if method == "rule" or (method == "auto" and not self.llm):
            return self._score_rule(question, answer, sources or [])
        return self._score_llm(question, answer, sources or [])

    def _score_rule(self, question: str, answer: str, sources: list[dict]) -> AnswerQualityScore:
        answer = (answer or "").strip()
        q = (question or "").strip()
        reasons = []

        if not answer:
            return AnswerQualityScore(
                score=0.0,
                faithfulness=0.0,
                relevance=0.0,
                grounding=0.0,
                readability=0.0,
                citation_coverage=0.0,
                method="rule",
                passed=False,
                reasons=["Empty answer"],
            )

        ans_tokens = max(count_tokens(answer), 1)
        q_terms = {t.lower() for t in re.split(r"\W+", q) if len(t) > 2}
        a_terms = {t.lower() for t in re.split(r"\W+", answer) if len(t) > 2}

        relevance = min(1.0, len(q_terms & a_terms) / len(q_terms)) if q_terms else DEFAULT_EMPTY_QUERY_RELEVANCE

        source_text = "\n".join((s.get("content", "") or "")[:1200] for s in sources)
        if source_text:
            source_terms = {t.lower() for t in re.split(r"\W+", source_text) if len(t) > 2}
            grounding = min(1.0, len(a_terms & source_terms) / max(len(a_terms), 1)) if a_terms else 0.0
        else:
            grounding = DEFAULT_NO_SOURCE_GROUNDING

        citation_count = len(re.findall(r"\[SOURCE_\d+\]", answer))
        # If there are no retrieved sources, use neutral fallback instead of penalizing to zero.
        citation_coverage = min(1.0, citation_count / max(len(sources), 1)) if sources else DEFAULT_NO_SOURCE_CITATION_COVERAGE

        readability = 1.0 if ans_tokens >= 20 else ans_tokens / 20.0
        faithfulness = (grounding + citation_coverage) / 2

        total = (faithfulness + relevance + grounding + readability + citation_coverage) / 5
        passed = total >= self.min_score

        if relevance < 0.4:
            reasons.append("Low question relevance")
        if grounding < 0.4:
            reasons.append("Weak grounding to retrieved context")
        if citation_coverage < 0.3:
            reasons.append("Insufficient citations")
        if readability < 0.4:
            reasons.append("Answer too short/unclear")

        return AnswerQualityScore(
            score=total,
            faithfulness=faithfulness,
            relevance=relevance,
            grounding=grounding,
            readability=readability,
            citation_coverage=citation_coverage,
            method="rule",
            passed=passed,
            reasons=reasons,
        )

    def _score_llm(self, question: str, answer: str, sources: list[dict]) -> AnswerQualityScore:
        if not self.llm:
            return self._score_rule(question, answer, sources)

        context = "\n\n".join(
            f"[SOURCE_{i+1}] {s.get('content', '')[:600]}" for i, s in enumerate(sources[:8])
        )
        prompt = f"""
Evaluate the final QA answer quality and return JSON only.

Question:
{question}

Answer:
{answer[:3000]}

Retrieved context:
{context[:4000]}

Return JSON with fields:
{{
  "faithfulness": 0.0-1.0,
  "relevance": 0.0-1.0,
  "grounding": 0.0-1.0,
  "readability": 0.0-1.0,
  "citation_coverage": 0.0-1.0,
  "reasons": ["short reason", "..."]
}}
"""
        try:
            response = self.llm.generate(prompt, temperature=0.0, max_tokens=600, json_mode=True)
            m = re.search(r'\{[\s\S]*\}', response)
            if m:
                data = json.loads(m.group())
                faithfulness = float(data.get("faithfulness", 0.7))
                relevance = float(data.get("relevance", 0.7))
                grounding = float(data.get("grounding", 0.7))
                readability = float(data.get("readability", 0.7))
                citation_coverage = float(data.get("citation_coverage", 0.7))
                reasons = data.get("reasons", [])
                if not isinstance(reasons, list):
                    reasons = [str(reasons)]

                total = (faithfulness + relevance + grounding + readability + citation_coverage) / 5
                return AnswerQualityScore(
                    score=total,
                    faithfulness=faithfulness,
                    relevance=relevance,
                    grounding=grounding,
                    readability=readability,
                    citation_coverage=citation_coverage,
                    method="llm",
                    passed=total >= self.min_score,
                    reasons=[str(r) for r in reasons],
                )
        except Exception as e:
            logger.warning(f"LLM answer quality scoring failed: {e}")

        return self._score_rule(question, answer, sources)
