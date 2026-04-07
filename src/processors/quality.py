"""
Quality Scorer
LLM self-assessment of text quality (completeness / clarity / independence / value)
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    score: float           # 0.0 - 10.0
    completeness: float    # information completeness
    clarity: float         # clarity
    independence: float    # independent readability
    value: float           # information value
    method: str            # llm / rule


class QualityScorer:
    """
    Text quality scorer
    
    Default: LLM self-assessment
    Fallback: Rule-based scoring (when LLM is unavailable)
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    def score(self, text: str, method: str = "auto") -> QualityScore:
        """
        Evaluate text quality
        
        Args:
            text: Text to evaluate
            method: auto / llm / rule
            
        Returns:
            QualityScore
        """
        if method == "rule" or (method == "auto" and not self.llm):
            return self._score_rule(text)
        else:
            return self._score_llm(text)
    
    def _score_rule(self, text: str) -> QualityScore:
        """
        Rule-based scoring (fallback when LLM is unavailable)
        
        Scoring dimensions:
        - Length: too short = low quality, too long needs splitting
        - Punctuation density: measures sentence completeness
        - Unique character ratio: measures information richness
        """
        if not text:
            return QualityScore(
                score=0.0, completeness=0, clarity=0,
                independence=0, value=0, method="rule"
            )
        
        # Length scoring
        length = len(text)
        length_score = min(length / 500, 1.0) * 8  # 0-8 points
        
        # Punctuation density (period/comma/semicolon density)
        punct_count = sum(1 for c in text if c in '。,，;；')
        punct_ratio = punct_count / length if length > 0 else 0
        punct_score = min(punct_ratio * 200, 2)  # 0-2 points
        
        # Unique character ratio (less repetition = richer information)
        unique_ratio = len(set(text)) / length if length > 0 else 0
        diversity_score = unique_ratio * 5  # 0-5 points
        
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
        """
        LLM self-assessment (requires llm_client)
        
        Prompt engineering guides LLM to score across four dimensions:
        - Completeness (no missing endings or taking things out of context)
        - Clarity (no ambiguity or fragmentation)
        - Independence (understandable without surrounding context)
        - Value (substantial content vs. empty/repetitive)
        """
        if not self.llm:
            return self._score_rule(text)
        
        truncated = text[:3000]  # Length limit
        
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
            
            import json, re
            # Extract JSON
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
        """
        Batch scoring and filtering
        
        Returns:
            [(QualityScore, text), ...] Only returns items with score >= min_score
        """
        scored = []
        for text in texts:
            score = self.score(text, method=method)
            if score.score >= min_score * 10:  # Normalize
                scored.append((score, text))
        
        return scored
