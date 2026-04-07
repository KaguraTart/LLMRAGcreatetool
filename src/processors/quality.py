"""
质量评分器
LLM 自评文本质量（完整性 / 清晰度 / 独立性 / 价值）
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    score: float           # 0.0 - 10.0
    completeness: float    # 信息完整性
    clarity: float        # 清晰度
    independence: float   # 独立可读性
    value: float          # 信息价值
    method: str           # llm / rule


class QualityScorer:
    """
    文本质量评分器
    
    默认：LLM 自评
    Fallback: 规则评分（无 LLM 时使用）
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
    
    def score(self, text: str, method: str = "auto") -> QualityScore:
        """
        评估文本质量
        
        Args:
            text: 待评估文本
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
        规则评分（无 LLM 时的降级方案）
        
        评分维度：
        - 长度：过短=低质量，过长需拆分
        - 标点密度：衡量句子完整性
        - 独特字符比例：衡量信息丰富度
        """
        if not text:
            return QualityScore(
                score=0.0, completeness=0, clarity=0,
                independence=0, value=0, method="rule"
            )
        
        # 长度评分
        length = len(text)
        length_score = min(length / 500, 1.0) * 8  # 0-8 分
        
        # 标点密度（句号/逗号/分号密度）
        punct_count = sum(1 for c in text if c in '。,，;；')
        punct_ratio = punct_count / length if length > 0 else 0
        punct_score = min(punct_ratio * 200, 2)  # 0-2 分
        
        # 独特字符比例（重复少=信息丰富）
        unique_ratio = len(set(text)) / length if length > 0 else 0
        diversity_score = unique_ratio * 5  # 0-5 分
        
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
        LLM 自评（需要 llm_client）
        
        提示工程引导 LLM 从四个维度打分：
        - 信息完整性（是否有头无尾）
        - 清晰度（是否有歧义）
        - 独立可读性（脱离上下文是否可理解）
        - 信息价值（是否有实质内容）
        """
        if not self.llm:
            return self._score_rule(text)
        
        truncated = text[:3000]  # 限制长度
        
        prompt = f"""
请评估以下文本片段的质量，从四个维度打分（每个维度 0-10 分）：

评估文本：
---
{truncated}
---

评分维度：
1. completeness（完整性）：信息是否完整，有头无尾/断章取义=低分
2. clarity（清晰度）：语义是否清晰连贯，有歧义/碎片=低分
3. independence（独立可读性）：脱离上下文是否可理解，过于依赖前后文=低分
4. value（信息价值）：是否包含实质性内容，空洞/重复=低分

输出严格按以下 JSON 格式（不要有其他内容）：
{{
  "completeness": 0-10的浮点数,
  "clarity": 0-10的浮点数,
  "independence": 0-10的浮点数,
  "value": 0-10的浮点数,
  "reasoning": "一句话评分理由"
}}
"""
        
        try:
            response = self.llm.generate(prompt)
            
            import json, re
            # 提取 JSON
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
            logger.warning(f"LLM 评分失败: {e}")
        
        return self._score_rule(text)
    
    def batch_score(
        self, 
        texts: list[str],
        min_score: float = 0.0,
        method: str = "auto"
    ) -> list[tuple[QualityScore, str]]:
        """
        批量评分并过滤
        
        Returns:
            [(QualityScore, text), ...] 仅返回分数 >= min_score 的项
        """
        scored = []
        for text in texts:
            score = self.score(text, method=method)
            if score.score >= min_score * 10:  # 归一化
                scored.append((score, text))
        
        return scored
