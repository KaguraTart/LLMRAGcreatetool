"""Multi-document synthesis: Map-Reduce + conflict detection."""
from __future__ import annotations
import json, logging, re
from .prompts import MAP_EXTRACT_PROMPT, REDUCE_MERGE_PROMPT, CONFLICT_DETECT_PROMPT

logger = logging.getLogger(__name__)

class Synthesizer:
    def __init__(self, provider):
        self.provider = provider

    def synthesize(self, question: str, chunks: list[dict]) -> str:
        extracts = []
        for i, chunk in enumerate(chunks, 1):
            prompt = MAP_EXTRACT_PROMPT.format(
                question=question, source_id=f"SOURCE_{i}",
                passage=chunk.get("content","")[:1500])
            try:
                result = self.provider.generate(prompt, temperature=0.2, max_tokens=300)
                if "NOT RELEVANT" not in result.upper():
                    extracts.append(f"[SOURCE_{i}]: {result.strip()}")
            except Exception as e:
                logger.warning(f"Map extract failed for chunk {i}: {e}")

        if not extracts:
            return "No relevant information found in the provided context."

        reduce_prompt = REDUCE_MERGE_PROMPT.format(
            question=question, extracts="\n".join(extracts))
        try:
            return self.provider.generate(reduce_prompt, temperature=0.3, max_tokens=1024)
        except Exception as e:
            logger.warning(f"Reduce merge failed: {e}")
            return "\n".join(extracts)

    def detect_conflicts(self, question: str, chunks: list[dict]) -> list[dict]:
        conflicts = []
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                prompt = CONFLICT_DETECT_PROMPT.format(
                    question=question,
                    source_a=f"SOURCE_{i+1}", text_a=chunks[i].get("content","")[:500],
                    source_b=f"SOURCE_{j+1}", text_b=chunks[j].get("content","")[:500])
                try:
                    resp = self.provider.generate(prompt, temperature=0, max_tokens=100, json_mode=True)
                    m = re.search(r'\{[\s\S]*\}', resp)
                    if m:
                        data = json.loads(m.group())
                        if data.get("contradiction"):
                            conflicts.append({"source_a": i+1, "source_b": j+1,
                                              "description": data.get("description","")})
                except Exception:
                    pass
        return conflicts
