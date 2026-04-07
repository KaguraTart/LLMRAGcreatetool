"""
Centralized Prompt Templates — all provider-neutral plain strings.
"""

ANSWER_SYSTEM = (
    "You are a precise, helpful RAG assistant.\n"
    "Answer based ONLY on the provided context passages.\n"
    "Cite sources inline as [SOURCE_1], [SOURCE_2], etc.\n"
    "If the context does not contain enough information, say so clearly.\n"
    "Do not fabricate facts."
)

ANSWER_WITH_COT = (
    "<think>\nLet me read the context and reason step by step before answering.\n</think>\n\n"
    "{context}\n\nQuestion: {question}\n\n"
    "Please answer based on the context above. Cite each fact with [SOURCE_N]."
)

ANSWER_FACTUAL = (
    "Context passages:\n{context}\n\nQuestion: {question}\n\n"
    "Answer concisely and cite sources as [SOURCE_N]."
)

ANSWER_PROCEDURAL = (
    "Context passages:\n{context}\n\nQuestion: {question}\n\n"
    "Provide a numbered step-by-step answer. Cite each step with [SOURCE_N]."
)

ANSWER_COMPARATIVE = (
    "Context passages:\n{context}\n\nQuestion: {question}\n\n"
    "Compare the items systematically. Use a structured format. Cite with [SOURCE_N]."
)

ANSWER_SUMMARIZATION = (
    "Context passages:\n{context}\n\nQuestion: {question}\n\n"
    "Write a comprehensive summary covering all key points. Cite with [SOURCE_N]."
)

SELF_VERIFY_SYSTEM = "You are a strict fact-checker. Only output JSON."

SELF_VERIFY_PROMPT = (
    "Given the answer and the source passages, evaluate:\n"
    "1. faithfulness (0-1): Is every claim in the answer supported by the context?\n"
    "2. relevance (0-1): Does the answer address the question?\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:\n{answer}\n\n"
    'Output JSON: {"faithfulness": 0.0, "relevance": 0.0, "issues": "brief note or empty"}'
)

ANSWER_QUALITY_SYSTEM = "You are an answer quality evaluator. Only output JSON."

ANSWER_QUALITY_PROMPT = (
    "Evaluate the following Q&A pair across four dimensions (each 0-10):\n"
    "- faithfulness: Is every claim grounded in the provided context?\n"
    "- relevance: Does the answer directly address the question?\n"
    "- completeness: Does the answer cover all important aspects?\n"
    "- coherence: Is the answer well-structured and easy to follow?\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:\n{answer}\n\n"
    "Output JSON:\n"
    '{"faithfulness":<float 0-10>,"relevance":<float 0-10>,'
    '"completeness":<float 0-10>,"coherence":<float 0-10>,'
    '"reasoning":"<one-sentence rationale>"}'
)

MAP_EXTRACT_PROMPT = (
    "Extract the key information from this passage relevant to the question.\n"
    "Be concise. If not relevant, output \"NOT RELEVANT\".\n\n"
    "Question: {question}\n\nPassage [{source_id}]:\n{passage}\n\nKey information:"
)

REDUCE_MERGE_PROMPT = (
    "Synthesize the extracted information into a coherent, comprehensive answer.\n"
    "Resolve contradictions by noting them. Cite sources as [SOURCE_N].\n\n"
    "Question: {question}\n\nExtracted information:\n{extracts}\n\nSynthesized answer:"
)

CONFLICT_DETECT_PROMPT = (
    "Compare these two passages and determine if they contradict each other.\n\n"
    "Question: {question}\n\n"
    "Passage A [{source_a}]: {text_a}\n\nPassage B [{source_b}]: {text_b}\n\n"
    'Output JSON: {"contradiction": true, "description": "brief explanation or empty"}'
)

FOLLOWUP_RESOLVE_PROMPT = (
    "Rewrite the follow-up question as a fully standalone question "
    "(no pronouns referencing prior context).\n\n"
    "Conversation history:\n{history}\n\nFollow-up question: {question}\n\n"
    "Standalone question:"
)

COMPACT_HISTORY_PROMPT = (
    "Summarize the conversation into a compact context summary (max 3 sentences) "
    "that preserves the key facts established.\n\nConversation:\n{history}\n\nSummary:"
)
