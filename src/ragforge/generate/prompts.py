SYSTEM = """You answer strictly from the numbered CONTEXT passages given to you.

Rules:
- Use only information present in the CONTEXT. Never use prior knowledge.
- If the CONTEXT does not contain the answer, say so plainly. Do not guess.
- Every factual sentence must be supported by a verbatim quote from the CONTEXT.

Return ONLY a JSON object, no prose before or after, of this exact shape:
{
  "answer": "<your answer, or an explicit statement that the context does not cover it>",
  "supported": true|false,
  "quotes": [
    {"passage": <passage number>, "quote": "<verbatim substring copied exactly from that passage>"}
  ]
}

A quote must be copied character-for-character from the passage. Do not paraphrase,
reformat, fix typos, or join text across passages. If you cannot answer from the
CONTEXT, set "supported" to false and return an empty "quotes" list."""


def build_prompt(question: str, passages: list[str]) -> str:
    blocks = "\n\n".join(f"[passage {i}]\n{p}" for i, p in enumerate(passages, start=1))
    return f"CONTEXT:\n{blocks}\n\nQUESTION: {question}\n\nJSON:"


REFUSAL = (
    "I could not find support for this in the indexed documents, so I am not going to "
    "answer. Try rephrasing, or check that the relevant document has been ingested."
)
