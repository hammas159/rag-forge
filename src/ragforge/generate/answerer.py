"""Retrieve -> generate -> verify citations -> gate on grounding -> answer or refuse."""

from __future__ import annotations

import json
import math
import re
import time

from ..config import get_settings
from ..embed import rerank
from ..llm import get_llm
from ..retrieve import retrieve
from ..types import Answer, ScoredChunk
from .citations import verify_quotes
from .prompts import REFUSAL, SYSTEM, build_prompt

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def _parse_json(raw: str) -> dict:
    """Small models wrap JSON in prose or fences. Recover rather than fail."""
    for candidate in (raw, raw.strip().strip("`")):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    m = _JSON_BLOCK.search(raw)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # Last resort: treat the whole output as an unsupported answer.
    return {"answer": raw.strip(), "supported": False, "quotes": []}


def _grounding_score(answer_text: str, retrieved: list[ScoredChunk]) -> float:
    """How well the retrieved evidence supports the answer, in [0, 1].

    The cross-encoder is reused as an entailment proxy: it is trained to score
    (query, passage) relevance, and scoring (answer, passage) is close enough to
    catch answers that drifted away from the evidence. A dedicated NLI model would
    be sharper; this avoids a third 2 GB download for a measurable share of the benefit.
    """
    if not answer_text.strip() or not retrieved:
        return 0.0
    scores = rerank(answer_text, [r.chunk.content for r in retrieved])
    if not scores:
        return 0.0
    best = max(scores)
    return 1.0 / (1.0 + math.exp(-best))  # logit -> probability


def answer_question(question: str, *, top_k: int | None = None) -> Answer:
    s = get_settings()
    started = time.perf_counter()
    llm = get_llm(s)

    retrieved = retrieve(question, top_k=top_k)
    if not retrieved:
        return Answer(
            text=REFUSAL,
            refused=True,
            grounding_score=0.0,
            latency_ms=int((time.perf_counter() - started) * 1000),
            backend=llm.name,
        )

    raw = llm.complete(
        build_prompt(question, [r.chunk.content for r in retrieved]),
        system=SYSTEM,
        max_tokens=1024,
    )
    parsed = _parse_json(raw)
    text = str(parsed.get("answer", "")).strip()
    citations, unverified = verify_quotes(parsed.get("quotes") or [], retrieved)
    grounding = _grounding_score(text, retrieved)

    # Three independent reasons to refuse. Any one is enough.
    refuse = (
        not parsed.get("supported", False)
        or grounding < s.grounding_threshold
        or (not citations and bool(unverified))
    )

    return Answer(
        text=REFUSAL if refuse else text,
        citations=[] if refuse else citations,
        refused=refuse,
        grounding_score=round(grounding, 4),
        latency_ms=int((time.perf_counter() - started) * 1000),
        backend=llm.name,
        retrieved=retrieved,
    )
