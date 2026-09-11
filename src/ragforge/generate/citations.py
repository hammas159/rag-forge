"""Verify model-emitted quotes against the real source text.

The model is never trusted to report where something came from. Every quote is
located inside the passage it claims, and only then converted into an absolute
character span of the original document. A quote that cannot be located is dropped -
which is itself a hallucination signal.
"""

from __future__ import annotations

import re

from ..types import Citation, ScoredChunk

_WS = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def _locate(haystack: str, needle: str) -> tuple[int, int] | None:
    """Find needle in haystack, tolerating whitespace differences only."""
    exact = haystack.find(needle)
    if exact != -1:
        return exact, exact + len(needle)

    # Map each non-space character of the normalised form back to its original index.
    idx_map: list[int] = []
    norm_chars: list[str] = []
    prev_space = True
    for i, ch in enumerate(haystack):
        if ch.isspace():
            if not prev_space:
                norm_chars.append(" ")
                idx_map.append(i)
            prev_space = True
        else:
            norm_chars.append(ch.lower())
            idx_map.append(i)
            prev_space = False

    norm = "".join(norm_chars)
    target = _normalise(needle)
    if not target:
        return None
    pos = norm.find(target)
    if pos == -1:
        return None
    start = idx_map[pos]
    end = idx_map[min(pos + len(target) - 1, len(idx_map) - 1)] + 1
    return start, end


def verify_quotes(
    quotes: list[dict], retrieved: list[ScoredChunk]
) -> tuple[list[Citation], list[str]]:
    """Return (verified citations, list of quotes that could not be located)."""
    citations: list[Citation] = []
    unverified: list[str] = []

    for q in quotes:
        text = (q.get("quote") or "").strip()
        if not text:
            continue

        idx = q.get("passage")
        # Prefer the passage the model named; fall back to searching all of them,
        # because a right quote with a wrong passage number is still a real quote.
        order = []
        if isinstance(idx, int) and 1 <= idx <= len(retrieved):
            order.append(retrieved[idx - 1])
        order.extend(r for r in retrieved if r not in order)

        for sc in order:
            span = _locate(sc.chunk.content, text)
            if span is None:
                continue
            citations.append(
                Citation(
                    source=sc.chunk.source,
                    section=sc.chunk.section,
                    # Offsets are chunk-relative + the chunk's own offset in the document.
                    char_start=sc.chunk.char_start + span[0],
                    char_end=sc.chunk.char_start + span[1],
                    quote=sc.chunk.content[span[0] : span[1]],
                )
            )
            break
        else:
            unverified.append(text)

    return citations, unverified
