"""Structure-aware chunking that preserves character offsets.

Offsets are the whole point: a citation has to be able to say "characters 4102-4288
of contracts.pdf", not "somewhere in chunk 7". Splitting on structure first
(headings, then paragraphs) keeps chunks semantically whole.
"""

from __future__ import annotations

import re

import tiktoken

from ..types import Chunk

_ENC = tiktoken.get_encoding("cl100k_base")

# Markdown ATX headings, and bare ALL-CAPS / numbered section lines common in PDFs.
_HEADING = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Z0-9 .,'\-]{6,}|\d+(\.\d+)*\s+\S.*)$", re.M)
_PARA_SPLIT = re.compile(r"\n\s*\n")


def _tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def _sections(text: str) -> list[tuple[str, int, int]]:
    """Split into (section_title, start, end) using headings as boundaries."""
    marks = [(m.start(), m.group().strip()) for m in _HEADING.finditer(text)]
    if not marks:
        return [("", 0, len(text))]

    out: list[tuple[str, int, int]] = []
    if marks[0][0] > 0:
        out.append(("", 0, marks[0][0]))
    for i, (pos, title) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out.append((title.lstrip("# ").strip(), pos, end))
    return out


def chunk_text(
    text: str, *, source: str = "", max_tokens: int = 512, overlap_tokens: int = 64
) -> list[Chunk]:
    """Pack paragraphs into chunks of <= max_tokens, overlapping by overlap_tokens."""
    chunks: list[Chunk] = []
    ordinal = 0

    for section, sec_start, sec_end in _sections(text):
        body = text[sec_start:sec_end]
        # (paragraph_text, absolute_start, absolute_end)
        paras: list[tuple[str, int, int]] = []
        cursor = 0
        for part in _PARA_SPLIT.split(body):
            idx = body.index(part, cursor) if part else cursor
            cursor = idx + len(part)
            if part.strip():
                paras.append((part, sec_start + idx, sec_start + cursor))

        buf: list[tuple[str, int, int]] = []
        buf_tokens = 0

        def flush() -> None:
            nonlocal buf, buf_tokens, ordinal
            if not buf:
                return
            content = "\n\n".join(p[0] for p in buf).strip()
            if content:
                chunks.append(
                    Chunk(
                        ordinal=ordinal,
                        content=content,
                        char_start=buf[0][1],
                        char_end=buf[-1][2],
                        section=section,
                        token_count=buf_tokens,
                        source=source,
                    )
                )
                ordinal += 1
            # Carry the tail forward so context is not severed at the boundary.
            carry: list[tuple[str, int, int]] = []
            carried = 0
            for p in reversed(buf):
                t = _tokens(p[0])
                if carried + t > overlap_tokens:
                    break
                carry.insert(0, p)
                carried += t
            buf = carry
            buf_tokens = carried

        for para, p_start, p_end in paras:
            t = _tokens(para)
            # A single oversized paragraph: emit it alone rather than dropping it.
            if t > max_tokens:
                flush()
                chunks.append(
                    Chunk(
                        ordinal=ordinal,
                        content=para.strip(),
                        char_start=p_start,
                        char_end=p_end,
                        section=section,
                        token_count=t,
                        source=source,
                    )
                )
                ordinal += 1
                buf, buf_tokens = [], 0
                continue

            if buf_tokens + t > max_tokens:
                flush()
            buf.append((para, p_start, p_end))
            buf_tokens += t

        flush()
        buf, buf_tokens = [], 0

    return chunks
