"""File -> text -> chunks -> embeddings -> Postgres."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from ..config import get_settings
from ..db import connection
from ..migrate import ensure_dim
from ..embed import embed_passages
from .chunker import chunk_text
from .readers import SUPPORTED, read_file

console = Console()


def _upsert_document(conn, source: str, title: str) -> int:
    """Re-ingesting a file replaces its chunks (ON DELETE CASCADE) rather than duplicating."""
    row = conn.execute(
        """
        INSERT INTO documents (source, title) VALUES (%s, %s)
        ON CONFLICT (source) DO UPDATE SET title = EXCLUDED.title
        RETURNING id
        """,
        (source, title),
    ).fetchone()
    conn.execute("DELETE FROM chunks WHERE document_id = %s", (row["id"],))
    return row["id"]


def ingest_file(path: Path) -> int:
    s = get_settings()
    text, title = read_file(path)
    if not text.strip():
        console.print(f"[yellow]skip[/] {path.name} - no extractable text")
        return 0

    chunks = chunk_text(
        text,
        source=str(path),
        max_tokens=s.chunk_tokens,
        overlap_tokens=s.chunk_overlap,
    )
    if not chunks:
        return 0

    vectors = embed_passages([c.content for c in chunks])

    with connection() as conn:
        ensure_dim(conn, len(vectors[0]))
        doc_id = _upsert_document(conn, str(path), title)
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks
                    (document_id, ordinal, content, char_start, char_end,
                     section, token_count, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        doc_id,
                        c.ordinal,
                        c.content,
                        c.char_start,
                        c.char_end,
                        c.section,
                        c.token_count,
                        v,
                    )
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
        conn.commit()

    console.print(f"[green]ok[/] {path.name} -> {len(chunks)} chunks")
    return len(chunks)


def ingest_path(target: str | Path) -> int:
    target = Path(target)
    files = (
        [target]
        if target.is_file()
        else sorted(p for p in target.rglob("*") if p.suffix.lower() in SUPPORTED)
    )
    if not files:
        console.print(f"[yellow]nothing to ingest under {target}[/]")
        return 0

    total = sum(ingest_file(p) for p in files)
    console.print(f"[bold green]ingested {len(files)} file(s), {total} chunks[/]")
    return total
