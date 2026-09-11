"""Keep the vector column's dimension in sync with the configured embedding model.

Swapping EMBED_MODEL changes the vector width. Rather than making that a manual
SQL chore (and a silent runtime error when someone forgets), the ingest path checks
the column and rebuilds it when it disagrees with the config.
"""

from __future__ import annotations

import re

from rich.console import Console

console = Console()

_DIM = re.compile(r"vector\((\d+)\)")


def current_dim(conn) -> int | None:
    row = conn.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod) AS coltype
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = 'chunks' AND a.attname = 'embedding' AND a.attnum > 0
        """
    ).fetchone()
    if not row:
        return None
    m = _DIM.search(row["coltype"] or "")
    return int(m.group(1)) if m else None


def ensure_dim(conn, dim: int) -> None:
    """No-op when already correct. Otherwise rebuild the column and its index.

    Existing embeddings cannot survive a dimension change - they were produced by a
    different model and are not comparable - so they are dropped deliberately and
    loudly, rather than left to poison retrieval.
    """
    found = current_dim(conn)
    if found == dim:
        return

    console.print(
        f"[yellow]embedding dimension {found} != configured {dim} — "
        f"rebuilding column and index; existing vectors will be re-embedded[/]"
    )
    conn.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
    conn.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS embedding")
    conn.execute(f"ALTER TABLE chunks ADD COLUMN embedding VECTOR({dim})")
    conn.execute(
        """
        CREATE INDEX chunks_embedding_idx ON chunks
        USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
        """
    )
    conn.commit()
    console.print(f"[green]schema now vector({dim})[/]")
