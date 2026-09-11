"""Postgres store: pgvector HNSW for dense, tsvector GIN for sparse.

The production path. One datastore carries both halves of hybrid retrieval, so
there is no second search engine to run, back up or keep in sync.
"""

from __future__ import annotations

from ..db import connection, healthcheck
from ..migrate import ensure_dim
from ..types import Chunk

_SELECT = """
    SELECT c.id, c.document_id, c.ordinal, c.content, c.char_start, c.char_end,
           c.section, c.token_count, d.source
"""


class PostgresStore:
    name = "postgres"

    def ensure_schema(self, dim: int) -> None:
        with connection() as conn:
            ensure_dim(conn, dim)

    def upsert_document(self, source: str, title: str) -> int:
        with connection() as conn:
            row = conn.execute(
                "INSERT INTO documents (source, title) VALUES (%s, %s) "
                "ON CONFLICT (source) DO UPDATE SET title = EXCLUDED.title RETURNING id",
                (source, title),
            ).fetchone()
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (row["id"],))
            conn.commit()
            return row["id"]

    def add_chunks(
        self, document_id: int, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        with connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, ordinal, content, char_start, char_end,"
                " section, token_count, embedding) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                [
                    (
                        document_id, c.ordinal, c.content, c.char_start, c.char_end,
                        c.section, c.token_count, v,
                    )
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
            conn.commit()

    def dense(self, vector: list[float], k: int) -> list[dict]:
        with connection() as conn:
            return conn.execute(
                _SELECT
                + ", 1 - (c.embedding <=> %s::vector) AS score"
                " FROM chunks c JOIN documents d ON d.id = c.document_id"
                " WHERE c.embedding IS NOT NULL"
                " ORDER BY c.embedding <=> %s::vector LIMIT %s",
                (vector, vector, k),
            ).fetchall()

    # websearch_to_tsquery joins every term with AND, so a natural-language question
    # only matches a chunk containing *all* of its words - which for a real question is
    # essentially never. The sparse half then returns nothing on every query, RRF has
    # one list to fuse instead of two, and the system is silently dense-only while
    # still calling itself hybrid.
    #
    # Rewriting the operators to OR keeps websearch's parsing - quoted phrases, negation
    # - and lets ts_rank_cd do what it is for: rank by how many terms matched and how
    # close together they are. Partial matching is the whole point of the sparse side.
    _TSQUERY = "replace(websearch_to_tsquery('english', %s)::text, '&', '|')::tsquery"

    def sparse(self, query: str, k: int) -> list[dict]:
        with connection() as conn:
            return conn.execute(
                _SELECT
                + f", ts_rank_cd(c.tsv, {self._TSQUERY}) AS score"
                " FROM chunks c JOIN documents d ON d.id = c.document_id"
                f" WHERE c.tsv @@ {self._TSQUERY}"
                " ORDER BY score DESC LIMIT %s",
                (query, query, k),
            ).fetchall()

    def counts(self) -> tuple[int, int]:
        with connection() as conn:
            docs = conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"]
            chunks = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
        return docs, chunks

    def healthy(self) -> bool:
        return healthcheck()
