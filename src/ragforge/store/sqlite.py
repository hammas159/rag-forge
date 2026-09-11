"""SQLite store: FTS5 for the sparse half, brute-force cosine for the dense half.

Deliberately stdlib-only - no numpy, no server, no Docker. A demo corpus is a few
hundred chunks, where an exact scan is both fast enough (single-digit milliseconds)
and more accurate than an approximate index. This is what lets the public demo run
on free CPU hosting, and what lets a reviewer clone the repo and get an answer
without installing anything.

Vectors are stored as raw float32 bytes rather than JSON: one third of the size and
no parse cost per row.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

from ..types import Chunk


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    """Both sides are L2-normalised at embed time, so the dot product is the cosine."""
    return sum(x * y for x, y in zip(a, b))


class SqliteStore:
    name = "sqlite"

    def __init__(self, path: str = "data/ragforge.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")

    def ensure_schema(self, dim: int) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                source  TEXT NOT NULL UNIQUE,
                title   TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ordinal     INTEGER NOT NULL,
                content     TEXT NOT NULL,
                char_start  INTEGER NOT NULL,
                char_end    INTEGER NOT NULL,
                section     TEXT NOT NULL DEFAULT '',
                token_count INTEGER NOT NULL DEFAULT 0,
                embedding   BLOB
            );
            CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);

            -- FTS5 gives BM25 ranking natively; kept in sync by triggers.
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(content, content='chunks', content_rowid='id');

            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            END;
            """
        )
        self.conn.commit()

    def upsert_document(self, source: str, title: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO documents(source, title) VALUES (?, ?) "
            "ON CONFLICT(source) DO UPDATE SET title=excluded.title RETURNING id",
            (source, title),
        )
        doc_id = cur.fetchone()[0]
        self.conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        self.conn.commit()
        return doc_id

    def add_chunks(
        self, document_id: int, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        self.conn.executemany(
            "INSERT INTO chunks(document_id, ordinal, content, char_start, char_end,"
            " section, token_count, embedding) VALUES (?,?,?,?,?,?,?,?)",
            [
                (
                    document_id, c.ordinal, c.content, c.char_start, c.char_end,
                    c.section, c.token_count, _pack(v),
                )
                for c, v in zip(chunks, vectors)
            ],
        )
        self.conn.commit()

    def _rows(self, where: str = "", params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT c.id, c.document_id, c.ordinal, c.content, c.char_start, c.char_end,"
            " c.section, c.token_count, c.embedding, d.source"
            " FROM chunks c JOIN documents d ON d.id = c.document_id " + where,
            params,
        ).fetchall()

    def dense(self, vector: list[float], k: int) -> list[dict]:
        scored = []
        for row in self._rows("WHERE c.embedding IS NOT NULL"):
            item = dict(row)
            item["score"] = _cosine(vector, _unpack(item.pop("embedding")))
            scored.append(item)
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:k]

    def sparse(self, query: str, k: int) -> list[dict]:
        # FTS5 treats punctuation as syntax; quote each term so user input cannot
        # produce a malformed MATCH expression.
        terms = " OR ".join(f'"{t}"' for t in query.split() if t.strip())
        if not terms:
            return []
        try:
            rows = self.conn.execute(
                "SELECT c.id, c.document_id, c.ordinal, c.content, c.char_start,"
                " c.char_end, c.section, c.token_count, d.source, -bm25(chunks_fts) AS score"
                " FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid"
                " JOIN documents d ON d.id = c.document_id"
                " WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                (terms, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(r) for r in rows]

    def counts(self) -> tuple[int, int]:
        docs = self.conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        chunks = self.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return docs, chunks

    def healthy(self) -> bool:
        try:
            self.conn.execute("SELECT 1")
            return True
        except Exception:
            return False
