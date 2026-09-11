"""Storage interface.

Two implementations, same contract:

  postgres  pgvector HNSW + tsvector GIN. What the project actually runs on.
  sqlite    FTS5 + brute-force cosine, stdlib only. What the public demo runs on,
            because free hosting gives you no database server and no Docker.

Retrieval and ingestion talk to this interface, so neither knows which is active.
"""

from __future__ import annotations

from typing import Protocol

from ..types import Chunk


class Store(Protocol):
    name: str

    def ensure_schema(self, dim: int) -> None:
        """Create or reconcile storage for vectors of width `dim`."""

    def upsert_document(self, source: str, title: str) -> int:
        """Insert or replace a document, clearing its old chunks. Returns its id."""

    def add_chunks(
        self, document_id: int, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None: ...

    def dense(self, vector: list[float], k: int) -> list[dict]:
        """Top-k by cosine similarity."""

    def sparse(self, query: str, k: int) -> list[dict]:
        """Top-k by lexical match."""

    def counts(self) -> tuple[int, int]:
        """(documents, chunks)"""

    def healthy(self) -> bool: ...
