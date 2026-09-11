"""Integration tests against a real Postgres.

Skipped when no database is reachable, so `pytest` still works on a clean clone and in
CI without services. What is tested here cannot be tested any other way: the bug these
exist for lived entirely in the SQL.
"""

from __future__ import annotations

import pytest

from ragforge.db import healthcheck
from ragforge.store import get_store
from ragforge.store.postgres import PostgresStore

pytestmark = pytest.mark.skipif(
    not healthcheck(), reason="no Postgres reachable; run `make up`"
)


@pytest.fixture(scope="module")
def seeded():
    """The seed corpus, ingested. Assumes `make ingest` has been run."""
    store = get_store()
    documents, chunks = store.counts()
    if not chunks:
        pytest.skip("index is empty; run `make ingest`")
    return store


class TestSparseRetrieval:
    def test_a_natural_language_question_matches_something(self, seeded):
        """The bug this exists for: websearch_to_tsquery joins every term with AND, so
        a real question only matches a chunk containing *all* of its words - which is
        essentially never. The sparse half returned nothing on every query, RRF had one
        list instead of two, and the system was silently dense-only while still calling
        itself hybrid.
        """
        rows = seeded.sparse("why use a cross encoder after the bi-encoder", 10)
        assert rows, "sparse retrieval returned nothing for a natural-language question"

    def test_partial_term_overlap_is_enough(self, seeded):
        """Partial matching is the entire point of the sparse side."""
        assert seeded.sparse("cross encoder zzzz qqqq", 10)

    def test_ranking_prefers_more_overlap(self, seeded):
        """ts_rank_cd should reward how many terms matched and how close they are."""
        rows = seeded.sparse("cross encoder bi-encoder precision recall", 10)
        scores = [r["score"] for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_a_quoted_phrase_still_works(self, seeded):
        """Rewriting the operators must not break websearch's phrase parsing."""
        assert seeded.sparse('"cross encoder"', 10) is not None

    def test_nonsense_matches_nothing(self, seeded):
        assert seeded.sparse("zzzzqqqq wwwwvvvv", 10) == []


class TestHybridFusion:
    def test_both_halves_contribute(self, seeded):
        """The assertion that would have caught the bug: a hit ranked by *both*
        retrievers. With the sparse side dead, every sparse_rank is None."""
        from ragforge.retrieve import retrieve

        hits = retrieve("why use a cross encoder after the bi-encoder", top_k=3)
        assert hits
        assert any(h.sparse_rank is not None for h in hits), (
            "no hit was found by sparse retrieval - hybrid search is dense-only"
        )
        assert any(h.dense_rank is not None for h in hits)

    def test_offsets_survive_the_round_trip(self, seeded):
        """The citation feature depends on this holding through the database."""
        from ragforge.retrieve import retrieve

        for hit in retrieve("reranking precision", top_k=3):
            assert 0 <= hit.chunk.char_start < hit.chunk.char_end


class TestSchema:
    def test_the_vector_column_matches_the_loaded_model(self, seeded):
        """ensure_dim reconciles these; a mismatch means inserts fail or, worse,
        retrieval silently compares incomparable vectors."""
        from ragforge.embed import embed_query
        from ragforge.migrate import current_dim
        from ragforge.db import connection

        with connection() as conn:
            column_width = current_dim(conn)
        assert column_width == len(embed_query("dimension probe"))

    def test_both_indexes_exist(self, seeded):
        from ragforge.db import connection

        with connection() as conn:
            names = {
                row["indexname"]
                for row in conn.execute(
                    "SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'"
                ).fetchall()
            }
        assert "chunks_embedding_idx" in names   # HNSW, dense half
        assert "chunks_tsv_idx" in names         # GIN, sparse half

    def test_the_store_is_the_postgres_one(self, seeded):
        assert isinstance(seeded, PostgresStore)
