"""The SQLite store is what the public demo runs on, so its behaviour is tested
directly rather than assumed to match the Postgres path."""

import sqlite3
import struct

import pytest


@pytest.fixture
def store(tmp_path):
    from ragforge.store.sqlite import SqliteStore

    s = SqliteStore(str(tmp_path / "t.db"))
    s.ensure_schema(3)
    return s


def _chunk(ordinal, content):
    from ragforge.types import Chunk

    return Chunk(
        ordinal=ordinal, content=content, char_start=0, char_end=len(content), source="t.md"
    )


def _seed(store):
    doc = store.upsert_document("t.md", "T")
    texts = [
        "Reciprocal Rank Fusion combines two ranked lists using rank position only.",
        "A cross encoder reads query and passage together and scores precision.",
        "The grounding gate refuses when evidence does not support the answer.",
    ]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    store.add_chunks(doc, [_chunk(i, t) for i, t in enumerate(texts)], vectors)
    return texts


def test_roundtrip_counts(store):
    _seed(store)
    assert store.counts() == (1, 3)


def test_dense_ranks_by_cosine(store):
    _seed(store)
    top = store.dense([0.0, 0.0, 1.0], k=1)
    assert "grounding gate" in top[0]["content"]


def test_sparse_ranks_by_bm25(store):
    _seed(store)
    top = store.sparse("cross encoder precision", k=1)
    assert "cross encoder" in top[0]["content"]


def test_punctuation_does_not_break_fts(store):
    """User input goes straight into a MATCH expression, where punctuation is syntax."""
    _seed(store)
    assert store.sparse('what? (is) "RRF" -- now', k=3) is not None


def test_reingest_replaces_rather_than_duplicates(store):
    _seed(store)
    _seed(store)
    assert store.counts() == (1, 3)


def test_float32_roundtrip_is_close_enough(store):
    """Vectors are stored as packed float32; the precision loss must not matter."""
    original = [0.1234567, -0.9876543, 0.5]
    packed = struct.pack("<3f", *original)
    back = list(struct.unpack("<3f", packed))
    assert all(abs(a - b) < 1e-6 for a, b in zip(original, back, strict=False))


def test_fts5_is_available():
    """The whole sparse half depends on it, and some sqlite builds omit it."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(content)")
