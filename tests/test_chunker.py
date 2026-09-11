from ragforge.ingest.chunker import chunk_text

SAMPLE = """# Title

First paragraph about retrieval augmented generation and its failure modes.

Second paragraph covering hybrid search, which combines dense and sparse signals.

## Section Two

Third paragraph discussing cross encoder reranking and why it improves precision.
"""


def test_chunks_are_produced():
    chunks = chunk_text(SAMPLE, source="sample.md", max_tokens=40, overlap_tokens=8)
    assert chunks, "expected at least one chunk"


def test_offsets_point_back_at_the_original_text():
    """The whole citation feature depends on this invariant holding."""
    chunks = chunk_text(SAMPLE, source="sample.md", max_tokens=40, overlap_tokens=8)
    for c in chunks:
        assert 0 <= c.char_start < c.char_end <= len(SAMPLE)
        original = SAMPLE[c.char_start : c.char_end]
        # Content is joined with blank lines, so compare on collapsed whitespace.
        assert "".join(c.content.split())[:40] in "".join(original.split())


def test_ordinals_are_sequential():
    chunks = chunk_text(SAMPLE, source="sample.md", max_tokens=40, overlap_tokens=8)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_oversized_paragraph_is_not_dropped():
    huge = "word " * 2000
    chunks = chunk_text(f"# T\n\n{huge}", source="x.md", max_tokens=100, overlap_tokens=10)
    assert any(c.token_count > 100 for c in chunks)


def test_empty_input():
    assert chunk_text("", source="empty.md") == []
