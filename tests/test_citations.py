from ragforge.generate.citations import verify_quotes
from ragforge.types import Chunk, ScoredChunk

CONTENT = "Hybrid retrieval fuses dense and sparse rankings using Reciprocal Rank Fusion."


def _retrieved() -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(
                id=1,
                document_id=1,
                ordinal=0,
                content=CONTENT,
                char_start=100,
                char_end=100 + len(CONTENT),
                section="Retrieval",
                source="notes.md",
            ),
            score=0.9,
        )
    ]


def test_exact_quote_maps_to_absolute_offsets():
    cits, unverified = verify_quotes(
        [{"passage": 1, "quote": "Reciprocal Rank Fusion"}], _retrieved()
    )
    assert not unverified
    assert cits[0].quote == "Reciprocal Rank Fusion"
    assert cits[0].char_start == 100 + CONTENT.index("Reciprocal Rank Fusion")


def test_whitespace_differences_are_tolerated():
    cits, unverified = verify_quotes(
        [{"passage": 1, "quote": "dense   and\n sparse rankings"}], _retrieved()
    )
    assert cits and not unverified


def test_fabricated_quote_is_rejected():
    """A quote that is not in the source must never become a citation."""
    cits, unverified = verify_quotes(
        [{"passage": 1, "quote": "guaranteed 99% accuracy on every dataset"}], _retrieved()
    )
    assert cits == []
    assert unverified == ["guaranteed 99% accuracy on every dataset"]


def test_wrong_passage_number_still_resolves():
    cits, _ = verify_quotes([{"passage": 7, "quote": "Hybrid retrieval"}], _retrieved())
    assert cits and cits[0].source == "notes.md"
