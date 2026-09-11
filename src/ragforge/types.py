"""Shared value objects. Typed end to end - no dicts floating between layers."""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable span of a document, with offsets back into the original text."""

    id: int | None = None
    document_id: int | None = None
    ordinal: int
    content: str
    char_start: int
    char_end: int
    section: str = ""
    token_count: int = 0
    source: str = ""


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    # Kept separately so the eval harness can show what each stage contributed.
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    """Points at an exact span of a real source, not just 'document 3'."""

    source: str
    section: str = ""
    char_start: int
    char_end: int
    quote: str


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    grounding_score: float = 0.0
    latency_ms: int = 0
    backend: str = ""
    retrieved: list[ScoredChunk] = Field(default_factory=list)
