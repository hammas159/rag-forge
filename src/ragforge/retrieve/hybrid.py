"""Hybrid retrieval: dense + sparse, fused with RRF, then cross-encoder reranked.

Dense alone misses exact terms (product codes, names, statute numbers). Sparse alone
misses paraphrase. RRF fuses the two rankings without needing their scores to be
comparable - it only uses rank position, which is why it is robust.
"""

from __future__ import annotations

from ..config import get_settings
from ..db import connection
from ..embed import embed_query, rerank
from ..types import Chunk, ScoredChunk

_SELECT = """
    SELECT c.id, c.document_id, c.ordinal, c.content, c.char_start, c.char_end,
           c.section, c.token_count, d.source
"""


def _dense(conn, vector: list[float], k: int) -> list[dict]:
    return conn.execute(
        _SELECT
        + """
        , 1 - (c.embedding <=> %s::vector) AS score
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (vector, vector, k),
    ).fetchall()


def _sparse(conn, query: str, k: int) -> list[dict]:
    return conn.execute(
        _SELECT
        + """
        , ts_rank_cd(c.tsv, websearch_to_tsquery('english', %s)) AS score
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE c.tsv @@ websearch_to_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT %s
        """,
        (query, query, k),
    ).fetchall()


def _to_chunk(row: dict) -> Chunk:
    return Chunk(
        id=row["id"],
        document_id=row["document_id"],
        ordinal=row["ordinal"],
        content=row["content"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        section=row["section"] or "",
        token_count=row["token_count"],
        source=row["source"],
    )


def _rrf(
    dense: list[dict], sparse: list[dict], k: int
) -> dict[int, tuple[float, int | None, int | None]]:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across both lists."""
    fused: dict[int, tuple[float, int | None, int | None]] = {}
    for rank, row in enumerate(dense, start=1):
        score, _, sp = fused.get(row["id"], (0.0, None, None))
        fused[row["id"]] = (score + 1.0 / (k + rank), rank, sp)
    for rank, row in enumerate(sparse, start=1):
        score, dn, _ = fused.get(row["id"], (0.0, None, None))
        fused[row["id"]] = (score + 1.0 / (k + rank), dn, rank)
    return fused


def retrieve(query: str, *, top_k: int | None = None) -> list[ScoredChunk]:
    s = get_settings()
    top_k = top_k or s.top_k_rerank
    qvec = embed_query(query)

    with connection() as conn:
        dense_rows = _dense(conn, qvec, s.top_k_dense)
        sparse_rows = _sparse(conn, query, s.top_k_sparse)

    if not dense_rows and not sparse_rows:
        return []

    by_id = {r["id"]: r for r in (*dense_rows, *sparse_rows)}
    fused = _rrf(dense_rows, sparse_rows, s.rrf_k)

    # Rerank the fused candidates. This is the single biggest quality lever in the
    # pipeline - the bi-encoder is a recall filter, the cross-encoder is precision.
    candidates = sorted(fused.items(), key=lambda kv: kv[1][0], reverse=True)
    cand_ids = [cid for cid, _ in candidates]
    scores = rerank(query, [by_id[cid]["content"] for cid in cand_ids])

    results = [
        ScoredChunk(
            chunk=_to_chunk(by_id[cid]),
            score=float(score),
            dense_rank=fused[cid][1],
            sparse_rank=fused[cid][2],
            rerank_score=float(score),
        )
        for cid, score in zip(cand_ids, scores, strict=True)
    ]
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
