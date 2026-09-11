"""Hybrid retrieval: dense + sparse, fused with RRF, then cross-encoder reranked.

Dense alone misses exact terms (product codes, names, statute numbers). Sparse alone
misses paraphrase. RRF fuses the two rankings without needing their scores to be
comparable - it only uses rank position, which is why it is robust.

Which storage engine serves the two halves is a config choice; see store/.
"""

from __future__ import annotations

from ..config import get_settings
from ..embed import embed_query, rerank
from ..store import get_store
from ..types import Chunk, ScoredChunk


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
    store = get_store()

    qvec = embed_query(query)
    dense_rows = [dict(r) for r in store.dense(qvec, s.top_k_dense)]
    sparse_rows = [dict(r) for r in store.sparse(query, s.top_k_sparse)]

    if not dense_rows and not sparse_rows:
        return []

    by_id = {r["id"]: r for r in (*dense_rows, *sparse_rows)}
    fused = _rrf(dense_rows, sparse_rows, s.rrf_k)

    # Rerank the fused candidates. This is the single biggest quality lever in the
    # pipeline - the bi-encoder is a recall filter, the cross-encoder is precision.
    cand_ids = [cid for cid, _ in sorted(fused.items(), key=lambda kv: kv[1][0], reverse=True)]
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
