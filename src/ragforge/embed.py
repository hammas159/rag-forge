"""Embedding and reranking models. Loaded once, kept on the GPU."""

from __future__ import annotations

from functools import lru_cache

from .config import get_settings


def _resolve_device(requested: str) -> str:
    """Fall back to CPU rather than crashing when there is no usable GPU."""
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    s = get_settings()
    return SentenceTransformer(s.embed_model, device=_resolve_device(s.device))


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    s = get_settings()
    return CrossEncoder(s.rerank_model, device=_resolve_device(s.device), max_length=512)


def embed_passages(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    vecs = get_embedder().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    return embed_passages([text])[0]


def rerank(query: str, passages: list[str], batch_size: int = 16) -> list[float]:
    """Cross-encoder relevance scores. Slower than the bi-encoder, far more accurate."""
    if not passages:
        return []
    scores = get_reranker().predict(
        [(query, p) for p in passages], batch_size=batch_size, show_progress_bar=False
    )
    return [float(s) for s in scores]
