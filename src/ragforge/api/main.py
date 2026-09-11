"""HTTP surface. Thin - all logic lives in the library modules."""

from __future__ import annotations

import hashlib
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import get_settings
from ..db import connection, healthcheck
from ..generate import answer_question
from ..types import Answer

app = FastAPI(
    title="rag-forge",
    version="0.1.0",
    description="Hybrid RAG with span-level citations and a grounding gate.",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_cache: bool = True


def _cache():
    """Redis is an optimisation, never a dependency - return None if it is not there."""
    try:
        import redis

        client = redis.Redis.from_url(get_settings().redis_url, decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if healthcheck() else "degraded", "postgres": healthcheck()}


@app.get("/stats")
def stats() -> dict:
    if not healthcheck():
        raise HTTPException(503, "postgres unavailable")
    with connection() as conn:
        docs = conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"]
        chunks = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    return {"documents": docs, "chunks": chunks}


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    if not healthcheck():
        raise HTTPException(503, "postgres unavailable")

    s = get_settings()
    key = "ask:" + hashlib.sha256(
        f"{req.question}|{req.top_k}|{s.llm_backend}".encode()
    ).hexdigest()

    client = _cache() if req.use_cache else None
    if client:
        hit = client.get(key)
        if hit:
            return Answer(**json.loads(hit))

    result = answer_question(req.question, top_k=req.top_k)

    if client and not result.refused:
        client.setex(key, s.cache_ttl_seconds, result.model_dump_json())
    return result
