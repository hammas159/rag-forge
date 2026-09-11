# rag-forge

Production RAG that refuses to guess.

Hybrid retrieval (dense + sparse, RRF-fused) → cross-encoder rerank → generation →
**every quote verified against the source** → a grounding gate that blocks the answer
when the evidence does not support it.

Most RAG demos answer every question. This one is built around the cases where it
should not.

---

## What makes it different

| | |
|---|---|
| **Verified citations** | The model returns quotes; each quote is *located in the actual source text* before it becomes a citation, and converted to a character span of the original document. Quotes that cannot be located are dropped — and a dropped quote is a hallucination signal. |
| **A real refusal path** | Three independent grounds for refusing: the model reporting no support, a grounding score under threshold, or every quote failing verification. Any one is enough. |
| **One datastore** | Postgres 17 holds both the `vector(1024)` HNSW index and the `tsvector` GIN index. Hybrid search without running a second search engine. |
| **Swappable LLM** | `LLM_BACKEND=ollama` (free, local, on your own GPU) or `anthropic`. Nothing above `llm.py` knows which is active. |
| **Measured, not claimed** | `make eval` writes `RESULTS.md` — hit@k, MRR, answer match, refusal rate, correct-refusal rate, p50/p95 latency. |

## Architecture

```
        ingest                      query
          │                           │
   readers (pdf/md/txt)          embed query
          │                           │
   structure-aware chunker    ┌───────┴────────┐
   (keeps char offsets)       │                │
          │              dense (HNSW)    sparse (GIN tsvector)
      bge-m3 embed            └───────┬────────┘
          │                     RRF fusion (k=60)
          ▼                           │
   ┌──────────────┐          cross-encoder rerank
   │  Postgres 17 │◄────┐            │
   │   + pgvector │     │      LLM (ollama │ anthropic)
   └──────────────┘     │            │
                        │     quote verification
                     Redis            │
                    (cache)   grounding gate ──► answer + citations
                                      └────────► refusal
```

Character offsets survive the whole pipeline. That invariant is what makes a citation
point at *characters 4102–4288 of contracts.pdf* rather than "document 3", and it is
covered by a test.

## Quick start

```bash
make up          # Postgres + Redis
make install     # uv sync, creates .venv
cp .env.example .env
make ingest      # indexes the seed corpus in data/raw/
make ask Q="What does Reciprocal Rank Fusion combine?"
make ui          # Streamlit on :8501
```

`ragforge status` checks every dependency — Postgres, Redis, GPU, LLM backend — and
tells you which one is missing.

### LLM backend

Local and free (default):
```bash
ollama pull qwen2.5:7b-instruct     # requires ollama installed
```

Or the API — set `LLM_BACKEND=anthropic` and `ANTHROPIC_API_KEY` in `.env`.

## Layout

```
src/ragforge/
  config.py        all settings, one place
  types.py         Chunk, ScoredChunk, Citation, Answer
  llm.py           dual backend behind one ABC
  db.py            pooled Postgres, pgvector registered
  embed.py         bge-m3 + bge-reranker-v2-m3, GPU with CPU fallback
  ingest/          readers → chunker (offset-preserving) → pipeline
  retrieve/        dense + sparse + RRF + rerank
  generate/        prompts, quote verification, grounding gate
  eval/            harness that writes RESULTS.md
  api/             FastAPI
ui/app.py          Streamlit, shows the retrieval trace
docker/init.sql    schema, both indexes
```

## Requirements

- Docker (Postgres + Redis)
- [uv](https://docs.astral.sh/uv/) — no system Python needed
- NVIDIA GPU with ~5 GB free for the two models. Falls back to CPU automatically.

## Results

Run `make eval` to generate [`RESULTS.md`](RESULTS.md). Numbers are not quoted here
until they have actually been measured on this machine.

## Tests

```bash
make test    # chunk offsets, citation verification, config invariants
make lint
```

The tests target the pure logic — offsets, quote location, refusal conditions — none
of which needs a database or a GPU. CI runs them plus a schema check against a real
pgvector container.

## License

MIT
