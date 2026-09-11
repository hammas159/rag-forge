---
title: rag-forge
emoji: "\U0001F50E"
colorFrom: indigo
colorTo: gray
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: mit
short_description: Hybrid RAG with verified citations and a refusal path
---

# rag-forge — live demo

Ask a question about the indexed documents and watch the whole pipeline:
hybrid retrieval, cross-encoder reranking, verified span citations, and a grounding
gate that refuses rather than guessing.

Try one of the deliberately unanswerable questions — the system is built to say no.

Running on free CPU hardware with the small model profile and a SQLite index.
Source, and the Postgres/GPU configuration it normally runs on:
**https://github.com/hammas159/rag-forge**
