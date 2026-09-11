"""Streamlit UI. Shows the answer, the verified citations, and the retrieval trace -
the trace matters as much as the answer, because it is what makes the system debuggable."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragforge.config import get_settings  # noqa: E402
from ragforge.generate import answer_question  # noqa: E402
from ragforge.store import get_store  # noqa: E402

st.set_page_config(page_title="rag-forge", page_icon="*", layout="wide")
settings = get_settings()

st.title("rag-forge")
st.caption("Hybrid retrieval · cross-encoder rerank · verified span citations · grounding gate")

with st.sidebar:
    st.subheader("System")
    store = get_store()
    if store.healthy():
        docs, chunks = store.counts()
        st.success(f"{docs} documents · {chunks} chunks · {store.name}")
    else:
        st.error("Store unreachable — run `make up`, or set STORE=sqlite")

    st.write(f"**LLM** `{settings.llm_backend}`")
    st.write(f"**Embed** `{settings.embed_model}`")
    st.write(f"**Rerank** `{settings.rerank_model}`")
    top_k = st.slider("Passages after rerank", 1, 10, settings.top_k_rerank)
    st.caption(f"Grounding threshold: {settings.grounding_threshold}")

question = st.text_input("Question", placeholder="Ask something about the indexed documents")

if question:
    with st.spinner("retrieving, reranking, generating…"):
        result = answer_question(question, top_k=top_k)

    if result.refused:
        st.warning(result.text)
        st.caption("The grounding gate blocked this answer rather than let it guess.")
    else:
        st.markdown(result.text)

    c1, c2, c3 = st.columns(3)
    c1.metric("Grounding", f"{result.grounding_score:.2f}")
    c2.metric("Latency", f"{result.latency_ms} ms")
    c3.metric("Citations", len(result.citations))

    if result.citations:
        st.subheader("Citations")
        for c in result.citations:
            name = Path(c.source).name
            with st.expander(f"{name} · {c.section or 'no section'} · chars {c.char_start}–{c.char_end}"):
                st.markdown(f"> {c.quote}")

    with st.expander(f"Retrieval trace ({len(result.retrieved)} passages)"):
        for i, r in enumerate(result.retrieved, start=1):
            st.markdown(
                f"**[{i}]** rerank `{r.score:.3f}` · dense rank `{r.dense_rank}` · "
                f"sparse rank `{r.sparse_rank}` · `{Path(r.chunk.source).name}`"
            )
            st.text(r.chunk.content[:800])
            st.divider()
