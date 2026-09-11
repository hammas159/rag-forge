"""File -> text -> chunks -> embeddings -> store."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from ..config import get_settings
from ..embed import embed_passages
from ..store import get_store
from .chunker import chunk_text
from .readers import SUPPORTED, read_file

console = Console()


def ingest_file(path: Path) -> int:
    s = get_settings()
    text, title = read_file(path)
    if not text.strip():
        console.print(f"[yellow]skip[/] {path.name} - no extractable text")
        return 0

    chunks = chunk_text(
        text, source=str(path), max_tokens=s.chunk_tokens, overlap_tokens=s.chunk_overlap
    )
    if not chunks:
        return 0

    vectors = embed_passages([c.content for c in chunks])

    store = get_store()
    # Width comes from the vectors themselves, never from config, so the schema and
    # the model cannot drift apart when EMBED_MODEL changes.
    store.ensure_schema(len(vectors[0]))
    doc_id = store.upsert_document(str(path), title)
    store.add_chunks(doc_id, chunks, vectors)

    console.print(f"[green]ok[/] {path.name} -> {len(chunks)} chunks")
    return len(chunks)


def ingest_path(target: str | Path) -> int:
    target = Path(target)
    files = (
        [target]
        if target.is_file()
        else sorted(p for p in target.rglob("*") if p.suffix.lower() in SUPPORTED)
    )
    if not files:
        console.print(f"[yellow]nothing to ingest under {target}[/]")
        return 0

    total = sum(ingest_file(p) for p in files)
    console.print(
        f"[bold green]ingested {len(files)} file(s), {total} chunks into {get_store().name}[/]"
    )
    return total
