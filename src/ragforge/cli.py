"""Command line entrypoint: ragforge ingest | ask | eval | status"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import get_settings

app = typer.Typer(add_completion=False, help="rag-forge")
console = Console()


@app.command()
def ingest(path: str = typer.Argument("data/raw", help="File or directory to index")) -> None:
    """Index documents into Postgres."""
    from .ingest import ingest_path

    ingest_path(path)


@app.command()
def ask(
    question: str,
    top_k: int = typer.Option(None, "--top-k", "-k", help="Passages to keep after rerank"),
    show_context: bool = typer.Option(False, "--show-context", help="Print retrieved passages"),
) -> None:
    """Ask a question against the index."""
    from .generate import answer_question

    result = answer_question(question, top_k=top_k)

    style = "red" if result.refused else "green"
    console.print(Panel(result.text, title="answer", border_style=style))

    if result.citations:
        table = Table("source", "section", "span", "quote", title="citations")
        for c in result.citations:
            quote = c.quote if len(c.quote) <= 70 else c.quote[:67] + "..."
            table.add_row(
                c.source.rsplit("\\", 1)[-1].rsplit("/", 1)[-1],
                c.section or "-",
                f"{c.char_start}-{c.char_end}",
                quote,
            )
        console.print(table)

    if show_context:
        for i, r in enumerate(result.retrieved, start=1):
            console.print(
                Panel(
                    r.chunk.content[:600],
                    title=f"[{i}] score={r.score:.3f} dense={r.dense_rank} sparse={r.sparse_rank}",
                    border_style="dim",
                )
            )

    console.print(
        f"[dim]backend={result.backend}  grounding={result.grounding_score}  "
        f"latency={result.latency_ms}ms  refused={result.refused}[/]"
    )


@app.command()
def status() -> None:
    """Check that every dependency this project needs is actually reachable."""
    import httpx

    from .db import connection, healthcheck

    s = get_settings()
    table = Table("component", "status", "detail")

    ok = healthcheck()
    table.add_row("postgres", "[green]up[/]" if ok else "[red]down[/]", s.postgres_dsn)

    if ok:
        with connection() as conn:
            docs = conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"]
            chunks = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
        table.add_row("index", "[green]ok[/]", f"{docs} documents, {chunks} chunks")

    try:
        import redis

        redis.Redis.from_url(s.redis_url).ping()
        table.add_row("redis", "[green]up[/]", s.redis_url)
    except Exception as exc:
        table.add_row("redis", "[red]down[/]", str(exc)[:60])

    try:
        import torch

        detail = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
        table.add_row("gpu", "[green]ok[/]" if torch.cuda.is_available() else "[yellow]cpu[/]", detail)
    except Exception as exc:
        table.add_row("gpu", "[red]error[/]", str(exc)[:60])

    if s.llm_backend == "ollama":
        try:
            r = httpx.get(f"{s.ollama_base_url}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            hit = any(m.startswith(s.ollama_model.split(":")[0]) for m in models)
            table.add_row(
                "llm (ollama)",
                "[green]up[/]" if hit else "[yellow]no model[/]",
                f"{s.ollama_model} | installed: {', '.join(models) or 'none'}",
            )
        except Exception as exc:
            table.add_row("llm (ollama)", "[red]down[/]", str(exc)[:60])
    else:
        has_key = bool(s.anthropic_api_key)
        table.add_row(
            "llm (anthropic)",
            "[green]ok[/]" if has_key else "[red]no key[/]",
            s.anthropic_model,
        )

    console.print(table)


@app.command()
def evaluate() -> None:
    """Run the eval harness over eval/questions.jsonl and write RESULTS.md."""
    from .eval.harness import run_eval

    run_eval()


if __name__ == "__main__":
    app()
