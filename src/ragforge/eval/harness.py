"""Eval harness. Produces the numbers table that opens the README.

Metrics, deliberately simple and reproducible - no LLM judge, no external service:
  hit@k          did any retrieved chunk contain the expected answer string
  mrr            mean reciprocal rank of the first such chunk
  answer_match   did the final answer contain the expected string
  refusal_rate   how often the grounding gate refused
  correct_refusal did it refuse on questions that are deliberately unanswerable
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ..config import get_settings
from ..generate import answer_question
from ..retrieve import retrieve

console = Console()

QUESTIONS = Path("eval/questions.jsonl")
RESULTS = Path("RESULTS.md")


def _load() -> list[dict]:
    if not QUESTIONS.exists():
        raise SystemExit(
            f"{QUESTIONS} not found. Each line: "
            '{"question": "...", "expected": "...", "answerable": true}'
        )
    return [
        json.loads(line)
        for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_eval() -> dict:
    s = get_settings()
    cases = _load()
    hits, rrs, matches, refusals, correct_refusals, latencies = [], [], [], [], [], []

    for case in cases:
        q, expected = case["question"], case.get("expected", "")
        answerable = case.get("answerable", True)

        started = time.perf_counter()
        chunks = retrieve(q)
        needle = expected.lower()
        rank = next(
            (i for i, c in enumerate(chunks, 1) if needle and needle in c.chunk.content.lower()),
            None,
        )
        if answerable:
            hits.append(1.0 if rank else 0.0)
            rrs.append(1.0 / rank if rank else 0.0)

        result = answer_question(q)
        latencies.append((time.perf_counter() - started) * 1000)
        refusals.append(1.0 if result.refused else 0.0)

        if answerable:
            matches.append(
                1.0 if needle and needle in result.text.lower() and not result.refused else 0.0
            )
        else:
            correct_refusals.append(1.0 if result.refused else 0.0)

        console.print(
            f"[dim]{'REFUSED' if result.refused else 'answered'}[/] "
            f"rank={rank} grounding={result.grounding_score} :: {q[:60]}"
        )

    def avg(xs: list[float]) -> float:
        return round(statistics.fmean(xs), 4) if xs else 0.0

    metrics = {
        "cases": len(cases),
        f"hit@{s.top_k_rerank}": avg(hits),
        "mrr": avg(rrs),
        "answer_match": avg(matches),
        "refusal_rate": avg(refusals),
        "correct_refusal": avg(correct_refusals),
        "p50_latency_ms": round(statistics.median(latencies)) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1])
        if len(latencies) > 1
        else 0,
        "backend": s.llm_backend,
        "embed_model": s.embed_model,
        "rerank_model": s.rerank_model,
    }

    table = Table("metric", "value", title="rag-forge eval")
    for k, v in metrics.items():
        table.add_row(k, str(v))
    console.print(table)

    rows = "\n".join(f"| {k} | {v} |" for k, v in metrics.items())
    RESULTS.write_text(
        "# Results\n\n"
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} · "
        f"`make eval` over `{QUESTIONS}`\n\n"
        "| metric | value |\n|---|---|\n" + rows + "\n",
        encoding="utf-8",
    )
    console.print(f"[green]wrote {RESULTS}[/]")
    return metrics
