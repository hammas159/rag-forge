# Build Log

How this project was actually built, what broke, and what the fix was.

Kept because the decisions are more interesting than the code, and because the same
problems recur on every new machine. Written as they happened, not reconstructed
afterwards.

**Machine:** Windows 11 · Quadro RTX 5000 16 GB · CUDA 13.0 · Docker 29.7.2 ·
uv 0.12.12 · no system Python · ~120 KB/s connection

---

## 1 · Environment problems, before a line of project code

### `gh` installed but "not recognized"

```
gh : The term 'gh' is not recognized as the name of a cmdlet...
```

**Cause:** `winget install` writes the new folder into the machine `PATH`, but an
already-open terminal keeps the environment it was born with. The binary was on disk
at `C:\Program Files\GitHub CLI\gh.exe` the whole time.

**Fix, in the open terminal:**

```powershell
$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' +
            [Environment]::GetEnvironmentVariable('Path','User')
```

**Fix, properly:** close the terminal, open a new one.

**Lesson:** after any installer, assume every open shell is stale.

### Terminal stuck on the wrong folder

New terminals kept opening in an old project directory. `cd` fixed the current tab and
nothing else, because **VS Code opens every new terminal at the folder the window has
open** — a window property, not a shell one.

```powershell
code -r D:\github      # -r replaces the current window instead of opening a second
```

Then kill the old terminal; a window reload does not re-home terminals that already
exist.

---

## 2 · Git configuration that would have bitten later

### Commit email vs account email

Git was committing as a personal address not verified on the GitHub account. Two
consequences, both silent until they are not:

1. Commits show as an unknown user and **never appear on the contribution graph**.
2. With *Block command line pushes that expose my email* enabled, the push is
   rejected outright with `GH007`.

**Fix — the noreply alias, which attributes correctly and keeps a personal address out
of every public commit:**

```powershell
git config --global user.email "<user-id>+<username>@users.noreply.github.com"
```

The numeric id comes from `gh api user --jq .id`.

### Line endings, the one that breaks Docker

On Windows, git will happily commit `entrypoint.sh` with CRLF. The Linux container then
fails with a message that looks like nothing:

```
/entrypoint.sh: line 2: $'\r': command not found
exec format error
```

**Fix, both halves.** The config:

```powershell
git config --global core.autocrlf input
```

and a committed `.gitattributes`, so the rule travels with the repo rather than living
on one machine:

```
* text=auto eol=lf
*.sh       text eol=lf
Dockerfile text eol=lf
```

### Git LFS installed but not activated

`git-lfs` being on `PATH` does not mean git uses it — the filters have to be registered.
Without it GitHub hard-rejects any file over 100 MB, and untangling that afterwards is
genuinely painful.

```powershell
git lfs install
```

### Long paths

ML repositories nest deep (`vae-lab/05-vq-vae-2/experiments/checkpoints/...`) and Windows
caps at 260 characters, producing `Filename too long` on clone.

```powershell
git config --global core.longpaths true
```

---

## 3 · The constraint that shaped the design: bandwidth

Measured mid-build with `curl`:

```
pypi:       18,872 B/s
later:           0 B/s      connection dropped entirely
recovered: 121,057 B/s
```

The `pgvector/pgvector:pg17` image — 627 MB — took roughly half an hour.

That number propagates into every decision:

| Planned | Size | Time at 120 KB/s |
|---|---|---|
| `bge-m3` + `bge-reranker-v2-m3` | 4.5 GB | ~10 h |
| PyTorch CUDA | 2.5 GB | ~6 h |
| `qwen2.5:7b-instruct` | 4.7 GB | ~11 h |

**~27 hours before the project could be run once.**

### Decision: default to a small model profile

| profile | embed | rerank | size |
|---|---|---|---|
| **small** (default) | `bge-small-en-v1.5` | `ms-marco-MiniLM-L-6-v2` | ~220 MB |
| large | `bge-m3` | `bge-reranker-v2-m3` | ~4.5 GB |

A project that cannot be executed cannot be verified, and an unverified repo is worth
less than no repo. Small models make the pipeline runnable in under an hour; the large
profile stays one `.env` edit away.

The trade-off became an asset: **both profiles are benchmarked in `RESULTS.md`.** A
measured quality-versus-cost curve is a stronger claim than having used the bigger model.

### The follow-on bug this created

Changing the embedding model changes the vector width — 1024 for `bge-m3`, 384 for
`bge-small`. The schema declared `VECTOR(1024)`, so switching profiles would fail on
insert or, worse, silently retrieve nonsense.

**Fix:** `src/ragforge/migrate.py::ensure_dim()` compares the column's real width — read
via `format_type(atttypid, atttypmod)` — against the width of the vectors actually
produced, and rebuilds the column and its HNSW index when they disagree. Stale vectors
are dropped **loudly**: embeddings from a different model are not comparable, and keeping
them would quietly poison retrieval.

The dimension comes from `len(vectors[0])`, not from config, so the two cannot drift apart.

---

## 4 · Design decisions worth defending

### One datastore instead of two

Hybrid retrieval usually means Postgres for vectors plus Elasticsearch for keywords. Here
Postgres 17 carries both: an HNSW index over `vector(384)` and a GIN index over a
generated `tsvector` column. One service, one backup, one connection pool — and
`websearch_to_tsquery` is good enough for the sparse half.

### RRF rather than score blending

Dense cosine similarity and `ts_rank_cd` are not on a comparable scale, and normalising
them requires tuning that does not survive a change of corpus. Reciprocal Rank Fusion uses
only rank position:

```
score(d) = Σ 1 / (k + rank(d))        k = 60
```

Nothing to normalise, nothing to retune.

### The model is never trusted about its own sources

Asking an LLM for citations produces citations — including for things it invented. So the
model returns verbatim quotes, and each quote is **located in the real passage text**
before it becomes a citation. Found → converted to a character span of the original
document. Not found → dropped, and that drop is recorded as a hallucination signal.

This is why character offsets are preserved from chunking all the way through, and why
that invariant has its own test. Without it a citation can only say "document 3" instead
of "characters 4102–4288 of contracts.pdf".

### Refusal as a first-class outcome

Three independent grounds, any one sufficient:

1. the model reports the context did not support an answer
2. the grounding score is below threshold
3. every quote failed verification

`correct_refusal` is a **scored metric**, with deliberately unanswerable questions in the
eval set. A system that cannot say "I don't know" cannot be trusted with the answers it
does give.

### A custom eval harness instead of RAGAS

RAGAS needs an LLM judge: it costs money per run and returns different numbers each time.
The harness here measures hit@k, MRR, answer-match, refusal rate, correct-refusal rate and
p50/p95 latency — reproducible, free, and every number is explainable in an interview.

### Streamlit rather than Next.js

The original plan specified Next.js for flagship projects. Streamlit reached a working,
screenshot-able UI in hours rather than days, and it shows the **retrieval trace** — rerank
score, dense rank, sparse rank per passage — which is what makes the system debuggable and
what another engineer actually wants to see.

### CPU torch inside the Docker image

The container exists so a reviewer can run the project without installing Python or CUDA.
CUDA wheels would take the image from ~1 GB to ~6 GB for a demo path that does not need
speed. The GPU path stays on the host via `make api`.

---

## 5 · Smaller things that cost time

| Problem | Fix |
|---|---|
| `perl -0pi -e` failed editing the Makefile — the shell ate the quoting | rewrote with `awk` into a temp file and moved it into place |
| PowerShell 5.1 has no `&&`, no ternary, no `??` | `; if ($?) { ... }` and explicit `if/else` |
| No system Python, so nothing could be syntax-checked before deps installed | compiled every source with uv's cached interpreter: `python -m compileall -q src tests ui` |
| Ports 5432 / 6379 risk clashing with an existing local install | mapped to **5433** and **6380** on the host |
| `docker compose up` that pulls images blocks for many minutes | ran it as a background task and kept writing code against the schema meanwhile |
| Two concurrent `uv sync` runs would contend on the same lock | stopped the interactive one before handing the job to the unattended script |

---

## 6 · Reproducing this setup on a new machine

```powershell
winget install --id GitHub.cli -e
# open a NEW terminal, then:
gh auth login                       # HTTPS, and answer YES to "authenticate Git"
git config --global user.email "<id>+<user>@users.noreply.github.com"
git config --global core.autocrlf input
git config --global core.longpaths true
git lfs install
```

Then, in the repo:

```bash
make up        # postgres + redis
make install   # uv sync, writes .env
make ingest    # index data/raw
make ask Q="What does Reciprocal Rank Fusion combine?"
```

`ragforge status` reports Postgres, Redis, GPU and LLM backend individually, so a broken
setup names its own broken piece.

On an intermittent connection, `scripts/overnight-setup.ps1` runs the whole download
sequence unattended: it disables sleep, retries every network step, and orders work by
priority so a 3 a.m. dropout still leaves the most useful subset finished.
