"""Turn a file into plain text. One function per format, all returning (text, title)."""

from __future__ import annotations

from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf"}


def read_pdf(path: Path) -> tuple[str, str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    # Page markers keep char offsets meaningful and let a citation name its page.
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        parts.append(f"\n\n[page {i}]\n\n{page.extract_text() or ''}")
    title = (reader.metadata.title if reader.metadata else None) or path.stem
    return "".join(parts).strip(), str(title)


def read_text(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # A leading markdown H1 is a better title than the filename.
    first = text.lstrip().split("\n", 1)[0].strip()
    title = first.lstrip("# ").strip() if first.startswith("#") else path.stem
    return text, title


def read_file(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix in {".txt", ".md"}:
        return read_text(path)
    raise ValueError(f"Unsupported file type {suffix!r}. Supported: {sorted(SUPPORTED)}")
