"""Space entrypoint.

Spaces give you no database server and no Docker, so the demo runs on the SQLite
store and the HuggingFace inference backend. The corpus is indexed once on first
boot and cached, which takes a few seconds for the seed documents.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Set before anything imports config, since settings are read once and cached.
os.environ.setdefault("STORE", "sqlite")
os.environ.setdefault("SQLITE_PATH", "/tmp/ragforge.db")
os.environ.setdefault("LLM_BACKEND", "huggingface")
os.environ.setdefault("DEVICE", "cpu")

import streamlit as st  # noqa: E402


@st.cache_resource(show_spinner="Indexing the demo corpus…")
def _bootstrap() -> int:
    from ragforge.ingest import ingest_path

    return ingest_path(ROOT / "data" / "raw")


_bootstrap()
exec(
    (ROOT / "ui" / "app.py").read_text(encoding="utf-8"), {"__file__": str(ROOT / "ui" / "app.py")}
)
