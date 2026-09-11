#!/usr/bin/env bash
# Runs once when a Codespace is created. Keep it to the demo path: SQLite store and
# CPU models, so a reviewer gets a working UI without provisioning a database.
set -euo pipefail

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync --all-groups --extra-index-url https://download.pytorch.org/whl/cpu

cp -n .env.example .env
{
  echo "STORE=sqlite"
  echo "SQLITE_PATH=data/ragforge.db"
  echo "DEVICE=cpu"
} >> .env

uv run ragforge ingest data/raw

echo
echo "Ready. Run:  uv run streamlit run ui/app.py"
