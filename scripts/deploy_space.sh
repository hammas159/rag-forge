#!/usr/bin/env bash
# Publish the live demo to a Hugging Face Space.
#
# A Space is its own git repo, and it expects app.py plus requirements.txt at its
# root - not this repo's layout. So the Space is assembled from deploy/space/ plus
# the source it needs, rather than kept as a duplicate copy that drifts.
#
#   ./scripts/deploy_space.sh <hf-username> [space-name]
#
# Prerequisites:  hf auth login      (a WRITE token)
set -euo pipefail

USER="${1:?usage: deploy_space.sh <hf-username> [space-name]}"
NAME="${2:-rag-forge}"
REPO="$USER/$NAME"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

command -v hf >/dev/null || { echo "hf CLI not found: pip install huggingface_hub"; exit 1; }

echo "==> creating space $REPO (no-op if it exists)"
hf repo create "$NAME" --repo-type space --space_sdk streamlit -y 2>/dev/null || true

echo "==> cloning"
git clone "https://huggingface.co/spaces/$REPO" "$WORK/space"

echo "==> assembling"
cp deploy/space/README.md       "$WORK/space/README.md"
cp deploy/space/requirements.txt "$WORK/space/requirements.txt"
cp deploy/space/app.py          "$WORK/space/app.py"
rm -rf "$WORK/space/src" "$WORK/space/ui" "$WORK/space/data"
cp -r src ui "$WORK/space/"
mkdir -p "$WORK/space/data/raw"
cp data/raw/*.md "$WORK/space/data/raw/"

cd "$WORK/space"
git add -A
if git diff --cached --quiet; then
  echo "==> nothing changed"
  exit 0
fi
git commit -m "deploy $(date -u +%Y-%m-%dT%H:%MZ)"
git push

echo
echo "==> live at https://huggingface.co/spaces/$REPO"
echo "    Set HF_TOKEN in Settings > Variables and secrets, or the demo cannot generate."
