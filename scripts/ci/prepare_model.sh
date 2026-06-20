#!/usr/bin/env bash
#
# Restore or build mistral.mog for CI.
#
# Re-exports only when export_mistral.py or requirements.txt change (cache miss).
# On cache hit, restores the ~10 GB file from actions/cache.
#
# Usage: ./scripts/ci/prepare_model.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL_MOG:-./mistral.mog}"
HF_DIR="${HF_MODEL_DIR:-./Mistral-7B-v0.1}"
HF_REPO="${HF_MODEL_REPO:-mistralai/Mistral-7B-v0.1}"

if [ -f "$MODEL" ]; then
    echo "Model already present: $MODEL ($(du -h "$MODEL" | cut -f1))"
    exit 0
fi

echo "mistral.mog not found — downloading checkpoint and exporting Q8F16..."

VENV="${CI_VENV:-.ci-venv}"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"
pip install --quiet -r requirements.txt huggingface_hub

if [ ! -f "$HF_DIR/config.json" ]; then
    echo "Downloading $HF_REPO to $HF_DIR ..."
    python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$HF_REPO", local_dir="$HF_DIR")
PY
fi

echo "Exporting Q8F16 model..."
python3 export_mistral.py --model_dir "$HF_DIR" --out "$MODEL"
deactivate 2>/dev/null || true

echo "Export complete: $MODEL ($(du -h "$MODEL" | cut -f1))"
