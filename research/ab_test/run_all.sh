#!/usr/bin/env bash
# =============================================================================
# A/B: AgentDecision-4B (JAH) vs Drex v1.5 (Nace.AI) on the JAH routing set.
#
# Runs on a fresh CUDA machine. It sets up both runtimes, pulls both models,
# feeds the IDENTICAL System One routing request to each, and prints one table.
#
# Usage:
#   bash research/ab_test/run_all.sh
#
# Overrides (env):
#   REPO_ROOT   repo root (default: two levels up from this script)
#   OUT         output dir   (default: $REPO_ROOT/research/ab_test/out)
#   WEIGHTS_DIR model weights (default: $REPO_ROOT/research/ab_test/weights)
#   LIMIT       only first N decisions (default: all 292)
#   SKIP_DREX=1 / SKIP_JAH=1  run only one arm
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
OUT="${OUT:-$REPO_ROOT/research/ab_test/out}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO_ROOT/research/ab_test/weights}"
DATA="$SCRIPT_DIR/data/routing_gold_v2.jsonl"   # self-contained copy; override with DATA=... for other sets
LIMIT="${LIMIT:-}"
mkdir -p "$OUT" "$WEIGHTS_DIR"

# HF download helper: prefer `hf`, fall back to `huggingface-cli`.
hf_dl() {
  if command -v hf >/dev/null 2>&1; then hf download "$@";
  elif command -v huggingface-cli >/dev/null 2>&1; then huggingface-cli download "$@";
  else python - <<'PY'
import sys, os
from huggingface_hub import snapshot_download
repo = sys.argv[1]; local_dir = sys.argv[2]
snapshot_download(repo_id=repo, local_dir=local_dir,
                  token=os.environ.get("HF_TOKEN"))
print("downloaded", repo, "->", local_dir)
PY
  fi
}

echo "==> [0/5] prerequisites"
command -v git >/dev/null || { echo "git missing"; exit 1; }
command -v python3 >/dev/null || { echo "python3 missing"; exit 1; }
if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; else echo "WARN: no nvidia-smi (CPU only)"; fi

JAH_PY=""
if [[ "${SKIP_JAH:-0}" != "1" ]]; then
  echo "==> [1/5] JAH research env (research/.venv)"
  ( cd "$REPO_ROOT/research"
    if command -v uv >/dev/null; then uv sync --quiet || pip install -e . >/dev/null
    else pip install -e . >/dev/null; fi )
  JAH_PY="$REPO_ROOT/research/.venv/bin/python"
  [[ -x "$JAH_PY" ]] || JAH_PY="$(command -v python3)"

  echo "==> [2/5] download AgentDecision-4B adapter"
  if [[ ! -f "$WEIGHTS_DIR/AgentDecision-4B/adapter_model.safetensors" ]]; then
    hf_dl JevAgentHarness/AgentDecision-4B --local-dir "$WEIGHTS_DIR/AgentDecision-4B"
  else echo "    (adapter already present, skipping)"; fi
fi

if [[ "${SKIP_DREX:-0}" != "1" ]]; then
  echo "==> [3/5] Drex repo + env + weights"
  DREX_REPO="$WEIGHTS_DIR/drex-decision-models"
  if [[ ! -d "$DREX_REPO" ]]; then
    git clone --depth 1 https://github.com/nace-ai/drex-decision-models.git "$DREX_REPO"
  fi
  ( cd "$DREX_REPO"
    python3 -m venv .venv
    .venv/bin/pip install -q -r requirements.txt )
  if [[ ! -f "$WEIGHTS_DIR/drex-v1.5/head.pt" ]]; then
    hf_dl nace-ai/drex-v1.5 --local-dir "$WEIGHTS_DIR/drex-v1.5"
  else echo "    (drex-v1.5 weights already present, skipping)"; fi
fi

LIMIT_ARG=""
[[ -n "$LIMIT" ]] && LIMIT_ARG="--limit $LIMIT"

if [[ "${SKIP_JAH:-0}" != "1" ]]; then
  echo "==> [4/5] run JAH arm (AgentDecision-4B)"
  "$JAH_PY" "$SCRIPT_DIR/run_jah.py" \
    --data "$DATA" \
    --base "Qwen/Qwen3-4B" \
    --adapter "$WEIGHTS_DIR/AgentDecision-4B" \
    $LIMIT_ARG \
    --out "$OUT/jah.json"
fi

if [[ "${SKIP_DREX:-0}" != "1" ]]; then
  echo "==> [4/5] run Drex arm (v1.5)"
  python3 "$SCRIPT_DIR/run_drex.py" \
    --drex-repo "$WEIGHTS_DIR/drex-decision-models" \
    --weights "$WEIGHTS_DIR/drex-v1.5" \
    --data "$DATA" \
    $LIMIT_ARG \
    --out "$OUT/drex.json"
fi

if [[ "${SKIP_JAH:-0}" != "1" && "${SKIP_DREX:-0}" != "1" ]]; then
  echo "==> [5/5] compare"
  python3 "$SCRIPT_DIR/compare.py" --jah "$OUT/jah.json" --drex "$OUT/drex.json" --out "$OUT/comparison.md"
fi

echo "done. results in $OUT"
