#!/usr/bin/env bash
# Full UCCI replication on CoNLL-2003 (paper Section 7 suggests exactly this run).
#
# Steps, in order:
#   1. latency.py   : cost ratio c_l / c_s from 100 queries per model, batch size 1
#   2. generate.py  : small model over the pooled validation + test sentences
#   3. generate.py  : large model over the same sentences
#   4. analyze.py   : 30/20/50 split, UCCI and baselines, results.json / results.md / figures
#
# Every step is resumable: re-running the script skips finished work
# (latency.json is kept if present, generation logs resume where they stopped).
#
# Usage (from anywhere):
#   PYTHON=/path/to/python bash benchmarks/conll2003/run_full.sh
# Environment overrides: PYTHON, OUT, DTYPE, BATCH_SIZE, SMALL, SMALL_REV, LARGE, LARGE_REV.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
OUT="${OUT:-$HERE/runs/full}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SMALL="${SMALL:-Qwen/Qwen2.5-1.5B-Instruct}"
SMALL_REV="${SMALL_REV:-989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
LARGE="${LARGE:-Qwen/Qwen2.5-7B-Instruct}"
LARGE_REV="${LARGE_REV:-a09a35458c702b33eeacc393d103063234e8bc28}"

log() { echo "[run_full $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

mkdir -p "$OUT"
log "python: $("$PY" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
log "output: $OUT"

# Record the software and hardware the numbers come from.
"$PY" -m pip freeze > "$OUT/environment-pip-freeze.txt" 2>/dev/null || true
{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "os: $(uname -srm)"  # kernel name, release and machine; no host name
  if command -v sysctl >/dev/null 2>&1; then
    echo "cpu: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
    echo "memory_bytes: $(sysctl -n hw.memsize 2>/dev/null || true)"
  fi
  if command -v sw_vers >/dev/null 2>&1; then
    echo "macos: $(sw_vers -productVersion 2>/dev/null || true)"
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
  fi
  "$PY" -c 'import platform, torch, transformers, datasets, numpy; print("python:", platform.python_version()); print("torch:", torch.__version__); print("transformers:", transformers.__version__); print("datasets:", datasets.__version__); print("numpy:", numpy.__version__)'
} > "$OUT/environment.txt"

if [ -s "$OUT/latency.json" ]; then
  log "latency.json exists, keeping it"
else
  log "step 1/4: latency (100 queries per model, batch size 1)"
  "$PY" "$HERE/latency.py" \
    --small "$SMALL" --small-revision "$SMALL_REV" \
    --large "$LARGE" --large-revision "$LARGE_REV" \
    --dtype "$DTYPE" --n-queries 100 --warmup 3 \
    --out "$OUT/latency.json"
fi

log "step 2/4: small model generation ($SMALL)"
"$PY" "$HERE/generate.py" --model "$SMALL" --revision "$SMALL_REV" \
  --dtype "$DTYPE" --batch-size "$BATCH_SIZE" --save-token-stats \
  --out "$OUT/small.jsonl"

log "step 3/4: large model generation ($LARGE)"
"$PY" "$HERE/generate.py" --model "$LARGE" --revision "$LARGE_REV" \
  --dtype "$DTYPE" --batch-size "$BATCH_SIZE" --save-token-stats \
  --out "$OUT/large.jsonl"

log "step 4/4: analysis"
"$PY" "$HERE/analyze.py" \
  --small "$OUT/small.jsonl" --large "$OUT/large.jsonl" \
  --latency "$OUT/latency.json" --out-dir "$OUT"

log "done: $OUT/results.md"
