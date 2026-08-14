#!/usr/bin/env bash
# End-to-end supervisor: train -> evaluate -> benchmark, detached and resumable.
#
# Two failure modes bit earlier runs and are designed out here:
#
#   1. `pgrep -f howto.train` matches the *watcher's own* command line, so a
#      loop waiting on it waits forever. Nothing here greps process names --
#      stages run in the foreground of this script and we use exit codes.
#   2. Background jobs died with the interactive session. Launch this with
#      setsid so it survives the terminal going away.
#
# Progress is written to STATUS after every stage, so a heartbeat is always
# visible without inspecting logs. Every stage is idempotent: training skips
# models that already have run_meta.json, so a restart resumes rather than
# redoing work.
set -uo pipefail
cd "$(dirname "$0")/.."

FOLD="${FOLD:-0}"
EPOCHS="${EPOCHS:-1}"
MAX_TEST="${MAX_TEST:-400}"
LOGDIR="${LOGDIR:-/tmp/howto-pipeline}"
STATUS="$LOGDIR/STATUS"
ALL="qwen3-0.6b,qwen25-coder-0.5b,qwen35-0.8b-base,qwen35-0.8b,gemma3-1b,qwen3-1.7b,qwen35-2b-base,qwen35-2b,gemma4-e2b,qwen3-0.6b-bd3lm"

mkdir -p "$LOGDIR"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

count_done() { ls runs/*/run_meta.json 2>/dev/null | wc -l; }

: > "$STATUS"
say "pipeline start (fold=$FOLD epochs=$EPOCHS max_test=$MAX_TEST)"

# ---- stage 1: training -------------------------------------------------------
# train_all.sh already retries per-model; wrap it so a hard crash of the whole
# script still gets a second chance before we give up on the stage.
for attempt in 1 2; do
  say "training attempt $attempt ($(count_done)/10 complete)"
  LOGDIR="$LOGDIR/train" bash scripts/train_all.sh >> "$LOGDIR/train_all.log" 2>&1
  say "training pass $attempt finished: $(count_done)/10 complete"
  [ "$(count_done)" -ge 10 ] && break
done

TRAINED=$(for d in runs/*/; do
  [ -f "$d/run_meta.json" ] && basename "$d" | sed "s/-fold${FOLD}\$//"
done | paste -sd, -)
say "trained models: ${TRAINED:-none}"

if [ -z "$TRAINED" ]; then
  say "FATAL: nothing trained, stopping"
  exit 1
fi

# ---- stage 2: evaluation -----------------------------------------------------
say "evaluating (max_test=$MAX_TEST)"
uv run python -m howto.evaluate --candidates "$TRAINED" --folds "$FOLD" \
  --max-test "$MAX_TEST" > "$LOGDIR/evaluate.log" 2>&1
say "evaluation exit=$?"

# ---- stage 3: latency benchmark ---------------------------------------------
say "benchmarking"
uv run python -m howto.benchmark --candidates "$TRAINED" --fold "$FOLD" \
  --n-prompts 20 > "$LOGDIR/benchmark.log" 2>&1
say "benchmark exit=$?"

say "PIPELINE COMPLETE"
