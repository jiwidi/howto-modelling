#!/usr/bin/env bash
# Train every candidate on the 5090, several at a time.
#
# A single training process leaves the GPU ~58% idle: the batches are small and
# the model is tiny, so most of the time goes on kernel launches rather than
# compute. Running independent models concurrently fills the card. Models are
# grouped into waves by VRAM so a wave never exceeds ~30 GB of the 32 GB card.
#
# A run counts as complete only when run_meta.json exists -- the output
# directory is created at trainer init, so its presence proves nothing. Any
# candidate missing that file is retried at the end.
set -uo pipefail
cd "$(dirname "$0")/.."

FOLD="${FOLD:-0}"
EPOCHS="${EPOCHS:-1}"
LOGDIR="${LOGDIR:-/tmp/howto-train-logs}"
mkdir -p "$LOGDIR"

# wave definitions: "batch_size candidate1 candidate2 ..."
WAVES=(
  "16 qwen3-0.6b qwen25-coder-0.5b qwen3-0.6b-bd3lm"
  "16 qwen35-0.8b qwen35-0.8b-base gemma3-1b"
  "16 qwen3-1.7b qwen35-2b-base"
  "8  qwen35-2b gemma4-e2b"
)

is_done() { [ -f "runs/$1-fold${FOLD}/run_meta.json" ]; }

run_wave() {
  local bs="$1"; shift
  local pids=()
  for m in "$@"; do
    if is_done "$m"; then echo "  skip $m (already complete)"; continue; fi
    echo "  start $m (batch $bs)"
    uv run python -m howto.train --candidates "$m" --folds "$FOLD" \
      --epochs "$EPOCHS" --batch-size "$bs" > "$LOGDIR/$m.log" 2>&1 &
    pids+=($!)
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
  return 0
}

for wave in "${WAVES[@]}"; do
  read -r bs rest <<< "$wave"
  echo "=== wave: $rest ==="
  run_wave "$bs" $rest
done

# Retry pass: anything that crashed or was killed gets one more attempt, alone
# and with a smaller batch in case the failure was memory pressure.
MISSING=()
for wave in "${WAVES[@]}"; do
  read -r _ rest <<< "$wave"
  for m in $rest; do is_done "$m" || MISSING+=("$m"); done
done

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "=== retrying incomplete: ${MISSING[*]} ==="
  for m in "${MISSING[@]}"; do
    rm -rf "runs/$m-fold${FOLD}"
    uv run python -m howto.train --candidates "$m" --folds "$FOLD" \
      --epochs "$EPOCHS" --batch-size 4 > "$LOGDIR/$m.retry.log" 2>&1
  done
fi

echo "=== complete ==="
for d in runs/*/; do [ -f "$d/run_meta.json" ] && echo "  $(basename "$d")"; done
