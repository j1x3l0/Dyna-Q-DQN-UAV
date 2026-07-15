#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
EPISODES="${EPISODES:-1000}"
LOG_FILE="${LOG_FILE:-k_experiment.log}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  echo "Create it first and install requirements; see readme.md." >&2
  exit 1
fi

mkdir -p results/k_experiments

echo "Starting k=0/1/3 x seeds=42/123/2026, episodes=$EPISODES"
echo "Log: $LOG_FILE"

nohup env \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" \
  "$PYTHON_BIN" scripts/run_k_experiment.py \
    --k-values 0 1 3 \
    --seeds 42 123 2026 \
    --episodes "$EPISODES" \
    --case 1 \
    > "$LOG_FILE" 2>&1 &

pid=$!
printf '%s\n' "$pid" > "${LOG_FILE}.pid"
echo "Started PID $pid"
echo "Follow progress: tail -f $LOG_FILE"
