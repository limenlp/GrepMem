#!/usr/bin/env bash
# Evaluate an already-running model. Each output resumes independently.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "${RUN_ENV:-}" ]]; then
  source "$RUN_ENV"
fi
AGENT=${AGENT:-summary_grep}
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
BASE_URL=${BASE_URL:-http://localhost:8000/v1}
STEP=${STEP:-0}
OUT=${OUT:-$PWD/outputs/$AGENT}
export BCRL_LLM_TIMEOUT=${BCRL_LLM_TIMEOUT:-900}
mkdir -p "$OUT"
common=(--agent "$AGENT" --model "$MODEL" --base-url "$BASE_URL")

retry() {
  local rc attempt
  for attempt in 1 2 3; do
    "$@" && return 0 || rc=$?
    [[ "$rc" == 2 ]] || return "$rc"
  done
  return 2
}

for benchmark in ${BENCHMARKS:-bcpm cb ama amafile}; do
  case "$benchmark" in
    bcpm)
      retry python3 eval.py "${common[@]}" --dataset browsecomp_plus_multi \
        --concurrency "${CONCURRENCY:-64}" --out "$OUT/eval_bcpm_step$STEP.jsonl"
      ;;
    cb)
      retry python3 cb_eval.py "${common[@]}" --concurrency "${CONCURRENCY:-128}" \
        --runs-dir "$OUT/cb_runs" --out "$OUT/eval_cb_step$STEP.jsonl"
      ;;
    ama|amafile)
      protocol=stream
      [[ "$benchmark" != amafile ]] || protocol=file
      retry python3 ama_eval.py "${common[@]}" --protocol "$protocol" \
        --concurrency "${CONCURRENCY:-64}" --out "$OUT/eval_${benchmark}_step$STEP.jsonl"
      ;;
    *) echo "Unknown benchmark: $benchmark" >&2; exit 1 ;;
  esac
done
