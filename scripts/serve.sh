#!/usr/bin/env bash
# Foreground server; run in a separate terminal. Check readiness at /health.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-Qwen/Qwen3.5-9B}
GPUS=${GPUS:-0}
IFS=',' read -ra devices <<< "$GPUS"
CUDA_VISIBLE_DEVICES=$GPUS exec vllm serve "$MODEL" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --served-model-name "${SERVED_MODEL:-Qwen/Qwen3.5-9B}" \
  --dtype bfloat16 --max-model-len 65536 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --gdn-prefill-backend triton --enable-prompt-tokens-details \
  --data-parallel-size "${#devices[@]}"
