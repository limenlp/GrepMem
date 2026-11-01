#!/usr/bin/env bash
# Foreground corpus server. Downloads and resource requirements: docs/reproduction.md.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
exec python3 retrieval/server.py --port "${PORT:-8100}" "$@"
