#!/usr/bin/env bash
# Download source datasets without printing their questions or answers.
set -euo pipefail
hf download OpenResearcher/OpenResearcher-Dataset --repo-type dataset --include 'seed_42/*.parquet'
hf download Tevatron/browsecomp-plus --repo-type dataset
hf download AMA-bench/AMA-bench --repo-type dataset
if [[ "${RETRIEVAL_DATA:-0}" == 1 ]]; then
  hf download Qwen/Qwen3-Embedding-8B
  hf download Tevatron/browsecomp-plus-corpus --repo-type dataset
  hf download Tevatron/browsecomp-plus-indexes --repo-type dataset --include 'qwen3-embedding-8b/*'
  hf download OpenResearcher/OpenResearcher-Corpus --repo-type dataset
  hf download OpenResearcher/OpenResearcher-Indexes --repo-type dataset --include 'qwen3-embedding-8b/*'
fi
