#!/usr/bin/env bash
set -euo pipefail

# Runs all Qwen configs sequentially

cd "$(dirname "$0")/.."  # move to repo root

CONFIGS=(
  # "qwen_0.6b_cpu"
  # "qwen_1.7b_cpu"
  # "qwen_4b_cpu"
  # "qwen_8b_cpu"
  "qwen_14b_cpu"
  "qwen_32b_cpu"
  # "qwen_0.6b_gpu"
  # "qwen_1.7b_gpu"
  # "qwen_4b_gpu"
  # "qwen_8b_gpu"
  # "qwen_14b_gpu"
  # "qwen_32b_gpu"
)

for cfg in "${CONFIGS[@]}"; do
  echo "=== Running $cfg ==="
  python -m src.main -c "$cfg" || { echo "Failed on $cfg"; exit 1; }
  echo "=== Completed $cfg ==="
  echo
done


