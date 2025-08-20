#!/usr/bin/env bash
set -euo pipefail

# Runs all Qwen configs sequentially
# Usage:
# chmod +x pivotal-research/evaluating-probes/scripts/run_qwen_configs.sh
# bash pivotal-research/evaluating-probes/scripts/run_qwen_configs.sh

cd "$(dirname "$0")/.."  # move to repo root (pivotal-research/evaluating-probes)

CONFIGS=(
  # "qwen_0.6b_cpu"
  # "qwen_1.7b_cpu"
  # "qwen_4b_cpu"
  # "qwen_8b_cpu"
<<<<<<< Updated upstream
  # "qwen_14b_cpu"
=======
  "qwen_14b_cpu"
>>>>>>> Stashed changes
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


