#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_all_visualizations.sh [--seeds "42 43 44 45 46 47 48 49 50 51"] [--force]
#
# Notes:
# - By default, seeds 42..51 are used. Override with --seeds.
# - --force will overwrite existing visualizations.
# - This script assumes you already ran training/evaluation and results exist under results/<run_name>.

REPO_ROOT="/home/riya/pivotal/pivotal-research/evaluating-probes"
cd "$REPO_ROOT"

# All config base names (without _config.yaml)
CONFIGS=(
  gemma_spam_cpu
  gemma_spam_gpu
  llama_mask_cpu
  llama_mask_gpu
  qwen_0.6b_cpu
  qwen_0.6b_gpu
  qwen_1.7b_cpu
  qwen_1.7b_gpu
  qwen_4b_cpu
  qwen_4b_gpu
  qwen_8b_cpu
  qwen_8b_gpu
  qwen_14b_cpu
  qwen_14b_gpu
  qwen_32b_cpu
  qwen_32b_gpu
)

# Defaults
DEFAULT_SEEDS="42 43 44 45 46 47 48 49 50 51"
SEEDS_STR="$DEFAULT_SEEDS"
FORCE_FLAG=""

# Parse args
while (( "$#" )); do
  case "$1" in
    -s|--seeds)
      shift
      SEEDS_STR=${1:-"$DEFAULT_SEEDS"}
      ;;
    -f|--force)
      FORCE_FLAG="--force"
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
  shift
done

echo "Running visualizations for configs: ${CONFIGS[*]}"
echo "Seeds: $SEEDS_STR"
echo "Force: ${FORCE_FLAG:-no}"
echo

for CFG in "${CONFIGS[@]}"; do
  CFG_FILE="configs/${CFG}_config.yaml"
  if [[ ! -f "$CFG_FILE" ]]; then
    echo "Skipping '$CFG' (missing $CFG_FILE)"
    continue
  fi

  echo "==> Visualizing: $CFG"
  # shellcheck disable=SC2086
  python -m src.visualize.run_all_viz \
    -c "$CFG" \
    --seeds $SEEDS_STR \
    $FORCE_FLAG

  echo "✓ Completed: $CFG"
  echo
done

echo "All visualizations done. Check results/<run_name>/visualizations for each config."


