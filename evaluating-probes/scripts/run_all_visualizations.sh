#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_all_visualizations.sh [--seeds "42 43 44 45 46 47 48 49 50 51"] [--force] [--configs "cfg1 cfg2 ..."]
#
# Notes:
# - By default, seeds 42..51 are used. Override with --seeds.
# - --force will overwrite existing visualizations.
# - If --configs is omitted, all configs matching configs/*_config.yaml are run.
# - This script assumes you already ran training/evaluation and results exist under results/<run_name>.

REPO_ROOT="/home/riya/pivotal/pivotal-research/evaluating-probes"
cd "$REPO_ROOT"

DEFAULT_SEEDS="42 43 44 45 46 47 48 49 50 51"
SEEDS_STR="$DEFAULT_SEEDS"
FORCE_FLAG=""
USER_CONFIGS=""

# Curated list of configs to visualize (basenames without _config.yaml)
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

print_help() {
  cat <<EOF
Run all visualizations for available configs.

Options:
  -s, --seeds   "SEED_LIST"   Space-separated seeds in quotes (default: "$DEFAULT_SEEDS")
  -f, --force                  Overwrite existing visualizations
      --configs "CFG_LIST"     Space-separated config base names to run (omit _config.yaml). Overrides curated list.
  -h, --help                   Show this help
EOF
}

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
    --configs)
      shift
      USER_CONFIGS=${1:-""}
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_help
      exit 1
      ;;
  esac
  shift
done

if [[ -n "$USER_CONFIGS" ]]; then
  # shellcheck disable=SC2206
  CONFIGS=($USER_CONFIGS)
fi

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No configs found. Provide --configs or add files to configs/*_config.yaml" >&2
  exit 1
fi

echo "Running visualizations for configs: ${CONFIGS[*]}"
echo "Seeds: $SEEDS_STR"
echo "Force: ${FORCE_FLAG:-no}"
echo

# Derive qwen-specific seeds as the first 5 seeds from the provided list
IFS=' ' read -r -a SEEDS_ARR <<< "$SEEDS_STR"
QWEN_SEEDS=""
for ((i=0; i<${#SEEDS_ARR[@]} && i<5; i++)); do
  if [[ -z "$QWEN_SEEDS" ]]; then
    QWEN_SEEDS="${SEEDS_ARR[$i]}"
  else
    QWEN_SEEDS="$QWEN_SEEDS ${SEEDS_ARR[$i]}"
  fi
done

for CFG in "${CONFIGS[@]}"; do
  CFG_FILE="configs/${CFG}_config.yaml"
  if [[ ! -f "$CFG_FILE" ]]; then
    echo "Skipping '$CFG' (missing $CFG_FILE)"
    continue
  fi

  echo "==> Visualizing: $CFG"
  # Use fewer seeds for qwen configs (first 5 of provided seeds)
  SEEDS_FOR_CFG="$SEEDS_STR"
  if [[ "$CFG" == qwen_* || "$CFG" == *qwen* ]]; then
    SEEDS_FOR_CFG="$QWEN_SEEDS"
  fi
  # shellcheck disable=SC2086
  python -m src.visualize.run_all_viz \
    -c "$CFG" \
    --seeds $SEEDS_FOR_CFG \
    $FORCE_FLAG

  echo "✓ Completed: $CFG"
  echo
done

echo "All visualizations done. Check results/<run_name>/visualizations for each config."


