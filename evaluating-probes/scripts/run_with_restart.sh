#!/usr/bin/env bash
set -euo pipefail

# Simple restarter: runs the given command, and if it exits non-zero,
# waits a bit and restarts. Useful for transient OOMs or flaky I/O.
#
# Usage examples:
#   scripts/run_with_restart.sh -- python -m src.main -c my_config
#   scripts/run_with_restart.sh --delay 15 --max-restarts 0 -- \
#       EP_BATCHED_NJOBS=56 python -m src.main -c my_config
#   scripts/run_with_restart.sh --backoff --delay 10 --max-restarts 20 -- \
#       python -m src.main -c my_config

DELAY_SECONDS=1            # Base delay between restarts
MAX_RESTARTS=0             # 0 = infinite
USE_EXP_BACKOFF=false      # If true, delay grows exponentially up to a cap
BACKOFF_CAP_SECONDS=10    # Max sleep when backoff is enabled

print_usage() {
    cat <<USAGE
Usage: $0 [--delay SECONDS] [--max-restarts N] [--backoff] -- COMMAND [ARGS...]

Options:
  --delay SECONDS       Seconds to wait between restarts (default: ${DELAY_SECONDS})
  --max-restarts N      Max number of restarts; 0 means unlimited (default: ${MAX_RESTARTS})
  --backoff             Enable exponential backoff (capped at ${BACKOFF_CAP_SECONDS}s)
  -h, --help            Show this help

All args after -- are treated as the command to run.
Example:
  $0 --delay 15 --max-restarts 0 -- python -m src.main -c my_config
USAGE
}

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --delay)
            [[ $# -ge 2 ]] || { echo "$(ts) Missing value for --delay" >&2; exit 2; }
            DELAY_SECONDS="$2"; shift 2 ;;
        --max-restarts)
            [[ $# -ge 2 ]] || { echo "$(ts) Missing value for --max-restarts" >&2; exit 2; }
            MAX_RESTARTS="$2"; shift 2 ;;
        --backoff)
            USE_EXP_BACKOFF=true; shift ;;
        -h|--help)
            print_usage; exit 0 ;;
        --)
            shift; break ;;
        *)
            echo "$(ts) Unknown option: $1" >&2
            print_usage
            exit 2 ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "$(ts) Error: No command provided. Put your command after --" >&2
    print_usage
    exit 2
fi

CMD=("$@")

restarts=0

trap 'echo "$(ts) Caught termination signal. Exiting." >&2; exit 130' INT TERM

while :; do
    echo "$(ts) Starting (attempt $((restarts+1))) → ${CMD[*]}"
    "${CMD[@]}"
    status=$?

    if [[ $status -eq 0 ]]; then
        echo "$(ts) Command exited successfully. Done."
        exit 0
    fi

    # Diagnose common OOM kill exit code (137)
    if [[ $status -eq 137 ]]; then
        echo "$(ts) Command exited with 137 (SIGKILL). Likely OOM-killed. Will restart." >&2
    else
        echo "$(ts) Command exited with status $status. Will restart." >&2
    fi

    ((restarts++))
    if [[ $MAX_RESTARTS -ne 0 && $restarts -ge $MAX_RESTARTS ]]; then
        echo "$(ts) Reached max restarts ($MAX_RESTARTS). Exiting with last status $status." >&2
        exit "$status"
    fi

    sleep_seconds="$DELAY_SECONDS"
    if [[ "$USE_EXP_BACKOFF" == true ]]; then
        # Exponential backoff: delay * 2^(restarts-1), capped
        pow=$((restarts - 1))
        # Avoid overflow by capping at ~30 doublings
        if (( pow > 30 )); then pow=30; fi
        sleep_seconds=$(( DELAY_SECONDS * (1 << pow) ))
        if (( sleep_seconds > BACKOFF_CAP_SECONDS )); then
            sleep_seconds=$BACKOFF_CAP_SECONDS
        fi
    fi

    echo "$(ts) Sleeping ${sleep_seconds}s before restart #$restarts..."
    sleep "$sleep_seconds"
done


