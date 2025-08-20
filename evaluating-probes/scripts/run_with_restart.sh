#!/usr/bin/env bash
set -uo pipefail

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
BACKOFF_CAP_SECONDS=10     # Max sleep when backoff is enabled
LOG_FILE=""                # If set, tee all output to this file as well as terminal
LOG_DIR=""                 # If set (and LOG_FILE not set), create timestamped log file in this dir
ROTATE_PER_RESTART=false   # If true and using --log-dir, create a new log file per restart attempt
LOG_CURRENT_FILE=""        # Internal: current attempt's log file when rotating

print_usage() {
    cat <<USAGE
Usage: $0 [--delay SECONDS] [--max-restarts N] [--backoff] -- COMMAND [ARGS...]

Options:
  --delay SECONDS       Seconds to wait between restarts (default: ${DELAY_SECONDS})
  --max-restarts N      Max number of restarts; 0 means unlimited (default: ${MAX_RESTARTS})
  --backoff             Enable exponential backoff (capped at ${BACKOFF_CAP_SECONDS}s)
  --log FILE            Tee all stdout/stderr to FILE (append) while still printing to terminal
  --log-dir DIR         Log under DIR. With --rotate-per-restart, creates per-attempt files; otherwise one timestamped file
  --rotate-per-restart  When used with --log-dir, create a new timestamped log file on every restart attempt
  -h, --help            Show this help

All args after -- are treated as the command to run.
Example:
  $0 --delay 15 --max-restarts 0 -- python -m src.main -c my_config
USAGE
}

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Log helper: echo to terminal and append to current attempt file if set
log_line() {
    local message="$*"
    if [[ -n "${LOG_CURRENT_FILE}" ]]; then
        echo "$(ts) ${message}" | tee -a "${LOG_CURRENT_FILE}"
    else
        echo "$(ts) ${message}"
    fi
}

child_pid=-1

cleanup_children() {
    # Best-effort: kill the whole process group of the last-started child and reap it
    if [[ ${child_pid} -gt 0 ]]; then
        # Send TERM to the child's process group first for graceful shutdown
        kill -TERM -"${child_pid}" 2>/dev/null || true
        # Short grace period
        sleep 2
        # Force kill if still alive
        kill -KILL -"${child_pid}" 2>/dev/null || true
        # Reap the child to avoid zombies
        wait "${child_pid}" 2>/dev/null || true
        child_pid=-1
    fi
}

on_term() {
    log_line "Caught termination signal. Killing child process group and exiting."
    cleanup_children
    exit 130
}

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
        --log)
            [[ $# -ge 2 ]] || { echo "$(ts) Missing value for --log" >&2; exit 2; }
            LOG_FILE="$2"; shift 2 ;;
        --log-dir)
            [[ $# -ge 2 ]] || { echo "$(ts) Missing value for --log-dir" >&2; exit 2; }
            LOG_DIR="$2"; shift 2 ;;
        --rotate-per-restart)
            ROTATE_PER_RESTART=true; shift ;;
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

# If user didn't specify any logging option, default to rotating logs under ./logs
if [[ -z "$LOG_FILE" && -z "$LOG_DIR" ]]; then
    ROTATE_PER_RESTART=true
    LOG_DIR="./logs"
fi

# Configure logging
if [[ "$ROTATE_PER_RESTART" == true ]]; then
    # If rotating but no dir specified, default to ./logs
    if [[ -z "$LOG_DIR" ]]; then
        LOG_DIR="./logs"
    fi
    mkdir -p "$LOG_DIR"
    if [[ -n "$LOG_FILE" ]]; then
        echo "$(ts) Warning: --rotate-per-restart used with --log. Rotation is ignored; using single file $LOG_FILE" >&2
    else
        echo "$(ts) Per-restart logging enabled. Logs will be written under $LOG_DIR" >&2
    fi
else
    # Single-file logging: prefer explicit --log, else create timestamped file under --log-dir
    if [[ -z "$LOG_FILE" && -n "$LOG_DIR" ]]; then
        ts_name=$(date +%Y%m%d_%H%M%S)
        LOG_FILE="${LOG_DIR%/}/run_${ts_name}.log"
    fi
    if [[ -n "$LOG_FILE" ]]; then
        mkdir -p "$(dirname "$LOG_FILE")" || true
        echo "$(ts) Logging enabled. Appending to $LOG_FILE" >> "$LOG_FILE"
        exec > >(stdbuf -oL -eL tee -a "$LOG_FILE") 2>&1
    fi
fi

restarts=0

trap 'on_term' INT TERM
trap 'cleanup_children' EXIT

while :; do
    # Set up per-restart log file if rotating; always append a fresh file per attempt
    LOG_CURRENT_FILE=""
    if [[ "$ROTATE_PER_RESTART" == true && -z "$LOG_FILE" ]]; then
        ts_attempt=$(date +%Y%m%d_%H%M%S)
        LOG_CURRENT_FILE="${LOG_DIR%/}/run_${ts_attempt}_attempt_$((restarts+1)).log"
        mkdir -p "$(dirname "$LOG_CURRENT_FILE")"
        echo "$(ts) Logging attempt $((restarts+1)) to $LOG_CURRENT_FILE" >> "$LOG_CURRENT_FILE"
    fi

    log_line "Starting (attempt $((restarts+1))) → ${CMD[*]}"

    # Run the command in its own process group so we can cleanly kill all descendants
    if [[ -n "$LOG_CURRENT_FILE" ]]; then
        # Per-restart rotating logs: tee command output to file and terminal using process substitution
        setsid "${CMD[@]}" > >(stdbuf -oL -eL tee -a "$LOG_CURRENT_FILE") 2>&1 &
        child_pid=$!
        wait "$child_pid"
        status=$?
    else
        # Global logging (if enabled earlier via exec) or plain terminal output
        setsid "${CMD[@]}" &
        child_pid=$!
        wait "$child_pid"
        status=$?
    fi
    # Child finished; ensure variable reset (already reaped)
    child_pid=-1

    # Diagnose common OOM kill exit code (137) and SIGKILL (-9)
    if [[ $status -eq 137 || $status -eq 9 ]]; then
        log_line "Command was SIGKILLed ($status). Likely OOM or external kill. Will restart."
    elif [[ $status -eq 0 ]]; then
        log_line "Command exited successfully. Will restart."
    else
        log_line "Command exited with status $status. Will restart."
    fi

    ((restarts++))
    if [[ $MAX_RESTARTS -ne 0 && $restarts -ge $MAX_RESTARTS ]]; then
        log_line "Reached max restarts ($MAX_RESTARTS). Exiting with last status $status."
        exit "$status"
    fi

    sleep_seconds="$DELAY_SECONDS"
    if [[ "$USE_EXP_BACKOFF" == true ]]; then
        pow=$((restarts - 1))
        if (( pow > 30 )); then pow=30; fi
        sleep_seconds=$(( DELAY_SECONDS * (1 << pow) ))
        if (( sleep_seconds > BACKOFF_CAP_SECONDS )); then
            sleep_seconds=$BACKOFF_CAP_SECONDS
        fi
    fi

    log_line "Sleeping ${sleep_seconds}s before restart #$restarts..."
    sleep "$sleep_seconds"
done


