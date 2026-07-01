#!/usr/bin/env bash
# relaunch.sh — clear a finished direction's STOP file and start its research loop again in tmux.
# Same as launch.sh, except a present STOP file is DELETED (to continue) instead of aborting.
# Usage:  ./relaunch.sh <direction_dir> [hours]
# Examples:
#   ./relaunch.sh dir3_manifold          # uses HOURS from BUDGET.md
#   ./relaunch.sh dir9_ood 6             # override to 6h for this relaunch
set -euo pipefail

# project root = the folder this script lives in (must hold run.sh + BUDGET.md)
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
[ -x run.sh ] || { echo "[relaunch] no executable run.sh in $ROOT — run this from the project root."; exit 1; }

RAW="${1:?usage: ./relaunch.sh <direction_dir> [hours]}"
DIR="$(basename "${RAW%/}")"                       # strip trailing slash / any path
SESSION="$(printf '%s' "$DIR" | tr '.:' '__')"     # tmux name = dir (dots/colons -> _)
HOURS_ARG="${2:-}"

# --- guards (so it never silently dies the way the raw tmux command could) ---
[ -d "$DIR" ]          || { echo "[relaunch] no such direction: $DIR/  (create with ./new_direction.sh)"; exit 1; }
[ -f "$DIR/PLAN.md" ]  || { echo "[relaunch] $DIR/ has no PLAN.md — not a valid direction."; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[relaunch] '$SESSION' already running.  attach: tmux attach -t $SESSION   kill: tmux kill-session -t $SESSION"; exit 1
fi
if [ -f "$DIR/STOP" ]; then
  rm -f "$DIR/STOP"
  echo "[relaunch] removed $DIR/STOP — continuing the direction."
fi

# --- timeout = HOURS (arg > BUDGET.md > 4) * 3600 + 5min grace ---
read_budget(){ grep -E "^$1:" BUDGET.md 2>/dev/null | head -1 | sed -E "s/^$1:[[:space:]]*//;s/[[:space:]].*$//"; }
HOURS="${HOURS_ARG:-$(read_budget HOURS)}"; HOURS="${HOURS:-4}"
TLIMIT=$(awk "BEGIN{printf \"%d\", $HOURS*3600+300}")

tmux new-session -d -s "$SESSION" "timeout $TLIMIT ./run.sh $DIR ${HOURS_ARG}"
echo "[relaunch] started '$SESSION'  (dir=$DIR, ${HOURS}h, hard-kill ${TLIMIT}s)"
echo "[relaunch] attach: tmux attach -t $SESSION    |    tail: tail -f $DIR/session.log"
