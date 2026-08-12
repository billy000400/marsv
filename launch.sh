#!/usr/bin/env bash
# launch.sh — start ONE research loop in tmux. Session name is parsed from the direction folder.
# Usage:  ./launch.sh <direction_dir> [hours]
# Examples:
#   ./launch.sh dir3_manifold          # uses HOURS from BUDGET.md
#   ./launch.sh dir9_ood 6             # override to 6h for this launch
set -euo pipefail

# project root = the folder this script lives in (must hold run.sh + BUDGET.md)
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
[ -x run.sh ] || { echo "[launch] no executable run.sh in $ROOT — run this from the project root."; exit 1; }
[ -f workflow.py ] && [ -f WRITING.md ] || { echo "[launch] missing workflow.py or WRITING.md"; exit 1; }

RAW="${1:?usage: ./launch.sh <direction_dir> [hours]}"
DIR="$(basename "${RAW%/}")"                       # strip trailing slash / any path
SESSION="$(printf '%s' "$DIR" | tr '.:' '__')"     # tmux name = dir (dots/colons -> _)
HOURS_ARG="${2:-}"

# --- guards (so it never silently dies the way the raw tmux command could) ---
[ -d "$DIR" ]          || { echo "[launch] no such direction: $DIR/  (create with ./new_direction.sh)"; exit 1; }
[ -f "$DIR/PLAN.md" ]  || { echo "[launch] $DIR/ has no PLAN.md — not a valid direction."; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[launch] '$SESSION' already running.  attach: tmux attach -t $SESSION   kill: tmux kill-session -t $SESSION"; exit 1
fi
python3 workflow.py prepare "$DIR" >/dev/null
TASK_MANIFEST="$(python3 workflow.py active "$DIR")"
if [ -n "$TASK_MANIFEST" ]; then
  echo "[launch] task manifest: $TASK_MANIFEST (loaded by every run.sh iteration)"
fi
if [ -f "$DIR/STOP" ]; then
  echo "[launch] $DIR/ already finished (STOP present). To continue:  rm $DIR/STOP  then relaunch."; exit 1
fi

# --- timeout = HOURS (arg > BUDGET.md > 4) * 3600 + 5min grace ---
read_budget(){ grep -E "^$1:" BUDGET.md 2>/dev/null | head -1 | sed -E "s/^$1:[[:space:]]*//;s/[[:space:]].*$//"; }
HOURS="${HOURS_ARG:-$(read_budget HOURS)}"; HOURS="${HOURS:-4}"
TLIMIT=$(awk "BEGIN{printf \"%d\", $HOURS*3600+300}")

tmux new-session -d -s "$SESSION" "timeout $TLIMIT ./run.sh $DIR ${HOURS_ARG}"
echo "[launch] started '$SESSION'  (dir=$DIR, ${HOURS}h, hard-kill ${TLIMIT}s)"
echo "[launch] attach: tmux attach -t $SESSION    |    tail: tail -f $DIR/session.log"
