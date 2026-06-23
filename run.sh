#!/usr/bin/env bash
# Iteration-loop wrapper for one autonomous research direction.
# Usage:  ./run.sh <research-subdir> [hours]   (hours may be fractional, e.g. 4.5)
# Example: ./run.sh dir3_manifold 4.5
# Run under tmux + an outer `timeout` as a guaranteed hard kill:
#   tmux new-session -d -s dir3 'timeout 14700 ./run.sh dir3_manifold 4'   # 4h + 5min grace
set -uo pipefail

# This pod runs as root, and Claude Code blocks --dangerously-skip-permissions for uid 0
# unless a sandbox is signalled. The pod is dedicated/disposable, so we signal sandbox here.
# (The deny-list — rm -rf /, .git/.claude writes, etc. — still applies.)
export IS_SANDBOX=1

DIR="${1:?usage: run.sh <research-subdir> [hours]}"

# Frequently-changed knobs live in BUDGET.md at the project root (this dir, before we cd).
BUDGET_FILE="BUDGET.md"
BUDGET_ABS="$(pwd)/$BUDGET_FILE"
read_budget() { grep -E "^$1:" "$BUDGET_FILE" 2>/dev/null | head -1 | sed -E "s/^$1:[[:space:]]*//; s/[[:space:]].*$//"; }

# Time budget: CLI arg $2 wins, else BUDGET.md HOURS, else 4.
HOURS="${2:-$(read_budget HOURS)}"; HOURS="${HOURS:-4}"
# CPU thread cap so two concurrent agents don't oversubscribe the 4 cores.
CPU_THREADS="$(read_budget CPU_THREADS_PER_AGENT)"; CPU_THREADS="${CPU_THREADS:-2}"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS"

FINALIZE_MIN=20                                  # reserve this long to wrap up
BUDGET_SEC=$(awk "BEGIN{printf \"%d\", $HOURS*3600}")   # float-safe (handles 4.5)
END=$(( $(date +%s) + BUDGET_SEC ))              # hard wall-clock deadline

cd "$DIR" || { echo "[run.sh] cannot cd into $DIR"; exit 1; }
mkdir -p experiments results

echo "[run.sh] start $(date '+%F %T')  dir=$DIR  budget=${HOURS}h  cpu_threads=${CPU_THREADS}"

while [ "$(date +%s)" -lt "$END" ] && [ ! -f STOP ]; do
  REMAIN=$(( (END - $(date +%s)) / 60 ))
  echo "[run.sh] $(date '+%F %T')  ~${REMAIN} min left  -----------------------------"

  claude -p "You are mid-project and your working memory RESETS every iteration. \
FIRST read BUDGET.md (at ${BUDGET_ABS}), then PLAN.md, JOURNAL.md, and RESULTS.md in full. \
BUDGET.md has the SHARED GPU/RAM/CPU limits you MUST stay within — you share one machine with \
another agent, so cap VRAM via torch.cuda.set_per_process_memory_fraction, memmap large caches, \
set torch threads to the CPU budget, and HALVE batch size on any OOM. \
You have ${REMAIN} minutes of wall-clock left. \
If that number is <= ${FINALIZE_MIN}: do ONLY finalization — write the final summary and headline \
into RESULTS.md, write/update REPORT.md from whatever exists, then create an empty STOP file and stop. \
Otherwise do ONE focused iteration: advance the plan by the smallest useful step, write/modify code \
under experiments/, RUN it, record any metrics in RESULTS.md, then append to JOURNAL.md (what you did, \
what you learned, the revised next step) and update PLAN.md's 'Current status', 'Next step', and stage \
checkboxes. End the JOURNAL entry with the required one-line 'On track?' check. \
Persist ALL state to disk before you finish; assume nothing carries over." \
    --append-system-prompt "Durable state lives only on disk (PLAN.md, JOURNAL.md, RESULTS.md, experiments/, results/). You share one GPU/CPU/RAM with another agent — respect the limits in BUDGET.md every iteration. Persist every iteration. Prefer small verifiable steps over big leaps. If an experiment breaks, debug minimally or record the failure and fall back per PLAN.md rather than rabbit-holing." \
    --dangerously-skip-permissions \
    2>&1 | tee -a session.log

  sleep 2
done

if [ -f STOP ]; then
  echo "[run.sh] STOP file present — goal reached. Ended $(date '+%F %T')."
else
  echo "[run.sh] deadline hit — ended $(date '+%F %T')."
fi
