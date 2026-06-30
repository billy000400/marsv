#!/usr/bin/env bash
# Iteration-loop wrapper for one autonomous research direction.
# Usage:  ./run.sh <research-subdir> [hours]   (hours may be fractional, e.g. 4.5)
# Example: ./run.sh dir3_manifold 4
# Run under tmux + an outer `timeout` as a guaranteed hard kill:
#   tmux new-session -d -s dir3 'timeout 14700 ./run.sh dir3_manifold 4'
set -uo pipefail

export IS_SANDBOX=1              # allow --dangerously-skip-permissions as root on this disposable pod
export MPLBACKEND=Agg            # headless matplotlib: figures save to file, never try to display

DIR="${1:?usage: run.sh <research-subdir> [hours]}"

# Inputs live in BUDGET.md at the project root (this dir, before we cd).
BUDGET_FILE="BUDGET.md"
BUDGET_ABS="$(pwd)/$BUDGET_FILE"
RULES_ABS="$(pwd)/CLAUDE.md"   # operator rules (read before write, etc.)
read_budget() { grep -E "^$1:" "$BUDGET_FILE" 2>/dev/null | head -1 | sed -E "s/^$1:[[:space:]]*//; s/[[:space:]].*$//"; }

HOURS="${2:-$(read_budget HOURS)}";              HOURS="${HOURS:-4}"
N_AGENTS="$(read_budget N_AGENTS)";              N_AGENTS="${N_AGENTS:-1}"
CPU_CORES="$(read_budget CPU_CORES_TOTAL)";      CPU_CORES="${CPU_CORES:-4}"
RAM_TOTAL="$(read_budget RAM_TOTAL_GB)";         RAM_TOTAL="${RAM_TOTAL:-16}"
HEADROOM="$(read_budget VRAM_HEADROOM_FRACTION)";HEADROOM="${HEADROOM:-0.1}"

# Derived per-agent shares (total / N_AGENTS).
CPU_THREADS=$(awk "BEGIN{n=$CPU_CORES/$N_AGENTS; if(n<1)n=1; printf \"%d\", n}")
RAM_PER_AGENT=$(awk "BEGIN{printf \"%.1f\", $RAM_TOTAL/$N_AGENTS}")
VRAM_FRACTION=$(awk "BEGIN{f=(1-$HEADROOM)/$N_AGENTS; if(f<=0)f=0.1; printf \"%.3f\", f}")
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS"

# Detect the CURRENT gpu (the pod may get a different card each launch).
GPU_NAME="unknown"; GPU_VRAM_GB="?"; VRAM_PER_AGENT="?"
if command -v nvidia-smi >/dev/null 2>&1; then
  _g=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
  if [ -n "${_g:-}" ]; then
    GPU_NAME=$(printf '%s' "$_g" | cut -d, -f1 | sed 's/^ *//;s/ *$//')
    _mib=$(printf '%s' "$_g" | cut -d, -f2 | grep -oE '[0-9]+' | head -1)
    GPU_VRAM_GB=$(awk "BEGIN{printf \"%.1f\", ${_mib:-0}/1024}")
    VRAM_PER_AGENT=$(awk "BEGIN{printf \"%.1f\", $GPU_VRAM_GB*$VRAM_FRACTION}")
  fi
fi

FINALIZE_MIN=20
BUDGET_SEC=$(awk "BEGIN{printf \"%d\", $HOURS*3600}")
END=$(( $(date +%s) + BUDGET_SEC ))

cd "$DIR" || { echo "[run.sh] cannot cd into $DIR"; exit 1; }
mkdir -p experiments results plots

echo "[run.sh] start $(date '+%F %T')  dir=$DIR  budget=${HOURS}h  agents=${N_AGENTS}"
echo "[run.sh] gpu='${GPU_NAME}' ${GPU_VRAM_GB}GB  | per-agent share: vram_frac=${VRAM_FRACTION} (~${VRAM_PER_AGENT}GB)  ram=${RAM_PER_AGENT}GB  threads=${CPU_THREADS}"

while [ "$(date +%s)" -lt "$END" ] && [ ! -f STOP ]; do
  REMAIN=$(( (END - $(date +%s)) / 60 ))
  echo "[run.sh] $(date '+%F %T')  ~${REMAIN} min left  -----------------------------"

  claude -p "You are mid-project and your working memory RESETS every iteration. \
FIRST read CLAUDE.md (operator rules, at ${RULES_ABS}) and BUDGET.md (at ${BUDGET_ABS}), then \
PLAN.md, JOURNAL.md, and RESULTS.md in full. OBEY every rule in CLAUDE.md (e.g. 'read before write'). \
SHARED-RESOURCE LIMITS (computed for THIS run): you are 1 of ${N_AGENTS} agents on GPU '${GPU_NAME}' \
(${GPU_VRAM_GB} GB). Your share: call torch.cuda.set_per_process_memory_fraction(${VRAM_FRACTION}) \
(~${VRAM_PER_AGENT} GB VRAM), keep RAM under ~${RAM_PER_AGENT} GB (memmap large caches), use \
torch.set_num_threads(${CPU_THREADS}), and HALVE batch size on any OOM. \
You have ${REMAIN} minutes of wall-clock left. \
If that number is <= ${FINALIZE_MIN}: do ONLY finalization — write the final summary and headline \
into RESULTS.md, write/update REPORT.md from whatever exists (embed the plots/ figures by filename), \
then create an empty STOP file and stop. \
Otherwise do ONE focused iteration: advance the plan by the smallest useful step, write/modify code \
under experiments/, RUN it, record any metrics in RESULTS.md AND save a PNG figure for every \
quantitative result into plots/ with a descriptive name (use plt.savefig then plt.close — NEVER \
plt.show — backend is headless Agg) and reference each figure's filename in RESULTS.md/REPORT.md, \
then append to JOURNAL.md (what you did, what you learned, the revised next step) and update \
PLAN.md's 'Current status', 'Next step', and stage checkboxes. \
End the JOURNAL entry with the required one-line 'On track?' check. \
Persist ALL state to disk before you finish; assume nothing carries over." \
    --append-system-prompt "Obey the operator rules in CLAUDE.md every iteration (read before write; never clobber a file you haven't read). Durable state lives only on disk (PLAN.md, JOURNAL.md, RESULTS.md, experiments/, results/, plots/). You share one GPU/CPU/RAM with other agents — respect the per-agent share given in the prompt and BUDGET.md every iteration. VISUALIZE every result you report: save figures as PNGs in plots/ (plt.savefig + plt.close, never plt.show) and reference them in RESULTS.md/REPORT.md. Persist every iteration. Prefer small verifiable steps. If an experiment breaks, debug minimally or record the failure and fall back per PLAN.md rather than rabbit-holing." \
    --dangerously-skip-permissions \
    2>&1 | tee -a session.log

  sleep 2
done

if [ -f STOP ]; then
  echo "[run.sh] STOP file present — goal reached. Ended $(date '+%F %T')."
else
  echo "[run.sh] deadline hit — ended $(date '+%F %T')."
fi
