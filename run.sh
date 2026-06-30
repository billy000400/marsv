#!/usr/bin/env bash
# Iteration-loop wrapper for one autonomous research direction.
# Usage:  ./run.sh <research-subdir> [hours]
# Example: ./run.sh dir3_manifold 4   (or launch via ./launch.sh dir3_manifold)
set -uo pipefail

export IS_SANDBOX=1              # allow --dangerously-skip-permissions as root on this disposable pod
export MPLBACKEND=Agg            # headless matplotlib: figures save to file, never try to display

DIR="${1:?usage: run.sh <research-subdir> [hours]}"

BUDGET_FILE="BUDGET.md"
BUDGET_ABS="$(pwd)/$BUDGET_FILE"
RULES_ABS="$(pwd)/CLAUDE.md"     # operator rules (read before write, curation, report structure)
read_budget() { grep -E "^$1:" "$BUDGET_FILE" 2>/dev/null | head -1 | sed -E "s/^$1:[[:space:]]*//; s/[[:space:]].*$//"; }

HOURS="${2:-$(read_budget HOURS)}";              HOURS="${HOURS:-4}"
N_AGENTS="$(read_budget N_AGENTS)";              N_AGENTS="${N_AGENTS:-1}"
CPU_CORES="$(read_budget CPU_CORES_TOTAL)";      CPU_CORES="${CPU_CORES:-4}"
RAM_TOTAL="$(read_budget RAM_TOTAL_GB)";         RAM_TOTAL="${RAM_TOTAL:-16}"
HEADROOM="$(read_budget VRAM_HEADROOM_FRACTION)";HEADROOM="${HEADROOM:-0.1}"

CPU_THREADS=$(awk "BEGIN{n=$CPU_CORES/$N_AGENTS; if(n<1)n=1; printf \"%d\", n}")
RAM_PER_AGENT=$(awk "BEGIN{printf \"%.1f\", $RAM_TOTAL/$N_AGENTS}")
VRAM_FRACTION=$(awk "BEGIN{f=(1-$HEADROOM)/$N_AGENTS; if(f<=0)f=0.1; printf \"%.3f\", f}")
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS"

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
echo "[run.sh] gpu='${GPU_NAME}' ${GPU_VRAM_GB}GB  | per-agent: vram_frac=${VRAM_FRACTION} (~${VRAM_PER_AGENT}GB)  ram=${RAM_PER_AGENT}GB  threads=${CPU_THREADS}"

while [ "$(date +%s)" -lt "$END" ] && [ ! -f STOP ]; do
  REMAIN=$(( (END - $(date +%s)) / 60 ))
  echo "[run.sh] $(date '+%F %T')  ~${REMAIN} min left  -----------------------------"

  claude -p "You are mid-project and your working memory RESETS every iteration. \
FIRST read CLAUDE.md (operator rules, at ${RULES_ABS}) and BUDGET.md (at ${BUDGET_ABS}), then \
PLAN.md, JOURNAL.md, RESULTS.md, and CHANGELOG.md in full. OBEY every rule in CLAUDE.md. \
KEY RULES: RESULTS.md and REPORT.md are FINAL, presentable deliverables — read them, then overwrite \
to current-best ONLY (no version history, no 'changed after review', no weaker/superseded variant of \
an experiment when a stronger one exists). Put ALL change history in CHANGELOG.md (append-only). \
SHARED-RESOURCE LIMITS (this run): you are 1 of ${N_AGENTS} agents on GPU '${GPU_NAME}' (${GPU_VRAM_GB} GB); \
call torch.cuda.set_per_process_memory_fraction(${VRAM_FRACTION}) (~${VRAM_PER_AGENT} GB), keep RAM under \
~${RAM_PER_AGENT} GB (memmap caches), torch.set_num_threads(${CPU_THREADS}), HALVE batch on OOM. \
You have ${REMAIN} minutes of wall-clock left. \
If that number is <= ${FINALIZE_MIN}: do ONLY finalization — refresh RESULTS.md (current-best only) and \
write a clean presentable REPORT.md per CLAUDE.md (Summary -> Methods -> Results -> Conclusion; the \
Methods section MUST give Data/Model/Layer, and DEFINE every metric and baseline with rendered \$\$LaTeX\$\$ \
equations; embed plots/ figures; current-best numbers only). Append a final CHANGELOG.md entry, then \
create an empty STOP file and stop. \
Otherwise do ONE focused iteration: advance the plan by the smallest useful step, write/modify code under \
experiments/, RUN it, then CURATE RESULTS.md to current-best (read it, overwrite clean — no history) and \
save a PNG for every quantitative result into plots/ (plt.savefig + plt.close, NEVER plt.show; headless Agg) \
referenced from RESULTS.md/REPORT.md. APPEND to CHANGELOG.md what changed in the deliverables this iteration \
(old -> new numbers if a result was superseded). Then append to JOURNAL.md (what you did, learned, next step) \
and update PLAN.md 'Current status'/'Next step'/checkboxes. End the JOURNAL entry with the 'On track?' line. \
Persist ALL state to disk before you finish; assume nothing carries over." \
    --append-system-prompt "Obey CLAUDE.md every iteration. File roles are STRICT: RESULTS.md and REPORT.md are curated, presentable, current-best — read then overwrite clean, NEVER keep history or superseded/weaker results in them. CHANGELOG.md and JOURNAL.md are append-only history. REPORT.md must have a Methods section defining every metric and baseline with \$\$LaTeX\$\$ equations plus the data/model/layer used. Visualize every reported result: PNGs in plots/ (savefig+close, never show). Read before write. Persist every iteration. Prefer small verifiable steps; on a broken experiment debug minimally or fall back per PLAN.md." \
    --dangerously-skip-permissions \
    2>&1 | tee -a session.log

  sleep 2
done

if [ -f STOP ]; then
  echo "[run.sh] STOP file present — goal reached. Ended $(date '+%F %T')."
else
  echo "[run.sh] deadline hit — ended $(date '+%F %T')."
fi
