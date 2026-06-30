#!/usr/bin/env bash
# setup_tree.sh — SAFE bootstrap of the executable TOOLING only (run.sh, new_direction.sh, launch.sh).
# - Creates a script ONLY if absent (never clobbers). Does NOT touch direction folders.
# - Does NOT seed CLAUDE.md or BUDGET.md: those are hand-owned, git-tracked source-of-truth.
#   Edit them directly; there is no duplicate copy to keep in sync.
# - FORCE=1 overwrites the scripts, backing up each to <file>.bak.<epoch> first.
# Usage:  bash setup_tree.sh [PROJECT_ROOT]   |   FORCE=1 bash setup_tree.sh [PROJECT_ROOT]
set -euo pipefail
ROOT="${1:-.}"
[ -d "$ROOT" ] || { echo "[setup] project root '$ROOT' not found"; exit 1; }
cd "$ROOT"
echo "[setup] target: $(pwd)   FORCE=${FORCE:-0}"

write_if_absent() {
  local f="$1"
  if [ -e "$f" ] && [ "${FORCE:-0}" != "1" ]; then echo "[setup] skip (exists): $f"; cat >/dev/null; return 0; fi
  if [ -e "$f" ]; then cp -a "$f" "$f.bak.$(date +%s)"; echo "[setup] backup+overwrite: $f"; else echo "[setup] create: $f"; fi
  cat > "$f"
}

write_if_absent run.sh <<'__EOF_run__'
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
__EOF_run__
[ -f run.sh ] && chmod +x run.sh

write_if_absent new_direction.sh <<'__EOF_new__'
#!/usr/bin/env bash
# new_direction.sh — scaffold ONE new research direction for the run.sh loop.
# Run from the project root (where run.sh, BUDGET.md, CLAUDE.md live).
# Usage:   ./new_direction.sh <dir_name> ["Title / one-line objective"]
# Safe by design: refuses to overwrite an existing direction.
set -euo pipefail

DIR="${1:?usage: ./new_direction.sh <dir_name> [\"Title\"]}"
TITLE="${2:-TODO — describe this direction}"

case "$DIR" in */*|*" "*) echo "[new] '$DIR' must be a plain folder name"; exit 1 ;; esac
if [ -e "$DIR/PLAN.md" ]; then
  echo "[new] REFUSING: $DIR/PLAN.md already exists — would overwrite existing work."; exit 1
fi
[ -f run.sh ]    || echo "[new] WARNING: no run.sh in $(pwd) — run from the project root."
[ -f BUDGET.md ] || echo "[new] WARNING: no BUDGET.md in $(pwd)."
[ -f CLAUDE.md ] || echo "[new] WARNING: no CLAUDE.md in $(pwd) — agents won't see the operator rules."

mkdir -p "$DIR/experiments" "$DIR/results" "$DIR/plots"
: > "$DIR/experiments/.gitkeep"; : > "$DIR/plots/.gitkeep"

# ---- PLAN.md ----
printf '# PLAN — Direction: %s\n\n' "$TITLE"                                  >  "$DIR/PLAN.md"
printf '> Working folder: `%s`. Agent REWRITES "Current status"/"Next step" + ticks stages each\n' "$DIR" >> "$DIR/PLAN.md"
printf '> iteration. Disk (PLAN/JOURNAL/RESULTS/CHANGELOG + ../BUDGET.md + ../CLAUDE.md) is the only memory.\n\n' >> "$DIR/PLAN.md"
cat >> "$DIR/PLAN.md" <<'EOF'
## Success criterion (definition of "done")
TODO — concrete artifact(s) that mean done, e.g. "RESULTS.md has <X>/<Y> (current-best) and REPORT.md
gives a clear verdict with Methods + figures." Null/negative results are COMPLETE if the question is
answered. When done, the loop writes an empty `STOP` file.

## Fallback (if time runs short)
TODO — minimum acceptable deliverable. The wrapper reserves the last 20 min to finalize + STOP.

## Setup (fixed)
- TODO — model / data / hook points. Default: GPT-2 small via HuggingFace `transformers` + forward hooks; STREAM data.
- **Shared limits in `../BUDGET.md`; operator rules in `../CLAUDE.md` — read both every iteration.**
- **Deliverable hygiene (see CLAUDE.md):** RESULTS.md/REPORT.md = current-best only, no history; CHANGELOG.md = the history.
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, flax** — they break the CUDA build.

## Stages (checklist)
- [ ] S1 — TODO
- [ ] S2 — TODO
- [ ] S3 — TODO  (each reported metric: produce + save figure to plots/ + define it in REPORT.md Methods)

## Out of scope (do NOT)
- TODO. Don't drift into other directions.

## On-track check (required every iteration)
End each JOURNAL.md entry with: `On track? <yes/no> — <stage, % done, blocker if any>`.

## Current status
(none yet — fresh start)

## Next step
TODO — first concrete action.
EOF

# ---- JOURNAL.md (append-only working log) ----
printf '# JOURNAL — Direction: %s\n\n' "$TITLE"                               >  "$DIR/JOURNAL.md"
cat >> "$DIR/JOURNAL.md" <<'EOF'
Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---
EOF

# ---- CHANGELOG.md (append-only history of deliverable changes) ----
printf '# CHANGELOG — Direction: %s\n\n' "$TITLE"                             >  "$DIR/CHANGELOG.md"
cat >> "$DIR/CHANGELOG.md" <<'EOF'
Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---
EOF

# ---- RESULTS.md (curated, current-best only) ----
printf '# RESULTS — Direction: %s\n\n' "$TITLE"                               >  "$DIR/RESULTS.md"
cat >> "$DIR/RESULTS.md" <<'EOF'
> CURRENT-BEST ONLY. One row per experiment. No history, no superseded/weaker variants
> (those live in CHANGELOG.md). Read this file before rewriting it.

## Metrics
_(current-best result table(s))_

## Figures
_(each plots/*.png with a one-line caption)_

## Headline
_(the single current takeaway)_
EOF

# ---- REPORT.md (presentable skeleton with required Methods section) ----
printf '# REPORT — Direction: %s\n\n' "$TITLE"                               >  "$DIR/REPORT.md"
cat >> "$DIR/REPORT.md" <<'EOF'
> Final, presentable, current-best only (no history — see CHANGELOG.md). Read before rewriting.

## Summary
TODO — 2-4 sentences: the question, the headline result, the verdict.

## Methods
### Data & Model
TODO — dataset, model (e.g. GPT-2 small, 124M), exact layer(s)/hook point, sample sizes.

### Metrics
TODO — define EACH metric with a rendered equation. Example:
$$\mathrm{AUROC} = \Pr\big(s(x^{+}) > s(x^{-})\big)$$
State exactly what `s(x)` scores and which direction means "more anomalous".

### Baselines
TODO — name and define EACH baseline. Example (Mahalanobis distance):
$$d_M(x) = \sqrt{(x-\mu)^{\top}\,\Sigma^{-1}\,(x-\mu)}$$

## Results
TODO — current-best numbers only (one row per experiment), referencing figures in plots/.

## Conclusion
TODO — what the result implies; limitations.
EOF

echo "[new] created $DIR/ with PLAN, JOURNAL, RESULTS (curated), REPORT (skeleton+Methods), CHANGELOG, experiments/, results/, plots/"
echo "[new] next: 1) fill the TODOs in $DIR/PLAN.md   2) ./launch.sh $DIR"
__EOF_new__
[ -f new_direction.sh ] && chmod +x new_direction.sh

write_if_absent launch.sh <<'__EOF_launch__'
#!/usr/bin/env bash
# launch.sh — start ONE research loop in tmux. Session name is parsed from the direction folder.
# Usage:  ./launch.sh <direction_dir> [hours]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
[ -x run.sh ] || { echo "[launch] no executable run.sh in $ROOT — run from the project root."; exit 1; }
RAW="${1:?usage: ./launch.sh <direction_dir> [hours]}"
DIR="$(basename "${RAW%/}")"
SESSION="$(printf '%s' "$DIR" | tr '.:' '__')"
HOURS_ARG="${2:-}"
[ -d "$DIR" ]         || { echo "[launch] no such direction: $DIR/  (create with ./new_direction.sh)"; exit 1; }
[ -f "$DIR/PLAN.md" ] || { echo "[launch] $DIR/ has no PLAN.md — not a valid direction."; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[launch] '$SESSION' already running.  attach: tmux attach -t $SESSION   kill: tmux kill-session -t $SESSION"; exit 1
fi
if [ -f "$DIR/STOP" ]; then
  echo "[launch] $DIR/ already finished (STOP present). To continue:  rm $DIR/STOP  then relaunch."; exit 1
fi
read_budget(){ grep -E "^$1:" BUDGET.md 2>/dev/null | head -1 | sed -E "s/^$1:[[:space:]]*//;s/[[:space:]].*$//"; }
HOURS="${HOURS_ARG:-$(read_budget HOURS)}"; HOURS="${HOURS:-4}"
TLIMIT=$(awk "BEGIN{printf \"%d\", $HOURS*3600+300}")
tmux new-session -d -s "$SESSION" "timeout $TLIMIT ./run.sh $DIR ${HOURS_ARG}"
echo "[launch] started '$SESSION'  (dir=$DIR, ${HOURS}h, hard-kill ${TLIMIT}s)"
echo "[launch] attach: tmux attach -t $SESSION    |    tail: tail -f $DIR/session.log"
__EOF_launch__
[ -f launch.sh ] && chmod +x launch.sh

for cfg in CLAUDE.md BUDGET.md; do
  [ -f "$cfg" ] || echo "[setup] NOTE: $cfg not found here — it is git-tracked, not seeded by this script. Restore it from git (git checkout $cfg) or your repo."
done
echo "[setup] done. Tooling in place. Add directions with:  ./new_direction.sh <dir_name> \"Title\""
