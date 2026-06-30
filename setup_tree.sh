#!/usr/bin/env bash
# setup_tree.sh — SAFE bootstrap of SHARED INFRA only (run.sh, new_direction.sh, BUDGET.md, CLAUDE.md).
# - Creates a file ONLY if it does not already exist (never clobbers your work).
# - Does NOT touch any direction folder (dirN/PLAN.md etc.) — use ./new_direction.sh for those.
# - FORCE=1 overwrites infra, but backs up the old file to <file>.bak.<epoch> first.
# Usage:  bash setup_tree.sh [PROJECT_ROOT]            (default: current dir)
#         FORCE=1 bash setup_tree.sh [PROJECT_ROOT]    (refresh infra, with backups)
set -euo pipefail
ROOT="${1:-.}"
[ -d "$ROOT" ] || { echo "[setup] project root '$ROOT' not found"; exit 1; }
cd "$ROOT"
echo "[setup] target: $(pwd)   FORCE=${FORCE:-0}"

write_if_absent() {            # usage: write_if_absent <path>   (content on stdin)
  local f="$1"
  if [ -e "$f" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[setup] skip (exists): $f"; cat >/dev/null; return 0
  fi
  if [ -e "$f" ]; then cp -a "$f" "$f.bak.$(date +%s)"; echo "[setup] backup+overwrite: $f";
  else echo "[setup] create: $f"; fi
  cat > "$f"
}

write_if_absent run.sh <<'__EOF_run__'
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
__EOF_run__
[ -f run.sh ] && chmod +x run.sh

write_if_absent new_direction.sh <<'__EOF_new__'
#!/usr/bin/env bash
# new_direction.sh — scaffold ONE new research direction for the run.sh loop.
# Run from the project root (where run.sh and BUDGET.md live).
#
# Usage:   ./new_direction.sh <dir_name> ["Title / one-line objective"]
# Example: ./new_direction.sh dir7_proxy "A differentiable plateau proxy"
#
# Safe by design: refuses to overwrite an existing direction (no clobbering results).
set -euo pipefail

DIR="${1:?usage: ./new_direction.sh <dir_name> [\"Title\"]}"
TITLE="${2:-TODO — describe this direction}"

# --- validation ---------------------------------------------------------------
case "$DIR" in
  */*|*" "*) echo "[new] '$DIR' must be a plain folder name (no spaces or slashes)"; exit 1 ;;
esac
if [ -e "$DIR/PLAN.md" ]; then
  echo "[new] REFUSING: $DIR/PLAN.md already exists — would overwrite existing work."
  echo "      Pick a different name, or edit that direction's files directly."
  exit 1
fi
[ -f run.sh ]    || echo "[new] WARNING: no run.sh in $(pwd) — run this from the project root."
[ -f BUDGET.md ] || echo "[new] WARNING: no BUDGET.md in $(pwd) — the new direction expects one."

mkdir -p "$DIR/experiments" "$DIR/results" "$DIR/plots"
: > "$DIR/experiments/.gitkeep"
: > "$DIR/plots/.gitkeep"

# --- PLAN.md (dynamic header via printf, static body via quoted heredoc) -------
printf '# PLAN — Direction: %s\n\n' "$TITLE"                                  >  "$DIR/PLAN.md"
printf '> Working folder: `%s`. The agent REWRITES "Current status" and "Next step" and ticks\n' "$DIR" >> "$DIR/PLAN.md"
printf '> the stage boxes every iteration. Disk (this file + JOURNAL.md + RESULTS.md + ../BUDGET.md)\n' >> "$DIR/PLAN.md"
printf '> is the only memory.\n\n'                                            >> "$DIR/PLAN.md"
cat >> "$DIR/PLAN.md" <<'EOF'
## Success criterion (definition of "done")
TODO — the concrete artifact(s) that mean this direction is finished, e.g. "Produce <X> and <Y>
in RESULTS.md, plus REPORT.md with a clear verdict and the supporting figures in plots/." A
null/negative result is still COMPLETE if the question is answered. When done, the loop writes an
empty `STOP` file.

## Fallback (if time runs short)
TODO — the minimum acceptable deliverable. The wrapper reserves the final 20 min to finalize
whatever exists into RESULTS.md + REPORT.md (with whatever plots/ figures exist), then STOP.

## Setup (fixed)
- TODO — model / data / hook points. Default: GPT-2 small via HuggingFace `transformers` (already
  installed) + forward hooks; STREAM data, do not bulk-download.
- **Shared hardware + time limits live in `../BUDGET.md` — read it every iteration.** You share one
  GPU / RAM / CPU with another agent, so stay within your half: cap VRAM with
  `torch.cuda.set_per_process_memory_fraction`, memmap caches, keep batches small, halve on OOM.
- **Visualize what you report.** Save every figure as a PNG under `plots/` (matplotlib is headless
  via `MPLBACKEND=Agg`; use `plt.savefig` then `plt.close`, never `plt.show`) and reference each
  filename in RESULTS.md/REPORT.md.
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, or flax** — they
  downgrade and break the cluster's CUDA build. Use the existing env; add pure-python deps with
  `--no-deps`.

## Stages (checklist — update marks each iteration)
- [ ] S1 — TODO
- [ ] S2 — TODO
- [ ] S3 — TODO  (include a "produce + save figures to plots/" step for each reported metric)

## Out of scope (do NOT)
- TODO — anything explicitly out of bounds for this direction.
- Don't drift into other directions.

## On-track check (required every iteration)
End each JOURNAL.md entry with one line: `On track? <yes/no> — <stage, % done, blocker if any>`.

## Current status
(none yet — fresh start)

## Next step
TODO — the first concrete action.
EOF

# --- JOURNAL.md ---------------------------------------------------------------
printf '# JOURNAL — Direction: %s\n\n' "$TITLE"                               >  "$DIR/JOURNAL.md"
cat >> "$DIR/JOURNAL.md" <<'EOF'
Append-only. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---
EOF

# --- RESULTS.md ---------------------------------------------------------------
printf '# RESULTS — Direction: %s\n\n' "$TITLE"                               >  "$DIR/RESULTS.md"
cat >> "$DIR/RESULTS.md" <<'EOF'
## Metrics
_(add result tables here as they are produced)_

## Figures
_(list each plots/*.png with a one-line caption as you create them)_

## Headline
_(filled at finalize)_
EOF

# --- done ---------------------------------------------------------------------
echo "[new] created $DIR/ with PLAN.md, JOURNAL.md, RESULTS.md, experiments/, results/, plots/"
echo "[new] next:"
echo "      1) edit $DIR/PLAN.md — fill every TODO (success criterion, stages, next step)"
if [ -f BUDGET.md ]; then
  H=$(grep -E '^HOURS:' BUDGET.md 2>/dev/null | head -1 | sed -E 's/^HOURS:[[:space:]]*//;s/[[:space:]].*$//'); H="${H:-4}"
  T=$(awk "BEGIN{printf \"%d\", $H*3600+300}")
  echo "      2) launch:  tmux new-session -d -s $DIR 'timeout $T ./run.sh $DIR'"
else
  echo "      2) launch:  tmux new-session -d -s $DIR 'timeout <Hh+grace_sec> ./run.sh $DIR'"
fi
__EOF_new__
[ -f new_direction.sh ] && chmod +x new_direction.sh

write_if_absent BUDGET.md <<'__EOF_budget__'
# BUDGET — shared resource & time limits (single source of truth)

> `run.sh` reads the INPUTS below and **derives** each agent's share at launch. The GPU is NOT
> listed — `run.sh` auto-detects the current card with `nvidia-smi` every launch (the pod may get
> a different GPU each time), and `set_per_process_memory_fraction` is a fraction of whatever card
> is present, so it adapts automatically. **To retune, edit these values — nothing else changes.**

## Inputs (keep the `KEY: value` format — run.sh greps these)
HOURS: 4
N_AGENTS: 2                 # how many loops you launch CONCURRENTLY — set to match reality
CPU_CORES_TOTAL: 4          # static
RAM_TOTAL_GB: 16            # static
VRAM_HEADROOM_FRACTION: 0.1 # leave this fraction of the card free (shared headroom)

## Derived by run.sh from the above, and told to each agent every iteration:
##   VRAM fraction / agent = (1 - VRAM_HEADROOM_FRACTION) / N_AGENTS
##   CPU threads   / agent = CPU_CORES_TOTAL / N_AGENTS   (min 1)
##   RAM budget    / agent = RAM_TOTAL_GB / N_AGENTS
## (e.g. N_AGENTS=2, 4 CPU, 16 GB  ->  vram_frac 0.45, 2 threads, 8 GB RAM each)

## Rules for the agent — you are 1 of N_AGENTS sharing this box; assume the others are busy
- **GPU.** Call `torch.cuda.set_per_process_memory_fraction(<the fraction run.sh gives you>)` at
  startup so you can't starve the others. Small batches; move tensors off-GPU when done;
  `torch.cuda.empty_cache()` between stages.
- **RAM.** Stay under your per-agent GB. Don't hold large activation matrices in RAM —
  `np.memmap` / sharded `.npy` and stream.
- **CPU.** `torch.set_num_threads(<your thread budget>)`; DataLoader `num_workers <= that`.
- **On CUDA OOM / swapping:** HALVE batch/sample size and retry — never re-run the same size.
- **Time.** You have `HOURS` hours total; the wrapper enforces it and reports remaining minutes.
  Reserve the last 20 min to finalize.
__EOF_budget__

write_if_absent CLAUDE.md <<'__EOF_rules__'
# CLAUDE.md — operator rules for every agent in this project

> Read and follow these rules EVERY iteration, in addition to BUDGET.md and the direction's
> PLAN.md. These are hard rules: when a rule conflicts with convenience or speed, the rule wins.
> (This file is shared across all directions; it lives at the project root.)

## Rules
1. **Read before write.** Never overwrite or edit a file without first reading its current
   contents. Prefer a targeted edit over a full rewrite; if you must rewrite, preserve everything
   you are not deliberately changing. Never blank-truncate or clobber a file you have not read —
   this includes RESULTS.md, JOURNAL.md, PLAN.md, and any cached data under experiments/.

<!-- Add more rules below, one numbered item each. Examples you might enable:
2. **Never delete results or cached data** without an explicit instruction in PLAN.md.
3. **Append, don't replace** in JOURNAL.md.
4. **Commit/checkpoint** after each completed stage.
-->
__EOF_rules__

echo "[setup] done. Infra in place. Add directions with:  ./new_direction.sh <dir_name> \"Title\""
