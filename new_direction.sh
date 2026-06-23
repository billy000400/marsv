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

mkdir -p "$DIR/experiments" "$DIR/results"
: > "$DIR/experiments/.gitkeep"

# --- PLAN.md (dynamic header via printf, static body via quoted heredoc) -------
printf '# PLAN — Direction: %s\n\n' "$TITLE"                                  >  "$DIR/PLAN.md"
printf '> Working folder: `%s`. The agent REWRITES "Current status" and "Next step" and ticks\n' "$DIR" >> "$DIR/PLAN.md"
printf '> the stage boxes every iteration. Disk (this file + JOURNAL.md + RESULTS.md + ../BUDGET.md)\n' >> "$DIR/PLAN.md"
printf '> is the only memory.\n\n'                                            >> "$DIR/PLAN.md"
cat >> "$DIR/PLAN.md" <<'EOF'
## Success criterion (definition of "done")
TODO — the concrete artifact(s) that mean this direction is finished, e.g. "Produce <X> and <Y>
in RESULTS.md, plus REPORT.md with a clear verdict." A null/negative result is still COMPLETE if
the question is answered. When done, the loop writes an empty `STOP` file.

## Fallback (if time runs short)
TODO — the minimum acceptable deliverable. The wrapper reserves the final 20 min to finalize
whatever exists into RESULTS.md + REPORT.md, then STOP.

## Setup (fixed)
- TODO — model / data / hook points. Default: GPT-2 small via HuggingFace `transformers` (already
  installed) + forward hooks; STREAM data, do not bulk-download.
- **Shared hardware + time limits live in `../BUDGET.md` — read it every iteration.** You share one
  GPU / RAM / CPU with another agent, so stay within your half: cap VRAM with
  `torch.cuda.set_per_process_memory_fraction`, memmap caches, keep batches small, halve on OOM.
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, or flax** — they
  downgrade and break the cluster's CUDA build. Use the existing env; add pure-python deps with
  `--no-deps`.

## Stages (checklist — update marks each iteration)
- [ ] S1 — TODO
- [ ] S2 — TODO
- [ ] S3 — TODO

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

## Headline
_(filled at finalize)_
EOF

# --- done ---------------------------------------------------------------------
echo "[new] created $DIR/ with PLAN.md, JOURNAL.md, RESULTS.md, experiments/, results/"
echo "[new] next:"
echo "      1) edit $DIR/PLAN.md — fill every TODO (success criterion, stages, next step)"
if [ -f BUDGET.md ]; then
  H=$(grep -E '^HOURS:' BUDGET.md 2>/dev/null | head -1 | sed -E 's/^HOURS:[[:space:]]*//;s/[[:space:]].*$//'); H="${H:-4}"
  T=$(awk "BEGIN{printf \"%d\", $H*3600+300}")
  echo "      2) launch:  tmux new-session -d -s $DIR 'timeout $T ./run.sh $DIR'"
else
  echo "      2) launch:  tmux new-session -d -s $DIR 'timeout <Hh+grace_sec> ./run.sh $DIR'"
fi