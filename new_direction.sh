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
[ -f WRITING.md ] || echo "[new] WARNING: no WRITING.md in $(pwd) — report writing rules are missing."

mkdir -p "$DIR/experiments" "$DIR/results" "$DIR/plots"
: > "$DIR/experiments/.gitkeep"; : > "$DIR/plots/.gitkeep"

# ---- PLAN.md ----
printf '# PLAN — Direction: %s\n\n' "$TITLE"                                  >  "$DIR/PLAN.md"
printf '> Working folder: `%s`. Agent REWRITES "Current status"/"Next step" + ticks stages each\n' "$DIR" >> "$DIR/PLAN.md"
printf '> iteration. run.sh supplies compact task/PLAN/latest-journal context; full files remain available on demand.\n\n' >> "$DIR/PLAN.md"
cat >> "$DIR/PLAN.md" <<'EOF'
## Success criterion (definition of "done")
TODO — concrete artifact(s) that mean done, e.g. "RESULTS.md has <X>/<Y> (current-best) and REPORT.md
gives a clear verdict with Methods + figures." Null/negative results are COMPLETE if the question is
answered. When done, the loop writes an empty `STOP` file.

## Fallback (if time runs short)
TODO — minimum acceptable deliverable. The wrapper reserves the last 20 min to finalize + STOP.

## Setup (fixed)
- TODO — model / data / hook points. Default: GPT-2 small via HuggingFace `transformers` + forward hooks; STREAM data.
- **Shared limits in `../BUDGET.md`; operator rules in `../CLAUDE.md`; report rules in `../WRITING.md`.**
- **Deliverable hygiene:** RESULTS.md/REPORT*.md = current-best only; CHANGELOG.md = history.
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, flax** — they break the CUDA build.

## Stages (checklist)
- [ ] S1 — TODO
- [ ] S2 — TODO
- [ ] S3 — TODO  (each reported metric: save a plot + define it in the affected REPORT*.md Methods)

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
Append-only ledger of changes to RESULTS.md / REPORT*.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and every REPORT*.md stay current-best with no history.

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
