# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---

## 2026-07-15 — Initial result: plateau emergence over training (3 seeds)

First populated RESULTS.md and REPORT.md (both were TODO templates → full current-best deliverables).

- **New experiment.** Trained the image-models d4/w200 ReLU MNIST MLP (n=1000, AdamW, MSE-on-one-hot,
  100k steps) for seeds 0/1/2, snapshotting 13 log-spaced checkpoints, then ran the frozen plateau
  protocol (perturb `h1`, measure `L3` displacement; matched-random control; plateau contrast on
  ρ∈[0,0.2]; agglomerative-clustering validated stable-region count).
- **Headline numbers (mean over 3 seeds):** plateau contrast rises 0.42 (step 100) → 0.80 (step 100k),
  *after* test accuracy peaks (0.90 by step ~300, decaying to ~0.87). Validated stable-region count
  converges to **10** by step ~300 in every seed. Confident-wrong contrast 0.73 vs confident-correct
  0.85 vs uncertain 0.49 at 100k → confidence, not correctness, drives the plateau.
- **Resolved candidate transient:** the step-10k contrast dip (0.30) is present only in seed 0; seeds
  1–2 rise monotonically (0.56, 0.45). Classified as seed noise, not a split/merge → no escalation.
- **Verdict recorded:** expected monotonic emergence, replicated across 3 seeds.
- **Figures added (all embedded in RESULTS.md + REPORT.md):** training_dynamics.png,
  plateau_curves_by_stage.png, plateau_contrast_and_region_count.png (3-seed band), contrast_by_group.png.
- Confidence metric chosen as max raw output (softmax saturates near 0.23 under MSE-to-one-hot);
  documented in Methods.

## 2026-07-15 — Iter 2: region composition + membership-overlap lineage (fig #4); finalize + STOP

- **New analysis (`experiments/lineage.py`, `experiments/make_lineage_plot.py`).** For seed 0, recomputed
  per-example L3 cluster labels at all 13 checkpoints using the frozen protocol (avg-linkage
  agglomerative, silhouette-selected k, cosine), then aligned adjacent checkpoints by membership overlap
  (same 500 fixed eval examples). Added the 4th required figure `plots/region_composition_and_lineage.png`.
- **New finding (lineage).** Validated regions are born one predicted-digit at a time (1→2→3→9→10 by step
  ~300) and then persist; **no predicted digit ever hosts ≥2 validated regions at any checkpoint** (max=1).
  Membership-overlap matrices for the birth transition (100→300) and a late transition (75k→100k) are
  clean near-permutations: **0 splits, 0 merges** among validated regions. Raw silhouette k oscillates
  10–12 late in training as a transient sub-threshold split of the uncertain/mixed group, but that extra
  cluster is never validated and never persists across two adjacent checkpoints → escalation criterion not
  met. Confirms the monotonic-emergence verdict.
- **Deliverables.** Embedded the new figure in RESULTS.md and REPORT.md; added a Methods paragraph defining
  the membership-overlap matrix and split/merge/escalation rule (new ```math block, GitHub-render-verified:
  3/3 js-display-math, 0 pre-lang-math); updated the region-count finding and limitations (lineage now
  done, seed-0 scope noted). No result numbers superseded. Marked S4/S5 complete; wrote empty `STOP`
  (no unaddressed feedback files present).

## 2026-07-16 — Iter 3: address operator feedback (human_feedback_07161227) — Methods rewrite + inline-math fixes

Operator feedback (4 points) addressed in REPORT.md (+ small RESULTS.md fixes); no experiments re-run,
no result numbers changed.

- **ε value added.** The response equation now states $\varepsilon = 10^{-8}$ explicitly (matching
  `EPS = 1e-8` in `experiments/analyze_sweep.py`), with a sentence explaining it is a division-by-zero
  guard that is orders of magnitude below typical L3 norms; also noted the same guard in the
  plateau-contrast denominator.
- **"Positive entries" defined + motivated.** Matched-random-control paragraph now explains that h1 is
  post-ReLU (entries ≥ 0), that its positive entries are exactly the neurons the input activates
  (sparsity), and why matching norm + positive-entry count matters: an unmatched control would differ in
  scale/sparsity regime, making the contrast an artifact rather than evidence of learned structure.
- **Inline-symbol rendering fixed.** All inline symbols previously in backtick code spans (`h1`, `L3`,
  `G_t`, `R_t`, `ρ`, `k ∈ {2..15}`, the raw un-rendered `\arg\max_i M_{ij}`, etc.) converted to proper
  inline `$…$` math across REPORT.md and RESULTS.md, using 8b-safe forms (`\lbrace/\rbrace`, no
  backslash-punctuation). Render verified via GitHub markdown API: REPORT.md 3/3 js-display-math,
  0 pre-lang-math, 34 inline-math spans all intact (spot-checked `\lbrace`, `\arg\max`, `\varepsilon`);
  hazard grep clean; all plots still embedded as `![…](…)`.
- **Metrics section rewritten as a motivated narrative** (was a bare list of definitions): new opening
  paragraph maps the four research questions to the four measurements; each metric now states the
  question it answers, why the obvious alternative fails (absolute distances not comparable across
  checkpoints; a bare cluster is not a stable region; counts/IDs cannot detect member swaps), and which
  Results figure consumes it.
- Renamed `human_feedback_07161227.txt` → `human_feedback_07161227.txt.addressed.md`. Re-wrote `STOP`
  (plan complete, zero unaddressed feedback).
