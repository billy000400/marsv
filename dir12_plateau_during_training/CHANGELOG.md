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

## 2026-07-16 — Iter 4: NEW primary experiment (SLERP interpolation movie) per rewritten PLAN; feedback 07161151 addressed; RESULTS/REPORT rewritten

The operator rewrote PLAN.md around a new primary experiment (the post's activation-interpolation
protocol animated across training checkpoints), deleted STOP, and dropped feedback
`human_feedback_07161151.txt` ("Use the first 2,000 of the 10K test images to test").

- **Feedback addressed.** The new protocol draws all 55 interpolation endpoints from the first
  2,000 test images (in fact the first 233) and computes per-checkpoint test accuracy on
  test[:2000]; stated explicitly in Methods. (The prior perturbation study already complied: its
  500-example eval set lies within test[:583] and its training script tested on test[:2000].)
  Renamed the file to `.addressed.md`.
- **New experiment (primary).** `experiments/plateau_protocol.py` (frozen 55-pair bank, vectorized
  norm-rescaled SLERP validated against the branch `slerp_path` to 9.5e-7, per-checkpoint records),
  `train_and_record.py` (seed 0: 205 checkpoints, steps 0,10,30,100,300 + every 500 to 100k; seeds
  1–2: 56 checkpoints every 2,000), `manifest_check.py` (all 317 records pass), `render_movie.py`,
  `seed_comparison.py`, `dense_zoom.py`, `s1_final_checkpoint.py` (endpoint patching reproduces
  unpatched outputs to 3.7e-4).
- **RESULTS.md / REPORT.md rewritten around the new primary result.** Headline: plateaus absent at
  init (d(α) = diagonal), form gradually with no synchronized transition (plateau fraction
  0.20 → 0.34 @100 → ~0.4 @10k → 0.54–0.61 @100k across 3 seeds; 22–29/45 clean
  plateau→boundary→plateau pairs at 100k), and keep sharpening AND relocating boundaries long
  after test accuracy saturates (largest late flip, pair 5→6 @82,000–82,500, resolved by a
  bit-exact deterministic 50-step rerun to a ~150-step relocation). Within-class controls: 8/10
  boundary-free; 2 exceptions have a misclassified endpoint.
- **Old perturbation results demoted, not deleted:** now the "perturbation control (secondary)"
  section (PLAN's optional control), keeping contrast 0.42→0.80, region count →10, and
  confidence-not-correctness numbers with one embedded figure
  (plateau_contrast_and_region_count.png). Figures training_dynamics/plateau_curves_by_stage/
  contrast_by_group/region_composition_and_lineage/s1_final_checkpoint_examples remain in plots/
  but are no longer embedded (superseded as primary evidence; history here).
- **New figures embedded in both deliverables:** plateau_evolution.gif (205 frames),
  frames_selected_steps.png, plateau_training_heatmap.png, layerwise_selected_steps.png,
  seed_comparison.png, dense_zoom.png, training_context.png. GitHub render checks pass (3/3
  js-display-math, 0 pre-lang-math, hazard grep clean, all plot refs embedded).
- New data: results/checkpoint_manifest.json, results/plateau_records/seed_{0,1,2}(+_dense)/,
  results/ckpts_movie/seed{0,1,2}/ (state dicts at every scheduled step).
- Plan complete → STOP re-written (zero unaddressed feedback files).
## 2026-07-16 — Iter 5: operator feedback 07161530 (8 points) — early-phase linear-time movie, loss insets, readability rewrite

Operator had deleted STOP and dropped `human_feedback_07161530.txt` (5 readability points + 3
substantive asks). All 8 addressed; no prior result numbers changed.

- **Pt 2 (focus on early training, linear time scale) — NEW experiment** `experiments/early_movie.py`:
  deterministic seed-0 rerun recording the frozen protocol **every 5 steps from 0 to 1,000** (201
  linearly spaced frames; bit-exact = 0.00e+00 vs the movie records at all 7 overlapping steps).
  New figures embedded in RESULTS+REPORT: `plots/plateau_evolution_early.gif` (linear-time animation)
  and `plots/plateau_early_heatmap.png` (linear-y heatmap). New records
  `results/plateau_records/seed_0_early/` (+ manifest, metrics). New numbers: train acc 1.0 at step
  145; test acc ~0.88 by step ~70–120; PF 0.19→0.27 (25)→0.34 (100)→~0.37 (200–1,000, frozen);
  curves flicker during first ~150–200 steps then settle into soft sigmoids.
- **Pt 8 (train/val loss in animation)** — `render_movie.py` inset split into two panels
  (acc/conf top, train/test loss bottom, log y); main `plateau_evolution.gif` re-rendered (205
  frames); early animation has the same insets on a linear step axis.
- **Pt 6 (how are accuracy/confidence calculated)** — new Methods subsection "Accuracy, confidence,
  and loss" with rendered equations for acc(t), MSE loss (incl. the 1/(10N) convention), and
  conf(t) = mean max raw output, with reading guidance.
- **Pt 7 (which layer)** — Methods now states "d(alpha) is computed on the logits unless a figure
  says otherwise"; every caption in both deliverables names logit-space (layerwise figure flagged
  as the only exception); RESULTS preamble states it too.
- **Pt 1 (shorter Summary)** — Summary cut to question / what-we-did / 3 findings / verdict; pair
  counts, seed counts, rerun resolutions, per-step numbers moved to Methods/Results.
- **Pt 3 (overloaded sentences)** — Checkpoints paragraph and perturbation-control paragraph
  rewritten as short single-claim sentences.
- **Pt 4 (terminology)** — new "How to read the plots" box at the end of Methods defining the shared
  axes/legend format and explicitly separating primary metric (logit d) / summary number (PF) /
  secondary-only quantities (plateau contrast, stable-region count).
- **Pt 5 (repetition, long captions)** — findings stated fully once (Summary brief, Results carry
  evidence, Conclusion one paragraph); the four flagged captions (main gif, frames, heatmap, seed
  comparison) shortened.
- Render checks: REPORT 6/6 js-display-math, 0 pre-lang-math (RESULTS 0/0), inline-math hazard grep
  clean, all plot refs embedded as images in both files.
- Renamed `human_feedback_07161530.txt` → `.addressed.md`; STOP re-written (plan complete, zero
  unaddressed feedback).
