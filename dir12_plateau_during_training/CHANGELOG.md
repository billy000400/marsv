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
## 2026-07-17 — Iter 6: operator feedback 07161650 — CE-loss rerun + pairwise AUROC for 3v5

Operator had deleted-and-readded feedback next to a stale STOP (`human_feedback_07161650.txt`,
2 questions). STOP deleted, both questions answered with new experiments; renamed the file to
`.addressed.md`; STOP re-written.

- **Q1 (train acc flat while loss falls — MSE artifact? CE version?).** Answered + NEW experiment:
  `train_and_record.py` gained a `--loss {mse,ce}` flag (identical RNG stream → same init/subset/
  batches as MSE seed 0); ran CE seed 0 on the full 205-checkpoint schedule (records in
  `results/plateau_records/seed_0_ce/`, state dicts at 16 anchor steps only to keep the repo small).
  Explanation in REPORT (new Results subsection): accuracy checks only the argmax, so ANY loss keeps
  falling after acc=1.0; CE reproduces it (train acc 1.0 from its step-300 ckpt, CE loss 1.7e-8 at
  100k). KEY NEW FINDING: under CE the logit-space d(alpha) stays near-diagonal all training (PF 0.22
  at 100k ~ 0.20 floor) but probability-space d has the sharpest plateaus of any run (PF 0.89 vs MSE
  0.55), forming earlier (~1k-10k steps); decision regions piecewise-constant under both losses →
  "the loss decides where plateaus live, not whether they exist" added as Summary finding 4. CE side
  facts: test acc 0.881 vs MSE 0.848; CE test loss rises late. New scripts `mse_vs_ce.py`,
  `ce_frames.py`; `render_movie.py` gained `--suffix` (+ softmax-confidence inset for CE). New
  figures embedded in RESULTS+REPORT: mse_vs_ce_training.png, frames_selected_steps_ce_prob.png,
  plateau_evolution_ce.gif (also on disk: *_ce variants of heatmap/frames/layerwise/context).
  New Methods content: CE loss + CE-confidence equations, probability-space d.
- **Q2 (3->5 "boundary looks like the plateau" — is 3v5 AUC worse?).** NEW experiment
  `pairwise_auc.py` (results/pairwise_auc.json): pairwise AUROC via rank estimator on logit
  differences over test[:2000] at step 100k. YES: AUROC(3,5)=0.9306 is the WORST of all 45 pairs
  (next 5v8 0.9512; median 0.987), pairwise confusion 5.2%; also worst under CE (0.9755). Curve
  reinterpretation in REPORT: the 3->5 mid-level shelf (d~0.45, 11 points) is a genuine third-class
  plateau (predicted 9) and the left "3" endpoint is misclassified as 2 — a 2/9/5 staircase, not a
  smeared boundary. Curve shape only weakly predicts difficulty (Spearman AUROC vs mid-frac -0.21,
  vs third-class-frac -0.48). New Methods subsection defines AUROC/confusion/mid-frac/third-frac;
  figure pairwise_auc.png embedded in both deliverables. (PLAN.md forbade AUC "unless required" —
  operator feedback explicitly requested it; logged as the overriding reason.)
- Deliverables: Summary gained findings 4-5; Conclusion + Limitations updated (CE = seed 0 only,
  AUROC at final ckpt); RESULTS gained two sections + 4 embedded figures + CE PF numbers under the
  seed table. Render checks: 9/9 js-display-math, 0 pre-lang-math, hazard grep clean, all plot refs
  embedded. No prior result numbers changed.
## 2026-07-17 — Iter 7: operator feedback 07161721 — ReduceLROnPlateau reruns (MSE + CE)

Operator dropped `human_feedback_07161721.txt`: "trainings are too chaotic in the end, not
converging — use a more sophisticated LR scheduler; expect training loss to plateau and LR to
shrink if training loss does not improve for ~10 steps". (A prior partial iteration had added the
`--sched plateau` flag and `sched_compare.py` but its runs were killed mid-training; this
iteration wiped the partial outputs and reran everything to completion.)

- **NEW experiment.** Seed-0 MSE and CE reruns with `ReduceLROnPlateau(factor=0.5, patience=10,
  threshold=1e-4 rel, min_lr=1e-8)` stepped after EVERY optimization step on the full-train-set
  loss (per-batch loss too noisy for patience 10); identical init/data/batch order to the
  constant-LR twins (scheduler consumes no RNG); full 205-checkpoint schedule + frozen protocol.
  Baseline per-step full-train-loss traces obtained by deterministic retrace (`sched_compare.py`,
  now cached in `base_trace.npz`; retrace final losses match stored history exactly:
  3.999e-9 MSE, 1.740e-8 CE). New data: results/plateau_records/seed_0_sched/,
  seed_0_ce_sched/, results/ckpts_movie/seed0{_sched,_ce_sched}/ (sched_trace.npz = per-step
  loss+LR), results/lr_scheduler.json.
- **Feedback confirmed + answered.** Constant-LR full-train loss spikes 3–4 orders of magnitude
  from step ~2k (MSE) / ~10k (CE) to the end — genuinely "chaotic, not converging". Scheduled:
  LR halves 16× (MSE steps 767–1,949; CE 4,350–11,298) to 1.5e-8; loss converges flat
  (2.9e-6 MSE / 2.4e-7 CE, spike-free).
- **KEY NEW FINDING (Summary finding 6).** Convergence freezes the plateau geometry: late curve
  motion M (new Methods metric, mean |Δd| per 500-step gap, ckpts ≥50k) drops 2.4e-2 → 3.2e-6
  (MSE logit) and 8.8e-3 → 1.3e-4 (CE prob); late boundary flips vanish; but PF freezes at the
  LR-collapse value — MSE 0.37 (vs 0.556 const), CE prob 0.856 (vs 0.892; survives because CE
  prob plateaus form before ~11k). The late "chaos" IS the engine of late sharpening (causal,
  not correlational). Side effect: scheduled MSE test acc 0.8795 (pinned from step ~1k) vs
  0.8475 const — the late test-acc decline is a constant-LR effect; CE flips (0.8595 vs 0.881).
- **Deliverables.** REPORT.md: Summary finding 6 + extended verdict, new Methods subsections
  (scheduler rule with equation; curve-motion M with equation), new Results subsection "Making
  training converge", Conclusion + Limitations updated (scheduler = seed 0, one setting).
  RESULTS.md: preamble + new headline paragraph + new section. New figure embedded in BOTH:
  plots/lr_scheduler_comparison.png (6 panels: per-step losses, LR cascade, test acc, PF,
  curve motion). Render checks: REPORT 11/11 js-display-math, 0 degraded, hazard grep clean,
  no bare plot refs. No prior result numbers changed.
- Renamed `human_feedback_07161721.txt` → `.addressed.md`; STOP re-written (plan complete, zero
  unaddressed feedback).
## 2026-07-17 — Iter 8: operator feedback 07161834 — figure indices + smooth-converged runs only (new primary runs)

Feedback: (1) add figure indices; (2) only show smoothly converged results — "optimize your LR
scheduler to find a better one" if needed. Both addressed; the deliverables were rebuilt around
NEW primary runs.

- **NEW experiment: LR-scheduler search** (`experiments/sched_search.py`,
  `results/lr_scheduler_search.json`, `plots/lr_scheduler_search.png` — search figure kept on
  disk, not embedded, since it contains the rejected spiky traces). Candidates on seed 0 MSE,
  identical init/data/batches: constant LR, cosine anneal (1e-3→1e-6), ReduceLROnPlateau
  {f0.5/p10 (the 07161721 run), f0.9/p50, f0.5/p100}. New smoothness/convergence metrics (spike
  ratio, tail range, defined in Methods). Constant AND cosine are non-smooth (spike max 5.8e5 /
  1.5e5); all three RLROP settings smooth. **Winner f=0.5/p=100**: tail range 1.006, final
  train loss 8.4e-9 (350× below p10's 2.9e-6, ≈ the constant run's 4.0e-9 floor), best test acc
  0.8815.
- **NEW primary runs** with the winning schedule (`train_and_record.py` gained
  `--sched cosine` + `--factor/--patience/--eta-min`): MSE seeds 0/1/2 (205/55/55 ckpts) + CE
  seed 0 (205 ckpts); all manifest-verified (520 records). Constant-LR runs demoted to
  numbers-only context. **Superseded headline numbers:** PF at 100k 0.556/0.54/0.61 (const,
  non-converged) → **0.365/0.365/0.351 (converged)**; final test acc 0.848/0.869/0.858 →
  **0.8815/0.893/0.885**; CE prob-space PF 0.892 → **0.863** (converged); AUROC(3,5) 0.9306 →
  **0.9772** (still worst of 45; CE 0.9755 → 0.9697, still worst); late curve motion M 2.4e-2 →
  5.6e-7 (MSE), 8.8e-3 → 9.1e-5 (CE prob). Early-phase zoom unchanged and now BIT-EXACT for the
  primary run (first LR cut at step 1,375 > zoom end 1,000; verified 0.00e+00 through step
  1,000). 3→5 curve on the converged model: segments 2/3/5 (was 2/9/5), mid fraction 0.84,
  endpoint "3" still misclassified as 2.
- **Corrected transcription error** in old RESULTS.md: CE prob-space PF at steps 100/1k/10k was
  stated as 0.35/0.51/0.77; the stored records and results/mse_vs_ce.json give 0.58/0.79/0.85
  (final 0.89 was correct). New table uses the recomputed values (0.58/0.79/0.85 → 0.863 for
  the converged CE run).
- **Deliverables rebuilt** (REPORT.md + RESULTS.md): every figure now carries an explicit
  "Figure N." index (1–13), identical numbering in both files. New Figure 1
  (plots/smooth_convergence.png — per-step loss/LR/test-acc of the four converged runs; new
  script `experiments/smooth_convergence.py`). Figures 2–4, 7–13 re-rendered from the converged
  runs (suffix `_pl_f0.5_p100`); Figures 5–6 (early zoom) unchanged. New Methods subsections
  (scheduler rule w/ equation; spike ratio + tail range w/ equations); new Results subsections
  "Choosing a schedule that converges smoothly" (search table) and "What converged training
  does NOT show" (constant-LR comparison, numbers only, plots deliberately omitted per operator
  preference). Verdict rewritten: structure forms in the first few hundred steps; converged
  training freezes it; loss picks the space (MSE soft logit plateaus 0.37; CE sharp prob
  plateaus 0.86); logit-PF beyond ~0.37 occurs only in non-converged training.
- **Removed from deliverables** (figures/plots remain on disk; history here): the constant-LR
  movie/frames/heatmaps (plateau_evolution.gif etc.), dense_zoom.png (the 82k boundary flip is
  a constant-LR phenomenon), lr_scheduler_comparison.png (contains spiky traces), and the
  radial-perturbation control section (plateau_contrast_and_region_count.png) — its
  contrast-keeps-rising-late result documents the non-converged constant-LR regime and no
  longer matches the smooth-run narrative the operator asked to present.
- Render checks: REPORT 12/12 js-display-math, 0 pre-lang-math, inline hazard grep clean, all
  plot refs embedded as images in both files. Renamed feedback → `.addressed.md`; STOP
  re-written (plan complete, zero unaddressed feedback).
## 2026-07-17 — Iter 9: reopened extension S7–S10 — fresh full-60k-MNIST runs, comparison vs 1k reference, verdict

Plan reopened by operator (S7–S10): train the SAME step-0 initializations from scratch on all
60,000 MNIST images and compare plateau/sub-plateau evolution against the 1k reference. All
four stages completed this iteration; STOP written.

- **NEW training script** `experiments/train_full60k.py` (S7): loads the original step-0
  untrained MSE checkpoint (asserted bit-identical to a fresh re-derivation of the seed init
  AND to the const-run step0.pt -> provably untrained), trains on all 60k images shuffled
  without replacement each epoch (300 steps/epoch, batch 200; first-epoch exact-coverage
  asserted), AdamW 1e-3/wd 0.01, MSE, fixed cosine LR 1e-3->1e-6 over 30,000 steps (PLAN's
  prescribed schedule; kept — no numerical instability: full-train loss at checkpoints decays
  1e-1 -> 2.3e-7 with transients <=17x running min, none after ~21k, tail range 1.32).
  Built-in manifest check. Smoke test (300 steps) passed all three S7 verifications before the
  long runs.
- **NEW runs** (S8+S9): seed 0 full schedule (104 ckpts: 0,10,30,100 then every 300 to 30k),
  seeds 1–2 fallback (25 ckpts); pair bank = original 55 pairs + NEW frozen 50-path 3->5 bank
  (rank-i test 3 with rank-i test 5, i<50, unfiltered, selected before viewing results). All
  154 records manifest-verified. Also NEW `experiments/eval_3v5_ref1k.py`: 1k converged model
  re-evaluated on the identical 105-pair bank at its 16 anchors (paired baseline).
- **NEW figures** (`experiments/render_full60k.py`): Figure 14 full_mnist_training_context.png,
  Figure 15 full_vs_1k_evolution.gif (synchronized side-by-side, 25 step-aligned frames),
  Figure 16 full_vs_1k_frames.png, Figure 17 full_mnist_3v5_training.gif (104 frames),
  Figure 18 full_mnist_3v5_summary.png. All embedded in BOTH RESULTS.md and REPORT.md.
- **HEADLINE RESULT (new):** the converged-PF ceiling was a small-data effect. Full-60k runs
  converge smoothly AND keep sharpening: PF 0.43 (step 300) -> 0.61 (1.5k) -> peak 0.73
  (~8.4k) -> 0.64/0.69/0.65 final across seeds (1k converged reference: frozen at 0.37); test
  acc 0.979/0.976/0.977 vs 0.881; late curve motion 3e-4; 0/90 cross-pair endpoints
  misclassified (1k: 3/90). REPORT finding 5 REVISED from "sharpening beyond 0.37 requires
  non-converged training" to "…on the 1k subset; with 60k data converged training reaches
  0.64–0.73". Finding 2 + constant-LR subsection + verdict + conclusion re-scoped accordingly.
- **3->5 verdict (new finding 6):** endpoint correction, not merging. 49/50 paths
  endpoint-correct at 30k under 60k-training vs 36/50 under 1k (both 0/50 at step 0 — the
  preregistered "correct at step 0" subset is empty, stated in REPORT). Original 3->5 path:
  2|3|5 -> 3|5 in ALL three 60k seeds (endpoint correction + segment disappearance per the
  plan's definitions). Detours 4% vs 2%; seg mean 2.04 vs 1.90; repeated-class RLEs (only
  merge-capable patterns): 2/50 transient (60k), 1/50 (1k), 0 at final checkpoints. No global
  topology claims.
- **Methods added:** full-60k subsection (init verification, shuffle, cosine equation,
  checkpoint/alignment scheme, frozen 3v5 bank) + segment metrics (RLE segment count, detour
  indicator, endpoint correctness — equations + merge/endpoint-correction/disappearance
  distinctions). RESULTS.md preamble/headline/table rebuilt around both regimes.
- Render checks: 15/15 js-display-math, 0 pre-lang-math, inline hazard grep clean, all plot
  refs embedded as images in both files. PLAN S7–S10 ticked; STOP written (zero unaddressed
  feedback).

## 2026-07-17 — Iter 10: operator feedback human_feedback_1 — smooth-scheduler 60k reruns; report refocused on 60k

Feedback (`human_feedback_1.txt`): (1) the 60k training loss is noisy — use a more
sophisticated LR scheduler to make it smooth; (2) rerun every 1k-only experiment on 60k data
("more data = more reliable"); (3) make REPORT/RESULTS focus on the 60k results, keeping only a
small section on the effect of training-set size. All three addressed; deliverables rebuilt
around the 60k runs. Renamed `human_feedback_1.txt` → `.addressed.md`.

- **NEW scheduler search on the 60k run** (`experiments/sched_search_60k.py`,
  `results/lr_scheduler_search_60k.json`, `plots/lr_scheduler_search_60k.png`): cosine (previous)
  vs `ReduceLROnPlateau` f=0.5 p=100 / p=300, seed 0, identical init/data/batch order, scored on
  the same smoothness metrics as the 1k search. The prior cosine schedule is NOT smooth on 60k
  (spike max 101×, 12.5% of steps >2× the running min, tail range 3.23). **Winner f=0.5 p=100**:
  spike max 1.56, 0% spikes, tail range 1.07, PF 0.674, test acc 0.9775 — the same schedule the
  1k phase chose, so the size comparison holds the scheduler fixed.
- **NEW primary 60k runs** with the winning schedule (`train_full60k.py --sched plateau`, already
  supported): MSE seeds 0/1/2 (104/25/25 ckpts) + CE seed 0 (104 ckpts); all 258 records +
  201 early-zoom records manifest-verified. Every 1k-only experiment re-run on 60k via
  `render_all_60k.sh` (parameterized scripts): smooth-convergence, main movie + frames + heatmap
  + layerwise + context, early-phase linear-time zoom, seed comparison, MSE-vs-CE, CE frames,
  pairwise AUROC, and the full-vs-1k comparison. New `experiments/collect_60k_numbers.py`
  (`results/numbers_60k_p100.json`) aggregates every reported number. Fixed `mse_vs_ce.py` to
  guard the train-acc=1.0 axvline (the 60k MSE run never hits exactly 1.0 — 0.9999).
- **REPORT.md + RESULTS.md rewritten to focus on 60k.** The 60k runs are now the primary result
  (Figures 1–14 all 60k); the 1,000-image comparison is one dedicated "effect of training-set
  size" section (Figures 15–18). Superseded framing: the report previously led with the 1k
  reference (PF frozen 0.37) and treated 60k as an extension; now the small-data ceiling is
  presented as the *size effect* and the 60k sharpening (PF → 0.674) is the headline.
- **KEY CORRECTION (finding 4).** On the 1k model 3v5 was the single hardest pair (worst AUROC
  0.977, rank 1/45). On the 60k model 3v5 is near-perfectly separated — **AUROC 0.9993, rank
  4/45** (worst pair now 4v9 at 0.9975; confusion 6% → 0.8%). Finding 4 rewritten from "3v5 is
  genuinely the hardest pair" to "full data resolves the 3v5 difficulty — it was a small-data
  effect." Conclusion + AUROC section updated accordingly.
- **Headline numbers (60k, current-best).** PF logit: 0.19 (0) → 0.35 (100) → 0.43 (300) → 0.62
  (1,500) → 0.674 (30k), peak 0.674 @ ~27k, across seeds 0.674/0.663/0.668; late curve motion
  7.6e-4. Test acc 0.9775/0.9795/0.9785. CE: logit PF ~0.25 (floor), prob PF 0.18→0.55(100)→
  0.90(30k). 0/90 cross-pair endpoints misclassified; 9/10 within-class controls single-class.
  3→5 bank: 49/50 endpoints correct at 30k (vs 36/50 for 1k); orig 3→5 = 3|5 in all 3 seeds; seg
  mean 1.98; 0% detours; no repeated-class RLE at any 60k checkpoint.
- Render checks: REPORT 14/14 js-display-math, 0 pre-lang-math; RESULTS 0 pre-lang-math; inline
  hazard grep clean; all 19 figures embedded as `![…](…)` images in both files and present on
  disk. Feedback renamed to `.addressed.md`; STOP re-written (plan complete, zero unaddressed
  feedback).
