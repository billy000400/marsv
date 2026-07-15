# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---

## 2026-07-15 — S2 weekday setup reproduced (initial deliverable)
- Direction initialized (was empty TODO). Built `experiments/common.py` + `experiments/s2_collect.py`.
- Reproduced paper-consistent weekday setup on Llama 3.1 8B base, layer 28, 49 prompts:
  task accuracy **0.939** (46/49), mean weekday mass **0.743**, mean other mass **0.257**.
- Fit PCA over the 49 activations, seven ground-truth centroids, periodic cubic spline (Appendix A.3).
- RESULTS.md: (empty TODO) -> current-best S2 metrics table + PCA figure.
- Deviations logged (forced by constraints/math), see JOURNAL:
  (1) **Model placement**: 8B bf16 = 16 GB > 7.2 GB VRAM share, so weights are split GPU(6 GiB)/CPU
      via accelerate device_map. Precision unchanged (bf16); only placement differs.
  (2) **PCA rank**: "PCA-64" requested but only 49 activations exist -> subspace rank <= 48; we retain
      all 48 non-degenerate components. First-32 optimization subspace and PCA-32 recovery metric
      are unaffected.

## 2026-07-15 — S3+S4 combined-objective lambda sweep (Tuesday->Wednesday); verdict added
- Built `pathlib_opt.py` (GPU-resident layer 28-31 tail runner; natural-cubic-spline path
  parameterization; activation/behavior kinetic energies) and `s4_sweep.py` / `s5_analyze.py`.
- Tail correctness validated: injecting a prompt's own layer-28 activation reproduces the full-model
  8-bin behavior (argmax always matches; max L1 = 0.05).
- Ran the coarse grid lambda in {0,0.1,1,10,100} + output-only, Tue->Wed, 16 base prompts, 3 seeds
  (linear + 2 perturbed).
- RESULTS.md / REPORT.md: filled from S2-only status -> full pilot verdict (**NEGATIVE**):
  - Recovery (mean nearest-spline dist, PCA-32): linear/lambda=0 = 0.961 (best optimized), worsens
    monotonically to output-only 1.023 (linear init) / 1.40-1.43 (perturbed inits); target spline = 0.004.
  - Energy trade-off: centroid spline dominated in BOTH energies (E_act 104.9 vs chord 88.8;
    E_out 1.118 vs 1.026) -> objective structurally cannot prefer it.
  - Sanity checks pass: lambda=0 recovers chord (endpoint err 0, E_act=88.8 global min);
    output-only reduces E_out (0.930 vs 1.026); endpoints fixed for all paths.
  - Initialization sensitivity: high-lambda/output-only paths init-dependent (E_act 306-313 vs 93),
    E_out flat ~0.94 -> behavior objective underdetermines the activation path.
- New figures: plots/s4_recovery_vs_lambda.png, s4_energy_tradeoff.png, s4_dt_curves.png,
  s4_pca_geometry.png.
- Deviations: (a) primary recovery = transparent PCA-32 nearest-spline distance; paper Appendix-A.9
  SVD recovery not reproducible from available materials. (b) d(t) defined from first principles
  (referenced slerp_relative_distance.py absent from repo); used only as diagnostic. (c) E_out
  normalized per-init, so cross-seed normalized losses not comparable (raw energies + recovery are).

## 2026-07-15 — S6 generalization to all 7 adjacent pairs; direction COMPLETE
- Added `experiments/s6_allpairs.py` (single model load, pair-independent bases reused across pairs).
- Ran coarse λ grid + output-only (linear init) for all 7 adjacent weekday pairs.
- Result generalizes: for EVERY pair the best optimized path over all λ equals the linear chord
  (best_opt == rec_linear, no λ improves recovery); mean chord recovery 0.988 vs spline 0.004
  (~235× gap); centroid spline Pareto-dominated in both energies for 7/7 pairs.
- RESULTS.md / REPORT.md: verdict upgraded from "Tue→Wed pilot" to "holds for all 7 adjacent pairs".
- New figure: plots/s6_allpairs_recovery.png. Wrote STOP (question answered decisively; criteria met).

## 2026-07-15 — Finalization: STOP re-created; deliverables re-verified
- No deliverable numbers changed (RESULTS.md/REPORT.md already at current-best; direction complete).
- Re-verified REPORT.md display math against GitHub markdown API: 6/6 `math` fences render as
  `js-display-math`, 0 degraded to `<pre lang="math">`; 0 inline backslash-punctuation hazards.
- The `STOP` file recorded as written in the prior JOURNAL entry was absent from disk (lost before
  commit) — re-created it. Direction remains COMPLETE (S1–S6; NEGATIVE verdict, generalizes 7/7 pairs).

## 2026-07-15 — Added per-pair energy trade-off figure; STOP re-created
- No result numbers changed (direction already at current-best, decisive NEGATIVE, 7/7 pairs).
- Added `experiments/s6_energy_plot.py` (pure post-processing of `results/allpairs_summary.json`,
  no model re-run) → new figure `plots/s6_allpairs_energy_tradeoff.png`: E_act vs E_out for all 7
  adjacent pairs, showing every fitted centroid spline (★) alone in the dominated top-right corner
  while the optimized-path families trace the lower-left frontier. Script re-confirms spline
  Pareto-dominated in both energies 7/7. Referenced from RESULTS.md and REPORT.md (fills the gap
  where the energy trade-off figure previously only covered Tuesday→Wednesday).
- Re-verified REPORT.md display math (6/6 render as js-display-math, 0 degraded, 0 inline hazards).
- Re-created `STOP` (again absent from disk on resume). Direction remains COMPLETE (S1–S6).

## 2026-07-15 — Operator feedback (human_feedback_07140930) addressed; +2 metrics, all figures embedded
- **Addressed `human_feedback_07140930.md`** (renamed to `.addressed.md`). Four asks, all answered in
  RESULTS.md + REPORT.md:
  1. *What are the stars in `s2_pca_weekday_manifold.png`?* → the **seven ground-truth weekday
     centroids** (mean PCA position of each weekday's 7 prompts). Regenerated the figure
     (`experiments/s7_feedback_plots.py`, from saved `weekday_setup.npz`, no model) with an explicit
     legend + weekday labels distinguishing dots (49 prompts) from ★ centroids.
  2. *How many sequences per weekday?* → **7** (49 = 7 weekdays × 7 increments); stated in both files.
  3. *Cumulative-variance vs #PCs plot* → NEW `plots/s2_pca_cumvar.png`. Answer: 2–3 PCs are **not**
     representative — PC1–2 = 31.4%, PC1–3 = 43.6%, 18 PCs for 90%, 32 for 98.1%. Motivates using
     PCA-32 for all conclusions rather than the 2-D scatter.
  4. *Plateau metric `|h−hA|/(|h−hA|+|h−hB|)`* → NEW `experiments/s7_plateau.py` (model tail on saved
     Tue→Wed sweep paths) + `plots/s7_plateau_metric.png` + `results/s7_plateau.json`. Result: no
     sharp plateau; p(t) rises smoothly 0→1 on every path, and the **centroid spline's curve is
     essentially identical to the linear chord's** (both ≈0.59 at midpoint) — downstream behavior does
     not single out the on-manifold path. Adds a fourth independent strike to the NEGATIVE verdict.
- **Rule-12 fix:** RESULTS.md and REPORT.md previously referenced figures only as bare paths (0
  embedded images). Now **all 9 figures are embedded as rendered `![caption](plots/…)` images** in
  BOTH files. Verified: 9/9 embeds each, 0 bare unembedded refs.
- **Infra fix:** HF cache moved `/network/hf_cache` → `/workspace/hf_cache`; `common.py` now picks
  whichever exists (MODEL_ID + HF_HOME), so the model tail loads again. No numeric results changed.
- Re-verified display math: REPORT 7/7 render as js-display-math (0 degraded), RESULTS 1/1; 0 inline
  backslash-punctuation hazards. No prior result numbers superseded — this iteration only adds
  feedback-requested metrics/figures and embeds existing figures.
