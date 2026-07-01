# CHANGELOG — Direction #3 (Manifold)

Append-only. History of changes to the deliverables (RESULTS.md / REPORT.md).
Current-best numbers live in those files; this file records how they got there.

---

## 2026-06-30 — Figures added; report/results history migrated here (consolidation pass)
- **plots/ was empty; matplotlib was not installed.** Installed `matplotlib==3.11.0` with
  numpy/torch pinned (numpy 2.3.3 / torch 2.9.0+cu130 unchanged — verified). Added
  `experiments/make_plots.py`, which renders every reported quantitative result from the saved
  `results/*.json` (no recompute, headless Agg, savefig+close). New figures:
  - `plots/id_per_layer.png` — nonlinear TwoNN/MLE (centered & standardized, n=50k) vs linear
    PCA d95 vs d_model per layer.
  - `plots/ae_fvu_sweep.png` — AE held-out FVU vs k: CPU vs GPU-raw (seed-mean ± std) vs
    param-matched vs standardized (knee-gone) curves.
  - `plots/ae_marginal_gain.png` — ΔFVU per doubling (raw) showing no plateau.
  - `plots/id_validation.png` — estimator accuracy on synthetic Gaussians (estimated vs true d).
  - `plots/id_by_position.png` — layer-6 ID per token-position bucket.
  - `plots/id_diagnostics.png` — bootstrap 95% CIs + naive-vs-robust (self-masking) bars.
  RESULTS.md and REPORT.md now reference these figures; **no numbers changed.**
- **Consolidation per CLAUDE.md rules 6–7:** moved the "What changed after review" /
  "Status note" version-history blocks OUT of REPORT.md (a curated deliverable) into the dated
  entries below. REPORT.md/RESULTS.md now read as clean current-best documents; all change
  history lives here.

## 2026-06-23 — Codex review `CODEX_REVIEW_20260623T001526Z` addressed (Iter 7)
- **ID headline band widened 11–13 → 11–15.** New artifact `results/id_diagnostics.json`
  (GPU, `experiments/id_diagnostics.py`):
  - (#4) Duplicate / self-masking: 92/50k exact duplicate rows (0.18%); explicit self-index
    masking moves TwoNN 0.00 / MLE +0.17 → ID is **not** a duplicate/self-masking artifact.
  - (rec#5) Bootstrap CIs (B=20 disjoint draws, n=20k): TwoNN 12.71 ± 0.13, MLE 15.18 ± 0.09.
    Sampling CI ±0.1 is tight; the 11–13→11–15 spread is finite-sample n-dependence
    (n=50k → n=20k), not noise.
- **Documented layer-11 post-final-layernorm caveat (#5):** `hidden_states[11+1]` carries
  `ln_f`, so all layer-11 numbers are post-LN; layers 0/3/6/9 (and the layer-6 headline) are
  genuine interior block outputs and unaffected.
- **Wording fixes:** AE reframed "weak corroboration" → "raw-variance reconstruction artifact
  consistent with low ID" (#1); param-matched section now states outer width h1 varies and
  drops "k is the only varying channel" (#2); token-position headline "stable" → "low across
  buckets; TwoNN stable, MLE estimator-dependent" (#6); filled matched-AE train_FVU for
  k=128/256 from the existing JSON (#3).

## 2026-06-21 — Codex review `CODEX_REVIEW_20260621T031919Z` addressed (Iter 6)
- **Parameter-matched AE sweep added** (Codex concern #4 / step #1): new artifacts
  `results/ae_results_matched.json`, `results/ae_matched_param_counts.json`. Total params held
  to 0.087% spread (1024 params) by trading outer hidden width h1 (576→512) against bottleneck
  width k. The low-k bend **survives matching** (within ≤0.0021 of unmatched at every k) → the
  bend is not a parameter-count artifact.
- Scoped "stable across position" → "roughly similar with estimator-dependent variation";
  scoped the synthetic validation to linear-Gaussian data only; reworded "8.3× train budget" →
  "8.3× steps / 16.7× examples"; reported the uncontrolled AE param count with confound
  direction; cleaned stale duplicated operator blocks from PLAN.md.

## 2026-06-20/21 — Operator review `REVIEW.md` addressed (Iter 5)
- **Retracted the strong original framing** "AE and ID converge → demonstrated low-dimensional
  curved manifold." Reframed the AE as weak, preprocessing-sensitive corroboration. Added
  standardized + multi-seed GPU AE sweeps (`results/ae_results_gpu.json`,
  `results/ae_results_gpu_v2.json`) proving the knee vanishes under standardization and is
  seed-stable (std ≤0.0018). AE FVU floor dropped 0.051 (CPU 1200 steps) → 0.033 (GPU 10000
  steps) at k=256; CPU run was genuinely under-trained.
- **Fixed false "everywhere" claims**, now layer-scoped: "standardization changes ID by <2
  everywhere" (false at L11); "TwoNN/MLE agree within ~3 everywhere" (false at L11, gap ~5.2);
  "nonlinear an order of magnitude below d95 across layers" (false at L3/L11, where d95
  collapses to 5–6).
- Corrected "curve flattens after k≈16" (no plateau). Saved the synthetic-validation artifact
  (`results/id_validation.json`). Added token-position-stratified ID
  (`results/id_by_position.json`) and scoped the conclusion to "this pooled FineWeb sample."

## 2026-06-20 — Initial deliverables (S1–S4)
- S1 collect activations (200k fp16 vectors/layer, layers {0,3,6,9,11}, pooled tokens,
  FineWeb REST API + GPT-2). S2 linear PCA-PR + nonlinear TwoNN/MLE
  (`results/pca_pr.json`, `results/id_nonlinear.json`). S3 AE bottleneck sweep layer 6
  (`results/ae_results.json`). S4 first REPORT.md comparing AE elbow vs nonlinear ID vs PCA
  d95 vs d_model.
