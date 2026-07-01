# CHANGELOG — Direction

Append-only.

---

## 2026-06-30 — Stage A reproduction (first results)
- Built `experiments/smoke_plateau.py`: end-to-end plateau pipeline (GPT-2 small + jbloom
  resid_pre@6 SAE, in-context forward-hook plateau sweep). Installed `transformers`,
  `tokenizers==0.22.2`, `safetensors`, `huggingface_hub`, `regex`, `httpx`, `hf-xet` via
  `pip --no-deps` (network available); did NOT touch torch/CUDA.
- RESULTS.md created (was placeholder). New numbers, N=200, 8 directions:
  - `plateau_auc_low` medians — real 0.200, recon 0.162, naive 0.066, norm_rand 0.035.
  - Paired gaps vs real: recon 0.017 [0.010,0.024]; naive 0.122 [0.108,0.138];
    norm_rand 0.160 [0.148,0.176].
  - Pooled covariate Spearman: plateau vs norm +0.06 (no norm shortcut); plateau vs
    distance-to-source −0.82 (gap tracks closeness to a real activation).
- Plot `plots/plateau_stageA.png` added (mean KL curves + per-condition AUC box).
- Verdict added: H1 reproduced (not a norm artifact); H2 distance-to-source caveat open,
  to be resolved by Stage B distance-matched analysis.

## 2026-06-30 — Stage B: distance-to-source matched control (decisive H2 test)
- Built `experiments/stageB_distance.py`: adds an `iso_displace` reference family
  `x_real + δ·d` (δ∈{15,30,60,120}, distance≡δ) tracing plateau-vs-distance for random
  off-manifold displacement of a real activation; adds `sparse_match` (naive with k=source
  L0 and coefficients rescaled to source coef-RMS); calibrates τ on a **held-out real
  split** (τ_heldout=1.33e-4) and scores all conditions on the eval half (N_eval=100, 6 dirs).
- RESULTS.md: replaced the Stage-A "Confound preview" Spearman teaser with the full Stage B
  distance-matched analysis. Stage A table unchanged.
  - iso_displace reference plateau: d=15→0.184, d=30→0.173, d=60→0.128, d=120→0.078
    (plateau falls monotonically with distance for random displacement alone).
  - Distance-matched residual (condition plateau − reference @ its distance), 95% CI:
    recon −0.016 [−0.021,−0.003]; naive −0.058 [−0.065,−0.053];
    sparse_match −0.063 [−0.067,−0.049].
  - Pooled Spearman(plateau, distance) over SAE+iso conds (eval) = −0.64.
- Verdict UPDATED: H2 — plateau gap does NOT survive distance matching as an SAE-validity
  signal. No SAE-decoded condition plateaus ABOVE random displacement at matched distance;
  recon's advantage = closeness-to-real (residual ≈0); naive/sparse_match sit BELOW the
  random-displacement curve; sparsity/coef matching does not recover plateau. Plateau-ness =
  closeness-to-real proxy + direction-family effect, not an independent SAE-validity diagnostic.
- Plot `plots/plateau_stageB.png` added (plateau-vs-distance overlay + distance-matched
  residual bars). Artifacts: `results/stageB_summary.json`, `results/stageB_metrics.csv`.

## 2026-06-30 — Stage D: downstream-validity prediction gate (H4/M4) + REPORT.md finalized
- Built `experiments/stageD_validity.py`: independent downstream-validity target
  `output_kl = KL(p_real || p_candidate)` (in-context last-token); pooled 7 candidate
  conditions (recon, naive, sparse_match, iso15/30/60/120) × N=200 = 1400 rows; split by
  source prompt; linear held-out R² for log10 output_kl. Added a single fixed-radius
  local-sensitivity baseline `locsens = log10 mean-KL@r=0.02` as the discriminator.
- RESULTS.md: added Stage D section; rewrote "Current verdict" into a project-level null.
  - Held-out test R²: baseline(dist,norm) 0.795; +plateau 0.869; baseline+locsens 0.873;
    all(+plateau) 0.878.
  - ΔR² plateau beyond {dist,norm} = +0.073 (partial Spearman −0.65) — plateau DOES predict
    validity beyond distance+norm.
  - ΔR² plateau beyond {dist,norm,locsens} = +0.005 (partial Spearman −0.16) — plateau adds
    ~NOTHING beyond a single local-sensitivity number. Marginal Spearman: plateau −0.85,
    locsens +0.84, dist +0.75, norm −0.22.
- Verdict (H4): plateau's downstream-validity prediction = LOCAL SENSITIVITY, not
  interpretability validity (decision-table Jacobian/sensitivity row).
- Project-level null FINALIZED: plateau-ness = closeness-to-real (Stage B) + local robustness
  (Stage D), NOT an SAE interpretability-validity diagnostic. Names the failing notion =
  "mere local robustness". Consistent with D9 (OOD weak) and D6 (local sensitivity).
- REPORT.md WRITTEN (was absent): Summary→Methods→Results→Conclusion; Methods defines model/
  layer/SAE, every metric and baseline with $$LaTeX$$ (plateau_auc_low, output_kl, distance,
  norm, iso_displace reference, locsens). Plot `plots/plateau_stageD.png` added.
- Artifacts: `results/stageD_summary.json`, `results/stageD_metrics.csv`.
