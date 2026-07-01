# CHANGELOG — Direction

Append-only.

---

## 2026-07-01 — Display-math STILL broke; real fix (no science changed)
- **Rendering fix only — no numbers moved.** The earlier fix (below) was wrong: an indented ```math
  fence inside a bullet still renders as a gray code box (not math) when the bullet has any inline
  `$…$`. On top of the 2 nested fences, the 4 column-0 `$$` blocks (`x(r)`, `KL(r)`,
  `plateau_auc_low`, `output_kl`) were glued to the prior line and rendered as raw text. Verified via
  `POST api.github.com/markdown`.
- **Fix:** the 4 top-level `$$` blocks → column-0 ```math fences with blank lines; the 2 short
  nested equations (SAE encode/decode, distance-matched residual `ρ_c`) → inline `$…$` (keeps the
  bullet lists intact). API check: 4/4 display equations render, 0 code blocks, 0 raw `$$`.
- See rewritten project `CLAUDE.md` rule **8a**: never nest display math in a list item; keep it at
  column 0; verify via the markdown API before committing.

## 2026-07-01 — Stage B-dir curated into deliverables (direction-family robustness)

- Prior iteration ran `experiments/stageB_directions.py` (created + committed) but **did not
  curate the result into RESULTS.md/REPORT.md**; they still listed "SAE-decoder-direction
  robustness (not run)" and "one direction family (isotropic)" as an open caveat. Re-ran the
  script at **full config** (N=200, N_eval=100, 8 dirs; prior committed artifact was the smoke
  N=24 / 2-dir run) and curated both deliverables.
- **New Stage B-dir section** (RESULTS.md + REPORT.md): distance-matched residual $\rho_c$ under
  three perturbation-direction families. Full-config medians [95% CI]:
  - iso: recon −0.015 [−0.025,+0.003], naive −0.061 [−0.068,−0.057], sparse −0.062 [−0.069,−0.052]
  - sae_single: recon −0.016 [−0.029,−0.003], naive −0.066 [−0.071,−0.058], sparse −0.062 [−0.068,−0.052]
  - sae_sparse: recon −0.015 [−0.032,+0.006], naive −0.077 [−0.084,−0.065], sparse −0.071 [−0.076,−0.063]
  - Pooled Spearman(plateau, distance): iso −0.64, sae_single −0.60, sae_sparse −0.62.
- **Finding:** the Stage B null is **direction-family robust** — recon ≈ on the random curve and
  naive/sparse sit BELOW random under every family; the naive deficit is if anything *larger*
  along SAE decoder directions. Closes the biggest open caveat.
- Updated H2 verdict (added "direction-family robust") and Scope paragraph (moved
  SAE-decoder-direction robustness from "not run" to "confirmed") in both deliverables. No
  earlier numbers superseded (new section). Plot `plots/plateau_stageB_dir.png` (full config);
  artifacts `results/stageB_dir_summary.json`, `results/stageB_dir_metrics.csv`.

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
