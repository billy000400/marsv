# REPORT — Direction #9: Plateau-ness as an OOD / Anomaly Detector

> **iter-2 (2026-06-21) — rewritten after the operator review (CODEX_REVIEW.md, human_feedback.md).**
> iter-1's conclusions are retracted; they rested on a false environment claim, a mislabeled metric,
> and an underpowered baseline. This report reflects the corrected GPU rerun (N=200, seq_len=64).

## Question
Can "plateau-ness" of GPT-2's loss landscape (how flat the model's next-token distribution is to
local perturbations of an internal activation) act as an out-of-distribution detector, and does
measuring it **inside** the residual stream beat the simpler **input-space** version?

## What was run
- Model: GPT-2 small (124M), **on GPU (NVIDIA A10, sm_86, CUDA 13.2 — works)**. VRAM capped to 0.45,
  2 CPU threads (BUDGET). The iter-1 "V100/no-CUDA-kernels/CPU-only" story was false (ENV_NOTES.md).
- ID = held-out FineWeb. OOD (3 sets): **random tokens, shuffled tokens, and Python source code**
  (a real domain shift, read offline from numpy/torch site-packages — valid-but-different text, the
  regime that separates a real OOD detector from a gibberish flag).
- Two plateau variants + a transparency control, each at 4 measurement points (input embeddings +
  resid after blocks 3/6/9):
  - **plateau-jacFrob** (the genuine metric): Hutchinson estimate of `||d logp(next)/d h||_F`
    (4 random Gaussian output directions). A label-free measure of output-distribution flatness.
  - **plateau-perturbation**: mean next-token KL after 16 unit perturbations (eps=6) at the point.
  - **selfNLL-grad** (control = iter-1's mislabeled "jacobian"): `||d(-logp[argmax])/d h||`. Kept,
    renamed honestly, to show it is confidence-adjacent.
- Baselines: **MSP**, **activation L2 norm**, **Mahalanobis** (Gaussian fit on **1000** ID seqs),
  and **cupbearer's own detectors** — **cup-RMD** (relative Mahalanobis) and **cup-QUE**
  (Quantum-Entropy / SPECTRE), using cupbearer's actual code (vendored, see below).
- Metric: AUROC (OOD positive), N=200/set, seq_len=64. **Noise ≈ ±0.035.** Numbers in RESULTS.md /
  `results/auroc_table.csv`; plots in `results/plots/`. Scores oriented a priori (higher=OOD); we do
  NOT post-hoc flip signs, so AUROC<0.5 means a reversed signal.

> **Numbers below are the iter-5 canonical-split run** (one ID split shared by the plateau table and
> the real-cupbearer table; see "Canonical split" note at the bottom). They shift <0.04 vs iter-2 and
> the verdict is unchanged.

## Verdict per OOD set
### random tokens
Best plateau **plateau-jacFrob@input 0.734** < best baseline **MSP 0.932** (also L2@input 0.859,
maha@input 0.834). **Plateau loses.** The confidence-adjacent selfNLL-grad@input (0.923) matches MSP.

### shuffled tokens (same unigram stats, wrong order)
Best plateau **plateau-perturbation@resid3 0.534** < best baseline **MSP 0.872**. **Plateau loses
clearly.** plateau-jacFrob is reversed here (0.07–0.37).

### code (real domain shift)
Best plateau **plateau-jacFrob@input 0.649** < best baseline **cup-RMD@resid6 0.918** (naive
Mahalanobis@resid6 0.913; real-package cup-RMD/cup-maha/cup-QUE@resid6 0.918/0.913/0.910). **Plateau
loses.** Note **MSP collapses to chance (0.359)** on code — the one place a confidence baseline fails —
but Mahalanobis-type detectors (incl. cupbearer's) handle it easily, so plateau-ness is not needed even
in MSP's blind spot.

### Internal vs input-space
For the genuine plateau metric (jacFrob) **input-space is best and the residual stream is worse or
reversed** → **no value in measuring plateau-ness internally**. (The strong baselines are the
opposite: Mahalanobis/cup-RMD need a deep residual layer to catch the code shift.)

## Bottom line
1. **Plateau-ness does NOT beat the baselines on any OOD set.** Measured honestly (Jacobian-Frobenius
   of the output distribution), it is weak (≤0.73) and loses to MSP on synthetic OOD and to
   cupbearer/Mahalanobis on real domain shift. **This is a clean negative result** — which PLAN.md
   declares complete and acceptable.
2. **iter-1 was wrong because of the mislabeled metric.** Its "winning jacobian" was `selfNLL-grad`,
   which here tracks MSP (random 0.923≈0.932) and collapses on code like MSP — a confidence signal,
   not plateau geometry.
3. **Measuring internally buys nothing** for the genuine plateau signal (input-space ≥ residual).
4. **Best detectors overall:** MSP (synthetic) and cupbearer relative-Mahalanobis / well-fit
   Mahalanobis in a deep residual layer (real domain shift). No single method is robust across all
   three OOD types, but the most robust ones are baselines, not plateau.

## How each operator-review point was addressed
- **CODEX H1 — false environment/provenance.** Confirmed: GPU is a working A10 (CUDA 13.2), not a
  V100, and CUDA ops succeed. Corrected ENV_NOTES.md and reran everything on GPU. The false claim is
  retracted.
- **CODEX H2 — "plateau-jacobian" mislabeled.** Confirmed. Added the genuine `plateau-jacFrob`
  (output-distribution Jacobian-Frobenius, label-free) and renamed the old quantity `selfNLL-grad`.
  The genuine metric is weak, and selfNLL-grad is shown to be MSP-adjacent — so the headline changed.
- **CODEX M — benchmark too weak.** Added the **code** real-domain-shift OOD set, and added
  **cupbearer's own detectors** (cup-RMD, cup-QUE) as mechanistic-anomaly baselines.
- **CODEX M — "Jacobian far cheaper" unsubstantiated.** Dropped that claim; on GPU the whole sweep is
  ~270 s and cost is not a differentiator we rely on.
- **CODEX M — Mahalanobis underpowered (40 samples / 768-dim).** Fixed: covariance now fit on **1000**
  ID sequences. Mahalanobis no longer "collapses" in deep layers and is in fact the best code
  detector — confirming the iter-1 "collapse" claim was a small-sample artifact.
- **human_feedback — use cupbearer (GitHub, not PyPI) as baselines.** Cloned the GitHub PyTorch build;
  it cannot be installed without violating the shared-package constraint (pins numpy<2 vs shared
  numpy 2.3.3, plus lightning/torchvision/torchattacks), so its self-contained detector math is
  **vendored verbatim** (`experiments/cupbearer_helpers.py`) and used as `cup-RMD` / `cup-QUE`.
- **human_feedback — do not change shared cuda/torch/numpy; check repo runs; use own env if needed.**
  No shared DL package versions were changed (verified). Compatibility documented in ENV_NOTES.md.

## Operator follow-up (iter 4, 2026-06-21) — REAL cupbearer in an isolated env
`human_feedback.md`: *"Can you create a new environment to evaluate OOD with cupbearer?"* (i.e. run the
genuine package, not iter-2's vendored math). **Done and addressed:**
- Built an **isolated conda env** `cupenv` (its own numpy 1.26.4 + torch 2.9.0+cu130) and installed
  cupbearer **editable from the GitHub clone** (`vendor/cupbearer-main`), per the "GitHub not PyPI"
  instruction. Build scripts: `experiments/{build_cupenv.sh, resume_cupenv.sh}`.
- **GPU compat checked** (per instruction): A10, driver 595, **CUDA 13.2 ≥ 13.0**; torch CUDA matmul
  verified *inside* `cupenv`. The full detector sweep ran on GPU.
- **Shared cuda/torch/numpy untouched** (per instruction): base env verified still numpy 2.3.3 /
  torch 2.9.0+cu130 before and after; cupbearer's numpy<2 lives only in the separate `cupenv`.
- **Ran cupbearer's real detectors as baselines** (`experiments/cup_eval.py`) on the same precomputed
  GPT-2 activations: `cup-maha`, `cup-RMD`, `cup-QUE`, `cup-spectral` → `results/auroc_cupbearer.csv`.
- **Validation vs iter-2's vendored math** (`results/cup_real_vs_vendored.csv`): vendored **cup-RMD was
  faithful** (code@resid6 |Δ|=0.001), but vendored **cup-QUE understated the real detector** on the
  code shift (real 0.910 vs vendored 0.572). **Verdict is unchanged and strengthened:** the genuine
  package gives three strong code-shift detectors (cup-RMD 0.918, cup-maha 0.913, cup-QUE 0.910), all
  far above the best plateau variant (0.628) and above the collapsing MSP (0.384). Plateau-ness still
  loses to standard *and* genuine-cupbearer baselines on every OOD set. Full numbers in RESULTS.md.

This iteration also closes every item in the second Codex review (`CODEX_REVIEW_20260621T031213Z.md`),
which observed that the earlier `cupenv` was broken (couldn't import torch/cupbearer) and that no real
cupbearer-package evaluation existed:
- **"Fix or rebuild cupenv; cupenv cannot import torch (missing libtorch_global_deps.so) or
  cupbearer."** Fixed — `cupenv` now imports `torch 2.9.0+cu130` (CUDA works) and the real `cupbearer`.
- **"Run cup_eval.py and save results/auroc_cupbearer.csv."** Done (48 rows).
- **"Compare real package to vendored cup-RMD/cup-QUE; validate or update."** Done
  (`results/cup_real_vs_vendored.csv`): RMD validated, QUE updated to the real-package numbers.
- **"Don't claim 'cupbearer's actual code' for vendored-only math."** Fixed — the writeup now clearly
  separates iter-2 *vendored* rows from the iter-4 *real-package* rows, and marks vendored QUE as
  superseded.
- **"Revisit cup-QUE protocol — ID/OOD scored under different covariance functions; caveat or fix."**
  Fixed by the real package: `QuantumEntropyDetector` is fit **once** on ID (`idtrain` as both trusted
  and untrusted reference) and ID-test + every OOD set are scored under that single fixed function — a
  consistent baseline, not the transductive per-set scoring of iter-2's vendored `cup_que`. The
  vendored cup-QUE rows are demoted to caveated/experimental; the real cup-QUE is headline-eligible.
- Items the review already marked resolved (provenance/A10, mislabeled Jacobian, code-domain OOD,
  1000-seq Mahalanobis, dropped "Jacobian cheaper" claim) remain resolved.

## Canonical split (iter 5, 2026-06-23) — Codex review `CODEX_REVIEW_20260622T230658Z.md`
That review confirmed the main conclusion is faithful but flagged **comparison-hygiene** issues, chiefly
that the plateau table and the real-cupbearer table were computed on **different ID splits** (plateau:
first-N FineWeb seqs; cupbearer acts: a shuffled `randperm(seed=7)` split). Addressed:
- **High — different ID split.** Reran `experiments/plateau_v2.py` on the **exact same**
  `randperm(seed=7)` split that `extract_acts.py` used for the cupbearer acts (`fit=perm[:1000]`,
  `test=perm[1000:1200]`). The split indices are saved to `results/split/canonical_split.npz`, and the
  alignment is **verified at the value level**: the new plateau-split idtest activations equal the
  precomputed `results/acts/idtest__resid6.npy` to max|Δ|=2.5e-5, and on the unified split the vendored
  baselines now match the real-package cupbearer numbers essentially exactly (code@resid6: vendored
  cup-RMD 0.918 = real 0.918; naive-maha 0.913 = real cup-maha 0.913). So `auroc_table.csv` and
  `auroc_cupbearer.csv` are now strictly apples-to-apples. The numbers moved <0.04; the verdict is
  unchanged (plateau loses on every set).
- **Medium — cup-QUE interpretation.** The real `cup-QUE` is now explicitly scoped as a *consistent
  cupbearer-code detector variant* (fit once on ID), not the definitive standard QUE protocol — see
  Limitations. The transductive caveat is scoped to the vendored iter-2 rows only.
- **Medium — cupbearer audit trail / fresh log.** `auroc_cupbearer.csv` (48 rows) was **not** recomputed
  because its inputs (`results/acts/`) are byte-identical and on the canonical split (the alignment was
  verified above), so the numbers are unchanged; a `cupenv` re-run would only refresh a log at the cost
  of a slow ceph-FS import, not change any result. The earlier partial `finish_cup.log` (24-row failed
  run) is superseded by the working `cup_eval.py` that produced the current 48-row CSV.
- **Low — stale writeup blocks.** The stale transductive-`cup-QUE` limitation was scoped to vendored
  rows; the stale "Operator review — address before continuing" block was removed from `PLAN.md`.

## Limitations / honest caveats
- N=200 ⇒ ±0.035 noise; the qualitative ranking (baselines > plateau on every set) is robust to it.
- **cup-QUE scope.** Only the *vendored* iter-2 `cup-QUE` rows in `auroc_table.csv` are transductive
  (each scored set used its own untrusted covariance) — those are superseded/caveated. The
  **real-package** `cup-QUE` (`auroc_cupbearer.csv`) is fit **once** on ID and applied uniformly, so it
  is a consistent fixed-function detector. Even so, it should be read as *a cupbearer-code detector
  variant* (untrusted_data = the ID set), not necessarily the definitive standard SPECTRE/QUE
  anomaly-mixture protocol; with 768-dim activations its covariance is rank-limited. This does not
  affect the conclusion, since cup-RMD and naive/cup Mahalanobis already beat plateau strongly on code.
- One model (GPT-2 small) and three OOD sets; a broader model/OOD sweep could change magnitudes but
  is unlikely to overturn a result this one-sided.
- The real-package run reuses precomputed last-token activations (not cupbearer's full task/data
  harness); this isolates the *detector* comparison, which is exactly what the baseline question asks.
