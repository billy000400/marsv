# CHANGELOG — Direction #9 (Plateau-ness as an OOD detector)

Append-only. History of changes to the deliverables (RESULTS.md / REPORT.md). Current-best numbers
live in those files; this file records what moved and why.

---

## 2026-07-01 — Documentation clarity pass (iter 7; `human_feedback_07011019.md`)
- **Deliverable clarity only — no science changed, no numbers moved.** Addressed three operator
  documentation requests:
  1. **Headline figure now names the methods.** Regenerated `results/plots/summary_best_per_set.png`
     via a new reproducible script `experiments/make_summary_plot.py` (derives best-per-set from
     `auroc_table.csv`): each bar is annotated with its exact `method@point` — random
     `plateau-jacFrob@input` 0.73 vs `MSP` 0.93; shuffled `plateau-perturbation@resid3` 0.53 vs `MSP`
     0.87; code `plateau-jacFrob@input` 0.65 vs `cup-RMD@resid6` 0.92. Expanded the figure caption in
     both RESULTS.md and REPORT.md to state which plateau variant / baseline each bar is and that the
     `selfNLL-grad` control is excluded from the plateau pool.
  2. **Defined "canonical split"** in REPORT.md Methods (and a short gloss in RESULTS.md): the one fixed
     `randperm(seed=7)` ID partition (`fit=perm[:1000]`, `test=perm[1000:1200]`) reused byte-for-byte by
     every method/table so comparisons are apples-to-apples.
  3. **Explained why MSP detects OOD** under the MSP baseline in REPORT.md (and a gloss in RESULTS.md):
     the model is more confident on ID than OOD, so $1-\max_y p$ rises for OOD; noted its
     confidently-wrong failure on `code` (collapses to 0.359).
- Env note: base env had again lost `matplotlib` — reinstalled it + pure-python deps with `--no-deps`
  (torch 2.9.0+cu130 / numpy 2.3.3 verified untouched).

## 2026-06-30 — Final consolidation (iter 6)
- **Deliverable hygiene only — no science changed.** Brought RESULTS.md / REPORT.md to CLAUDE.md
  current-best form: removed the inline version-history blockquotes ("iter-5 supersedes iter-2
  supersedes iter-1", "rewritten after operator review") from both files — that history now lives
  here.
- **REPORT.md restructured to `Summary → Methods → Results → Conclusion → Limitations`.** Added the
  required Methods section with Data/Model/Layer and **`$$LaTeX$$` definitions for every metric**
  (plateau-jacFrob, plateau-perturbation, selfNLL-grad) **and baseline** (MSP, L2, Mahalanobis,
  cup-RMD, cup-QUE) plus the AUROC estimator. The per-iteration "how each review point was addressed"
  narrative was moved out of REPORT.md into the entries below.
- Added a headline figure `results/plots/summary_best_per_set.png` (best plateau variant vs best
  baseline per OOD set) referenced from both files.
- No numbers changed (canonical-split values from 2026-06-23 are current-best): random plateau 0.734
  < MSP 0.932; shuffled plateau 0.534 < MSP 0.872; code plateau 0.649 < cup-RMD@resid6 0.918.
- Wrote `STOP`.

## 2026-06-23 — Canonical-split rerun (iter 5; Codex `…20260622T230658Z`)
- **Fixed comparison hygiene (High):** the plateau/standard-baseline table and the real-cupbearer
  table had been computed on *different* ID splits (plateau = first-N FineWeb seqs; cupbearer acts =
  shuffled `randperm(seed=7)`). Reran `plateau_v2.py` on the **exact `randperm(seed=7)` split** used by
  `extract_acts.py`; saved indices to `results/split/canonical_split.npz`. Verified at value level
  (new idtest acts = precomputed `idtest__resid6.npy` to max|Δ|=2.5e-5; on the unified split vendored
  cup-RMD@resid6 0.918 = real 0.918, naive-maha 0.913 = real cup-maha 0.913).
- **Result deltas (old → new), all < 0.04, verdict unchanged:** representative — random
  jacFrob@input ~0.73 → 0.734; shuffled best plateau ~0.53 → 0.534; code jacFrob@input 0.628 → 0.649;
  code cup-RMD@resid6 0.917/0.918 → 0.918. `auroc_table.csv` now 87 data rows on the canonical split.
- Did **not** recompute `auroc_cupbearer.csv`: its inputs (`results/acts/`) are byte-identical and
  already on the canonical split, so the real-package numbers are unchanged.
- Scoped the transductive cup-QUE caveat to vendored rows only; removed the stale "do NOT STOP"
  operator-review block from PLAN.md.

## 2026-06-21 — Real cupbearer package in an isolated env (iter 4; `human_feedback.md` + Codex `…20260621T031213Z`)
- Built isolated conda env `cupenv` (numpy 1.26.4 + torch 2.9.0+cu130) and installed cupbearer
  **editable from the GitHub clone** (`vendor/cupbearer-main`), not PyPI. Shared base env verified
  untouched (numpy 2.3.3 / torch 2.9.0+cu130 before & after). Ran the genuine detectors on the
  precomputed activations → `results/auroc_cupbearer.csv` (48 rows); compared to iter-2 vendored math
  → `results/cup_real_vs_vendored.csv`.
- **Corrections to iter-2 numbers:** vendored **cup-RMD was faithful** (code@resid6 vendored 0.917 ≈
  real 0.918, |Δ|=0.001). Vendored **cup-QUE understated the real detector** on code: 0.572 → **0.910**
  (|Δ|=0.338) — because the real `QuantumEntropyDetector` is fit **once** on ID and applied uniformly,
  fixing the transductive per-set covariance of the vendored `cup_que`.
- **Verdict strengthened, not changed:** code domain shift now has three strong baselines
  (cup-RMD 0.918, cup-maha 0.913, cup-QUE 0.910), all ≫ best plateau (~0.63) and ≫ MSP (collapsed
  ~0.38). cup-spectral weak (≤0.78).

## 2026-06-21 — Corrected GPU rerun after operator review (iter 2–3; `CODEX_REVIEW.md` + `human_feedback.md`)
- **Major correction — iter-1 headline retracted.** Reran on **GPU** (A10, CUDA 13.2 works; iter-1's
  "V100/CPU-only" claim was false) at **N=200, seq_len=64** via `experiments/plateau_v2.py`.
- **Mislabeled metric fixed:** iter-1's "plateau-jacobian" was grad-norm of the model's own argmax NLL
  (confidence-adjacent); renamed **`selfNLL-grad`** and added the **genuine `plateau-jacFrob`**
  (Hutchinson estimate of the output log-prob Jacobian-Frobenius norm).
- **Number changes (iter-1 → iter-2):** the "strong jacobian" win evaporated — genuine plateau-jacFrob
  is weak (≤0.73, reversed in deep resid: 0.07–0.37) where iter-1 had reported 0.92–0.97; the 0.92–0.97
  belonged to selfNLL-grad, which tracks MSP (random 0.923 ≈ MSP 0.932) and collapses on code like MSP.
- **Baselines strengthened:** Mahalanobis covariance fit on **1000** ID seqs (was 40) — no longer
  "collapses" in deep layers and becomes the best code detector (maha@resid6 0.913). Added the **code**
  real-domain-shift OOD set and **cupbearer** detectors (cup-RMD / cup-QUE, vendored from GitHub).
- **Verdict flipped from iter-1's "competitive with MSP" to NEGATIVE:** plateau-ness does NOT beat the
  baselines on any OOD set. `auroc_table.csv` grew to cover 3 OOD sets × all methods × 4 points.
- iter 3 was finalization only (created the STOP that iter 2 missed; reconciled PLAN.md).

## 2026-06-20 — Initial full sweep (iter 1)
- First RESULTS.md / REPORT.md: 2 plateau variants × {input, resid3/6/9} × {MSP, L2, Mahalanobis} on
  OOD {random, shuffled}, N=40/seq=32. Reported a "strong jacobian" competitive with MSP.
- **Superseded entirely by the 2026-06-21 rerun** — that headline rested on a mislabeled metric, an
  underpowered Mahalanobis fit, and a false CPU-only environment claim (all corrected above).
