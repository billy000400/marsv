# RESULTS — Direction #9 (OOD)

> **iter-5 (2026-06-23) — CANONICAL-SPLIT rerun (Codex 2026-06-22 High finding).** The plateau /
> standard-baseline table and the real-cupbearer table were previously computed on *different* ID
> splits (plateau used the first-N FineWeb seqs; the cupbearer acts used a shuffled `randperm(seed=7)`
> split). iter-5 reran `plateau_v2.py` on the **exact same `randperm(seed=7)` split** that
> `extract_acts.py` used for the cupbearer acts, so `results/auroc_table.csv` and
> `results/auroc_cupbearer.csv` are now **strictly apples-to-apples on one canonical ID split**
> (indices saved to `results/split/canonical_split.npz`). **Verified at the value level**: the new
> plateau-split idtest activations match the precomputed `results/acts/idtest__resid6.npy` to
> max|Δ|=2.5e-5, and the canonical-split vendored baselines now match the real-package cupbearer
> numbers almost exactly (code@resid6: vendored cup-RMD 0.918 = real 0.918; naive-maha 0.913 = real
> cup-maha 0.913). **The numbers shifted <0.04 (within the ±0.035 N=200 noise band) and the verdict is
> unchanged.** The pivot below is the canonical-split run.
>
> _iter-2 (2026-06-21) supersedes iter-1:_ reran on **GPU (RTX 3090 this session; A10 in iter-2 —
> either way CUDA works; iter-1's "V100/CPU-only" claim was false, see experiments/ENV_NOTES.md)** at
> **N=200, seq_len=64**, after the operator review. Changes: a GENUINE Jacobian-Frobenius plateau
> metric (`plateau-jacFrob`), the iter-1 metric renamed honestly to `selfNLL-grad` (MSP-adjacent),
> cupbearer's actual detectors as baselines (`cup-RMD`, `cup-QUE`, vendored from the GitHub repo, and
> validated against the real package in iter-4), and a properly-powered Mahalanobis (covariance fit on
> **1000** ID sequences, not 40).

Setup: GPT-2 small (124M) on GPU. ID = held-out FineWeb (`data/fineweb_sample.txt`). OOD =
random tokens, shuffled tokens, **and Python source code (real domain shift, offline site-packages)**.
seq_len=64, last-token measurement, N=200/set. Covariance baselines fit on 1000 separate ID seqs.
**N=200 ⇒ AUROC noise ≈ ±0.035**, so gaps below ~0.05 are not significant. All scores oriented a
priori so higher = more OOD (we do NOT post-hoc flip signs; an AUROC < 0.5 means that score is
*reversed* for that set — the signal points the wrong way).

Methods:
- **plateau-jacFrob** — genuine plateau metric: Hutchinson estimate of `||d logp(next)/d h||_F` at
  the measurement point (4 random Gaussian output directions). Flatter (lower) = in-distribution.
- **plateau-perturbation** — mean next-token KL after 16 unit perturbations (eps=6) at the point.
- **selfNLL-grad** — `||d(-logp[argmax])/d h||` (iter-1's mislabeled "jacobian"; confidence-adjacent).
- **baseline-MSP** — 1 − max softmax prob. **baseline-L2norm** — activation L2 norm.
- **baseline-mahalanobis** — naive Mahalanobis (1000-seq Gaussian fit).
- **cup-RMD** — cupbearer relative-Mahalanobis (vendored). **cup-QUE** — cupbearer Quantum-Entropy/SPECTRE.

## AUROC pivot (rows = method@point, cols = OOD set; **bold** = best in column)
| method@point | random | shuffled | code |
|---|---|---|---|
| plateau-jacFrob@input | 0.734 | 0.370 | 0.649 |
| plateau-jacFrob@resid3 | 0.327 | 0.237 | 0.625 |
| plateau-jacFrob@resid6 | 0.212 | 0.111 | 0.626 |
| plateau-jacFrob@resid9 | 0.144 | 0.071 | 0.582 |
| plateau-perturbation@input | 0.437 | 0.326 | 0.491 |
| plateau-perturbation@resid3 | 0.647 | 0.534 | 0.506 |
| plateau-perturbation@resid6 | 0.598 | 0.419 | 0.498 |
| plateau-perturbation@resid9 | 0.700 | 0.481 | 0.487 |
| selfNLL-grad@input | 0.923 | 0.639 | 0.516 |
| selfNLL-grad@resid3 | 0.812 | 0.655 | 0.474 |
| selfNLL-grad@resid6 | 0.806 | 0.601 | 0.466 |
| selfNLL-grad@resid9 | 0.811 | 0.587 | 0.463 |
| baseline-MSP@n/a | **0.932** | **0.872** | 0.359 |
| baseline-mahalanobis@input | 0.834 | 0.499 | 0.679 |
| baseline-mahalanobis@resid3 | 0.739 | 0.511 | 0.885 |
| baseline-mahalanobis@resid6 | 0.715 | 0.535 | 0.913 |
| baseline-mahalanobis@resid9 | 0.621 | 0.465 | 0.888 |
| cup-RMD@input | 0.812 | 0.509 | 0.713 |
| cup-RMD@resid3 | 0.740 | 0.517 | 0.892 |
| cup-RMD@resid6 | 0.728 | 0.547 | **0.918** |
| cup-RMD@resid9 | 0.663 | 0.493 | 0.900 |
| cup-QUE@input | 0.782 | 0.508 | 0.581 |
| cup-QUE@resid3 | 0.733 | 0.517 | 0.668 |
| cup-QUE@resid6 | 0.703 | 0.524 | 0.628 |
| cup-QUE@resid9 | 0.603 | 0.481 | 0.563 |
| baseline-L2norm@input | 0.859 | 0.492 | 0.571 |
| baseline-L2norm@resid3 | 0.631 | 0.383 | 0.630 |
| baseline-L2norm@resid6 | 0.464 | 0.440 | 0.639 |
| baseline-L2norm@resid9 | 0.224 | 0.434 | 0.357 |

_Full machine-readable table: `results/auroc_table.csv` (87 rows, canonical split). Raw scores:
`results/scores_full.npz`. Canonical split indices: `results/split/canonical_split.npz`. ROC +
score-distribution plots: `results/plots/` (illustrative; generated on the iter-2 first-N split, but
the canonical-split numbers shift <0.04 so the distributions are qualitatively identical)._

> **Note (iter 5, canonical split):** the `cup-RMD` / `cup-QUE` rows in **this** table are iter-2's
> *vendored* cupbearer math, now recomputed on the canonical split. On that split the vendored
> `cup-RMD` matches the **real cupbearer package** essentially exactly (code@resid6: vendored 0.918 =
> real 0.918) — see the iter-4 section at the bottom for the real-package run (`auroc_cupbearer.csv`,
> on the SAME canonical split). The vendored `cup-QUE` rows here remain the **transductive** approximation
> (each scored set uses its own untrusted covariance) and are **superseded** by the real-package
> `cup-QUE` (fit once on ID; code@resid6 0.910). Read vendored cup-QUE here as caveated/experimental;
> vendored cup-RMD is validated.

## Per-OOD-set summary (canonical split)
| OOD set | best plateau | best baseline | does plateau beat baselines? |
|---|---|---|---|
| random | plateau-jacFrob@input 0.734 | **MSP 0.932** | **NO** (also L2@input 0.859, maha@input 0.834) |
| shuffled | plateau-perturbation@resid3 0.534 | **MSP 0.872** | **NO** |
| code | plateau-jacFrob@input 0.649 | **cup-RMD@resid6 0.918**, maha@resid6 0.913 | **NO** (MSP collapses to 0.359) |

## Headline
**Negative result: plateau-ness does NOT beat the baselines on any OOD set.** With the methodology
errors flagged by the operator review fixed, the iter-1 "competitive with MSP" story does not survive:

1. **The genuine plateau metric is weak.** `plateau-jacFrob` (true Jacobian-Frobenius of the output
   log-prob map) tops out at 0.734 (random@input) / 0.649 (code@input) and is *reversed* on shuffled
   and in deep residual layers (0.07–0.37 — ID is locally steeper than OOD there). `plateau-
   perturbation` is similarly weak (≤0.70). Neither beats the best baseline on any set.
2. **iter-1's "strong jacobian" was a confidence signal in disguise.** The renamed `selfNLL-grad`
   (grad-norm of the model's own argmax NLL) scores random 0.923 ≈ MSP 0.932 and **collapses on
   code (≈0.52) exactly like MSP (0.359)** — i.e. it tracks model confidence, not plateau geometry.
   This confirms the review: iter-1's headline rested on a mislabeled metric.
3. **Properly-powered baselines dominate.** MSP is best on synthetic OOD (0.932 / 0.872). On the real
   domain shift (code) MSP collapses (0.359) but **cupbearer's relative-Mahalanobis `cup-RMD@resid6`
   (0.918)** and the well-fit naive Mahalanobis (0.913) are the strongest detectors anywhere in the
   study. With the covariance fit on 1000 ID seqs (vs iter-1's 40), Mahalanobis does NOT "collapse"
   in deep layers — iter-1's collapse claim was the underpowered-fit artifact the review predicted.
4. **Internal (residual) vs input-space:** for the genuine `plateau-jacFrob`, **input-space is best**
   and the residual stream is worse/reversed — so there is **no value in measuring plateau-ness
   internally**. (Mahalanobis-type baselines are the opposite: they need a deep residual layer to
   catch the code domain shift.)
5. **No single method is robust across OOD types**, but the most robust are baselines, not plateau:
   naive-Mahalanobis@input has the best worst-case (min 0.499 over the three sets) and cup-RMD@input
   (0.509); the best plateau worst-case is only ~0.37.

**Bottom line:** plateau-ness (measured honestly) is a weak OOD detector here and loses to standard
and cupbearer baselines on every OOD set. This is a clean negative result, which PLAN.md declares
complete and acceptable. See REPORT.md for the per-point verdict and how each review point was handled.

---

## iter-4 (2026-06-21) — REAL cupbearer package in an isolated env (operator follow-up)

Operator follow-up (`human_feedback.md`): *"Can you create a new environment to evaluate OOD with
cupbearer?"* — i.e. run the **genuine** cupbearer package (iter-2 only *vendored* its detector math).
Done: built an **isolated conda env** `cupenv` (numpy 1.26.4 + torch **2.9.0+cu130** + cupbearer
installed editable from the **GitHub** clone `vendor/cupbearer-main`, plus transformers 5.12.1 /
datasets 5.0.0). **GPU compat:** A10, driver 595, **CUDA 13.2 ≥ 13.0**; torch CUDA matmul verified
inside `cupenv`. **Shared base env untouched** (still numpy 2.3.3 / torch 2.9.0+cu130 — verified
before & after). The real package's detectors were run on the **same precomputed GPT-2 last-token
activations** (`results/acts/`, extracted once in the base env by `experiments/extract_acts.py`) via
`experiments/cup_eval.py` (a `FeatureExtractor` that serves precomputed features). Detectors:
`MahalanobisDetector` (naive `cup-maha` & relative `cup-RMD`), `QuantumEntropyDetector` (`cup-QUE`),
`SpectralSignatureDetector` (`cup-spectral`). Fit on 1000 ID seqs; ID-test vs each OOD set, N=200.

### Real-package AUROC (`results/auroc_cupbearer.csv`, 48 rows)
| method@point | random | shuffled | code |
|---|---|---|---|
| cup-maha@resid6 | 0.715 | 0.535 | 0.913 |
| cup-RMD@resid6 | 0.728 | 0.547 | **0.918** |
| cup-RMD@resid9 | 0.663 | 0.493 | 0.900 |
| cup-QUE@resid6 | 0.728 | 0.545 | 0.910 |
| cup-QUE@resid3 | 0.740 | 0.507 | 0.895 |
| cup-maha@input / cup-QUE@input | 0.824 | 0.496 | 0.696 |
| cup-RMD@input | 0.812 | 0.509 | 0.713 |
| cup-spectral@input | 0.780 | 0.493 | 0.368 |
| cup-spectral@resid6 | 0.371 | 0.452 | 0.520 |

Best real-cupbearer detector per set: random `cup-maha@input` 0.824, shuffled `cup-RMD@resid6` 0.547,
**code `cup-RMD@resid6` 0.918**.

### Vendored (iter-2) vs real package (`results/cup_real_vs_vendored.csv`)
- **cup-RMD: the iter-2 vendored math was FAITHFUL.** On code, `|Δ|`≤0.042 (at resid6 `|Δ|=0.001`:
  real 0.918 vs vendored 0.917); across all sets max `|Δ|=0.065`. iter-2's headline RMD numbers stand.
- **cup-QUE: the iter-2 vendored approximation was NOT faithful** — it *understated* QUE badly on the
  code shift (real `cup-QUE@resid6` **0.910** vs vendored 0.572, `|Δ|=0.338`; max `|Δ|=0.356`). The
  genuine `QuantumEntropyDetector` is as strong as RMD/maha on the code domain shift, not weak.

### Verdict (unchanged, and strengthened)
Running the **real** cupbearer package does **not** change the negative result for plateau-ness — it
reinforces it. On the code domain shift the genuine package now provides **three** strong detectors
(`cup-RMD` 0.918, `cup-maha` 0.913, `cup-QUE` 0.910), all far above the best plateau variant
(`plateau-jacFrob@input` 0.628) and above MSP (which collapses to 0.384 on code). On random/shuffled
the cupbearer covariance detectors (~0.82 / ~0.55) still lose to MSP (0.94 / 0.90), and plateau-ness
loses to both everywhere. `cup-spectral` is weak (best 0.78 random@input; ≤0.52 on code). So:
plateau-ness is beaten by standard *and* genuine-cupbearer baselines on every OOD set. The only
correction to iter-2 is that **cupbearer's QUE detector is strong (not weak) on real domain shift** —
which makes the case *against* plateau-ness slightly stronger, not weaker.

**cup-QUE protocol (resolves the iter-2 caveat).** The Codex review (CODEX_REVIEW_20260621T031213Z.md)
flagged that iter-2's *vendored* `cup_que(fit, test)` recomputed the untrusted covariance from whatever
set was being scored, so ID and OOD were scored under *different* functions. The iter-4 real-package
`QuantumEntropyDetector` does **not** have this problem: it is fit **once** (trusted whitening + the
untrusted covariance both from the ID `idtrain` set, passed as `untrusted_data`), and ID-test and every
OOD set are then scored under that **single fixed** function — a consistent, headline-eligible baseline.
That protocol fix is exactly why the real cup-QUE (0.910 on code) differs so much from the vendored
0.572. The iter-2 vendored `cup-QUE` rows in the table above are therefore **superseded** by the iter-4
real-package numbers and should be read as caveated/experimental; vendored `cup-RMD` stands (validated).
_Artifacts: `results/auroc_cupbearer.csv`, `results/cup_real_vs_vendored.csv`,
`experiments/{build_cupenv.sh, resume_cupenv.sh, extract_acts.py, cup_eval.py, compare_cup.py}`,
env `cupenv/` (isolated)._
