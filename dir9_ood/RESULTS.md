# RESULTS — Direction #9 (Plateau-ness as an OOD detector)

**Negative result: plateau-ness does NOT beat the baselines on any OOD set.** Full methods, metric and
baseline definitions (with equations) are in REPORT.md.

**Setup.** GPT-2 small (124M) on GPU. ID = held-out FineWeb (`data/fineweb_sample.txt`). OOD = random
tokens, shuffled tokens, **and Python source code** (real domain shift, offline site-packages).
`seq_len=64`, last-token measurement, $N=200$/set; covariance baselines fit on 1000 separate ID seqs.
**One canonical ID split** (`randperm(seed=7)`, indices in `results/split/canonical_split.npz`) is shared
byte-for-byte by the plateau/standard table and the real-cupbearer table, so all rows are strictly
apples-to-apples. **$N=200 \Rightarrow$ AUROC noise $\approx \pm0.035$**, so gaps below ~0.05 are not
significant. Scores are oriented a priori (higher = more OOD); an AUROC < 0.5 means a *reversed* signal.

**Methods (summary; equations in REPORT.md):** `plateau-jacFrob` = Hutchinson estimate of
$\|\partial \log p/\partial h\|_F$ (4 dirs; flatter=ID); `plateau-perturbation` = mean next-token KL
after 16 unit perturbations (eps=6); `selfNLL-grad` = $\|\partial(-\log p[\arg\max])/\partial h\|$
(confidence-adjacent control); `baseline-MSP` = $1-\max$ softmax; `baseline-L2norm` = $\|h\|_2$;
`baseline-mahalanobis` = naive Mahalanobis (1000-seq fit); `cup-RMD` = cupbearer relative Mahalanobis;
`cup-QUE` = cupbearer Quantum-Entropy/SPECTRE.

![Best plateau variant vs best baseline per OOD set](results/plots/summary_best_per_set.png)

**Figure — what each bar is.** Each red bar is the strongest *plateau variant* for that OOD set, each
blue bar the strongest *baseline*, both annotated with the exact `method@point`: **random**
`plateau-jacFrob@input` 0.73 vs `MSP` 0.93; **shuffled** `plateau-perturbation@resid3` 0.53 vs `MSP`
0.87; **code** `plateau-jacFrob@input` 0.65 vs `cup-RMD@resid6` 0.92. The plateau pool is
{`plateau-jacFrob`, `plateau-perturbation`} (the `selfNLL-grad` confidence control is excluded); the
baseline pool is {MSP, L2, naive-Mahalanobis, cup-RMD, cup-QUE}. Baselines win every set. Regenerate
with `experiments/make_summary_plot.py` (derives best-per-set from `auroc_table.csv`).

**Two concepts used below** (full definitions in REPORT.md). **Canonical split:** the one fixed ID
partition — `randperm(seed=7)` over the FineWeb pool, `fit=perm[:1000]` (fits every ID statistic),
`test=perm[1000:1200]` (held-out ID scored for AUROC) — reused *byte-for-byte* by every method and
table, so all comparisons are apples-to-apples on identical ID examples. **Why MSP detects OOD:** the
model is on average more confident (higher $\max_y p$) on in-distribution text and flatter/less confident
on OOD, so $s_{\text{MSP}}=1-\max_y p$ rises for OOD — except when the model is *confidently wrong* on a
fluent out-of-domain input (the `code` set), where MSP collapses to 0.359.

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

_Full machine-readable table: `results/auroc_table.csv` (87 rows). Raw scores: `results/scores_full.npz`.
Canonical split: `results/split/canonical_split.npz`. ROC + score-distribution plots: `results/plots/`
(ROC plots generated on the iter-2 first-N split; canonical-split numbers shift <0.04, so the
distributions are qualitatively identical)._

The `cup-RMD` / `cup-QUE` rows above are cupbearer's detector math **vendored** and recomputed on the
canonical split. On that split vendored `cup-RMD` matches the **real cupbearer package** essentially
exactly (code@resid6: vendored 0.918 = real 0.918). Vendored `cup-QUE` is the **transductive**
approximation and is superseded by the real-package `cup-QUE` below; read it as caveated.

## Per-OOD-set summary
| OOD set | best plateau | best baseline | does plateau beat baselines? |
|---|---|---|---|
| random | plateau-jacFrob@input 0.734 | **MSP 0.932** | **NO** (also L2@input 0.859, maha@input 0.834) |
| shuffled | plateau-perturbation@resid3 0.534 | **MSP 0.872** | **NO** |
| code | plateau-jacFrob@input 0.649 | **cup-RMD@resid6 0.918**, maha@resid6 0.913 | **NO** (MSP collapses to 0.359) |

## Headline
With the methodology errors flagged by the operator review fixed, the iter-1 "competitive with MSP"
story does not survive:

1. **The genuine plateau metric is weak.** `plateau-jacFrob` tops out at 0.734 (random@input) / 0.649
   (code@input) and is *reversed* on shuffled and in deep residual layers (0.07–0.37). `plateau-
   perturbation` is similarly weak (≤0.70). Neither beats the best baseline on any set.
2. **iter-1's "strong jacobian" was a confidence signal in disguise.** The renamed `selfNLL-grad`
   scores random 0.923 ≈ MSP 0.932 and **collapses on code (≈0.52) exactly like MSP (0.359)** — it
   tracks model confidence, not plateau geometry.
3. **Properly-powered baselines dominate.** MSP is best on synthetic OOD (0.932 / 0.872). On the code
   domain shift MSP collapses (0.359) but **cup-RMD@resid6 (0.918)** and well-fit naive Mahalanobis
   (0.913) are the strongest detectors anywhere. With a 1000-seq covariance fit, Mahalanobis does NOT
   collapse in deep layers.
4. **Internal (residual) vs input-space:** for the genuine `plateau-jacFrob`, **input-space is best**
   and the residual stream is worse/reversed → **no value in measuring plateau-ness internally**.
   (Mahalanobis-type baselines are the opposite: they need a deep residual layer for the code shift.)
5. **No single method is robust across OOD types**, but the most robust are baselines, not plateau:
   naive-Mahalanobis@input has the best worst-case (min 0.499) and cup-RMD@input (0.509); the best
   plateau worst-case is only ~0.37.

**Bottom line:** plateau-ness (measured honestly) is a weak OOD detector and loses to standard and
cupbearer baselines on every OOD set. This is a clean negative result, complete and acceptable per
PLAN.md. See REPORT.md for the per-point verdict and full equations.

---

## Real cupbearer package (isolated env)

The genuine cupbearer package (installed editable from the GitHub clone in an isolated conda env
`cupenv`; shared base env verified untouched, numpy 2.3.3 / torch 2.9.0+cu130) was run on the **same
precomputed GPT-2 last-token activations** (`results/acts/`, canonical split) via
`experiments/cup_eval.py`. Detectors: `MahalanobisDetector` (naive `cup-maha` & relative `cup-RMD`),
`QuantumEntropyDetector` (`cup-QUE`), `SpectralSignatureDetector` (`cup-spectral`); fit on 1000 ID seqs.

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

### Vendored vs real package (`results/cup_real_vs_vendored.csv`)
- **cup-RMD: the vendored math was FAITHFUL** — on code, $|\Delta|\le0.042$ (resid6 $|\Delta|=0.001$:
  real 0.918 vs vendored 0.917); max $|\Delta|=0.065$ across all sets.
- **cup-QUE: the vendored approximation was NOT faithful** — it *understated* QUE on the code shift
  (real `cup-QUE@resid6` **0.910** vs vendored 0.572, $|\Delta|=0.338$). The genuine
  `QuantumEntropyDetector` is fit **once** on ID and applied uniformly (no transductive per-set
  covariance), which is why it differs so much.

### Verdict (unchanged, strengthened)
The real package gives **three** strong code-shift detectors (cup-RMD 0.918, cup-maha 0.913, cup-QUE
0.910), all far above the best plateau variant (≈0.65) and above MSP (collapsed to 0.359 on code). On
random/shuffled the cupbearer covariance detectors (~0.82 / ~0.55) still lose to MSP (0.93 / 0.87), and
plateau-ness loses to both everywhere. `cup-spectral` is weak (≤0.78). Plateau-ness is beaten by
standard *and* genuine-cupbearer baselines on every OOD set.
