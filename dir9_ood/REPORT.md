# REPORT — Direction #9: Plateau-ness as an OOD / Anomaly Detector

## Summary
**Question.** Can the *plateau-ness* of GPT-2's loss landscape — how flat the model's next-token
distribution is to local perturbations of an internal activation — act as an out-of-distribution (OOD)
detector? And does measuring it **inside** the residual stream beat the simpler **input-space** version?

**Answer — negative.** Measured honestly (Jacobian-Frobenius of the output distribution), plateau-ness
is a **weak** OOD detector and **loses to standard baselines on every OOD set tested**: it tops out at
AUROC ≈ 0.73 and is reversed in deep residual layers. Maximum-Softmax-Probability (MSP) wins synthetic
OOD, and Mahalanobis / cupbearer relative-Mahalanobis win the real domain shift where MSP collapses.
For the genuine plateau metric **input-space is best**, so there is **no value in measuring internally**.
A null result is complete and acceptable per PLAN.md.

![Best plateau variant vs best baseline per OOD set](results/plots/summary_best_per_set.png)

**Figure — what each bar is.** For every OOD set the figure shows the single strongest *plateau
variant* (red) against the single strongest *baseline* (blue), each bar annotated with the exact
`method@measurement-point` it represents. Concretely: **random** — best plateau `plateau-jacFrob@input`
(0.73) vs best baseline `MSP` (0.93); **shuffled** — `plateau-perturbation@resid3` (0.53) vs `MSP`
(0.87); **code** — `plateau-jacFrob@input` (0.65) vs `cup-RMD@resid6` (0.92). "Plateau variants" are the
methods under test (`plateau-jacFrob`, `plateau-perturbation`); the `selfNLL-grad` confidence *control*
is excluded from the red bar. The baseline pool is {MSP, L2 norm, naive Mahalanobis, cup-RMD, cup-QUE}.
The blue bar wins in every set. (Regenerate with `experiments/make_summary_plot.py`, which derives the
best-per-set directly from `results/auroc_table.csv`.)

## Methods

### Data & Model
- **Model:** GPT-2 small (124M) for the main study; a scaling check additionally runs the `rand-points`
  detector and MSP/Mahalanobis baselines on **gpt2-large (774M, ~6×)** (GPT-2 XL is not in the offline
  cache). Run on GPU (RTX 3090 / A10), VRAM capped per BUDGET.md.
- **In-distribution (ID):** held-out FineWeb text (`data/fineweb_sample.txt`).
- **OOD sets (3):** `random` tokens (uniform over the vocab), `shuffled` tokens (ID tokens with order
  permuted — same unigram statistics, wrong order), and `code` (Python source read offline from
  numpy/torch site-packages — a *real* domain shift of valid text).
- **Measurement points (4):** token-embedding input space, and the residual stream `resid_post` after
  transformer blocks **{3, 6, 9}**. The last-token activation $h \in \mathbb{R}^{768}$ is used.
- **Sample sizes:** $N=200$ sequences per set, `seq_len=64`. Covariance baselines are fit on a separate
  **1000** ID sequences. The ID split is the **canonical split** (see below), saved to
  `results/split/canonical_split.npz` and shared byte-for-byte by the plateau table and the
  real-cupbearer table.

**What "canonical split" means.** The FineWeb ID pool is shuffled once with a fixed permutation
`randperm(seed=7)` and cut into two disjoint index sets: the first 1000 sequences (`fit = perm[:1000]`)
are the *reference* used to fit every ID statistic (the Gaussian $(\mu,\Sigma)$ for Mahalanobis, the
cupbearer detectors), and the next 200 (`test = perm[1000:1200]`) are the held-out *ID test* examples
scored against the OOD sets for AUROC. It is "canonical" because it is the **one fixed split every method
and every table uses** — the plateau/standard-baseline table, the vendored-cupbearer table, and the
real-cupbearer-package table all read these exact same indices. Fixing the seed makes the ID fit
reproducible; sharing the indices byte-for-byte makes every method-vs-method comparison strictly
apples-to-apples (same ID examples, same held-out set), which is what an earlier review required.

### Evaluation metric
All scores are oriented *a priori* so that **higher = more OOD** (no post-hoc sign flipping). Detection
quality is the area under the ROC curve, equal to the probability a random OOD example outscores a
random ID example:

```math
\mathrm{AUROC} = \Pr\big(s(x_{\text{OOD}}) > s(x_{\text{ID}})\big)
   = \frac{1}{N_{\text{ID}}N_{\text{OOD}}}\sum_{i}\sum_{j} \mathbb{1}\!\left[s(x^{\text{OOD}}_j) > s(x^{\text{ID}}_i)\right].
```

An $\mathrm{AUROC} < 0.5$ therefore means the signal is **reversed** for that set (ID scores higher than
OOD). With $N=200$, sampling noise on AUROC is $\approx \pm 0.035$; gaps below ~0.05 are not significant.

### Plateau scores (the methods under test)
Let $h$ be the activation at the measurement point and $p(\cdot\mid x)=\mathrm{softmax}(f(h))$ the
next-token distribution obtained by continuing the forward pass from $h$.

**plateau-jacFrob** (the genuine metric) — Frobenius norm of the Jacobian of the log-probabilities
w.r.t. $h$, via a Hutchinson estimator with $k=4$ random standard-Gaussian output directions $v_i$:

```math
s_{\text{jacFrob}}(x) \;=\; \Big\| \tfrac{\partial \log p(\cdot\mid x)}{\partial h} \Big\|_F
   \;\approx\; \sqrt{\tfrac{1}{k}\sum_{i=1}^{k}\Big\| \tfrac{\partial \langle v_i,\,\log p\rangle}{\partial h} \Big\|_2^2 }.
```

A label-free measure of output-distribution flatness: **flatter (lower) = in-distribution**.

**plateau-perturbation** — mean next-token KL divergence after $M=16$ random unit perturbations of
fixed magnitude $\epsilon=6$ applied at $h$:

```math
s_{\text{pert}}(x) \;=\; \frac{1}{M}\sum_{j=1}^{M} D_{\mathrm{KL}}\!\big(p(\cdot\mid h)\,\big\|\,p(\cdot\mid h+\epsilon u_j)\big),\qquad \|u_j\|_2=1.
```

Sharper response (higher KL) = more OOD. The magnitude $\epsilon=6$ is fixed for the main table; its
sensitivity is scanned separately (see *Epsilon sensitivity* under Results) over
$\epsilon\in\lbrace0.25,\dots,24\rbrace$.

**selfNLL-grad** (transparency control = iter-1's mislabeled "jacobian") — gradient norm of the
model's own argmax negative log-likelihood; kept to show it is confidence-adjacent:

```math
s_{\text{selfNLL}}(x) \;=\; \Big\| \tfrac{\partial\,[-\log p(\hat y\mid x)]}{\partial h} \Big\|_2,
   \qquad \hat y = \arg\max_y p(y\mid x).
```

**rand-points** (operator request 2026-07-09, forward-only) — "OOD detection with randomly sampled
points in the residual stream". Sample $K$ points around $h$, $h_k = h + \sigma z_k$ with
$z_k\sim\mathcal{N}(0,I)$ and $\sigma = 0.1\Vert h\Vert$, continue the forward pass from each to get
$p_k = p(\cdot\mid h_k)$, and summarise their spread two ways — **dispersion** (the epistemic /
"plateau-width" term, a.k.a. BALD mutual information) and **entropy** of the mean:

```math
s_{\text{disp}}(x) = \frac{1}{K}\sum_{k=1}^{K} D_{\mathrm{KL}}\!\big(p_k \,\Vert\, \bar p\big),
\qquad
s_{\text{ent}}(x) = -\sum_y \bar p(y)\log \bar p(y),
\qquad \bar p = \frac{1}{K}\sum_{k=1}^{K} p_k .
```

Both are oriented higher = more OOD. The plateau hypothesis predicts **low** dispersion on a flat
in-distribution plateau. This differs from `plateau-perturbation` (distance from the *clean* output):
`rand-points-disp` measures the spread *among the sampled outputs*, a Monte-Carlo epistemic estimate.

### Baselines
**MSP** — one minus the maximum softmax probability:

```math
s_{\text{MSP}}(x) = 1 - \max_y p(y\mid x).
```

*Why it detects OOD:* a model trained on in-distribution text is, on average, **more confident** on
inputs like its training data — one next-token candidate takes most of the probability mass, so
$\max_y p$ is high and $s_{\text{MSP}}$ is low. On OOD inputs the model is more often uncertain, the
softmax is flatter, $\max_y p$ drops, and $s_{\text{MSP}}$ rises. So higher $s_{\text{MSP}}$ = more
OOD. This is the classic maximum-softmax-probability baseline (Hendrycks & Gimpel, 2017). Its failure
mode is exactly the `code` set: GPT-2 can be *confidently wrong* on a fluent but out-of-domain input,
which is why MSP collapses to 0.359 there while distance-based baselines still fire.

**L2 norm** — activation magnitude at the point: $s_{\text{L2}}(x) = \Vert h\Vert_2.$

**Mahalanobis** — squared distance to a Gaussian $(\mu,\Sigma)$ fit on 1000 ID activations:

```math
s_{\text{maha}}(x) = (h-\mu)^\top \Sigma^{-1} (h-\mu).
```

**cup-RMD** — cupbearer's relative Mahalanobis: the ID-class distance minus a background-class
distance $(\mu_0,\Sigma_0)$ fit on the pooled data, which cancels generic-norm effects:

```math
s_{\text{RMD}}(x) = (h-\mu)^\top \Sigma^{-1}(h-\mu) \;-\; (h-\mu_0)^\top \Sigma_0^{-1}(h-\mu_0).
```

**cup-QUE** — cupbearer's Quantum-Entropy / SPECTRE detector: a quadratic form on the ID-whitened
activation $\tilde h=\Sigma^{-1/2}(h-\mu)$ that up-weights directions of excess untrusted covariance
$\tilde\Sigma$,

```math
s_{\text{QUE}}(x) \;\propto\; \tilde h^\top\,\exp\!\big(\alpha(\tilde\Sigma - I)\big)\,\tilde h.
```

It is fit **once** on the ID set (used as both trusted and untrusted reference) and applied uniformly.

Baselines `cup-RMD` / `cup-QUE` are reported two ways: **vendored** (cupbearer's detector math copied
verbatim into `experiments/cupbearer_helpers.py`, in `auroc_table.csv`) and from the **real cupbearer
package** run in an isolated conda env `cupenv` (`auroc_cupbearer.csv`). On the canonical split the two
agree essentially exactly for RMD (code@resid6: 0.918 = 0.918); for QUE the real package is the
reference (see Limitations).

## Results
Full machine-readable numbers: `results/auroc_table.csv` (87 rows) and `results/auroc_cupbearer.csv`
(48 rows). ROC and score-distribution plots: `results/plots/`. The pivot table and per-set summary are
in **RESULTS.md**. Verdict per OOD set:

| OOD set | best plateau | best baseline | plateau beats baselines? |
|---|---|---|---|
| **random** | plateau-jacFrob@input **0.734** | MSP **0.932** (L2@input 0.859, maha@input 0.834) | **NO** |
| **shuffled** | plateau-perturbation@resid3 **0.534** | MSP **0.872** | **NO** (jacFrob reversed, 0.07–0.37) |
| **code** | plateau-jacFrob@input **0.649** | cup-RMD@resid6 **0.918** (maha@resid6 0.913, cup-QUE 0.910) | **NO** (MSP collapses to 0.359) |

- **The genuine plateau metric is weak.** `plateau-jacFrob` peaks at 0.734 (random@input) / 0.649
  (code@input) and is *reversed* on shuffled and in deep residual layers (ID is locally steeper than
  OOD there). `plateau-perturbation` is similarly weak (≤0.70).
- **iter-1's "strong jacobian" was a confidence signal in disguise.** The renamed `selfNLL-grad` scores
  random 0.923 ≈ MSP 0.932 and **collapses on code (≈0.52) just like MSP (0.359)** — it tracks model
  confidence, not plateau geometry.
- **Properly-powered baselines dominate.** MSP wins synthetic OOD. On the code domain shift MSP
  collapses, but cupbearer relative-Mahalanobis (cup-RMD@resid6 0.918) and well-fit naive Mahalanobis
  (0.913) are the strongest detectors in the study. With a 1000-sample fit, Mahalanobis does **not**
  collapse in deep layers.
- **Internal vs input-space:** for the genuine `plateau-jacFrob`, **input-space is best** and the
  residual stream is worse/reversed → **no value in measuring plateau-ness internally**. (The strong
  baselines are the opposite: Mahalanobis/cup-RMD need a deep residual layer to catch the code shift.)

### Epsilon sensitivity (robustness of plateau-perturbation)
The main table fixes the perturbation magnitude at $\epsilon=6$; an operator asked whether a *different*
$\epsilon$ makes plateau-perturbation competitive. `experiments/eps_scan.py` sweeps
$\epsilon\in\lbrace0.25,0.5,1,2,4,6,8,12,16,24\rbrace$ at all four measurement points on the same canonical split
(same 16 random directions reused across magnitudes, so the $\epsilon=6$ column reproduces the main table
exactly). Full numbers in `results/auroc_perturbation_eps.csv` (120 rows).

![plateau-perturbation AUROC vs epsilon](results/plots/perturbation_eps_scan.png)

- **Residual-stream points are nearly $\epsilon$-insensitive** (resid3/6/9 vary <0.05 across two decades).
- **Input space is $\epsilon$-sensitive and $\epsilon=6$ was a poor choice there:** random@input is
  ~0.87 for $\epsilon\le2$, 0.44 at $\epsilon=6$, and *reverses* to 0.12 at $\epsilon\ge8$ — the fixed
  value sat on the cliff, understating the metric.
- **Even an oracle $\epsilon$ (best $\epsilon$ and point per set, an upper bound that peeks at labels)
  loses on every set:** random 0.873 (input, $\epsilon=0.25$) < MSP 0.932; shuffled 0.554 (input,
  $\epsilon=4$) < MSP 0.872; code 0.614 (input, $\epsilon=24$) < cup-RMD@resid6 0.918. No single $\epsilon$
  is jointly best (random wants small, code wants large). The negative verdict is unchanged and
  strengthened — "wrong $\epsilon$" does not rescue plateau-perturbation.

### Randomly-sampled residual points + GPT-2 scaling (operator request 2026-07-09)
An operator asked to *"try GPT-2 XL, and OOD detection with randomly sampled points in the residual
stream."* We added the `rand-points` detector (Methods) and re-ran it plus MSP/Mahalanobis on gpt2-large
(GPT-2 XL is not cached). Full numbers: `results/auroc_randpoints.csv` (78 rows).

![rand-points vs baselines — GPT-2 small](results/plots/randpoints_gpt2.png)
![rand-points vs baselines — GPT-2 large](results/plots/randpoints_gpt2-large.png)

**Observation.** Best AUROC over measurement points: the genuine dispersion signal `rand-points-disp` is
weak and mostly *reversed* — random 0.518 / shuffled 0.267 / code 0.707 (gpt2), and 0.436 / 0.256 / 0.596
(gpt2-large) — losing to Mahalanobis on `code` (0.913 / 0.842) and to MSP on the synthetic sets. The
`rand-points-ent` variant is near-perfect on `random`/`shuffled` (up to 1.000) but **collapses/reverses
on `code`** (gpt2 0.566; gpt2-large 0.30–0.43), the same failure mode as MSP (0.359 / 0.326). At ~6× scale
MSP rises to 0.957 / 0.914 on synthetic OOD and Mahalanobis@resid18 (0.842) still leads on `code`.

**Interpretation.** `rand-points-disp` measures epistemic dispersion, the honest "plateau-width" idea;
its reversal on synthetic OOD suggests random/shuffled inputs put the model in an *already-saturated*
(near-uniform, low-dispersion) output state, the opposite of the flat-ID / scattered-OOD hypothesis.
`rand-points-ent` is simply predictive entropy — a confidence baseline — which is why it tracks MSP and
shares its confident-wrong collapse on `code`. The scaling check indicates the negative result is not an
artifact of GPT-2 small's capacity.

**Limitations.** gpt2-large used slightly smaller $K$/$N$ (8/150 vs 16/200) under the shared budget;
GPT-2 XL itself was not run (not in the offline cache). Two model sizes, three OOD sets.

**Next check.** Fitting `rand-points-ent`/`disp` as calibrated detectors on a non-code real domain shift,
and running the actual GPT-2 XL if its weights become available, would test whether the collapse-on-code
pattern is domain-specific.

## Conclusion
Plateau-ness, measured honestly as the flatness of the output distribution, is a **weak OOD detector
that is beaten by standard *and* cupbearer baselines on every OOD set**, and measuring it inside the
residual stream provides no advantage over the simpler input-space signal. The best detectors are MSP
(synthetic OOD) and relative/naive Mahalanobis in a deep residual layer (real domain shift). The
operator's randomly-sampled-residual-points detector fits the same pattern — its epistemic-dispersion
term is weak/reversed, and its entropy term is a confidence baseline that collapses on the real domain
shift — and the negative result **holds at ~6× scale (gpt2-large)**. This is a clean negative result,
which PLAN.md declares complete and acceptable.

## Limitations / honest caveats
- $N=200 \Rightarrow \pm0.035$ AUROC noise; the qualitative ranking (baselines > plateau on every set)
  is far larger than this and robust to it.
- **cup-QUE scope.** The *vendored* cup-QUE rows in `auroc_table.csv` are **transductive** (each scored
  set used its own untrusted covariance) and are superseded/caveated. The **real-package** cup-QUE
  (`auroc_cupbearer.csv`) is fit once on ID and applied uniformly — a consistent fixed-function
  detector — but should be read as *a cupbearer-code detector variant* (untrusted_data = the ID set),
  not necessarily the definitive SPECTRE/QUE anomaly-mixture protocol; with 768-dim activations its
  covariance is rank-limited. This does not affect the conclusion: cup-RMD and naive/cup Mahalanobis
  already beat plateau strongly on code.
- The real-package run reuses precomputed last-token activations (not cupbearer's full task/data
  harness); this isolates the *detector* comparison, which is exactly the baseline question asked.
- One model (GPT-2 small) and three OOD sets. A broader model/OOD sweep could shift magnitudes but is
  unlikely to overturn a result this one-sided.
- The shared base env intermittently lost `transformers`/`tokenizers` (reinstalled `--no-deps` each
  time, torch/numpy untouched and verified); details in `experiments/ENV_NOTES.md`.
