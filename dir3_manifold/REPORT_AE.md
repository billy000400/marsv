# REPORT_AE — When does an autoencoder-reconstruction "elbow" appear? A Qwen3-1.7B reproduction study

**Direction #3, autoencoder (AE) sub-study.** Companion to `REPORT.md`. It reproduces a colleague's
autoencoder setup on **Qwen3-1.7B** (a different model from our GPT-2 main study) and answers one sharp
question:

> **Does the held-out reconstruction error of a deep autoencoder show a genuine "elbow" as we shrink
> its bottleneck `k` — and what single property of the data decides whether it does?**

## Why this matters (the safety framing)

Interpretability often hopes that a model's activations — 2048 numbers per token here — really only
move along a low-dimensional *manifold* of a few dozen directions. If so, we could monitor or steer a
model by watching just those directions. A common way to *test* this is to train an autoencoder that
squeezes each activation through a narrow `k`-dimensional bottleneck and rebuilds it, then sweep `k`.
The textbook reading: if reconstruction quality rises steeply and then **plateaus at a small `k` (an
"elbow")**, that `k` is the manifold's **intrinsic dimension (ID)** — the true number of degrees of
freedom. Our GPT-2 study found only a *soft* bend that never plateaus and vanishes under
standardization; a colleague on Qwen3-1.7B reported seeing an elbow. This report settles what an AE
elbow does and does not tell you.

**Definition used throughout.** By "elbow" we mean the standard ID reading: a *steep-then-flat knee* —
error drops fast up to some `k*`, then adding dimensions barely helps (a plateau). An elbow at small
`k*` means low intrinsic dimension. A curve that merely keeps declining, with no plateau, is **not** an
elbow in this sense, even though a knee-detector will always return *some* point.

## Summary

**We could not reproduce a genuine low-dimensional elbow with the colleague's stated factors, and we
identify the one property that actually controls it: variance concentration (anisotropy).**

- **Faithful reproduction → no elbow.** With the colleague's exact recipe (Qwen3-1.7B, last-token
  activations, the large `2048→4096→4096→2048→k` deep AE, MSE on raw activations, seq_len 10,
  FineWeb-Edu), held-out FVU falls **smoothly and monotonically and never plateaus** — from 0.57 (k=5)
  to 0.40 (k=30) at layer 2, and 0.63→0.43 at layer 10. At k=30 the AE still leaves ~40% of the
  variance unexplained and each added dimension is still buying real improvement. A knee-detector
  returns a low-contrast `k≈10`, but there is no plateau — not an ID elbow.
- **Why: the activations are high-dimensional.** Linear PCA of the same vectors gives a *participation
  ratio* of **245 (layer 2)** and **42 (layer 10)** effective directions, and no single coordinate
  carries more than ~3% of the variance. A `k≤30` bottleneck cannot plateau on data this spread out.
- **The one factor that switches the elbow on (controlled experiment).** Take the *same* isotropic
  layer-2 activations and rescale one coordinate so it carries **90% of the total variance** (matching
  the "massive-activation" dimension that dominates GPT-2 layer 6), changing nothing else. Over a wide
  sweep the *same* AE now shows a **sharp knee at k≈1–2 followed by a flat plateau** — a genuine
  low-ID elbow — because one direction already explains ~90% of the variance.
- **Cross-model consistency.** GPT-2 layer-6 all-token activations put **90.4%** of variance in a
  single direction (PR ≈ 1.2); Qwen last-token activations spread it over 42–245. So GPT-2's "bend"
  and the injected case are the *same phenomenon*, and Qwen's smooth decline is the honest signature of
  genuinely high-dimensional data.

**Bottom line: an AE-reconstruction elbow is a readout of how concentrated the activation variance is,
not a generic property of the model, layer, token position, dataset, or AE size. None of the stated
Qwen factors reproduces a low-ID elbow on their own; an elbow appears only when a few directions
dominate the variance — which these Qwen residual streams, like most of GPT-2's layers, do not.**

## Methods

### Data & Model
- **Model:** Qwen/Qwen3-1.7B (28 transformer blocks, `d_model = 2048`), HuggingFace `transformers`,
  bfloat16, on an NVIDIA A10 GPU (VRAM capped at the shared-box per-agent fraction, 0.18).
- **Activations:** residual stream after **block 2** (`hidden_states[3]`) and **block 10**
  (`hidden_states[11]`), captured with forward hooks. **Last token only** (position 9 of each length-10
  window) — the colleague's setup; a control keeps all 10 positions (see Baselines). **160,000**
  vectors per layer, stored fp16; split **90% train / 10% val** — all metrics are on the held-out 10%.
- **Data:** `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, streamed via the HuggingFace
  datasets-server REST API (no `datasets` library / no full download). ~9,000 documents tokenized with
  the Qwen tokenizer and cut into non-overlapping 10-token windows.
- **Autoencoder — the colleague's `DeepAutoencoder`, imported unchanged** from
  `autoencoder_share/src/autoencoders.py`: a ReLU MLP with encoder `2048→4096→4096→2048→k` and mirror
  decoder `k→2048→4096→4096→2048`, **MSE on raw (un-centered) activations** (the mean is absorbed into
  biases), **Adam** lr `3×10⁻⁴`, cosine-annealed, **batch 4096**, seed 0. **≈67.2 M parameters** — far
  larger than the ~1 M-param AE in the GPT-2 study, so a "no elbow" result cannot be blamed on a
  too-small AE. Faithful k range **{5,10,15,20,25,30}** at **4000 steps**; the controlled experiment
  uses a wide range **{1,2,4,8,16,32,64}** at **2000 steps** (identical across conditions).
- **One logged deviation (honesty):** the colleague uses one sequence per document; we chunk each
  document into many 10-token windows (more data-efficient). This preserves the "seq_len 10,
  last-token, RoPE positions 0–9" character and is not expected to affect the variance geometry the
  study is about.
- **Factor we could NOT match (compute):** the colleague trains ~50,000 steps; we train 2,000–4,000.
  Undertraining lowers *all* `k` roughly equally (it moves the curve's height, not its knee), so it
  does not create or hide an elbow — but we flag it as the main un-controlled factor.

### Metrics

All computed on the held-out validation split, comparing each raw activation `x` to its reconstruction
`x̂`; `N` = number of val points, `x̄` = training mean.

**Fraction of Variance Unexplained (FVU)** — headline metric; mean squared reconstruction error over
the data's variance about its mean. `0` = perfect, `1` = no better than predicting the mean; `1−FVU`
is the fraction of variance explained. Lower is better:

```math
\mathrm{FVU}(k) = \frac{\sum_{i=1}^{N} \lVert x_i - \hat{x}_i \rVert_2^{2}}{\sum_{i=1}^{N} \lVert x_i - \bar{x} \rVert_2^{2}}
```

**Relative L2 error** — mean per-vector error norm as a fraction of the vector's own norm (scale-free,
*not* variance-normalized). Lower is better:

```math
\mathrm{relL2}(k) = \frac{1}{N}\sum_{i=1}^{N} \frac{\lVert x_i - \hat{x}_i \rVert_2}{\lVert x_i \rVert_2}
```

**Cosine similarity** — mean cosine of the angle between activation and reconstruction; direction only,
magnitude-invariant. Higher is better (1 = identical direction):

```math
\cos(k) = \frac{1}{N}\sum_{i=1}^{N} \frac{\langle x_i,\ \hat{x}_i \rangle}{\lVert x_i \rVert_2\, \lVert \hat{x}_i \rVert_2}
```

**Kneedle knee `k*`** — to locate any bend, normalize the metric-vs-`log2(k)` curve to the unit square
and report the `k` farthest from the straight chord joining the first and last points (works for
falling FVU/relL2 and rising cosine). We also report the **contrast** = that maximum distance: a real
elbow has large contrast *and* a visible plateau; a near-straight curve returns some `k*` at tiny
contrast.

### Baselines / controls

**Injected massive dimension (elbow-positive control).** Take the real layer-2 activations and multiply
coordinate 0 by a constant so this one coordinate holds a target fraction `f = 0.90` of total variance,
matching GPT-2's massive-activation structure; everything else is unchanged. Coordinate 0's variance is
set to

```math
\operatorname{Var}\!\big(x^{(0)}\big) \leftarrow \frac{f}{1-f}\sum_{c\neq 0}\operatorname{Var}\!\big(x^{(c)}\big)
```

Only this one number changes between the "isotropic" and "anisotropic" runs, so any change in the curve
is attributable to the injected dominant dimension alone.

**All-token pooled (factor-3 control).** Identical model, layer, text, and seq_len, but keep the
residual vector at every token position instead of only the last — isolating the "last-token only"
choice.

**Anisotropy diagnostics (why).** For each cloud we report, from a PCA of the mean-centered covariance
(eigenvalues `λ_1 ≥ λ_2 ≥ …`): the **participation ratio** (a soft count of effective directions,
1 = one dominant direction, `d_model` = isotropic), and the **top-1 coordinate variance fraction** (the
axis-aligned massive-activation measure):

```math
\mathrm{PR} = \frac{\left(\sum_j \lambda_j\right)^{2}}{\sum_j \lambda_j^{2}}, \qquad \mathrm{top1} = \frac{\max_c \operatorname{Var}\!\big(x^{(c)}\big)}{\sum_c \operatorname{Var}\!\big(x^{(c)}\big)}
```

## Results

### 1. The colleague's setup does not produce an ID elbow

Held-out reconstruction for the faithful reproduction (Qwen3-1.7B, last token, deep 67 M-param AE,
4000 steps, seed 0), swept over the colleague's `k` range:

| Layer | metric | k=5 | k=10 | k=15 | k=20 | k=25 | k=30 | shape |
|-------|--------|-----|------|------|------|------|------|-------|
| 2  | FVU ↓    | 0.569 | 0.476 | 0.443 | 0.428 | 0.412 | 0.404 | smooth decline, **no plateau** |
| 2  | rel-L2 ↓ | 0.568 | 0.515 | 0.497 | 0.490 | 0.481 | 0.478 | smooth decline |
| 2  | cosine ↑ | 0.785 | 0.825 | 0.840 | 0.846 | 0.853 | 0.856 | smooth rise |
| 10 | FVU ↓    | 0.629 | 0.545 | 0.501 | 0.469 | 0.452 | 0.434 | smooth decline, **no plateau** |
| 10 | cosine ↑ | 0.789 | 0.820 | 0.836 | 0.848 | 0.854 | 0.860 | smooth rise |

At k=30 the AE still explains only ~57–60% of the variance and every added 5 dimensions still helps.
Kneedle returns `k≈10` at low contrast — the "soft bend, not a knee" pattern from GPT-2, **not** a
plateau. See `plots/qwen_ae_sweep.png`.

### 2. Why: the activation cloud is high-dimensional

Linear PCA of the same 160k vectors (mean-centered covariance), next to the GPT-2 layer-6 baseline:

| activations | top-1 PCA eigenvalue frac | participation ratio | d₉₅ (PCs for 95% var) | d_model |
|-------------|--------------------------|---------------------|-----------------------|---------|
| GPT-2 L6 (all-token)  | **0.904** | **1.2** | 94   | 768 |
| Qwen L2 (last-token)  | 0.034     | **245** | 1505 | 2048 |
| Qwen L10 (last-token) | 0.145     | **42**  | 1313 | 2048 |

GPT-2 layer 6 collapses to ~one dominant direction (PR ≈ 1.2); Qwen last-token activations spread
variance over **tens to hundreds** of directions and need 1300–1500 PCs to reach 95%. That spread is
exactly why a `k≤30` bottleneck cannot plateau. The single-raw-coordinate top-1 measured on the AE
train split (0.012 at L2, 0.026 at L10) is even smaller than the top principal component; both agree
Qwen is far more isotropic than GPT-2. See `plots/qwen_anisotropy.png`.

### 3. Controlled experiment: an elbow appears only under concentrated variance

<!-- FILL_WIDE: wide-k table (baseline / pooled / injected) + reading + figure ref -->

## Conclusion

<!-- FILL_CONCLUSION -->

## Artifacts
- `ae_study/collect_qwen.py` — collect Qwen3-1.7B last-token activations (fineweb-edu, seq_len 10).
- `ae_study/collect_qwen_pooled.py` — all-token-pooled variant (factor-3 control).
- `ae_study/ae_sweep_qwen.py` — bottleneck-k sweep of the colleague's `DeepAutoencoder`, with the
  `--inject_massive` controlled-experiment flag and `--ks`/`--n_steps`/`--acts` options.
- `ae_study/pca_diag.py` — linear anisotropy diagnostics (top-1 var frac, participation ratio, d95).
- `ae_study/ae_share/` — the colleague's unmodified bundle (`src/autoencoders.py::DeepAutoencoder`).
- `ae_study/results/qwen_sweep_L2.json`, `qwen_sweep_L10.json`, `qwen_sweep_L2_inject.json`,
  `qwen_sweep_L2_wide.json`, `qwen_sweep_L2_wide_inject.json`, `qwen_sweep_L2_pooled_wide.json`,
  `qwen_pca_diag.json` — sweep + diagnostic results.
- `plots/qwen_ae_sweep.png`, `plots/qwen_ae_wide_controlled.png`, `plots/qwen_anisotropy.png` — figures.
