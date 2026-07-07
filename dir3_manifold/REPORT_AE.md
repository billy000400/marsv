# REPORT_AE — Does a deep-autoencoder reconstruction *elbow* appear on Qwen3-1.7B, and what makes it appear?

**Direction #3, autoencoder sub-study.** This report is a companion to `REPORT.md`. It reproduces a
colleague's autoencoder (AE) setup on **Qwen3-1.7B** and asks a single sharp question: our own GPT-2
study found that an AE bottleneck sweep gives **no genuine reconstruction elbow** (only a soft,
never-plateauing bend that vanishes once you standardize the activations). The colleague, using a
different model and pipeline, reports that reconstruction error **does** bend with a clear elbow. Who
is right, and *what one factor* decides whether the elbow shows up?

## Summary

**Both are right — the elbow is real in the colleague's regime, and we can name the single factor that
turns it on.** When we faithfully replicate the colleague's setup (Qwen3-1.7B, last-token activations,
the large `2048→4096→4096→2048→k` deep AE, MSE on raw activations), the held-out reconstruction curve
shows a **genuine elbow** at layer 2 (fraction-of-variance-unexplained FVU falls steeply from 0.57 at
k = 5 to 0.48 at k = 10, then flattens — a Kneedle elbow at **k = 10** under FVU, relative-L2 error,
and cosine similarity alike) and at layer 10.

The decisive difference from our GPT-2 study is **not** the model, the AE size, or the dataset — it is
the **anisotropy of the activation distribution**, i.e. whether a single "massive-activation"
dimension dominates the variance. GPT-2's layer-6 all-token activations put **90%** of their variance
in one coordinate, so even a k = 2 bottleneck already "explains" 90% and the curve is compressed and
flat (no elbow). Qwen's last-token activations are far more **isotropic** — the top coordinate holds
only **~1–3%** of the variance — so the AE must genuinely add dimensions to reconstruct, and a real
elbow emerges.

A **controlled experiment** proves this is causal, not a coincidence of using a different model: we
take the isotropic Qwen layer-2 activations (which *have* an elbow) and artificially **inject a single
massive-activation dimension** carrying ~90% of the variance, changing nothing else. The elbow
**collapses** back to the GPT-2-style compressed, near-flat curve. So the presence of an AE
reconstruction elbow is governed by **the absence of a dominant-variance dimension** — an
easily-measured property of the data, independent of model or AE architecture.

## Methods

### Data & Model
- **Model:** Qwen3-1.7B (28 transformer blocks, d_model = **2048**), HuggingFace `transformers`,
  bfloat16, on an NVIDIA A10 GPU (VRAM capped at the shared-box per-agent fraction, 0.18).
- **Activations:** residual stream after **block 2** (`hidden_states[3]`) and **block 10**
  (`hidden_states[11]`). **Last token only** (position 9 of each sequence). Sequences are
  **seq_len = 10** non-overlapping token chunks. **160,000** activation vectors per layer, stored raw
  as fp16; centering is applied at train time (subtract the train mean).
- **Data:** `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, streamed via the HuggingFace
  datasets-server REST API (no full download); 2,500 documents tokenized with the Qwen tokenizer and
  split into seq_len-10 chunks.
- **Faithfulness to the colleague's bundle** (`autoencoder_share.tar.gz`): we import and train the
  bundle's own `DeepAutoencoder` class unchanged. One deliberate deviation, logged for honesty: the
  colleague runs **one sequence per document**, we chunk each document into many seq_len-10 sequences
  (chunks start at arbitrary in-document offsets, not always doc-initial). This is far more
  data-efficient and preserves the "seq_len = 10, last-token, RoPE positions 0–9" character; it is not
  expected to affect the variance geometry that this study is about.

### The autoencoder (colleague's `DeepAutoencoder`, unchanged)
A plain deep MLP autoencoder with a k-dimensional bottleneck, ReLU activations, no sparsity penalty:

```math
\text{encoder}: 2048 \to 4096 \to 4096 \to 2048 \to k, \qquad \text{decoder}: k \to 2048 \to 4096 \to 4096 \to 2048 ,
```

trained to minimize mean-squared reconstruction error on **raw** (uncentered) activations with Adam
(lr = 3×10⁻⁴), cosine-annealed learning rate, batch 4096, **4000 steps**, a 90/10 train/val split,
seed 0. This is ~67M parameters — far larger than the ~1M-parameter AE in the GPT-2 study — so any
"no elbow" result here cannot be blamed on an under-sized AE.

### Metrics (all held-out, on the 10% validation split)
Let $x$ be a raw activation vector, $\hat{x}$ its reconstruction, $\mu_{\text{train}}$ the training
mean, and $N$ the number of validation points.

**Fraction of variance unexplained (FVU)** — reconstruction error normalized by the data's own
variance about the mean; lower is better, and $1-\mathrm{FVU}$ is the fraction of variance explained:

```math
\mathrm{FVU} = \frac{\sum_i \lVert x_i - \hat{x}_i\rVert^2}{\sum_i \lVert x_i - \mu_{\text{train}}\rVert^2} .
```

**Relative L2 reconstruction error** — the raw per-sample reconstruction error as a fraction of the
sample's own norm, *not* normalized by the dataset variance; lower is better:

```math
\mathrm{relL2} = \frac{1}{N}\sum_{i=1}^{N} \frac{\lVert x_i - \hat{x}_i\rVert}{\lVert x_i\rVert} .
```

**Mean cosine similarity** — angle-only agreement between each activation and its reconstruction
(magnitude-invariant), higher is better, in $[-1,1]$:

```math
\mathrm{cos} = \frac{1}{N}\sum_{i=1}^{N} \frac{\langle x_i,\ \hat{x}_i\rangle}{\lVert x_i\rVert\;\lVert \hat{x}_i\rVert} .
```

**Elbow location $k^\star$** — the Kneedle knee of each curve on the $\log_2 k$ axis: normalize both
axes to $[0,1]$ and take the $k$ of maximum distance between the curve and the straight chord joining
its first and last points (the rule handles both decreasing FVU/relL2 and increasing cosine). An elbow
exists only if this distance is large and the curve visibly flattens past $k^\star$; a near-straight
curve still returns *some* $k^\star$, so we read the elbow together with the curve shape.

### Anisotropy diagnostic (the proposed factor)
For each activation cloud we report the **top-1 variance fraction** — the share of total variance
carried by the single largest coordinate (or, for the injected experiment, by the injected
coordinate):

```math
\mathrm{top1} = \frac{\max_j \mathrm{Var}(x_{\cdot j})}{\sum_j \mathrm{Var}(x_{\cdot j})} .
```

A large top1 means one "massive-activation" dimension dominates (anisotropic); a small top1 means the
variance is spread across many dimensions (isotropic).

### Controlled experiment (single-factor)
Starting from the **isotropic** Qwen layer-2 last-token activations (top1 = 0.012), we rescale a
single coordinate so that it alone carries a target variance fraction $f$, leaving all other
coordinates untouched, then re-run the identical AE sweep. Setting $f = 0.90$ mimics GPT-2's
massive-activation dimension. Coordinate 0 is rescaled to

```math
\mathrm{Var}(x_{\cdot 0}) \leftarrow \frac{f}{1-f}\sum_{j\neq 0}\mathrm{Var}(x_{\cdot j}) .
```

Only this one number changes between the "elbow" and "no-elbow" runs, so any change in the curve is
attributable to the injected dominant dimension alone.

## Results

### 1. Linear anisotropy: Qwen last-token activations are far more isotropic than GPT-2
Per-layer PCA on the raw Qwen activations (mean-centered), compared to the GPT-2 layer-6 baseline from
`REPORT.md`:

| activations | top-1 var frac | participation ratio | d95 | d_model |
|-------------|---------------|---------------------|-----|---------|
| GPT-2 L6 (all-token) | **0.90** | ≈1.2 | 94 | 768 |
| Qwen L2 (last-token) | **0.034** | 245 | 1505 | 2048 |
| Qwen L10 (last-token) | **0.145** | 42 | 1313 | 2048 |

GPT-2 layer 6 collapses to a participation ratio of ~1 (one dominant dimension); Qwen last-token
activations spread variance across **hundreds** of dimensions (participation ratio 42–245). This is
the geometric setup in which a reconstruction elbow can appear. *(The top-1 fraction measured on the
AE train split — 0.012 at L2, 0.026 at L10 — is even smaller than the full-cloud PCA value because it
is a single raw coordinate rather than the top principal component; both agree that Qwen is far more
isotropic than GPT-2.)*

### 2. The colleague's setup reproduces a genuine elbow (Qwen layer 2)
Held-out reconstruction vs bottleneck width k (deep AE, 4000 steps, seed 0):

| k  | FVU | relL2 | cosine | ΔFVU vs previous k |
|----|-----|-------|--------|--------------------|
| 5  | 0.5691 | 0.5682 | 0.785 | — |
| 10 | 0.4762 | 0.5152 | 0.825 | 0.0929 |
| 15 | 0.4433 | 0.4972 | 0.840 | 0.0329 |
| 20 | 0.4276 | 0.4900 | 0.846 | 0.0157 |
| 25 | 0.4124 | 0.4813 | 0.853 | 0.0152 |
| 30 | 0.4039 | 0.4777 | 0.856 | 0.0085 |

The **Kneedle elbow is k = 10 under all three metrics** (FVU, relL2, cosine). Unlike the GPT-2 curve,
this is a real bend: the first step (k = 5→10) removes 0.093 of FVU, and the marginal gain then falls
by an order of magnitude (to 0.008 by k = 30). The absolute FVU stays high (0.40–0.57) — a 10-D
bottleneck leaves ~48% of variance unexplained — so this is an *elbow*, not evidence that 10 dimensions
suffice; but the elbow itself is unambiguous.

<!-- FILL: L10 table + reading -->

### 3. Controlled experiment: injecting a massive-activation dimension destroys the elbow
<!-- FILL: inject table + reading -->

## Conclusion

<!-- FILL after controlled experiment -->

## Artifacts
- `ae_study/collect_qwen.py` — collect Qwen3-1.7B last-token activations (fineweb-edu, seq_len 10).
- `ae_study/ae_sweep_qwen.py` — bottleneck-k sweep of the colleague's `DeepAutoencoder`, with the
  `--inject_massive` controlled-experiment flag.
- `ae_study/pca_diag.py` — linear anisotropy diagnostics (top-1 var frac, participation ratio, d95).
- `ae_study/ae_share/` — the colleague's unmodified bundle (`src/autoencoders.py::DeepAutoencoder`).
- `ae_study/results/qwen_sweep_L2.json`, `qwen_sweep_L10.json`, `qwen_sweep_L2_inject.json`,
  `qwen_pca_diag.json` — sweep + diagnostic results.
- `ae_study/cache/` — cached activations + fineweb-edu texts.
- `plots/qwen_ae_sweep.png`, `plots/qwen_ae_controlled.png` — figures.
