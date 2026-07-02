# ColdSteer — on-manifold correction for activation steering

> Final, presentable, current-best only (history in CHANGELOG.md).

## Summary

**Activation steering** is a popular way to control a language model's behavior at
inference time: you find a direction `v` in the model's hidden state that corresponds to a
concept (say, "positive sentiment"), then add `α·v` to the activations as the model runs,
where `α` sets the strength. The problem is that pushing hard on `α` drags the activation
away from the region of activations the model actually produces on real text — it goes
**off-manifold** — and the model's fluency collapses.

This direction asks whether a small **corrector** can preserve the intended steering effect
while pulling the activation back onto the manifold. This report covers two steps.

**Step 1 — quantifying the failure mode.** Using GPT-2 small and a sentiment steering
direction, we show that as steering strength `α` grows, the steered activation moves
monotonically off-manifold by three independent measures, and the model's language modeling
loss degrades by up to +2.78 nats (≈16× perplexity). This establishes both the problem and
the metrics a corrector is judged against.

**Step 2 — a surprising negative result.** We build the natural first corrector: it stays in
the ColdSteer form `ĥ = z + P_{v⊥}r` (a correction orthogonal to `v`, so the steering
projection along `v` is preserved *exactly*), and it uses the **provably optimal** correction
for a Gaussian model of the manifold — the shift that lowers the Mahalanobis distance the most
at matched projection. It does lower that distance (by 22% at `α=8`) and preserves the
projection to the digit. **Yet it makes the language model dramatically worse**, not better:
`ΔLM` rises to +4.2 nats, and even at weak steering where raw steering is nearly harmless
(+0.08 nats) the "corrected" activation is catastrophic (+3.31 nats). The lesson is sharp and
useful: **statistical "on-manifold" and "safe for the LM" are decoupled** — you can reduce the
off-manifold distance while multiplying LM loss ~40×. An effective corrector must be trained
against the **downstream LM objective**, not a manifold-distance surrogate.

## Methods

### Data & Model

**Model.** GPT-2 small (124M parameters), via HuggingFace `transformers`. We hook the
**residual stream after block 6** (`resid_post`, the middle of GPT-2's 12 blocks), a 768-dim
vector per token. In HuggingFace terms this is `hidden_states[7]` (index `layer+1`, since
`hidden_states[0]` is the embedding).

**Text data.** FineWeb documents (a large open web-text corpus). We fit activation
statistics on **49,218 token activations** from 400 documents (128 tokens each) and measure
LM degradation on a **held-out 100 documents**.

**Steering vector.** A **DiffMean** sentiment direction: mean activation over 20 clearly
positive sentences minus mean activation over 20 matched negative sentences, at block 6.
It is used in **raw activation units** (not renormalized), giving `|v| = 11.1`; for
reference the mean clean-activation norm is `|h| = 112.2`. Steering strength `α` is therefore
in multiples of one "natural sentiment shift." We form the steered activation for a clean
activation `h`:

```math
z = h + \alpha\, v
```

and sweep `α ∈ {0, 1, 2, 3, 4, 6, 8}`.

### Metrics

All three metrics measure how far steering pushes the activation off the real-text
manifold; **higher = more off-manifold / more damage** for all three.

**Mahalanobis distance `D_M`** — distance from the steered activation to the cloud of real
activations, fit as a single Gaussian with mean `μ` and covariance `Σ` (estimated on the
49,218 clean tokens, with a small ridge `10⁻³·I` added to `Σ` for invertibility). It counts
how many standard deviations, in the whitened activation space, the point sits from typical
activations:

```math
D_M(x) = \sqrt{(x-\mu)^{\top}\, \Sigma^{-1}\, (x-\mu)}
```

We report the mean `D_M` over a 20,000-token sample. The reference value for **real
activations** is `D_M = 27.3`; a steered activation scoring well above this is off-manifold.

**Norm inflation ratio** — how much steering blows up the activation's length relative to a
clean activation:

```math
\rho(\alpha) = \frac{\mathbb{E}\,\lVert z \rVert}{\mathbb{E}\,\lVert h \rVert}
```

A value of 1 means the steered activation has a typical length; `ρ > 1` means it is
abnormally large.

**Δ LM loss** — the practical cost of steering. We patch `z = h + α·v` into `resid_post` at
block 6 for **every token position** during a forward pass and measure the mean next-token
cross-entropy (natural log), then subtract the unsteered baseline `α = 0`:

```math
\Delta\mathrm{LM}(\alpha) = \mathrm{CE}_{\text{next-token}}(z=h+\alpha v)\; -\; \mathrm{CE}_{\text{next-token}}(z=h)
```

`ΔLM` is in nats; `+ln(k)` nats means perplexity got `k×` worse. Larger = more fluency
damage.

**Projection retention (Experiment 2)** — how much of the intended steering edit survives
in the corrected activation, measured as the component of the net edit along the unit
steering direction `v̂ = v/|v|`. This is the quantity a corrector must NOT destroy (destroying
it would just be "turn off the steer"):

```math
\mathrm{retention} = \big\langle\, \hat{h} - h,\; \hat{v} \,\big\rangle
```

For raw steering and any correction of the form `ĥ = z + P_{v⊥}r`, this equals `α|v|` exactly,
so those methods are compared at **matched projection**.

### The corrector and its baselines (Experiment 2)

**ColdSteer parameterization.** The corrector returns a residual `r` and adds only its
component orthogonal to `v`, which guarantees the steering projection is preserved:

```math
\hat{h} = z + P_{v^{\perp}}\, r, \qquad P_{v^{\perp}} = I - \hat{v}\hat{v}^{\top}
```

**`cov_corr` — analytic optimal correction under the Gaussian manifold.** Among all constant
shifts `Δ` that achieve the target projection `⟨Δ, v̂⟩ = α|v|`, the one that minimizes the
whitened movement cost `ΔᵀΣ⁻¹Δ` (i.e. the smallest step in Mahalanobis geometry, hence the
lowest added off-manifold distance) has a closed form:

```math
\Delta \;=\; \Sigma\,\hat{v}\;\frac{\alpha\,\lVert v\rVert}{\hat{v}^{\top}\Sigma\,\hat{v}}
```

This is a rotation of the raw shift `α v` toward how activations actually covary; it satisfies
`Δ - αv ⟂ v`, so it is exactly a ColdSteer residual. Kantorovich's inequality guarantees its
Mahalanobis penalty is `≤` that of raw steering, so it always lowers `D_M`.

**`norm_clip` — norm-clipping baseline.** Rescale each steered activation to the clean mean
norm `|h| = 112.2`: `ĥ = z·|h|/|z|`. Fixes the norm-inflation symptom but does not preserve the
projection exactly.

**`naive_inversion` — negative control.** Set `ĥ = h` (undo the steer). By construction
projection retention is 0 and there is no LM damage; it exists only to confirm the evaluation
is not "rewarded" for silently erasing the steer.

### Baselines

This first experiment has no learned method yet — it characterizes the phenomenon. The
**reference points** are:

- **Unsteered activation (`α = 0`)** — the clean baseline for `ΔLM` (zero by construction)
  and for norm (ratio ≈ 1).
- **Real-activation Mahalanobis (`D_M = 27.3`)** — the on-manifold reference line. A steered
  activation is "off-manifold" precisely when its `D_M` climbs above this.

A learned corrector (next iterations) will be judged on whether it lowers `D_M` and `ΔLM`
at a given steering strength **while preserving the projection of the edit along `v`**.

## Results

![off-manifold phenomenon](plots/01_offmanifold_phenomenon.png)

As steering strength `α` rises from 0 to 8, all three off-manifold measures increase
monotonically:

| α | Mahalanobis `D_M` | `|z|/|h|` | Δ LM loss (nats) |
|---|-------------------|-----------|------------------|
| 0 | 27.3 | 0.98 | 0.00 |
| 1 | 27.8 | 0.98 | +0.08 |
| 2 | 29.2 | 1.00 | +0.32 |
| 3 | 31.4 | 1.03 | +0.74 |
| 4 | 34.1 | 1.07 | +1.22 |
| 6 | 41.0 | 1.17 | +2.11 |
| 8 | 49.0 | 1.30 | +2.78 |

**Interpretation.** The damage is small at weak steering (`α ≤ 2`: `ΔLM < 0.35` nats,
`D_M` barely above the real-activation reference) but accelerates. By `α = 8` the steered
activation is ~1.8× as far from the real-activation cloud as a typical real activation is,
its norm is inflated by 30%, and the model's next-token loss is +2.78 nats worse — roughly a
**16× increase in perplexity**. This is exactly the regime where practitioners want strong
steering but where raw linear steering fails, and it is the regime where an on-manifold
corrector should help most.

### Experiment 2 — the Gaussian-manifold corrector reduces `D_M` but breaks the LM

![corrector vs raw steering](plots/02_corrector.png)

The corrector behaves exactly as designed on the manifold metric and the projection, and
exactly *wrong* on the LM:

| α | `D_M` raw | `D_M` cov_corr | ΔLM raw (nats) | ΔLM cov_corr (nats) | retention (raw = cov_corr) |
|---|-----------|----------------|----------------|---------------------|----------------------------|
| 1 | 27.8 | **27.5** | +0.08 | **+3.31** | 11.1 |
| 2 | 29.2 | **28.1** | +0.33 | **+3.84** | 22.2 |
| 4 | 34.1 | **30.4** | +1.22 | **+4.09** | 44.3 |
| 6 | 41.0 | **33.9** | +2.11 | **+4.18** | 66.5 |
| 8 | 49.0 | **38.1** | +2.78 | **+4.20** | 88.6 |

**Interpretation.** Reading across the row at `α=8`: the corrector cut the off-manifold
distance from 49.0 to 38.1 and kept the steering projection identical to raw (88.6), so by the
two quantities it was built to control it *succeeded*. But its LM loss is +4.2 nats versus raw
steering's +2.78 — **worse, not better.** The contradiction is starkest at weak steering: at
`α=1` raw steering costs only +0.08 nats while the "on-manifold" correction costs +3.31 nats,
a ~40× larger fluency hit despite sitting *closer* to the Gaussian manifold. The norm-clip
baseline (see figure) neither helps `ΔLM` nor stays on-manifold at low `α`. The mechanism: the
Mahalanobis-minimizing correction direction `Σv̂` concentrates in GPT-2's handful of very
high-variance "outlier" activation dimensions. Moving there is cheap in Mahalanobis terms
(large variance ⇒ small whitened cost) but those dimensions are precisely the ones the LM
reads most sharply, so the edit is maximally destructive downstream.

**Why this matters.** It falsifies the tempting assumption behind manifold-projection steering
methods — that pulling a steered activation back toward the statistical activation cloud makes
it safer for the model. Here the two objectives are not just different, they are **anti-correlated**
in the operative regime. Any corrector that optimizes a statistical manifold surrogate (Gaussian
Mahalanobis, and likely PCA/whitening variants that share the same geometry) can be actively
harmful. The corrector must instead see the downstream LM loss.

## Conclusion

Raw linear activation steering in GPT-2 trades off strength against fluency in a sharp,
measurable way: the stronger the steer, the further off-manifold the activation and the worse
the language model behaves. But the natural fix — pulling the activation back toward the
statistical manifold while preserving the steering projection — **backfires**. A
provably-optimal Gaussian-manifold corrector lowers the off-manifold distance by 22% yet
raises LM loss to +4.2 nats, because it moves along the LM's most sensitive (high-variance)
directions. Statistical "on-manifold" and "safe for the LM" are decoupled.

The concrete consequence for ColdSteer: keep the projection-preserving parameterization

```math
\hat{h} = z + P_{v^{\perp}}\, r_\theta(h, z, v, \alpha)
```

but train `r_θ` (a small MLP) against the **downstream LM loss** (plus a stay-near-`z` term),
rather than any manifold-distance surrogate. Experiment 2 shows why that supervision signal is
not optional but essential — it is the only one that reflects the quantity we actually care
about.

**Limitations.** (1) The manifold is modeled as a single Gaussian, so `D_M` captures
scale/correlation but not multimodal or nonlinear structure — though Experiment 2 shows this is
a feature-defining flaw of the *metric as a training target*, not merely a modeling nicety.
(2) Off-manifold distance and `ΔLM` are activation- and loss-level proxies; they do not yet
measure downstream *concept strength* or generated-text quality, which later iterations will
add. (3) One layer, one steering direction so far. (4) The corrector tested here is analytic;
the learned downstream-supervised corrector is the next step.
