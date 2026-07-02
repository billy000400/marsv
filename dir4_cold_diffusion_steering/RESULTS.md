# RESULTS — ColdSteer: on-manifold correction for activation steering

> CURRENT-BEST ONLY. One row per experiment. History lives in CHANGELOG.md.

## Metrics

**Experiment 1 — Motivating phenomenon: raw linear steering goes off-manifold.**
GPT-2 small (124M), resid_post at block 6. Steering vector `v` = DiffMean sentiment
direction (positive − negative), raw units, `|v| = 11.1`; mean clean activation norm
`|h| = 112.2`. Steered activation `z = h + α·v`. Real-activation reference:
mean Mahalanobis distance `D_M = 27.3`.

| α | Mahalanobis `D_M` | `|z|/|h|` | Δ LM loss (nats) |
|---|-------------------|-----------|------------------|
| 0 | 27.3 | 0.98 | 0.00 |
| 1 | 27.8 | 0.98 | +0.08 |
| 2 | 29.2 | 1.00 | +0.32 |
| 3 | 31.4 | 1.03 | +0.74 |
| 4 | 34.1 | 1.07 | +1.22 |
| 6 | 41.0 | 1.17 | +2.11 |
| 8 | 49.0 | 1.30 | +2.78 |

Gaussian fit on 49,218 clean tokens; ΔLM evaluated on 100 held-out FineWeb documents (128 tokens each).

**Reading it:** all three quantities rise monotonically with steering strength. By α=8 the
steered activation sits at ~1.8× the typical Mahalanobis distance of real activations, its
norm is inflated 30%, and next-token cross-entropy is +2.78 nats worse (≈ 16× higher
perplexity). This is the "steering pushes activations off-manifold and breaks the LM"
failure mode that a corrector must fix.

**Experiment 2 — Projection-preserving correction: statistical "on-manifold" ≠ LM-safe.**
Same model/layer/vector. We test the ColdSteer parameterization `ĥ = z + P_{v⊥}r` (a correction
`r` orthogonal to `v`, so the steering projection along `v` is preserved *exactly*) with an
**analytic, provably-optimal** choice of `r`: the constant shift `Δ = Σv̂·α|v|/(v̂ᵀΣv̂)` that
minimizes the Gaussian whitened-movement cost `ΔᵀΣ⁻¹Δ` at the matched projection α|v| (`cov_corr`).
Baselines: raw steering, per-token norm-clipping to the clean mean norm (`norm_clip`), and a
naive-inversion negative control (`ĥ=h`, which erases the steer). Gaussian fit on 49,218 clean
tokens; ΔLM on 100 held-out FineWeb documents.

| α | `D_M` raw | `D_M` cov_corr | ΔLM raw (nats) | ΔLM cov_corr (nats) | proj. retention raw = cov_corr |
|---|-----------|----------------|----------------|---------------------|--------------------------------|
| 1 | 27.8 | **27.5** | +0.08 | **+3.31** | 11.1 |
| 2 | 29.2 | **28.1** | +0.33 | **+3.84** | 22.2 |
| 4 | 34.1 | **30.4** | +1.22 | **+4.09** | 44.3 |
| 6 | 41.0 | **33.9** | +2.11 | **+4.18** | 66.5 |
| 8 | 49.0 | **38.1** | +2.78 | **+4.20** | 88.6 |

**Reading it:** the corrector does exactly what it is built to do on paper — it **lowers** the
Mahalanobis distance (49.0→38.1 at α=8) and preserves the steering projection to the digit
(retention column identical to raw). Yet it makes the **language model much worse**: ΔLM jumps to
+4.2 nats and, most tellingly, at small α where raw steering is nearly harmless (+0.08 nats) the
"corrected" activation is catastrophic (+3.31 nats). Norm-clipping gives essentially no ΔLM
improvement over raw and even inflates `D_M` on clean activations. So **reducing the statistical
off-manifold distance actively damages the LM.** The Mahalanobis-minimizing correction direction
`Σv̂` loads onto GPT-2's few high-variance "outlier" activation dimensions — cheap in Mahalanobis
terms but exactly the directions the LM is most sensitive to.

**Takeaway:** a purely statistical manifold prior (Gaussian Mahalanobis) is a **misleading proxy**
for real steering harm — the two are decoupled (you can lower `D_M` while raising LM loss ~40×).
An effective corrector must therefore be trained against the **downstream LM objective**, not a
manifold-distance surrogate. This motivates the next step: a learned, downstream-supervised `r_θ`.

## Figures
- `plots/01_offmanifold_phenomenon.png` — (a) Mahalanobis distance, (b) norm inflation,
  (c) ΔLM loss, each vs steering strength α. All monotonically increasing.
- `plots/02_corrector.png` — (a) `D_M`, (b) ΔLM, (c) projection retention vs α for raw steering,
  the cov-aligned corrector, norm-clip, and the naive-inversion control. The corrector lowers `D_M`
  but raises ΔLM; retention curves for raw and cov_corr coincide (matched projection).

## Headline
Raw linear steering `h + α·v` in GPT-2 drives activations progressively off-manifold (by α=8:
Mahalanobis distance nearly doubles, norm +30%, LM loss +2.78 nats). But **correcting toward the
Gaussian manifold backfires**: an analytic projection-preserving corrector cuts the off-manifold
distance 22% yet worsens LM loss to +4.2 nats — showing statistical "on-manifold" and LM-safe are
decoupled, and that steering correction must optimize the downstream objective directly.
