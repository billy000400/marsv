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

**Experiment 2 — Statistical "on-manifold" ≠ LM-safe (why a manifold prior fails).**
Same model/layer/vector. We test the ColdSteer parameterization `ĥ = z + P_{v⊥}r` (a correction
`r` orthogonal to `v`, so the steering projection along `v` is preserved *exactly*) with an
**analytic, provably-optimal** choice of `r`: the constant shift `Δ = Σv̂·α|v|/(v̂ᵀΣv̂)` that
minimizes the Gaussian whitened-movement cost `ΔᵀΣ⁻¹Δ` at the matched projection α|v| (`cov_corr`).

| α | `D_M` raw | `D_M` cov_corr | ΔLM raw (nats) | ΔLM cov_corr (nats) | proj. retention raw = cov_corr |
|---|-----------|----------------|----------------|---------------------|--------------------------------|
| 1 | 27.8 | **27.5** | +0.08 | **+3.31** | 11.1 |
| 2 | 29.2 | **28.1** | +0.33 | **+3.84** | 22.2 |
| 4 | 34.1 | **30.4** | +1.22 | **+4.09** | 44.3 |
| 6 | 41.0 | **33.9** | +2.11 | **+4.18** | 66.5 |
| 8 | 49.0 | **38.1** | +2.78 | **+4.20** | 88.6 |

**Reading it:** the corrector does exactly what it is built to do on paper — it **lowers** the
Mahalanobis distance (49.0→38.1 at α=8) and preserves the steering projection to the digit
(retention identical to raw). Yet it makes the **language model much worse**: ΔLM jumps to
+4.2 nats and, most tellingly, at small α where raw steering is nearly harmless (+0.08 nats) the
"corrected" activation is catastrophic (+3.31 nats). So **reducing the statistical off-manifold
distance actively damages the LM.** The Mahalanobis-minimizing direction `Σv̂` loads onto GPT-2's
few high-variance "outlier" dimensions — cheap in Mahalanobis terms but exactly the directions
the LM reads most sharply. Statistical on-manifold distance and real LM damage are **decoupled**
(you can lower `D_M` while raising LM loss ~40×): a manifold-distance surrogate is the wrong
training target.

**Experiment 3 — Learned, LM-supervised corrector: it works (the direction's payoff).**
Same parameterization `ĥ = z + P_{v⊥}r_θ`, but now `r_θ` is a **4-layer MLP (4.46M params)**
trained end-to-end against the **downstream LM loss**: for each batch we patch `ĥ` into
resid_post block 6, run the frozen upper GPT-2 (blocks 7–11 + head), and backprop the real
next-token cross-entropy into `r_θ` only (`h` detached, LM weights frozen), with steering
strength α sampled U(0.5, 8) per step and a light minimal-correction penalty. Trained on 300
FineWeb docs, evaluated on the **same held-out 100** docs as above. Projection retention is
identical (α|v|) for all three methods — a **matched-projection** comparison.

| α | ΔLM raw (nats) | ΔLM cov_corr | **ΔLM learned** | `D_M` raw | `D_M` learned | retention (all matched) |
|---|----------------|--------------|------------------|-----------|----------------|--------------------------|
| 1 | +0.08 | +3.31 | **−0.07** | 27.8 | 31.9 | 11.1 |
| 2 | +0.33 | +3.84 | **−0.05** | 29.2 | 36.1 | 22.2 |
| 4 | +1.22 | +4.09 | **+0.06** | 34.1 | 49.9 | 44.3 |
| 6 | +2.11 | +4.18 | **+0.22** | 41.0 | 65.4 | 66.5 |
| 8 | +2.78 | +4.20 | **+0.44** | 49.0 | 79.5 | 88.6 |

**Reading it:** the learned corrector **beats raw steering at every strength**, at matched
steering projection. At α=8 it cuts the fluency damage from +2.78 nats to **+0.44 nats — an 84%
reduction** — recovering almost all of the loss caused by strong steering while keeping the full
intended edit along `v`. At weak/medium steering it is essentially free or slightly *better* than
no correction (ΔLM ≈ −0.05). Crucially, it does this while moving **further** off the Gaussian
manifold than raw steering (`D_M` 49.0→79.5 at α=8), not closer — the mirror image of Experiment
2. This is the decoupling made constructive: the LM-safe correction is off-Gaussian-manifold, and
only a downstream-supervised objective can find it. A statistical manifold prior would have
pushed in exactly the wrong direction.

**Experiment 4 — Generalization: the corrector extrapolates beyond its training range.**
The learned corrector was trained with steering strength sampled `α ∼ U(0.5, 8)`. Here we
evaluate the *same* corrector at `α = 10` and `α = 12` — strictly **beyond** what it ever saw
in training — on the same held-out 100 docs, at matched projection.

| α | in-training-range? | ΔLM raw (nats) | **ΔLM learned** | reduction | `D_M` raw | `D_M` learned |
|---|--------------------|----------------|------------------|-----------|-----------|----------------|
| 8 | yes (boundary) | +2.78 | **+0.44** | 84% | 49.0 | 79.5 |
| 10 | **no (extrap.)** | +3.31 | **+0.76** | 77% | 57.7 | 91.2 |
| 12 | **no (extrap.)** | +3.74 | **+1.50** | 60% | 66.8 | 101.2 |

**Reading it:** the corrector keeps helping well outside its training range. At `α=10` it still
removes **77%** of raw steering's fluency damage, and even at `α=12` — 50% past the training
ceiling — it removes **60%**. The recovery fraction shrinks gradually as α leaves the training
region (84%→77%→60%), so the corrector **degrades gracefully rather than collapsing** at
unseen strengths. In-range α values (1–8) reproduce Experiment 3 to the digit (same seed, same
data). This is evidence the MLP learned a genuine correction rule that transfers, not a lookup
table over the trained α grid.

**Experiment 5 — Held-out steering vector: is the corrector overfit to one direction?**
The corrector `r_θ(h, z, α)` never receives the steering vector `v` as an input — it sees the
direction only implicitly through `z = h + α·v`. So the sharpest overfit test is a **new**
direction. We build a second DiffMean vector `v₂` for a semantically unrelated concept —
**formality** (formal ↔ informal), `|v₂| = 34.0`, nearly orthogonal to the sentiment vector
(`cos(v₁, v₂) = 0.014`) — and evaluate three methods on `v₂` at matched projection:
raw steering; the **transfer** corrector (trained on *sentiment* `v₁`, applied unchanged to `v₂`);
and a **native** corrector (the identical architecture/recipe retrained on `v₂` — the
direction-specific oracle).

| α | ΔLM raw (nats) | ΔLM transfer | ΔLM native | recovery transfer | recovery native |
|---|----------------|--------------|------------|-------------------|-----------------|
| 1 | +0.57 | +0.53 | **−0.03** | 7% | 104% |
| 2 | +2.09 | +2.02 | **+0.07** | 4% | 97% |
| 4 | +4.47 | +4.52 | **+0.35** | −1% | 92% |
| 6 | +5.78 | +5.82 | **+0.73** | −1% | 87% |
| 8 | +6.49 | +6.53 | **+1.12** | −1% | 83% |

**Reading it:** two findings, both informative. **(1) The correction rule is direction-specific.**
The sentiment-trained corrector does **not** transfer to the formality direction — its ΔLM is
indistinguishable from raw steering (recovery ≈ 0%, even slightly negative at high α). This
confirms proposal Failure Mode 4: a single trained corrector overfits to the one vector it saw.
**(2) The *method* generalizes.** Retraining the *same* 4-layer MLP with the *same* recipe on the
new direction recovers **83–104%** of raw steering's fluency damage (ΔLM +6.49 → +1.12 at α=8) —
reproducing Experiment 3's result on a completely different, larger, near-orthogonal behavior
family. So ColdSteer is a working *recipe* that must be instantiated **per steering direction**
(or conditioned on `v` / trained on a vector bank), not a single frozen operator you can reuse
across concepts. As in Experiment 3, the native corrector wins on fluency while moving *further*
off the Gaussian manifold than raw (`D_M` 66.6 → 123.1 at α=8).

## Figures
- `plots/01_offmanifold_phenomenon.png` — (a) Mahalanobis distance, (b) norm inflation,
  (c) ΔLM loss, each vs steering strength α. All monotonically increasing.
- `plots/02_corrector.png` — (a) `D_M`, (b) ΔLM, (c) projection retention vs α for raw steering,
  the analytic cov-aligned corrector, norm-clip, and the naive-inversion control. The corrector
  lowers `D_M` but raises ΔLM; retention curves for raw and cov_corr coincide (matched projection).
- `plots/03_learned_corrector.png` — (a) ΔLM, (b) `D_M`, (c) projection retention vs α for raw,
  analytic cov-aligned, and the learned LM-supervised corrector. The learned corrector's ΔLM sits
  near zero across α while its `D_M` rises above raw — winning on fluency by going *off* the
  Gaussian manifold.
- `plots/04_generalization.png` — (a) ΔLM, (b) `D_M` vs α for raw steering and the learned
  corrector, with the shaded region marking α>8 (beyond training). The learned ΔLM stays far below
  raw across the extrapolation region, its advantage narrowing smoothly.
- `plots/05_heldout_vector.png` — (a) ΔLM, (b) `D_M` vs α on the held-out formality vector `v₂` for
  raw steering, the transfer corrector (trained on sentiment `v₁`), and the native corrector
  (retrained on `v₂`). The transfer curve sits on top of raw (no transfer); the native curve
  collapses ΔLM toward zero (the recipe generalizes when retrained).

## Headline
Raw linear steering `h + α·v` in GPT-2 drives activations off-manifold and breaks the LM (+2.78
nats at α=8). Correcting toward the **Gaussian manifold backfires** — an analytic projection-
preserving corrector cuts off-manifold distance 22% but *worsens* LM loss to +4.2 nats. But a
**learned corrector supervised by the LM loss** — same projection-preserving form, so the steering
edit is untouched — **recovers 84% of the damage** (ΔLM +2.78→+0.44 at α=8) while moving *further*
from the Gaussian manifold. Statistical "on-manifold" and "LM-safe" are decoupled; only the
downstream objective finds the safe, on-behavior correction. The correction is **direction-specific**
— a corrector trained on one concept does not transfer to a near-orthogonal one — but the **recipe
generalizes**: retraining it on a new formality direction recovers 83–104% of the damage there too.
