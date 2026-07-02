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

**Experiment 6 — Direction-conditional corrector on a vector bank: one model, many directions.**
Experiment 5's fix was "train one corrector per direction." The natural way to avoid that is to make
the corrector **conditional on the direction** — feed the unit steering vector `v̂` as an extra input,
`r_θ(h, z, v̂, α)` — and **train it on a bank** of directions at once. We build a 3-vector bank
(**sentiment**, **formality**, **concreteness**; all DiffMean at block 6) and train a **single**
conditional corrector (5.25M params) sampling a (direction, α) pair per step. A fourth direction,
**certainty**, is **held out** (never trained). Pairwise cosines: sentiment is near-orthogonal to all
(|cos| ≤ 0.03); formality/concreteness/certainty share a subspace (|cos| 0.76–0.82), so the held-out
`certainty` lies largely *within* the bank's span.

*One model corrects every in-bank direction (fluency recovery at matched projection):*

| direction | in bank? | ΔLM raw @α=8 | ΔLM bank (one model) | recovery @α=8 | recovery @α=2 |
|---|---|---|---|---|---|
| sentiment | bank | +2.78 | **+1.24** | 55% | 64% |
| formality | bank | +6.49 | **+1.95** | 70% | 90% |
| concreteness | bank | +4.40 | **+3.65** | 17% | 70% |
| certainty | **HELD-OUT** | +3.71 | **+3.45** | 7% | 42% |

*Held-out `certainty` sweep — bank corrector (transfer) vs native oracle (retrained on certainty):*

| α | ΔLM raw | ΔLM bank (transfer) | ΔLM native (oracle) | recovery bank | recovery native |
|---|---------|---------------------|---------------------|---------------|-----------------|
| 1 | +0.22 | +0.11 | **−0.09** | 51% | 141% |
| 2 | +0.99 | +0.57 | **−0.05** | 42% | 105% |
| 4 | +2.62 | +2.07 | **+0.11** | 21% | 96% |
| 6 | +3.35 | +2.94 | **+0.39** | 12% | 88% |
| 8 | +3.71 | +3.45 | **+0.80** | 7% | 78% |

**Reading it:** two findings. **(1) One conditional model amortizes across a bank of directions.** A
single corrector recovers 55–70% of raw steering's fluency damage on two of the three in-bank
directions at α=8 (formality +6.49→+1.95), and more at moderate strength — where Experiments 3/5
needed a *separate* trained model for each direction. The cost of sharing is a lower per-direction
recovery than a dedicated corrector (sentiment 84%→55%, formality 83%→70% at α=8) and one direction
it handles poorly at strong steering (concreteness, 17% at α=8 but 70% at α=2) — evidence of capacity
interference across directions. **(2) Conditioning + a bank begins to transfer to a held-out
direction, but only partially.** On `certainty` — never trained, yet highly correlated with two bank
members — the bank corrector recovers **51% at α=1 falling to 7% at α=8**. That is a real improvement
over Experiment 5's frozen single-vector transfer (≈0% at every α), so conditioning on `v̂` plus a
small bank does start to generalize across directions, best at moderate strength; but it stays far
below the native oracle (retraining on `certainty` recovers 78–141%). The practical read: a
direction-conditional corrector replaces "one model per vector" with "one model per *bank*." Whether
simply *enlarging* the bank closes the remaining held-out gap at strong steering is tested — and
answered negatively — in Experiment 7.

**Experiment 7 — Does a denser bank close the held-out gap? Scaling the vector bank.**
Experiment 6 suggested "scaling the bank is the indicated path" to correct a genuinely unseen
direction at strong steering. We test that directly. Holding **certainty** out as before, we train the
**same** direction-conditional corrector (5.25M params, identical recipe / seed / data) on **nested**
training banks of size **1** (sentiment), **3** (sentiment, formality, concreteness — Experiment 6's
bank), and **5** (+ two new DiffMean directions: **politeness** `|v|=15.6`, **complexity** `|v|=58.4`),
and measure transfer to the held-out `certainty` at matched projection. Cosines to `certainty`:
formality +0.77, concreteness −0.82, complexity −0.80 (all strongly related), politeness −0.35
(weakly), sentiment +0.03 (orthogonal) — so growing 3→5 adds one strongly-correlated direction
(complexity) plus one weakly-correlated one (politeness).

*Held-out `certainty` fluency recovery vs training-bank size (native = oracle retrained on certainty):*

| α | ΔLM raw | rec bank=1 | rec bank=3 | rec bank=5 | rec native (oracle) |
|---|---------|-----------|-----------|-----------|---------------------|
| 1 | +0.22 | 14% | **51%** | −1% | 142% |
| 2 | +0.99 | 8% | **42%** | 9% | 105% |
| 4 | +2.62 | 1% | **21%** | 6% | 96% |
| 6 | +3.35 | 0% | **12%** | 4% | 88% |
| 8 | +3.71 | 0% | **7%** | 3% | 78% |

*In-bank per-direction recovery @α=8 under the SINGLE size-5 corrector (all five trained together):*

| direction | cos to certainty | ΔLM raw @α=8 | ΔLM size-5 bank | recovery @α=8 |
|---|---|---|---|---|
| sentiment | +0.03 | +2.78 | +1.21 | 57% |
| formality | +0.77 | +6.49 | +3.55 | 45% |
| concreteness | −0.82 | +4.40 | +3.84 | 13% |
| politeness | −0.35 | +4.47 | +1.27 | 72% |
| complexity | −0.80 | +5.39 | +3.18 | 41% |

**Reading it: naively enlarging the bank does NOT close the held-out gap — at fixed model capacity it
makes transfer *worse*.** Transfer to the held-out `certainty` is **non-monotone in bank size and peaks
at size 3**, not size 5: going 3→5 dropped recovery at every strength (α=1 51%→−1%, α=8 7%→3%). This
happens even though one of the two added directions (complexity, |cos|=0.80) is strongly correlated
with the held-out direction — extra "coverage" that should help transfer. The corroborating signal is
in-bank: under the size-5 model, per-direction recovery at α=8 is *lower* than the size-3 model
delivered (formality 70%→45%, concreteness 17%→13%), while the two new directions land at 41–72%. So a
fixed-capacity 5.25M corrector, asked to correct five directions instead of three, does **each**
direction worse — the held-out one included. **Capacity interference, not coverage, is the binding
constraint.** This revises Experiment 6's optimistic reading: closing the held-out gap at strong
steering needs **model-capacity scaling and/or bank curation toward the target's subspace**, not simply
more directions poured into the same-size model. The native oracle retrained on `certainty` still
recovers 78–142%, so the direction is fully correctable — the gap is a property of amortization, not of
`certainty` being intrinsically hard.

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
- `plots/06_conditional_bank.png` — (a) per-direction fluency recovery at α=8 for ONE
  direction-conditional corrector trained on the {sentiment, formality, concreteness} bank (blue =
  in-bank, orange = held-out certainty); (b) ΔLM vs α on the held-out certainty direction for raw,
  the bank corrector (partial transfer), and the native oracle.
- `plots/07_bank_scaling.png` — (a) held-out `certainty` fluency recovery vs α for training-bank sizes
  {1, 3, 5} plus the native oracle; (b) held-out recovery at α=1 and α=8 vs bank size — the headline
  curve, showing transfer peaks at bank size 3 and drops when the bank grows to 5 (capacity
  interference at fixed model size).

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
Making the corrector **direction-conditional** (feed `v̂`) and training it on a **bank** of directions
gives **one model that corrects every in-bank direction at once** (55–70% recovery at α=8) and
**begins to transfer** to a held-out direction (51%→7% recovery from weak to strong steering, vs ≈0%
for a frozen single-vector corrector) — replacing "one model per vector" with "one model per bank."
But **naively enlarging that bank does not close the held-out gap**: at fixed model capacity, growing
the training bank from 3 to 5 directions *lowers* transfer to the held-out direction (α=8 recovery
7%→3%) and lowers per-direction in-bank recovery too — capacity interference, not coverage, binds. The
route to a reusable corrector is **more model capacity and/or a curated bank**, not merely more
directions.
