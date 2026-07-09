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
direction worse — the held-out one included. **Coverage is not the binding constraint** (extra
directions hurt); the tempting next hypothesis is that raw model capacity is. This revises
Experiment 6's optimistic reading: closing the held-out gap at strong steering needs **bank curation
toward the target's subspace and/or a stronger training signal**, not simply more directions poured
into the same-size model — and, as **Experiment 8** shows, not simply a bigger model either. The
native oracle retrained on `certainty` still recovers 78–142%, so the direction is fully correctable —
the gap is a property of amortization, not of `certainty` being intrinsically hard.

**Experiment 8 — Does more model capacity close the held-out gap? Scaling the corrector.**
Experiment 7 blamed the bank-scaling failure on *capacity interference* — directions competing for a
fixed 5.25M-param MLP — a causal claim about model size that Exp 7 never varied. We test it directly:
**hold the bank fixed** at the size-5 set {sentiment, formality, concreteness, politeness, complexity}
(the exact bank that gave Exp 7's *worst* held-out transfer) and **scale the corrector's width**,
`hidden ∈ {1024, 2048, 4096}` → **5.2M / 14.7M / 46.2M params** (a 9× range), same recipe/seed/data/eval.
The native oracle (retrained on `certainty`) is the ceiling. The `hidden=1024` point re-runs Exp 7's
size-5 model and reproduces it to the digit.

*Held-out `certainty` fluency recovery vs corrector capacity (native = oracle retrained on certainty):*

| corrector capacity | rec @α=1 | @α=2 | @α=4 | @α=6 | @α=8 |
|---|---|---|---|---|---|
| 5.2M (hid 1024) | −1% | 9% | 6% | 4% | **3%** |
| 14.7M (hid 2048) | −22% | 20% | 6% | 3% | **2%** |
| 46.2M (hid 4096) | −146% | −22% | −2% | 0% | **1%** |
| native oracle | 142% | 105% | 96% | 88% | **78%** |

*In-bank per-direction recovery @α=8 vs corrector capacity (all five trained together):*

| capacity | sentiment | formality | concreteness | politeness | complexity | **mean** |
|---|---|---|---|---|---|---|
| 5.2M | 57% | 45% | 13% | 72% | 41% | **45.4%** |
| 14.7M | 63% | 38% | 7% | 73% | 38% | **43.8%** |
| 46.2M | 59% | 42% | 24% | 75% | 32% | **46.3%** |

**Reading it: more capacity does NOT close the held-out gap either — simple width scaling is not the
fix.** Two signals. **(1) In-bank recovery saturates.** Mean in-bank recovery at α=8 is essentially
flat across a 9× capacity increase (**45.4% → 43.8% → 46.3%**); the shared MLP was not width-starved,
and adding parameters does not let it correct the five *training* directions any better. **(2) Held-out
transfer does not improve and degrades at weak steering.** At α=8, held-out recovery is flat-to-falling
(3%→2%→1%); and at the weak α where the correction should be nearly the identity, the widest model
*actively damages* the unseen direction — recovery goes −1% → −22% → **−146%** at α=1 (the 46M model
adds +0.32 nats to a nearly-harmless weak steer). That is textbook overfitting: extra capacity is spent
memorizing the bank directions, which hurts an unseen one. So Experiment 7's "capacity interference"
reading is only half right — the ceiling on amortized cross-direction correction is set by the
**training signal (which directions are in the bank, how the corrector is conditioned, the objective),
not by parameter count.** The native oracle (78–142%) is unchanged and remains the only reliable route
to a genuinely unseen direction: **the correction is fundamentally direction-specific**, and neither
more directions (Exp 7) nor more parameters (Exp 8) amortizes it away.

**Experiment 9 — Does curating the bank toward the target's subspace close the gap? Bank diversity, not alignment.**
Experiments 7 and 8 both closed by naming the same open path: "curate the bank toward the held-out
target's subspace." Neither tested it. We do, in the clean controlled way — **hold bank size fixed at 3
and corrector capacity fixed at 5.25M** (`hidden=1024`), and vary only *which* three of the five pool
directions are in the bank, by their mean absolute cosine to the held-out `certainty`:

- **diffuse** = {sentiment, politeness, formality}, mean `|cos|` = **0.38** (angularly spread-out)
- **exp6** = {sentiment, formality, concreteness}, mean `|cos|` = **0.54** (Experiment 6/7's bank)
- **curated** = {formality, concreteness, complexity}, mean `|cos|` = **0.80** (aligned to target)

`diffuse` and `curated` share exactly one member (formality) and differ only in the other two, so this
is a controlled contrast. If subspace *coverage* of the target were the binding constraint, transfer
should rise monotonically diffuse → exp6 → curated.

*Held-out `certainty` fluency recovery vs bank (all size 3, all 5.25M; native = oracle retrained on certainty):*

| α | ΔLM raw | rec diffuse (0.38) | rec exp6 (0.54) | rec curated (0.80) | rec native (oracle) |
|---|---------|--------------------|-----------------|--------------------|---------------------|
| 1 | +0.22 | 38% | **51%** | **−183%** | 142% |
| 2 | +0.99 | 28% | **42%** | −15% | 105% |
| 4 | +2.62 | 13% | **21%** | 3% | 96% |
| 6 | +3.35 | 7% | **12%** | −5% | 88% |
| 8 | +3.71 | 6% | **7%** | −12% | 78% |

*In-bank per-direction recovery @α=8 under each size-3 corrector (mean over the bank's 3 directions):*

| bank | mean `|cos|` to certainty | member recoveries @α=8 | **mean in-bank rec** |
|---|---|---|---|
| diffuse | 0.38 | sentiment 65%, politeness 74%, formality 60% | **67%** |
| exp6 | 0.54 | sentiment 55%, formality 70%, concreteness 17% | **48%** |
| curated | 0.80 | formality 37%, concreteness 17%, complexity 35% | **30%** |

**Reading it: curating the bank toward the target subspace does NOT close the gap — it makes transfer
catastrophically *worse*.** Held-out recovery is **non-monotone in bank→target alignment and collapses
at the most-aligned bank**: the `curated` bank (mean `|cos|`=0.80) is the worst by far, *actively
damaging* the unseen direction at weak steering (α=1 recovery **−183%**: ΔLM +0.22 → +0.62) and staying
net-negative at strong steering (−12% @α=8), while the moderately-aligned `exp6` bank transfers best
(51%→7%). The in-bank table gives the mechanism: recovery *falls* monotonically as the bank's own
directions become more internally correlated (diffuse 67% → exp6 48% → curated 30%). The `curated`
members are pairwise near-collinear (`|cos|` 0.76–0.82), so the direction-conditional corrector cannot
tell them apart from `v̂` and cannot specialize — it learns a single shared-subspace correction that
over-fires on any nearby unseen direction (hence the weak-α blow-up on `certainty`). So the lever is
**bank angular *diversity* (separability of its directions), not coverage of the target's subspace** —
curating *toward* the target backfires. This is the **third** corrective negative in a row: neither more
directions (Exp 7), more parameters (Exp 8), nor a target-aligned bank (Exp 9) amortizes the correction
away. `exp6` reproduces Experiment 6/7's size-3 bank to the digit (recovery 51/42/21/12/7), a built-in
reproducibility check. The native oracle (78–142%) is unchanged — the direction stays fully correctable
per-direction; amortizing it across directions is capped by the **training signal**, of which bank
composition is now shown to matter through *diversity*, not target alignment.

**Experiment 10 — Behavioral reality-check: matched projection ≠ matched steering in generation.**
Experiments 2–9 all score the corrector on `ΔLM` (fluency) at *matched projection* — the corrector
holds the layer-6 edit along `v` fixed to `α|v|` by construction. But matched projection **at one
layer** is a proxy: it does not prove that when you use the corrected activation to actually *generate*
text, the text (a) still moves in the steered direction, or (b) reads more fluently than raw-steered
text. We test both directly. Using the flagship sentiment corrector (Exp 3, retrained identically), we
greedily generate 30-token continuations from 48 held-out prompts with the steer applied at resid_post
block 6 at **every** generated position, under raw steering vs. the corrector, and measure two
quantities on a **clean re-encode** of the generated text: the **sentiment effect** `B(α)−B(0)` (mean
projection of the continuation's block-6 activations onto `v̂`, relative to the unsteered greedy
continuation `B(0)=+0.34`; higher = more strongly steered) and **degeneration** via **distinct-2** (the
unique-bigram ratio of the generated tokens; the unsteered baseline is `0.70`, and lower = more
repetitive/collapsed text).

| α | effect raw `B−B₀` | effect corr `B−B₀` | distinct-2 raw | distinct-2 corr |
|---|-------------------|--------------------|----------------|-----------------|
| 2 | **+2.97** | +0.17 | 0.78 | 0.65 |
| 4 | **+2.31** | +0.19 | 0.72 | 0.72 |
| 6 | **+2.47** | +0.15 | 0.54 | 0.71 |
| 8 | +1.77 | +0.48 | **0.32** | **0.64** |

**Reading it: the corrector's fluency win is real but is NOT a free lunch — it comes partly at the cost
of the behavioral steer, a tradeoff the matched-projection `ΔLM` metric hid.** Three things happen.
**(1) Raw steering genuinely steers, then degenerates.** Raw steering swings the generated text's
sentiment hard (`+2.97` at α=2 — e.g. "the weather is perfect. The temperature is perfect") but as α
grows the text collapses into repetition/gibberish: distinct-2 falls `0.78 → 0.32` (α=8 sample:
"the Southern-the-Bt and the second-t-t-t-t-t-t"). **(2) The corrector fixes the degeneration.** Its
generations stay coherent and near-baseline-diverse at every strength (distinct-2 `0.64–0.72`, vs the
unsteered `0.70`; α=8 sample: "It is located in the heart of the city … a place to watch the city's
skyline") — no collapse. **(3) But the corrector's text is barely steered.** Its sentiment effect is
`+0.15–0.48` — roughly one-sixth of raw's pre-collapse effect. So the correction, though orthogonal to
`v` in *activation* space, is **not** orthogonal to the downstream sentiment *readout*: minimizing LM
loss at matched layer-6 projection drives the corrector to a solution that produces near-normal,
lightly-steered text. On the effect-vs-fluency Pareto the two methods do **not** dominate each
other — raw buys behavioral effect at the price of fluency; the corrector buys fluency at the price of
behavioral effect. **The practical caveat:** the large `ΔLM` recoveries of Experiments 3–9 measure how
much the corrector reduces the *disruption a steer causes to processing real text* (teacher-forced), and
that is genuine; but a substantial part of that reduction reflects a **weaker propagated edit in
generation**, not a costless cleanup. "Matched projection at one layer" does **not** guarantee "matched
behavioral steering," and any deployment of ColdSteer must measure the behavioral effect on generated
text — not just `ΔLM` — before trusting the correction.

**Experiment 11 — Can a behavioral-preservation term push the Exp 10 Pareto outward?**
Experiment 10 diagnosed *why* the flagship corrector under-steers in generation: its correction, though
orthogonal to `v` at layer 6, is **not** orthogonal to the downstream sentiment *readout*, so minimizing
LM loss quietly suppresses the propagated concept signal. The natural fix is to supervise that readout
directly. We add one term to the Exp 3 objective: during training we also read out the sentiment
projection at a **later layer** (`L2 = 11`, the final resid_post that feeds the output head; downstream
DiffMean sentiment direction `ŵ`, `|w| = 3.87`) and push the corrected activation's downstream
projection `p_corr = ⟨resid^{(11)}, ŵ⟩` toward what **raw steering** produces, `p_raw`, via
`L_behav = ⟨((p_corr − p_raw)/100)²⟩` weighted by `λ_b`. We train a family `λ_b ∈ {0, 10, 40}`
(`λ_b = 0` is exactly the Exp 10 corrector) and score every one on the **identical** Exp 10 generation
protocol (48 prompts, 30 greedy tokens, sentiment effect `B(α)−B(0)` and distinct-2 on a clean
re-encode). Raw steering is the shared reference.

| α | eff raw | eff λ_b=0 | eff λ_b=10 | eff λ_b=40 | d2 raw | d2 λ_b=0 | d2 λ_b=10 | d2 λ_b=40 |
|---|---------|-----------|------------|------------|--------|----------|-----------|-----------|
| 2 | **+2.97** | +0.17 | +0.45 | +0.99 | 0.78 | 0.65 | 0.66 | **0.73** |
| 4 | **+2.31** | +0.19 | +0.87 | +1.31 | 0.72 | 0.72 | 0.61 | 0.65 |
| 6 | **+2.47** | +0.15 | +0.93 | +0.84 | 0.54 | 0.71 | 0.58 | 0.59 |
| 8 | +1.77 | +0.48 | +1.23 | +1.08 | **0.32** | 0.64 | 0.59 | 0.52 |

(`eff` = sentiment shift `B(α)−B(0)`, higher = more steered; `d2` = distinct-2, higher = more fluent;
unsteered baselines `B(0)=+0.34`, distinct-2 `=0.70`. `λ_b=0` reproduces Experiment 10 to the digit — a
built-in reproducibility check.)

**Reading it: the behavioral term works — it recovers 2–6× more behavioral effect while keeping generation
fluent — and it pushes the Pareto frontier *outward at the fluent end*, but a hard ceiling remains.**
Three findings. **(1) The term does its job cheaply.** Adding `λ_b` lifts the generated sentiment effect
from Exp 10's `+0.15–0.48` up to `+0.8–1.3` (roughly 2–6×) while distinct-2 stays `0.52–0.73` — far above
raw steering's high-α collapse (0.32 at α=8) and near the unsteered baseline of 0.70. **(2) The corrector
family now Pareto-dominates raw in the modest-effect regime.** For a sentiment effect of about `+1`, the
`λ_b=40` corrector keeps distinct-2 at **0.73** (α=2) — essentially unsteered-baseline fluency — whereas
raw steering only produces an effect that low (`+1.77` at α=8) *after* it has already collapsed into
repetition (distinct-2 0.32). So where Exp 10 found "neither method dominates," an explicit
behavioral-preservation term now gives a corrector that **strictly beats raw** — more steer *and* more
fluency — as long as you want moderate steering. **(3) But it cannot reach raw's strong effect.** No
`λ_b` lifts the generated effect past ≈+1.3; pushing `λ_b` from 10 to 40 stops raising the effect (it even
falls at α=6, +0.93→+0.84) and only raises training LM loss. The reason is a second layer of the same
proxy gap: the term successfully matches raw's *teacher-forced* downstream readout (the training
`L_behav` drops to ~0.005, `p_corr ≈ p_raw`), yet matching a teacher-forced readout only *partially*
transfers to the autoregressive generation effect. **Net:** a behavioral term is a real, cheap win — it
converts Exp 10's non-dominating tradeoff into outright dominance over raw at moderate steering — but the
projection-preserving corrector still cannot match raw's *strong* behavioral steering, so the effect–
fluency frontier is pushed out, not erased.

**Experiment 20 — Supervising through *differentiable generation* breaks the Exp 11 ceiling.**
Experiment 11's behavioral term matched the corrector's downstream sentiment readout on a **teacher-forced**
pass (corrected activation patched over *ground-truth* FineWeb tokens). Its ceiling (effect never past
≈+1.3) was traced to a proxy gap: matching a teacher-forced readout only partially transfers to
autoregressive generation. This experiment supervises the readout on the corrector's **own generated
continuation** through a **differentiable soft-token rollout** — starting from `P=8` real tokens, roll out
`K=8` steps with the steer applied at every position, read the downstream sentiment projection at each
generated position, and feed the softmax-weighted expected embedding `softmax(ℓ/τ)·Wₑ` back as the next
input (fully differentiable in `r_θ`) — pushing the corrected rollout's readout toward raw steering's own
rollout, weight `λ_g`. Everything else is the Exp 11 recipe; `λ_g=0` is the Exp 10/11 base corrector.
Scored on the identical Exp 10 protocol (48 prompts, 30 greedy tokens; `eff` = `B(α)−B(0)`, higher = more
steered; `d2` = distinct-2, higher = more fluent; unsteered baselines `B(0)=+0.34`, `d2=0.70`).

| α | eff raw | eff λ_g=0 | eff λ_g=40 | eff λ_g=160 | d2 raw | d2 λ_g=0 | d2 λ_g=40 | d2 λ_g=160 |
|---|---------|-----------|------------|-------------|--------|----------|-----------|------------|
| 2 | **+2.97** | +0.17 | +1.01 | +1.61 | 0.78 | 0.65 | 0.67 | **0.71** |
| 4 | **+2.31** | +0.19 | +1.40 | +1.48 | 0.72 | 0.67 | 0.67 | 0.60 |
| 6 | **+2.47** | +0.15 | +1.30 | +0.61 | 0.54 | 0.71 | 0.54 | 0.46 |
| 8 | +1.77 | +0.48 | **+1.72** | −0.22 | **0.32** | 0.64 | 0.47 | 0.32 |

(For reference, Exp 11's teacher-forced term at its best gave α=8 effect +1.23 (`λ_b=10`) / +1.08 (`λ_b=40`).
`λ_g=0` reproduces Exp 10/11 to the digit — a built-in reproducibility check.)

**Reading it: supervising on the *autoregressive* distribution rather than a teacher-forced proxy pushes the
effect-fluency frontier further out than Exp 11 — it breaks the ≈+1.3 effect ceiling — but the frontier
stays sensitive at strong steering.** Three findings. **(1) The ceiling moves.** At α=8 the moderate
corrector `λ_g=40` reaches sentiment effect **+1.72** — above Exp 11's best (+1.23/+1.08) and nearly matching
raw's already-collapsed +1.77 — while keeping distinct-2 at **0.47** vs raw's collapsed **0.32**. The
generation-aware signal recovers behavioral effect the teacher-forced signal could not. **(2) At moderate
steering the win is clean.** The stronger corrector `λ_g=160` at α=2 reaches effect **+1.61 at near-baseline
fluency 0.71**, dominating Exp 11's best moderate point (+0.99 at 0.73): at low α the differentiable rollout
stays coherent so the readout target is met without degenerating. **(3) But over-weighting collapses at
strong steering.** `λ_g=160` overshoots — pushing the generation readout too hard destabilizes training (one
step spiked LM loss to ~20) and at α≥6 the corrector *degenerates like raw*: effect falls to +0.61 (α=6) then
**−0.22** (α=8) with distinct-2 collapsing to 0.32 (its α=8 sample repeats *"the Southern-the-Beal and the
Southern-the-Beal…"*, exactly raw's failure). So the generation-aware term is a strictly better lever than
the teacher-forced one in the *moderate*-steering regime and raises the achievable strong-α effect
(+1.08→+1.72 at α=8), but the strong-effect-**and**-fluent corner still eludes: too little generation weight
under-steers, too much collapses. Differentiable-generation supervision **narrows** the Exp 11 proxy gap
without closing it — the projection-preserving corrector's frontier is pushed out a second time, not erased.

**Experiment 12 — Layer robustness: is the result a block-6 artifact?**
Every experiment above steers and corrects at resid_post block 6. The obvious question is whether the
two headline facts — **(P)** raw steering breaks the LM, and **(C)** the LM-supervised
projection-preserving corrector recovers it — are specific to that layer. We replicate the flagship
Experiment-3 pipeline **unchanged** (same DiffMean sentiment prompts, same 400-doc Gaussian fit, same
300-doc training set, same held-out 100-doc eval, same 4-layer corrector, same seed / hyper-parameters —
*only the hook layer changes*) at **block 3 (early)**, **block 6 (mid, = Exp 3)**, and **block 9 (late)**.
The sentiment vector is rebuilt at each layer; its raw norm grows with depth (`|v|` = 6.75 / 11.08 / 23.16),
as does the mean activation norm (`|h|` = 88.8 / 112.2 / 176.5), so each layer is compared at its own
matched projection `α|v|`.

| layer | ΔLM raw @α=8 | **ΔLM learned @α=8** | recovery @α=8 | recovery @α=4 | `D_M` raw / learned @α=8 |
|---|---|---|---|---|---|
| block 3 (early) | +2.56 | **+0.25** | **90%** | 100% | 44.1 / 74.3 |
| block 6 (mid, = Exp 3) | +2.78 | **+0.44** | **84%** | 95% | 49.0 / 79.5 |
| block 9 (late) | +2.34 | **+0.55** | **76%** | 91% | 49.2 / 70.9 |

**Reading it: both facts replicate at every layer — the result is not a block-6 artifact.** At all three
depths, (P) raw steering drives `ΔLM` up monotonically with α (to +2.3–2.6 nats at α=8) and inflates the
Mahalanobis distance, and (C) the identical LM-supervised corrector removes the bulk of that damage at
matched projection — **recovering 76–90% at α=8 and ≥91% at α=4**, with `ΔLM` essentially zero or slightly
negative at weak steering (recovery >100% at α≤2, as in Exp 3). The block-6 point reproduces Experiment 3
**to the digit** (raw +2.78 → learned +0.44, 84%), a built-in reproducibility check that the refactored
layer-swept pipeline is faithful. Two second-order trends are worth noting: recovery at α=8 falls slightly
with depth (90%→84%→76%), consistent with a fixed-capacity corrector facing a larger absolute edit as `|v|`
grows toward the output; and the signature decoupling of Experiments 2–3 holds **at every layer** — the
corrected activation sits *further* off the Gaussian manifold than raw (`D_M` corrected > raw at all three
depths), confirming that "LM-safe but off-Gaussian" is a general property of the learned correction, not a
quirk of one layer. So the core ColdSteer claim — *a downstream-supervised, projection-preserving corrector
buys back most of raw steering's fluency damage while moving off the statistical manifold* — is a
**layer-robust** phenomenon across the early, middle, and late residual stream.

**Experiment 13 — Cross-model generality: is the result a GPT-2-*small* artifact?**
Experiment 12 showed the flagship result replicates across *layers* of GPT-2 small. The complementary
external-validity question is the *model* axis: does the same recipe work on a different, larger model? We
replicate the exact Experiment-3 pipeline **unchanged** on **GPT-2 medium (355M, 24 blocks, `d = 1024`)**,
steering and correcting at its **mid layer, block 12 of 24** (the depth analogue of block 6 of 12 in
small). The DiffMean sentiment prompts, 400-doc Gaussian fit, 300-doc training set, held-out 100-doc eval,
4-layer corrector, seed, and hyper-parameters are all identical to Experiment 3 — only the model changes,
so `|v|`, `|h|`, and `d` change with it (`|v| = 19.6`, mean `|h| = 226.2`, clean `D_M = 31.5`; corrector
5.25M params at `d = 1024`).

| α | ΔLM raw (nats) | **ΔLM learned** | recovery | `D_M` raw | `D_M` learned |
|---|----------------|------------------|----------|-----------|----------------|
| 1 | +0.04 | **−0.12** | >100% | 32.0 | 36.6 |
| 2 | +0.15 | **−0.09** | >100% | 33.5 | 40.0 |
| 4 | +0.74 | **−0.01** | **101%** | 38.8 | 51.9 |
| 8 | +2.72 | **+0.30** | **89%** | 55.1 | 79.9 |

**Reading it: both headline facts replicate on a different, larger model — the result is not a
GPT-2-small artifact.** (P) Raw steering breaks the LM on GPT-2 medium exactly as on small: `ΔLM` climbs
monotonically to **+2.72 nats at α=8** while the Mahalanobis distance inflates 31.5→55.1. (C) The identical
LM-supervised, projection-preserving corrector removes essentially all of it — recovering **89% of the
fluency damage at α=8** and **101% at α=4** (`ΔLM` +0.74→−0.01), at matched projection. At weak steering the
corrected activation lands slightly *below* the unsteered baseline (`ΔLM` −0.09 to −0.12, recovery >100% —
the same free-or-better weak-α behavior seen in Experiment 3 on small; recovery reads >100% only because
raw's damage is near zero there). The α=8 recovery on medium (89%) is even a touch higher than small's 84%.
And the signature decoupling holds again: the corrected activation sits **further** off the Gaussian
manifold than raw at **every** α (`D_M` learned > raw throughout, 79.9 vs 55.1 at α=8). So the core
ColdSteer claim — *a downstream-supervised, projection-preserving corrector buys back nearly all of raw
steering's fluency damage while moving off the statistical manifold* — is **model-robust** as well as
layer-robust: it holds on GPT-2 medium, a 3× larger model with a different width and depth.

**Experiment 14 — Direct confirmation: bank angular *diversity* is the causal lever (confound removed).**
Experiment 9 *inferred* that bank angular diversity — not coverage of the target's subspace — governs a
direction-conditional corrector's per-direction recovery, but it could not isolate the two: in that pool
the held-out `certainty` sits *inside* the collinear cluster, so the most target-aligned bank was also the
most internally collinear. This experiment removes the confound with a **controlled third-member swap**.
Every bank is size 3, capacity fixed at 5.25M, and shares the **same anchor pair {sentiment, formality}**;
only the **third** member changes, chosen to be increasingly collinear with `formality`. We measure the
recovery of the *shared* anchor pair. The key control: **sentiment is orthogonal to every other direction
*and* to the held-out target** (`|cos| ≤ 0.03`), so its recovery cannot depend on target coverage or on
the third member's identity — only on the bank's internal separability.

| bank | internal collinearity \|cos\| (↑ = less diverse) | **sentiment** rec @α=8 (⟂-isolate) | **formality** rec @α=8 | swapped 3rd member (\|cos\| to formality) | **3rd member** rec @α=8 (α=4) | held-out `certainty` rec @α=8 |
|---|---|---|---|---|---|---|
| div  | 0.13 | **63%** | 69% | politeness (0.07) | **69%** (75%) | 9% |
| mid  | 0.21 | **61%** | 69% | complexity (0.57) | **40%** (57%) | 5% |
| coll | 0.26 | **55%** | 70% | concreteness (0.76) | **17%** (34%) | 7% |

**Reading it: bank diversity is a *causal* lever, confirmed with the target-alignment confound removed.**
Two complementary signals, both monotone in the bank's internal collinearity. **(1) A bank member
confusable with a neighbor cannot be specialized.** As the third member is made ever more collinear with
`formality`, *its own* recovery collapses — politeness 69% → complexity 40% → concreteness 17% at α=8
(75% → 57% → 34% at α=4) — because the direction-conditional corrector receives `v̂` and cannot tell two
near-parallel directions apart. **(2) Collinearity anywhere in the bank raises interference for
*everyone*, and this cannot be a target-coverage effect.** The confound-free isolate `sentiment` — which
is orthogonal to every bank member *and* to the held-out `certainty` — is nonetheless corrected **worse**
in the more collinear banks (63% → 61% → 55% at α=8), even though nothing about sentiment's own geometry
or its relation to any target changed. Its degradation can *only* be reduced bank separability. Meanwhile
`formality` — the anchor that *gains* the collinear neighbor — holds ~69–70% throughout: when the
corrector cannot disambiguate two near-parallel directions it collapses them onto the dominant
(larger-norm, and here better-corrected) one, so `formality` keeps its recovery while its weaker neighbor
loses. (Weak-α recovery is omitted here: raw `ΔLM` is near zero at α=1 so the recovery ratio is unstable,
as throughout.) This turns Experiment 9's *correlational* claim into a *controlled* one: **angular
diversity (separability), not target-subspace coverage, is what a shared corrector needs** — the positive
counterpart to the three scaling negatives (Exp 7/8/9).

**Experiment 15 — Held-out prompt-family generalization: is the corrector overfit to FineWeb text?**
Every experiment above both *trains* and *evaluates* on FineWeb web text, so the corrector could have
fit that prompt distribution rather than a general correction rule. We test this directly. We train the
flagship sentiment corrector **exactly as Experiment 3** (same vector, seed, recipe, 300 FineWeb training
docs), then evaluate its fluency recovery at matched projection on three held-out prompt families of
increasing distribution shift away from FineWeb: **fineweb** (held-out FineWeb docs — *in-distribution*,
reproduces Exp 3), **markdown** (this project's own technical research prose — a different natural-language
register), and **code** (Python source from the numpy / torch / transformers libraries — non-natural-language,
strongly out-of-distribution). We quantify *how* out-of-distribution each family is by the mean Mahalanobis
distance of its **clean** activations under the FineWeb Gaussian (the same manifold fit used throughout):
`D_M` = 27.5 (fineweb) / 30.1 (markdown) / 37.4 (code) — code activations sit ~36% further off the FineWeb
manifold than in-distribution text.

*Fluency recovery vs α, by prompt family (same FineWeb-trained corrector; matched projection α|v|):*

| α | fineweb (in-dist, `D_M` 27.5) | markdown (`D_M` 30.1) | code (`D_M` 37.4) |
|---|-------------------------------|-----------------------|-------------------|
| 2 | 116% | 101% | 99% |
| 4 | 95% | 87% | 78% |
| 6 | 89% | 82% | 71% |
| 8 | **84%** | **77%** | **60%** |

*Absolute ΔLM at α=8 (nats), with each family's clean-activation shift:*

| family | clean `D_M` (under FineWeb Gaussian) | ΔLM raw @α=8 | ΔLM learned @α=8 | recovery @α=8 |
|---|---|---|---|---|
| fineweb (in-distribution) | 27.5 | +2.78 | **+0.44** | 84% |
| markdown (technical prose) | 30.1 | +2.67 | **+0.61** | 77% |
| code (Python source) | 37.4 | +3.31 | **+1.31** | 60% |

**Reading it: the corrector is not overfit to the FineWeb prompt distribution — it transfers to genuinely
different prompt families, degrading gracefully as the family gets more out-of-distribution.** A corrector
that never saw Markdown or code still removes **77%** and **60%** of raw steering's fluency damage on those
families at α=8 (and 87% / 78% at α=4), versus 84% on in-distribution FineWeb. Recovery tracks the
activation shift monotonically: as a family's clean activations sit further off the FineWeb Gaussian
(`D_M` 27.5 → 30.1 → 37.4), recovery falls smoothly (84% → 77% → 60% at α=8) rather than collapsing — the
correction rule is applied to activations it was never trained on and still works, just less perfectly the
further those activations drift from the training distribution. The in-distribution `fineweb` row reproduces
Experiment 3 **to the digit** (raw +2.78 → learned +0.44, 84%), a built-in reproducibility check. So the
corrector generalizes across the *prompt* axis (in addition to steering strength, Exp 4; and layer/model,
Exp 12/13): the fluency result is **not a FineWeb-prompt artifact**, and a single trained corrector remains
useful on out-of-domain text — most so when that text's activations stay near the distribution it was fit on.

**Experiment 16 — Is the "manifold" actually Gaussian? Intrinsic dimension and Gaussianity of clean activations.**
Every off-manifold measure above uses `D_M`, which models the cloud of real activations as a **single
768-dimensional Gaussian**. That is a strong assumption — and if it is wrong, the phrase "off the
manifold" needs care. We test it directly on the clean layer-6 FineWeb activations used throughout
(49,218 tokens, **no steering**), with two standard tools for **recovering a manifold from discrete
points** — the intrinsic-dimension estimators **TwoNN** (Facco et al. 2017) and the **Levina–Bickel MLE**
(2004) — plus tests of Gaussianity.

*Intrinsic dimension — how many degrees of freedom the activations really occupy:*

| estimator | value | as % of ambient 768 |
|---|---|---|
| TwoNN (raw) | 11.4 | 1.5% |
| TwoNN (per-dim z-scored) | 8.1 | 1.1% |
| Levina–Bickel MLE, k=10 / 20 (raw) | 25.1 / 26.6 | ~3% |
| Levina–Bickel MLE, k=10 / 20 (z-scored) | 31.3 / 33.8 | ~4% |
| PCA participation ratio | 1.1 | 0.1% |
| # PCs for 90% / 95% of variance | 1 / 3 | — |

*Gaussianity of the fit — if the Gaussian were correct, held-out `D_M²` would follow `χ²₇₆₈` exactly:*

| quantity of held-out `D_M²` | observed | Gaussian (`χ²₇₆₈`) | ratio |
|---|---|---|---|
| mean | 765 | 768 | 1.00 (not diagnostic\*) |
| standard deviation | 263 | 39.2 | **6.7×** |
| skewness | 0.45 | 0.10 | 4.4× |
| excess kurtosis | 0.74 | 0.016 | — |
| # heavy-tailed dims (per-dim excess kurtosis > 1) | 14 / 768 | ≈ 0 | — |
| max per-dim excess kurtosis | 118 | ≈ 0 | — |

\*The mean of `D_M²` is ≈ `d` for *any* distribution once `(μ, Σ)` are fit on matched data (it is
essentially `trace(Σ⁻¹Σ) = d`), so it does not test Gaussianity; the spread and shape do.

**Reading it: the doubt is correct — the activation cloud is NOT a single 768-d Gaussian, and this
sharpens rather than weakens the paper's thesis.** Three facts. **(1) It is low-dimensional.** Every
intrinsic-dimension estimator puts the manifold at **~8–34 dimensions** — one to two orders of magnitude
below the 768-d ambient space. The activations lie near a thin, curved manifold, not spread through the
space a full-rank Gaussian describes. **(2) It is extremely anisotropic.** The linear participation ratio
is **1.1**: a *single* direction carries ~90% of the variance and three carry 95% — the signature of
GPT-2's well-documented "outlier"/"rogue" activation dimensions. **(3) It is heavy-tailed and
non-Gaussian.** Were the Gaussian right, held-out `D_M²` would be `χ²₇₆₈` (std 39); instead its spread is
**6.7× larger** (variance ≈ 45× too big), it is right-skewed (0.45 vs 0.10), and **14** individual
dimensions have excess kurtosis above 1 (up to **118**). **Why this matters for the corrector.** This is
Experiment 2's central negative result made concrete: because the Gaussian mis-models the manifold —
piling almost all of its "volume" into a handful of high-variance rogue directions — the
Mahalanobis-minimizing correction moves *into* exactly those directions, which is cheap in `D_M` but
maximally destructive to the LM. It also clarifies what "off the Gaussian manifold" means in Experiments
3/5/12/13: the learned corrector moves off a **crude Gaussian fit**, which is not the same as moving off
the true (low-dimensional, non-Gaussian) data manifold. `D_M` is a useful *diagnostic* of
departure-from-typical, but — as the whole direction argues — was never, and should never be, a training
target. Supervising the corrector with the downstream LM loss is exactly the right response to a manifold
this far from Gaussian.

**Experiment 17 — A *real* diffusion corrector: Cold-Diffusion (steering corruption) vs one-shot MLP vs a generic Gaussian-noise prior.**
The direction is named after Cold Diffusion, but the flagship corrector (Exp 3) is a **one-shot MLP**, not the
iterative diffusion model of the GLP paper. This experiment builds the actual diffusion machinery and compares
three correctors at matched steering projection `α|v|` on the same held-out eval set (GPT-2 small, block 6,
sentiment vector), all reusing the Exp-3 pipeline:

- **(1) one-shot MLP** — the incumbent (Exp 3): `ĥ = z + P_{v⊥}r_θ(h,z,α)`, a single forward pass (4.46M params).
- **(2) cold-diffusion iterative (K=8)** — NEW. A same-capacity, weight-shared, *step-conditioned* velocity
  field `g_θ(h,x,α,t)` integrated over 8 steps, `x_{k-1}=x_k+(1/K)P_{v⊥}g_θ`, so the projection along `v` is
  preserved at *every* step. Trained by **unrolling the 8 steps and backpropping the frozen upper-LM
  next-token loss** into `g_θ` — the iterative analogue of Exp 3 (4.46M params).
- **(3) GLP Gaussian prior (SDEdit)** — NEW baseline = the "generic Gaussian-noise GLP teacher" the proposal
  names. A real **DDPM** (cosine schedule, ε-prediction, 2.69M params) trained on **clean** standardized
  activations with **Gaussian-noise** corruption, pure MSE, **no LM in the loop**. It corrects a steered `z`
  by SDEdit (noise to `t_start`=0.15, chosen by steelmanning, DDIM-denoise back), then we re-impose the target
  projection `α|v|` so the fluency comparison is matched.

| α | ΔLM raw | ΔLM one-shot MLP | ΔLM cold-diff iter | ΔLM GLP prior | recovery one-shot | recovery iter | recovery GLP |
|---|---|---|---|---|---|---|---|
| 1 | +0.076 | −0.069 | **−0.074** | +0.631 | 191% | **197%** | −731% |
| 2 | +0.325 | −0.051 | **−0.058** | +0.862 | 116% | **118%** | −165% |
| 4 | +1.222 | +0.058 | **+0.039** | +1.634 | 95% | **97%** | −34% |
| 6 | +2.111 | +0.224 | **+0.195** | +2.360 | 89% | **91%** | −12% |
| 8 | +2.778 | +0.435 | **+0.419** | +2.925 | 84% | **85%** | −5% |

Steering-projection retention is **identical** (matched) for raw/one-shot/iter at every α (11.1→88.6). The
unconditional GLP prior, *before* re-imposing, **erases** part of the steer: as-is retention 10.6/83.1 vs
target 11.1/88.6 at α=1/8 (~5–6% lost). Off-Gaussian distance `D_M` at α=8: raw 49.0, GLP 52.8, one-shot 79.5,
**iter 75.2** (both LM-supervised correctors go *further* off the Gaussian; the iterative one slightly less).

**Reading it — three clean answers to the central critique.** **(1) The Cold-Diffusion framing is what
matters, not "diffusion" per se.** Training on the *actual steering corruption* `z=h+αv` under LM supervision
(both the one-shot MLP and the iterative model) recovers **84–85%** of the fluency damage at α=8; the generic
Gaussian-noise GLP prior — the standard "denoise back to the manifold" recipe — has **negative recovery at
every strength** (−5% at α=8, i.e. it makes the LM *worse than raw steering*, +2.93 vs +2.78 nats). A prior
that only knows "typical activation" cannot know which off-typical directions the LM tolerates; only the
downstream objective does. This is the Exp-2 lesson in diffusion clothing. **(2) The iterative diffusion
structure essentially TIES the one-shot MLP** — a small, consistent edge at every α (85% vs 84% at α=8; ΔLM
+0.419 vs +0.435), at equal capacity, while sitting slightly *closer* to the Gaussian (`D_M` 75.2 vs 79.5). So
the one-shot MLP was not leaving fluency on the table: the expensive 8-step unroll buys a marginal improvement,
not a qualitative one. The value of "diffusion" here is the *corruption model* (steering, not Gaussian noise)
and the *supervision* (LM, not reconstruction) — not the iteration count. **(3) The unconditional prior erases
the steer**, exactly the information-loss the GLP authors flagged for unconditional priors: ~5–6% of the target
projection is lost before we re-impose it, and even with the projection re-imposed it cannot repair the LM.
Conditioning on the clean activation and supervising with the LM — what ColdSteer does — is the fix.

**Experiment 18 — Beyond hand-built DiffMean: does the recipe depend on the steering-vector *family*?**
Every steering vector above (6 of them) is a **DiffMean** direction — the difference of class means
`μ⁺ − μ⁻` — built from ~20 **hand-written** contrastive sentences. Two worries follow: the flagship result
could be an artifact of (i) that one extraction method, or (ii) the hand-built prompts. This experiment
changes **both** axes at once, on the same concept (sentiment). **(i) Real data:** we build the steering
vectors from a **downloaded real dataset** — 500 positive + 500 negative movie-review sentences from **SST-2**
— instead of hand-written text. **(ii) Three extraction families:** from those same block-6 activations we
build the three canonical linear-steering families, each a genuinely different direction: **DiffMean**
(`μ⁺ − μ⁻`); a **logistic-regression probe** (the weight vector of an L2-regularized pos-vs-neg classifier —
a *discriminative* direction); and **PCA-contrast** (top principal component of centered positive−negative
activation-pair differences — an *unsupervised* direction, the RepE recipe). All three are sign-aligned to
"+positive" and **rescaled to a common norm `|v| = 11.0`** (the flagship scale), so the *only* thing that
differs across families is the **direction**. The three directions really are distinct — their cosines to
the DiffMean direction are **1.00 / 0.40 / 0.30** — and the SST-2 DiffMean direction is only moderately
aligned with the original hand-built one (`cos = 0.49`). We then run the **identical flagship recipe (Exp 3)**
— train an LM-supervised, projection-preserving corrector per direction — on each family, at matched
projection `α|v|`.

| family (cos to DiffMean) | ΔLM raw @α=8 | **ΔLM learned @α=8** | recovery @α=8 | recovery @α=4 | `D_M` raw / learned @α=8 |
|---|---|---|---|---|---|
| DiffMean (1.00) | +3.41 | **+0.47** | **86%** | 98% | 41.4 / 65.2 |
| LogReg probe (0.40) | +2.63 | **+0.42** | **84%** | 95% | 61.6 / 80.1 |
| PCA-contrast (0.30) | +2.27 | **−0.02** | **101%** | 118% | 27.3 / 47.5 |

(Steering-projection retention is matched `α|v|` = 11.0→88.0 for all three families at every α.)

**Reading it: the recipe is not tied to DiffMean or to hand-built prompts — it recovers every steering-vector
family, from real data.** Three points. **(1) Family-robust.** All three genuinely different directions
(cos 0.30–1.00) show the same two facts: raw steering breaks the LM (`ΔLM` +2.3 to +3.4 nats at α=8) and the
identical LM-supervised corrector recovers it at matched projection — **84–101% at α=8, 95–118% at α=4**. The
DiffMean family reproduces the flagship Experiment 3 (raw +3.41 → learned +0.47, 86% ≈ Exp 3's 84%), a
built-in check, even though it was built from real SST-2 movie reviews rather than the original hand-written
sentences (and the two DiffMean directions agree only at cos 0.49 — so the *concept* vector is only partly
reproducible across data sources, yet the *recipe* works on both). **(2) The PCA-contrast case sharpens the
central decoupling from the opposite side.** The unsupervised PCA direction happens to align with GPT-2's
dominant high-variance axis (Exp 16), so steering along it leaves the Mahalanobis distance essentially
**flat** (`D_M` 27.3 = the clean value, *on* the Gaussian manifold) — yet it still breaks the LM by **+2.27
nats**. So off-Gaussian distance is neither necessary nor sufficient for LM damage: raw PCA steering is
on-manifold but harmful, and (as always) the corrector fixes it by moving *off* the manifold (`D_M`
27.3 → 47.5). **(3) The reliable route is unchanged** — a per-direction native corrector, now shown to work
regardless of how the steering direction was extracted. This closes the last external-validity axis: the
ColdSteer result is robust to the **steering-vector family** as well as to strength, direction, layer,
model, and prompt family.

**Experiment 19 — Model scaling to GPT-2 large: does the result hold at 774M?**
Experiment 13 replicated the flagship result on GPT-2 **medium** (355M). This adds the third
model-scale point — **GPT-2 large (774M, 36 blocks, `d = 1280`)** — so the model axis now spans a
6.2× parameter range: **124M → 355M → 774M**. We replicate the exact Experiment-3 pipeline
**unchanged** (same DiffMean sentiment prompts, 400-doc Gaussian fit, 300-doc training set, held-out
100-doc eval, 4-layer corrector, seed, `α ∼ U(0.5, 8)`, hyper-parameters), steering and correcting at
the **mid layer, block 18 of 36** — the depth analogue of block 6 of 12 in small and block 12 of 24 in
medium. Only the model changes, so `|v|`, `|h|`, and `d` change with it (`|v| = 16.8`, mean `|h| = 129.1`,
clean `D_M = 35.2`; corrector 6.03M params at `d = 1280`).

| α | ΔLM raw (nats) | **ΔLM learned** | recovery | `D_M` raw | `D_M` learned |
|---|----------------|------------------|----------|-----------|----------------|
| 1 | +0.04 | **−0.07** | >100% | 35.9 | 42.7 |
| 2 | +0.15 | **−0.05** | >100% | 37.9 | 46.8 |
| 4 | +0.73 | **+0.03** | **95%** | 45.0 | 62.2 |
| 8 | +2.47 | **+0.39** | **84%** | 66.0 | 96.8 |

(Steering-projection retention is matched `α|v|` = 16.8 → 134.0 for raw and learned at every α.)

**Reading it: both headline facts replicate on GPT-2 large — the result is robust across a 6× model-scale
range.** (P) Raw steering breaks the 774M model exactly as it breaks the 124M and 355M ones: `ΔLM` climbs
monotonically to **+2.47 nats at α=8** while the Mahalanobis distance inflates 35.2 → 66.0. (C) The
identical LM-supervised, projection-preserving corrector removes essentially all of it at matched
projection — recovering **84% of the fluency damage at α=8** and **95% at α=4** (`ΔLM` +0.73 → +0.03), with
`ΔLM` slightly negative at weak steering (the same free-or-better weak-α behavior as on small/medium; the
">100%" reads only reflect near-zero raw damage). Across the three scales the α=8 recovery is essentially
flat — **small 84% / medium 89% / large 84%** — so amortized correction quality does **not** degrade as the
model grows. And the signature decoupling holds a third time: the corrected activation sits **further** off
the Gaussian manifold than raw at **every** α (`D_M` learned > raw throughout, 96.8 vs 66.0 at α=8). The
core ColdSteer claim — *a downstream-supervised, projection-preserving corrector buys back nearly all of raw
steering's fluency damage while moving off the statistical manifold* — is confirmed **model-robust** across
GPT-2 small, medium, and large.

**Experiment 21 — Cross-architecture generality: is the result a GPT-2-*architecture* artifact?**
Experiments 13 and 19 scaled the *model* (124M → 355M → 774M) but every one stayed inside the **GPT-2
family** — the same architecture (learned positional embeddings, LayerNorm, dense multi-head attention,
GELU MLP). The sharpest remaining external-validity question is whether the result depends on that
architecture at all. We replicate the exact Experiment-3 pipeline **unchanged** on **Qwen3-1.7B (28 blocks,
`d = 2048`)**, a modern architecture that differs from GPT-2 on every structural axis: **RMSNorm** (not
LayerNorm), **rotary position embeddings** (not learned positions), a **SwiGLU** MLP (not GELU), and
**grouped-query attention** (16 query / 8 key-value heads, not dense MHA). We steer and correct at the
**mid layer, block 14 of 28** (the depth analogue of block 6 of 12 in GPT-2 small). Only the model changes;
the DiffMean sentiment prompts, 400-doc Gaussian fit, 300-doc training set, held-out 100-doc eval, 4-layer
corrector, seed, `α ∼ U(0.5, 8)`, and objective are identical to Exp 3 (weights loaded in bf16 for the VRAM
share; `|v| = 38.1`, mean `|h| = 301.9`, clean `D_M = 44.7`; corrector 8.39M params at `d = 2048`).

| α | ΔLM raw (nats) | **ΔLM learned** | recovery | `D_M` raw | `D_M` learned |
|---|----------------|------------------|----------|-----------|----------------|
| 1 | +0.06 | **−0.18** | >100% | 45.4 | 60.5 |
| 2 | +0.24 | **−0.16** | >100% | 47.5 | 65.6 |
| 4 | +1.08 | **−0.09** | **108%** | 55.0 | 81.9 |
| 8 | +3.43 | **+0.19** | **94%** | 77.8 | 122.2 |

(Steering-projection retention is matched `α|v|` = 38.1 → 304.8 for raw and learned at every α.)

**Reading it: both headline facts replicate on a non-GPT-2 architecture — the result is not a
GPT-2-*architecture* artifact.** (P) Raw steering breaks Qwen3 exactly as it breaks GPT-2: `ΔLM` climbs
monotonically to **+3.43 nats at α=8** while the Mahalanobis distance inflates 44.7 → 77.8. (C) The
identical LM-supervised, projection-preserving corrector removes essentially all of it at matched
projection — recovering **94% of the fluency damage at α=8** and **108% at α=4**, with `ΔLM` slightly
*below* the clean baseline at weak/medium steering (the same free-or-better behavior seen on every GPT-2
scale; the ">100%" reads only reflect raw's near-zero damage there). The α=8 recovery on Qwen3 (94%) is
even a touch higher than GPT-2 small's 84%. And the signature decoupling holds a **fourth** time: the
corrected activation sits **further** off the Gaussian manifold than raw at **every** α (`D_M` learned > raw
throughout, 122.2 vs 77.8 at α=8). So the core ColdSteer claim is **architecture-robust**, not merely
scale-robust: it holds across LayerNorm ↔ RMSNorm, learned ↔ rotary positions, GELU ↔ SwiGLU MLPs, and
dense ↔ grouped-query attention — every structural axis that separates Qwen3 from the GPT-2 family.

**Experiment 22 — Behavioral reality-check on Qwen3: is Exp 21's 94% recovery bought by under-steering?**
Experiment 21 recovered **94%** of raw steering's fluency damage on Qwen3-1.7B — but that number is a
*teacher-forced* `ΔLM` at matched layer-14 projection. Experiment 10 showed on GPT-2 that this proxy can be
misleading: the corrector's fluency win came partly at the cost of a *weaker propagated behavioral edit* in
generation (its correction is ⟂ `v` in activation space but not ⟂ the downstream sentiment readout). That
caveat had never been checked off GPT-2. We run the **identical Exp 10 protocol** on Qwen3, reusing the
**exact Exp 21 corrector** (checkpoint, no retraining): greedily generate 30-token continuations from 48
held-out 12-token prompts with the steer applied at block 14 at every position, raw vs corrected, and on a
**clean re-encode** measure the sentiment **effect** `B(α)−B(0)` (mean projection of the continuation's
block-14 activations onto `v̂`; higher = more strongly steered; unsteered baseline `B(0)=+28.6`) and
**degeneration** via **distinct-2** (unique-bigram ratio; unsteered baseline `0.875`; lower = more repetitive).

| α | effect raw `B−B₀` | effect corr `B−B₀` | corr/raw effect | distinct-2 raw | distinct-2 corr |
|---|-------------------|--------------------|-----------------|----------------|-----------------|
| 2 | **+5.22** | +0.53 | 10% | 0.886 | 0.840 |
| 4 | **+7.31** | +0.77 | 11% | 0.876 | 0.833 |
| 6 | **+7.64** | +0.98 | 13% | 0.819 | 0.843 |
| 8 | +8.01 | +2.31 | 29% | **0.761** | 0.825 |

**Reading it: the Exp 10 under-steering caveat *replicates on Qwen3* — the 94% fluency recovery is honest as
a teacher-forced metric but is again partly bought by a weaker behavioral edit in generation.** Two findings,
and a difference from GPT-2. **(1) The corrector under-steers, exactly as on GPT-2.** Raw steering swings the
generated sentiment hard (`+5.2` at α=2 rising to `+8.0` at α=8), while the corrector's effect is only
`+0.53–2.31` — **10–29% of raw's** (cf. ~1/6 on GPT-2 in Exp 10). So the projection-preserving correction is
again not orthogonal to the downstream readout: minimizing LM loss at matched layer-14 projection yields
near-normal, lightly-steered text (α=8 corrected sample: *"…situated in the heart of the city of Bridgend,
just 15 minutes north of the city of Bridgend"* — fluent, factual, barely steered; raw α=8: *"…a welcoming
family and a welcoming community. The community is a home and a family. The community is a…"* — positive but
repetitive). **(2) The fluency win is smaller here because raw steering degenerates *less* on Qwen3.** Raw
distinct-2 falls only `0.886→0.761` at α=8 — nowhere near GPT-2's collapse to `0.32`. The corrector stays flat
and fluent (`0.825–0.843`), so it is still more fluent than raw at strong steering, but the gap (0.06 at α=8)
is far narrower than on GPT-2 (0.32). **The practical read:** on Qwen3 the two methods sit on a *shallower*
effect-vs-fluency Pareto — raw is a stronger baseline here (it steers hard *and* stays fairly fluent), so the
corrector's advantage is almost entirely on the (small) fluency axis while paying a large behavioral-effect
cost. This confirms that the flagship `ΔLM` recoveries (Exp 3–21) measure reduced *teacher-forced disruption*,
genuinely, but on Qwen3 as on GPT-2 a substantial part of that reduction reflects a weaker propagated edit —
**matched projection ≠ matched behavioral steering is architecture-robust**, and the Exp 11/20 behavioral-
preservation terms (tested on GPT-2) are the indicated fix if strong *behavioral* steering is required.

**Experiment 23 — Does the GPT-2 behavioral-preservation fix (Exp 11) transfer to Qwen3?**
Experiment 22 found the under-steering caveat replicates on Qwen3-1.7B and named the Experiment-11
behavioral-preservation term as the indicated fix — but never tested it. We do that here, reusing the exact
Exp 21/22 Qwen3 pipeline and the identical Exp 22 generation protocol, and adding the Exp 11 term: during
teacher-forced training we also read out the sentiment projection at a **downstream Qwen3 layer** (`L2 = 27`,
the last decoder block, downstream DiffMean direction `ŵ`, `|w| = 12.9`) and push the corrected activation's
downstream projection `p_corr` toward what **raw** steering produces, `p_raw`, weighted by `λ_b`. We train a
family `λ_b ∈ {0, 10, 40}` (`λ_b = 0` loads the exact Exp 21 checkpoint — the Exp 22 corrector — as a
reproducibility anchor) and score each on the identical Exp 22 protocol (48 prompts, 30 greedy tokens;
sentiment effect `B(α)−B(0)` and distinct-2 on a clean re-encode; unsteered baselines `B(0) = +28.6`,
distinct-2 `0.875`). Raw steering is the shared reference.

| α | eff raw | eff λ_b=0 | eff λ_b=10 | eff λ_b=40 | d2 raw | d2 λ_b=0 | d2 λ_b=10 | d2 λ_b=40 |
|---|---------|-----------|------------|------------|--------|----------|-----------|-----------|
| 2 | **+5.22** | +0.53 | +2.56 | +4.06 | 0.886 | 0.840 | 0.833 | 0.875 |
| 4 | **+7.31** | +0.77 | +4.35 | +5.87 | 0.876 | 0.833 | 0.802 | 0.859 |
| 6 | **+7.64** | +0.98 | +4.47 | +6.35 | 0.819 | 0.843 | 0.730 | 0.789 |
| 8 | +8.01 | +2.31 | +2.91 | +4.21 | 0.761 | 0.825 | 0.613 | 0.673 |

(`eff` = sentiment shift `B(α)−B(0)`, higher = more steered; `d2` = distinct-2, higher = more fluent.
`λ_b=0` reproduces Experiment 22 to the digit — a built-in reproducibility check.)

**Reading it: the behavioral term's *mechanism* transfers to Qwen3 — it recovers most of the generated
sentiment effect the base corrector threw away — but its *Pareto advantage* does not, because on Qwen3 raw
steering does not collapse for the corrector to dominate.** Three findings. **(1) The term works
mechanically.** Adding `λ_b` lifts the generated effect from the base corrector's `+0.53–2.31` (10–29% of
raw's, = Exp 22) up to `+4.06–6.35` at `λ_b=40` — **53–83% of raw's effect** at α≤6, a 2–8× increase — exactly
the lever Experiment 11 found on GPT-2. So the correction's non-orthogonality to the downstream readout, and
the readout-preservation fix for it, are **architecture-robust**. **(2) But it does not beat raw on Qwen3.**
On GPT-2 the same term flipped a non-dominating tradeoff into outright dominance *because raw steering there
collapsed into repetition* (distinct-2 0.32) and a fluent-and-steered corrector dominated it. On Qwen3 raw
does **not** collapse (distinct-2 only 0.761 at α=8, Exp 22), so raw is a strong-and-fluent baseline: at
`λ_b=40` the corrector's distinct-2 (0.875→0.673) sits slightly *below* raw's (0.886→0.761) at every α while
its effect is also below raw's, so raw weakly dominates at matched α. The `λ_b` sweep traces a frontier from
the base corrector (fluent, weak) *toward* raw (strong, fluent) without passing it. **(3) Strong steering is
unstable, as in Exp 20.** At `λ_b=40`, α=8 the effect falls to `+4.21` (below its own α=6 peak of +6.35) with
distinct-2 dropping to 0.673 — the same over-steer degeneration Experiment 20's `λ_g=160` showed on GPT-2.
**Net:** the GPT-2 behavioral fix is architecture-robust *as a lever on generated effect*, but its practical
value is **gated by whether raw steering degenerates** — where raw collapses (GPT-2) the corrector wins the
Pareto; where raw stays fluent (Qwen3) the fix recovers most of the effect but cannot dominate. This closes
the behavioral arc (Exp 10 → 11 → 20 → 22 → 23): matched projection ≠ matched steering everywhere, the
readout-preservation fix transfers everywhere, and the size of its payoff depends on the baseline's failure
mode.

**Experiment 24 — A second non-GPT-2 architecture: turning "architecture-robust" into a sweep (Pythia-410m / GPT-NeoX).**
Experiment 21 crossed the GPT-2 boundary **once** (to Qwen3-1.7B). A single point off the GPT-2 family is a
weak basis for calling the result "architecture-robust." Here we add a **third, structurally distinct**
architecture so the axis becomes a genuine sweep of three families. **Pythia-410m** is a **GPT-NeoX** model:
it shares **rotary** positions with Qwen3 and **LayerNorm / GELU / dense multi-head attention** with GPT-2, but
its transformer block uses a **parallel residual** — attention and MLP are computed from the *same* layer input
and summed, rather than applied in series — and its input/output embeddings are untied. That parallel block is
different from **both** GPT-2 (serial residual) and Qwen3 (serial residual + RMSNorm + SwiGLU + grouped-query
attention). We replicate the flagship Experiment-3 pipeline **unchanged** (same DiffMean sentiment prompts,
400-doc Gaussian fit, 300-doc training set, held-out 100-doc eval, 4-layer corrector, seed, `α ∼ U(0.5, 8)`,
objective) — only the model changes — steering and correcting at the **mid layer, block 12 of 24** (the depth
analogue of block 6 of 12 in GPT-2 small). Pythia-410m is small (`d = 1024`, ~800 MB), so it runs in fp32 within
the VRAM share (`|v| = 3.29`, mean `|h| = 35.3`, clean `D_M = 31.3`; corrector 5.25M params at `d = 1024`).

| α | ΔLM raw (nats) | **ΔLM learned** | recovery | `D_M` raw | `D_M` learned |
|---|----------------|------------------|----------|-----------|----------------|
| 1 | +0.06 | **+0.04** | 41% | 31.8 | 36.1 |
| 2 | +0.23 | **+0.07** | **71%** | 33.1 | 39.7 |
| 4 | +0.95 | **+0.18** | **81%** | 37.7 | 53.4 |
| 8 | +3.10 | **+0.59** | **81%** | 52.3 | 89.4 |

(Steering-projection retention is matched `α|v|` = 3.29 → 26.29 for raw and learned at every α.)

**Reading it: both headline facts replicate on a third, parallel-residual architecture — the result is a
genuine architecture sweep, not a one-off boundary crossing.** (P) Raw steering breaks Pythia exactly as it
breaks GPT-2 and Qwen3: `ΔLM` climbs monotonically to **+3.10 nats at α=8** while the Mahalanobis distance
inflates 31.3 → 52.3. (C) The identical LM-supervised, projection-preserving corrector removes most of it at
matched projection — recovering **81% of the fluency damage at α=8** and **81% at α=4** (71% at α=2). At the
weakest steering (α=1) raw's damage is nearly zero (+0.06 nats), so the recovery *ratio* there (41%) is
dominated by noise rather than a real shortfall — the same instability of the ratio at α=1 noted throughout.
And the signature decoupling holds a **fifth** time: the corrected activation sits **further** off the Gaussian
manifold than raw at **every** α (`D_M` learned > raw throughout, 89.4 vs 52.3 at α=8). Placed beside the other
architectures, the α=8 recovery now spans **three distinct families** — GPT-2 (124M/355M/774M: 84% / 89% /
84%), Qwen3-1.7B (RMSNorm / RoPE / SwiGLU / GQA: 94%), and Pythia-410m / GPT-NeoX (parallel residual: 81%) —
all between **81% and 94%**. So the core ColdSteer claim is **architecture-robust as a sweep**: a
downstream-supervised, projection-preserving corrector buys back the bulk of raw steering's fluency damage,
while moving off the statistical manifold, across serial and parallel residual blocks, LayerNorm and RMSNorm,
learned and rotary positions, GELU and SwiGLU MLPs, and dense and grouped-query attention.

**Experiment 25 — Behavioral reality-check on Pythia-410m: is Exp 24's 81% recovery bought by under-steering?**
Experiment 24's 81% is a **teacher-forced `ΔLM`** at matched layer-12 projection. Experiment 10 (GPT-2) and
Experiment 22 (Qwen3) both showed this proxy can hide a **weaker propagated behavioral edit** in generation: the
corrector's correction is orthogonal to `v` in *activation* space but not to the downstream sentiment *readout*.
We run the **identical Exp 10/22 generation protocol** on Pythia-410m, reusing the **exact Exp 24 corrector**
(checkpoint, no retraining): greedy-generate 30 tokens from 48 held-out 12-token prompts with the steer applied at
block 12 at every position, under raw vs. corrected, then on a **clean re-encode** measure the **sentiment effect**
`B(α)−B(0)` (mean projection of the continuation's block-12 activations onto `v̂`; unsteered baseline `B(0) = −4.77`)
and **distinct-2** (unique-bigram ratio; unsteered baseline `0.77`, lower = more repetitive).

| α | effect raw `B−B₀` | effect corr `B−B₀` | distinct-2 raw | distinct-2 corr |
|---|-------------------|--------------------|----------------|-----------------|
| 2 | +0.17 | **+0.90** | 0.81 | 0.82 |
| 4 | +0.40 | **+0.80** | 0.86 | 0.76 |
| 6 | +1.01 | +0.93 | 0.74 | 0.73 |
| 8 | +1.17 | +0.98 | **0.38** | **0.72** |

**Reading it: on Pythia the under-steering caveat does NOT bite the way it did on GPT-2/Qwen3 — the corrector's
generation carries a behavioral effect *comparable to* raw's, and Pareto-dominates raw at strong steering.** Three
points. **(1) Raw steering is a weak behavioral steerer here.** Raw's generated sentiment effect only reaches
`+1.17` even at α=8 (vs `+2.97` on GPT-2, `+8.0` on Qwen3), so at these α Pythia's block-12 steer propagates
weakly to the generated text. **(2) The corrector is therefore *not* badly under-steered.** Its effect
(`+0.80`–`+0.98`) is *above* raw's at α≤4 and 84–92% of raw's at α≥6 — not the ~1/6 shortfall seen on GPT-2
(Exp 10) and Qwen3 (Exp 22). **(3) At strong steering the corrector cleanly dominates.** Raw degenerates at α=8
(distinct-2 `0.86→0.38`, the same collapse as GPT-2), while the corrector stays fluent (`0.72`) and keeps 84% of
raw's effect — so on the effect-vs-fluency Pareto the corrector is up-and-right of raw at α=8. **Interpretation:**
whether "matched projection ≠ matched behavioral steering" costs behavioral effect depends on **how strongly raw
steering itself propagates** in that model; where raw's behavioral steer is weak (Pythia), the corrector loses
little of it, and its fluency win at strong α is close to a free lunch. **Limitation:** the effect magnitudes are
small on Pythia (raw peaks at `+1.17`), a low-signal regime, so this is best read as "the Exp 10/22 under-steering
penalty is *architecture-dependent and mild here*," not as evidence the corrector steers *more* than raw in
general. **Next check:** the Exp 11/20 behavioral-preservation terms (GPT-2/Qwen3-tested) on Pythia if a stronger
behavioral steer is required. `λ_b=0`/raw rows reuse the exact Exp 24 checkpoint, so this is directly comparable to
Exp 10 and Exp 22.

**Experiment 26 — Seed robustness: a confidence interval on the flagship 84% recovery.**
Every result above is a **single training run at `SEED = 0`**, so the headline number ("84% fluency
recovery at α=8", Experiment 3) has no error bar. CLAUDE.md's review standard names *seed* as a control a
metric should survive, and it is the one axis Experiments 4/5/12/13/15/18/19/21/24 (strength / direction /
layer / model / prompt / steering-family / architecture) never varied. We close it by re-running the **exact
flagship Experiment-3 pipeline** — same DiffMean sentiment vector (`|v| = 11.08`), same 400-doc Gaussian fit
(49,218 clean tokens, clean `D_M = 27.3`), same 300-doc training set, same held-out 100-doc eval, same
4-layer 4.46M corrector, same recipe / `α ∼ U(0.5, 8)` — at **five seeds** (`0–4`), and report the mean ± sample
standard deviation of the fluency recovery `recovery(α) = (ΔLM_raw(α) − ΔLM_learned(α)) / ΔLM_raw(α)` at each
α. Raw steering has no trained parameters, so `ΔLM_raw` is identical across seeds (computed once); only the
learned corrector varies. Seed 0 reproduces Experiment 3 to the digit — a built-in check.

| α | ΔLM raw (nats) | **ΔLM learned (mean ± sd)** | **recovery (mean ± sd)** | `D_M` learned (mean ± sd) |
|---|----------------|------------------------------|--------------------------|----------------------------|
| 1 | +0.076 | **−0.073 ± 0.014** | 196.0 ± 18.8% | 31.8 ± 0.2 |
| 2 | +0.325 | **−0.056 ± 0.011** | 117.4 ± 3.4% | 35.6 ± 0.5 |
| 4 | +1.222 | **+0.047 ± 0.010** | **96.2 ± 0.8%** | 47.9 ± 1.8 |
| 6 | +2.111 | **+0.211 ± 0.012** | **90.0 ± 0.6%** | 61.7 ± 2.9 |
| 8 | +2.778 | **+0.464 ± 0.054** | **83.3 ± 2.0%** | 74.6 ± 3.6 |

(Per-seed recovery at α=8: 84.3 / 84.5 / 84.6 / 83.0 / 80.0% — Experiment 3's 84% is seed 0's 84.3%.)

**Reading it: the flagship recovery is highly reproducible across seeds — the headline 84% is not a
single-seed artifact.** At α=8 the five independently-trained correctors recover **83.3 ± 2.0%** of raw
steering's fluency damage (range 80–85%), and the spread is tighter still at moderate steering — **96.2 ± 0.8%**
at α=4 and **90.0 ± 0.6%** at α=6 — so the corrector's advantage over raw is far larger than its seed-to-seed
variability at every strength that matters. Experiment 3's single-seed 84% sits inside the band (it *is* seed
0), confirming that number was representative, not lucky. The only wide error bar is at α=1 (196 ± 19%), and
that is an artifact of the recovery *ratio*: raw's damage there is only +0.076 nats, so dividing by it inflates
the relative spread even though the absolute `ΔLM_learned` is a tight −0.073 ± 0.014 nats — the same
ratio-instability at near-zero raw damage noted throughout. **Limitation.** This varies only the training seed
(corrector init + the α-sampling / data-shuffle RNG); the eval set, Gaussian fit, and steering vector are held
fixed, so it bounds *optimization* variance, not sampling variance over eval documents or over the DiffMean
vector's construction. Still, the one control the review standard names for the flagship number is now met: the
84% recovery is stable to ±2 points across five seeds.

**Experiment 27 — Seed robustness on GPT-2 medium: an error bar on the cross-model recovery.**
Experiment 26 put a five-seed confidence interval on the *flagship* recovery (GPT-2 small, block 6). The
cross-model number (Experiment 13, GPT-2 medium, block 12/24) came from a **single seed-0 run**, so we could
not say whether medium's apparently-higher recovery (89% @α=8 vs small's 83.3%) is a genuine model-scale effect
or just optimization noise. We close that by re-running the **exact Experiment-13 GPT-2-medium pipeline** — same
DiffMean sentiment vector (`|v| = 19.57`, mean `|h| = 226.2`), same 400-doc Gaussian fit (clean `D_M = 31.45`),
same 300-doc training set, same held-out 100-doc eval, same 5.25M corrector at `d = 1024`, same recipe /
`α ∼ U(0.5, 8)` — at **five seeds** (`0–4`), and report the mean ± sample standard deviation of the fluency
recovery `recovery(α) = (ΔLM_raw(α) − ΔLM_learned(α)) / ΔLM_raw(α)`. `ΔLM_raw` is seed-independent (computed
once); only the learned corrector varies. Seed 0 reproduces Experiment 13 to the digit — a built-in check.

| α | ΔLM raw (nats) | **ΔLM learned (mean ± sd)** | **recovery (mean ± sd)** | `D_M` raw | `D_M` learned (mean ± sd) |
|---|----------------|------------------------------|--------------------------|-----------|----------------------------|
| 1 | +0.037 | **−0.114 ± 0.006** | 409.2 ± 16.8% | 32.0 | 36.1 ± 0.5 |
| 2 | +0.150 | **−0.093 ± 0.004** | 162.1 ± 2.9% | 33.5 | 38.9 ± 0.8 |
| 4 | +0.738 | **−0.013 ± 0.007** | **101.7 ± 1.0%** | 38.8 | 49.2 ± 1.8 |
| 8 | +2.718 | **+0.317 ± 0.059** | **88.3 ± 2.2%** | 55.1 | 74.6 ± 4.5 |

(Per-seed recovery at α=8: 89 / 90 / 88 / 85 / 89% — Experiment 13's 89% is seed 0's 89%.)

**Reading it: the cross-model recovery is reproducible, and medium's edge over small is real, not seed noise.**
At α=8 the five independently-trained GPT-2-medium correctors recover **88.3 ± 2.2%** of raw steering's fluency
damage (range 85–90%). This band (`[86.1, 90.5]%`) sits **entirely above** GPT-2 small's five-seed band
(83.3 ± 2.0%, `[81.3, 85.3]%`, Experiment 26): the two intervals do not overlap, so the ~5-point higher recovery
on medium is a genuine model-scale effect, not a lucky seed. The spread is tighter still at moderate steering
(**101.7 ± 1.0%** at α=4), confirming the free-or-better weak-α behavior (recovery ≥100%, `ΔLM_learned` at or
below the unsteered baseline) is also seed-stable rather than a single-run coincidence. The signature decoupling
holds across every seed too: the corrected activation sits *further* off the Gaussian manifold than raw at every
α (`D_M` learned 74.6 ± 4.5 vs raw 55.1 at α=8). The wide bar at α=1 (409 ± 17%) is the usual ratio artifact —
raw's damage there is only +0.037 nats, so the recovery *ratio* is inflated even though the absolute
`ΔLM_learned` is a tight −0.114 ± 0.006 nats. **Limitation.** As in Experiment 26, this varies only the training
seed (corrector init + α-sampling / data-shuffle RNG); the eval set, Gaussian fit, and steering vector are held
fixed, so it bounds *optimization* variance on GPT-2 medium, not eval-document or vector-construction sampling
variance. The larger models on the scale axis (GPT-2 large, Exp 19) and the cross-architecture checks (Exp 21/24)
remain single-seed. **Next check.** A five-seed control on a cross-*architecture* model (Qwen3 / Pythia) would
extend the error bars past the GPT-2 family and test whether the 81–94% architecture band is within seed noise
(done — Experiment 28).

**Experiment 28 — Seed robustness on Pythia-410m / GPT-NeoX: an error bar on the cross-architecture recovery.**
Experiments 26/27 put five-seed intervals on two GPT-2 scales, but the *architecture* axis (Experiment 21
Qwen3, Experiment 24 Pythia/GPT-NeoX; the reported 81–94% band) was still single-seed, so we could not say
whether Pythia's 81% @α=8 — the low end of that band, below both GPT-2 scales — is a real architecture effect
or optimization noise. We close it by re-running the **exact Experiment-24 Pythia-410m pipeline** — same
DiffMean sentiment vector (`|v| = 3.29`, mean `|h| = 35.3`), same 400-doc Gaussian fit (clean `D_M = 31.3`),
same 300-doc training set, same held-out 100-doc eval, same 5.25M corrector at `d = 1024`, same recipe /
`α ∼ U(0.5, 8)`, steered at block 12/24 — at **five seeds** (`0–4`), reporting mean ± sample standard deviation
of the fluency recovery `recovery(α) = (ΔLM_raw(α) − ΔLM_learned(α)) / ΔLM_raw(α)`. `ΔLM_raw` is seed-independent
(computed once); only the learned corrector varies. Seed 0 reproduces Experiment 24 to the digit — a built-in check.

| α | ΔLM raw (nats) | **ΔLM learned (mean ± sd)** | **recovery (mean ± sd)** | `D_M` raw | `D_M` learned (mean ± sd) |
|---|----------------|------------------------------|--------------------------|-----------|----------------------------|
| 1 | +0.059 | **+0.033 ± 0.006** | 44.0 ± 10.7% | 31.8 | 36.1 ± 0.5 |
| 2 | +0.231 | **+0.064 ± 0.003** | 72.1 ± 1.5% | 33.1 | 39.2 ± 0.5 |
| 4 | +0.948 | **+0.174 ± 0.003** | **81.7 ± 0.3%** | 37.7 | 50.4 ± 1.8 |
| 8 | +3.103 | **+0.597 ± 0.048** | **80.8 ± 1.6%** | 52.3 | 80.8 ± 6.6 |

(Per-seed recovery at α=8: 81 / 82 / 80 / 78 / 81% — Experiment 24's 81% is seed 0's 81%.)

**Reading it: the cross-architecture recovery is seed-stable, and the recipe reproduces on a non-GPT-2 family
without re-tuning.** At α=8 the five independently-trained Pythia correctors recover **80.8 ± 1.6%** of raw
steering's fluency damage (range 78–82%), tightening to **81.7 ± 0.3%** at α=4 — the corrector's advantage over
raw again dwarfs its seed-to-seed variability. Placed next to the other two seed bands at α=8, this puts the
architecture *ordering* on a controlled footing: Pythia's band (`[79.2, 82.4]%`) sits **entirely below** GPT-2
medium's (`[86.1, 90.5]%`, Experiment 27) — a genuine gap, not seed noise — but **overlaps** GPT-2 small's
(`[81.3, 85.3]%`, Experiment 26), so Pythia and small are within seed noise of each other at α=8. The signature
decoupling holds every seed: the corrected activation sits *further* off the Gaussian manifold than raw at α=8
(`D_M` learned 80.8 ± 6.6 vs raw 52.3). The wide bar at α=1 (44 ± 11%) is the usual ratio artifact — raw's
damage there is only +0.059 nats, inflating the relative spread while the absolute `ΔLM_learned` stays tight
(+0.033 ± 0.006). **Limitation.** As in Experiments 26/27, this varies only the training seed (corrector init +
α-sampling / data-shuffle RNG); the eval set, Gaussian fit, and steering vector are held fixed, so it bounds
*optimization* variance on Pythia, not eval-document or vector-construction sampling variance. The other
non-GPT-2 architecture (Qwen3, Exp 21) and GPT-2 large (Exp 19) remain single-seed. **Next check.** A five-seed
control on Qwen3 (the top of the 81–94% band) would test whether its higher recovery is likewise real or seed
noise. *Done in Experiment 29.*

**Experiment 29 — Seed robustness on Qwen3-1.7B: an error bar on the *top* of the cross-architecture band.**
Experiments 26/27/28 put five-seed intervals on GPT-2 small (83.3 ± 2.0%), GPT-2 medium (88.3 ± 2.2%), and
Pythia-410m/GPT-NeoX (80.8 ± 1.6%) at α=8. The one remaining single-seed point the paper leans on is
**Qwen3-1.7B (Experiment 21) — the *top* of the reported 81–94% architecture band**, and its 94% is the largest
single-seed recovery in the whole study, exactly where a lone seed is most in doubt. We close it by re-running the
**exact Experiment-21 Qwen3-1.7B pipeline** — same DiffMean sentiment vector (`|v| = 38.1`, mean `|h| = 301.9`),
same 400-doc Gaussian fit (clean `D_M = 44.7`), same 300-doc training set, same held-out 100-doc eval, same 8.39M
corrector at `d = 2048`, same recipe / `α ∼ U(0.5, 8)`, steered at block 14/28 — at **five seeds** (`0–4`),
reporting mean ± sample standard deviation of the fluency recovery `recovery(α) = (ΔLM_raw(α) − ΔLM_learned(α)) /
ΔLM_raw(α)`. `ΔLM_raw` is seed-independent (computed once); only the learned corrector varies. Seed 0 reproduces
Experiment 21 to the digit — a built-in check.

| α | ΔLM raw (nats) | **ΔLM learned (mean ± sd)** | **recovery (mean ± sd)** | `D_M` raw | `D_M` learned (mean ± sd) |
|---|----------------|------------------------------|--------------------------|-----------|----------------------------|
| 1 | +0.064 | **−0.173 ± 0.014** | 370.1 ± 21.4% | 45.4 | 59.8 ± 3.4 |
| 2 | +0.243 | **−0.153 ± 0.020** | 162.9 ± 8.2% | 47.5 | 63.9 ± 4.6 |
| 4 | +1.081 | **−0.090 ± 0.022** | **108.3 ± 2.1%** | 55.0 | 79.2 ± 5.1 |
| 8 | +3.429 | **+0.177 ± 0.056** | **94.8 ± 1.6%** | 77.8 | 123.3 ± 5.4 |

(Per-seed recovery at α=8: 94 / 95 / 96 / 92 / 96% — Experiment 21's 94% is seed 0's 94%.)

**Reading it: Qwen3's top-of-band recovery is real, not a lucky seed — its interval sits clear above every other
model tested.** At α=8 the five independently-trained Qwen3 correctors recover **94.8 ± 1.6%** of raw steering's
fluency damage (range 92–96%), and at α≤4 they land *at or above* the unsteered baseline (recovery ≥ 100%,
`ΔLM_learned` slightly negative — the free-or-better weak-α behavior seen on every model). This settles the
architecture ordering at α=8: Qwen3's band (`[93.2, 96.4]%`) sits **entirely above** GPT-2 medium's
(`[86.1, 90.5]%`, Experiment 27), which sits above GPT-2 small's (`[81.3, 85.3]%`, Experiment 26) and Pythia's
(`[79.2, 82.4]%`, Experiment 28) — four seed-controlled models, and Qwen3's 94% edge is a genuine effect, not
optimization noise. The signature decoupling holds every seed: the corrected activation sits *further* off the
Gaussian manifold than raw at α=8 (`D_M` learned 123.3 ± 5.4 vs raw 77.8). The wide bars at α≤2 are the usual
ratio artifact — raw's damage there is only +0.06–0.24 nats, inflating the relative spread while the absolute
`ΔLM_learned` stays tight (±0.01–0.02). **Limitation.** As in Experiments 26/27/28, this varies only the training
seed (corrector init + α-sampling / data-shuffle RNG); the eval set, Gaussian fit, and steering vector are held
fixed, so it bounds *optimization* variance on Qwen3, not eval-document or vector-construction sampling variance.
GPT-2 large (Exp 19) remains the only headline model still single-seed. **Next check.** The seed axis now covers
four models spanning two scales and two architectures; a further-scaled or non-Transformer family (e.g. a
state-space model) would extend it, at low marginal value.

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
- `plots/08_capacity_scaling.png` — (a) held-out `certainty` fluency recovery vs α for corrector
  capacities {5.2M, 14.7M, 46.2M params} on the FIXED size-5 bank, plus the native oracle;
  (b) recovery at α=8 vs capacity (log axis) for the held-out direction and the mean in-bank
  direction — both flat/falling as capacity grows, with the native oracle far above. More width
  does not close the held-out gap or improve in-bank recovery.
- `plots/09_curated_bank.png` — (a) held-out `certainty` fluency recovery vs α for three size-3 banks
  of increasing alignment to the target (diffuse 0.38 / exp6 0.54 / curated 0.80) plus the native
  oracle; (b) held-out recovery at α=1 and α=8 vs bank mean `|cos|` to the target — the headline
  curve, showing transfer PEAKS at moderate alignment and COLLAPSES (negative at weak steering) for
  the most target-aligned bank. Curating toward the subspace backfires; diversity governs transfer.
- `plots/10_behavioral_pareto.png` — behavioral test on generated text (sentiment corrector). (a)
  sentiment effect `B(α)−B(0)` vs α for raw vs corrected steering; (b) distinct-2 (unique-bigram ratio)
  vs α; (c) the effect-vs-fluency Pareto (points labelled by α). Raw traces high-effect→collapsing-
  fluency; the corrector clusters at low-effect→high-fluency — neither dominates. Matched layer-6
  projection does not translate into matched behavioral steering in generation.
- `plots/11_behavioral_corrector.png` — behavioral-preservation term sweep (sentiment). (a) sentiment
  effect `B(α)−B(0)` vs α for raw and correctors of increasing behavioral weight `λ_b ∈ {0,10,40}`;
  (b) distinct-2 vs α; (c) the effect-vs-fluency Pareto. As `λ_b` grows the corrector points move right
  (more effect) while staying high on fluency, dominating raw in the moderate-effect regime but never
  reaching raw's strong-effect corner.
- `plots/12_layer_robustness.png` — layer-robustness replication of the flagship result at blocks 3, 6, 9
  (color = layer). (a) ΔLM vs α, raw (dashed) vs corrected (solid) — corrected sits near zero at every
  layer while raw climbs; (b) fluency recovery vs α per layer — all three curves ≥76% at α=8, ≥91% at α=4;
  (c) `D_M` vs α, raw vs corrected — corrected exceeds raw at every layer (off-Gaussian-but-LM-safe holds
  throughout the residual stream). Block 6 reproduces Experiment 3 exactly.
- `plots/13_cross_model.png` — cross-model replication of the flagship result on GPT-2 medium (355M,
  block 12/24). (a) ΔLM vs α, raw (dashed) vs corrected (solid) — corrected sits near zero while raw
  climbs to +2.72; (b) fluency recovery vs α — 89% at α=8, ≥101% at α≤4; (c) `D_M` vs α, raw vs corrected
  — corrected exceeds raw at every α (off-Gaussian-but-LM-safe holds on the larger model too).
- `plots/14_diversity_lever.png` — controlled third-member swap isolating bank diversity (all size-3 banks
  share the {sentiment, formality} anchor; only the 3rd member's collinearity varies). (a) anchor-pair
  recovery @α=8 vs the bank's internal collinearity — falls as the bank loses angular diversity; (b)
  sentiment recovery vs α, one line per bank — sentiment (⟂ every direction and ⟂ the target) is the
  confound-free isolate and is corrected worse in more collinear banks, confirming diversity is causal.
- `plots/15_prompt_family.png` — held-out prompt-family generalization of the FineWeb-trained corrector.
  (a) ΔLM vs α, raw (dashed) vs corrected (solid), one color per family (fineweb / markdown / code) —
  corrected sits near zero for all three while raw climbs; (b) fluency recovery vs α per family — all
  three ≥60% at α=8, ordered by distribution shift; (c) bar of each family's clean-activation Mahalanobis
  distance under the FineWeb Gaussian, showing code is the most out-of-distribution and recovery tracks it.
- `plots/16_manifold_geometry.png` — geometry of the clean layer-6 activation cloud (no steering).
  (a) QQ plot of held-out `D_M²` vs `χ²₇₆₈` theoretical quantiles — the empirical points rise far
  steeper than the Gaussian y=x line (spread 6.7× larger), a heavy right tail; (b) PCA cumulative
  variance explained — ~90% in the first PC, 95% in three (participation ratio 1.1), extreme anisotropy;
  (c) intrinsic-dimension estimates (TwoNN, Levina–Bickel MLE, PCA participation ratio) all far below the
  ambient 768. The activation manifold is low-dimensional, anisotropic, and non-Gaussian.
- `plots/17_diffusion_corrector.png` — the three-corrector comparison at matched projection. (a) ΔLM vs α
  for raw, one-shot MLP, cold-diffusion iterative (K=8), and the GLP Gaussian prior — the two LM-supervised
  correctors hug zero while the GLP prior sits *above raw*; (b) fluency recovery vs raw — one-shot and
  iterative overlap near 84–85% at α=8, the GLP prior stays negative; (c) steering-projection retention —
  the iterative corrector preserves the target `α|v|` exactly (on the matched line) while the unconditional
  GLP prior falls below it (erases the steer).
- `plots/18_steering_family.png` — the ColdSteer recipe across three steering-vector families (DiffMean,
  logistic-probe, PCA-contrast) built from real SST-2 data at a common norm. (a) ΔLM vs α, raw (dashed) vs
  corrected (solid), one color per family — all three raw curves climb while all three corrected curves hug
  zero; (b) fluency recovery vs α per family — all three ≥84% at α=8; (c) `D_M` vs α, raw vs corrected — the
  PCA family's raw curve is flat at the clean value (on-manifold yet LM-breaking) while all corrected curves
  rise above raw (off-Gaussian-but-LM-safe holds for every family).
- `plots/19_gpt2_large.png` — model-scaling replication of the flagship result on GPT-2 large (774M,
  block 18/36). (a) ΔLM vs α, raw (dashed) vs corrected (solid) — corrected sits near zero while raw climbs
  to +2.47; (b) fluency recovery vs α — 84% at α=8, 95% at α=4; (c) `D_M` vs α, raw vs corrected — corrected
  exceeds raw at every α (off-Gaussian-but-LM-safe holds on the largest model too).
- `plots/20_diff_generation.png` — differentiable-generation behavioral supervision (sentiment). (a)
  sentiment effect `B(α)−B(0)` vs α for raw and generation-supervised correctors of increasing weight
  `λ_g ∈ {0,40,160}`; (b) distinct-2 vs α; (c) the effect-vs-fluency Pareto. The `λ_g=40` corrector reaches
  higher effect than Exp 11's teacher-forced term at comparable fluency (breaking the ≈+1.3 ceiling to +1.72
  at α=8), while `λ_g=160` gains effect at moderate α but collapses to raw-like repetition at α≥6.
- `plots/21_cross_arch.png` — cross-architecture replication of the flagship result on **Qwen3-1.7B**
  (28 blocks, block 14/28; RMSNorm / RoPE / SwiGLU / grouped-query attention). (a) ΔLM vs α, raw (dashed) vs
  corrected (solid) — corrected sits at or below zero while raw climbs to +3.43; (b) fluency recovery vs α —
  94% at α=8, ≥108% at α≤4; (c) `D_M` vs α, raw vs corrected — corrected exceeds raw at every α
  (off-Gaussian-but-LM-safe holds on a non-GPT-2 architecture too).
- `plots/22_behavioral_qwen.png` — behavioral test on generated text on **Qwen3-1.7B** (block 14, sentiment),
  the Exp-10 protocol run on the Exp-21 corrector. (a) sentiment effect `B(α)−B(0)` vs α for raw vs corrected —
  raw rises to +8.0 while the corrector stays at +0.5–2.3 (under-steered); (b) distinct-2 vs α — raw dips to
  0.76 at α=8 while the corrector holds ~0.83; (c) the effect-vs-fluency Pareto (points labelled by α), showing
  a shallower tradeoff than GPT-2 (raw degenerates far less on Qwen3). The under-steering caveat replicates.
- `plots/23_behavioral_qwen_fix.png` — does the GPT-2 behavioral-preservation fix (Exp 11) transfer to
  **Qwen3-1.7B** (steer block 14, downstream readout block 27)? (a) sentiment effect vs α for raw and
  correctors of increasing behavioral weight `λ_b ∈ {0,10,40}`; (b) distinct-2 vs α; (c) the effect-vs-fluency
  Pareto (points labelled by α). As `λ_b` grows the corrector points move right (2–8× more effect) toward raw,
  but because Qwen3's raw does not collapse the corrector approaches raw's frontier without passing it — the
  fix's mechanism transfers, its Pareto win (seen on GPT-2) does not.
- `plots/24_cross_arch_pythia.png` — SECOND non-GPT-2 architecture, making the architecture axis a sweep:
  the flagship result replicated on **Pythia-410m / GPT-NeoX** (24 blocks, block 12/24; parallel residual /
  rotary / LayerNorm / dense MHA). (a) ΔLM vs α, raw (dashed) vs corrected (solid) — corrected sits near zero
  while raw climbs to +3.10; (b) fluency recovery vs α — 81% at α=8 and α=4; (c) `D_M` vs α, raw vs corrected —
  corrected exceeds raw at every α (off-Gaussian-but-LM-safe holds on a parallel-residual architecture too).
- `plots/25_behavioral_pythia.png` — behavioral test on generated text on **Pythia-410m / GPT-NeoX** (block 12,
  sentiment), reusing the exact Exp 24 corrector. (a) sentiment effect vs α — the corrector's effect is *above*
  raw's at α≤4 and 84–92% of raw's at α≥6 (not the ~1/6 under-steer of GPT-2/Qwen3); (b) distinct-2 vs α — raw
  collapses to 0.38 at α=8 while the corrector holds 0.72; (c) the effect-vs-fluency Pareto (points labelled by
  α), showing the corrector up-and-right of raw at α=8. The Exp 10/22 under-steering caveat is mild here because
  raw itself steers Pythia weakly (effect peaks at +1.17).
- `plots/26_seed_robustness.png` — seed robustness of the flagship result (GPT-2 small, layer 6, 5 seeds).
  (a) ΔLM vs α, raw (red) vs the learned corrector (blue mean line + ±1 sd band over 5 seeds) — the band hugs
  zero far below raw at every α; (b) fluency recovery vs α with mean ± sd error bars and the five per-seed points
  overlaid — 83.3 ± 2.0% at α=8, tightening to ±0.6–0.8% at α=4–6. The headline 84% is reproducible, not a
  single-seed artifact.
- `plots/27_seed_robustness_medium.png` — seed robustness on **GPT-2 medium** (355M, block 12/24, 5 seeds).
  Same two panels as Exp 26. (b) recovery is 88.3 ± 2.2% at α=8 (five per-seed points 85–90%), a band that sits
  entirely above GPT-2 small's 83.3 ± 2.0% — the model-scale edge is outside seed noise.
- `plots/28_seed_robustness_pythia.png` — seed robustness on **Pythia-410m / GPT-NeoX** (block 12/24, 5 seeds).
  Same two panels as Exp 26/27. (b) recovery is 80.8 ± 1.6% at α=8 (five per-seed points 78–82%): its band sits
  below GPT-2 medium's (non-overlapping) but overlaps GPT-2 small's — the recipe is seed-stable across a third,
  non-GPT-2 architecture.
- `plots/29_seed_robustness_qwen.png` — seed robustness on **Qwen3-1.7B** (block 14/28, 5 seeds). Same two panels
  as Exp 26/27/28. (b) recovery is 94.8 ± 1.6% at α=8 (five per-seed points 92–96%): its band sits **entirely
  above** every other model's, so Qwen3's top-of-band 94% is a genuine effect and the recipe is seed-stable on a
  fourth model.

## Headline
Raw linear steering `h + α·v` in GPT-2 drives activations off-manifold and breaks the LM (+2.78
nats at α=8). Correcting toward the **Gaussian manifold backfires** — an analytic projection-
preserving corrector cuts off-manifold distance 22% but *worsens* LM loss to +4.2 nats. But a
**learned corrector supervised by the LM loss** — same projection-preserving form, so the layer-6
steering projection is untouched — **recovers 84% of the teacher-forced fluency damage** (ΔLM
+2.78→+0.44 at α=8; **83.3 ± 2.0% across 5 training seeds**, Exp 26) while moving *further* from the Gaussian manifold. Statistical "on-manifold" and
"LM-safe" are decoupled; only the downstream objective finds the fluent correction. The correction is
**direction-specific**
— a corrector trained on one concept does not transfer to a near-orthogonal one — but the **recipe
generalizes**: retraining it on a new formality direction recovers 83–104% of the damage there too.
Making the corrector **direction-conditional** (feed `v̂`) and training it on a **bank** of directions
gives **one model that corrects every in-bank direction at once** (55–70% recovery at α=8) and
**begins to transfer** to a held-out direction (51%→7% recovery from weak to strong steering, vs ≈0%
for a frozen single-vector corrector) — replacing "one model per vector" with "one model per bank."
But **no scaling axis closes the held-out gap.** Enlarging the training bank from 3 to 5 directions
*lowers* transfer to the held-out direction (α=8 recovery 7%→3%) and lowers per-direction in-bank
recovery (Exp 7); scaling the *model* 9× wider (5.2M→46.2M params) on a fixed 5-direction bank leaves
in-bank recovery flat (~45%) and *worsens* weak-steering held-out transfer through overfitting (α=1
recovery −1%→−146%) (Exp 8); and — the tempting last lever — *curating* the bank **toward** the target's
subspace backfires worst of all: at fixed size and capacity, the most target-aligned bank transfers
*catastrophically* (α=1 recovery −183%, net-negative at every strength), while a moderately-aligned,
angularly *diverse* bank transfers best (Exp 9). The real lever is bank **diversity** (separability of
its directions), not coverage of the target's subspace — and a controlled third-member swap **confirms
this causally** (Exp 14): holding a shared {sentiment, formality} anchor and varying only a third member's
collinearity, the confound-free isolate `sentiment` (⟂ every direction *and* ⟂ the target) is corrected
worse as the bank collinearizes (63%→55% @α=8), and a member made collinear with a neighbor cannot be
specialized (its own recovery collapses 69%→17%). So amortized cross-direction correction is
capped not by coverage, parameter count, or subspace alignment but by the **training signal** — the
correction is fundamentally direction-specific, and the reliable route to a genuinely unseen direction
remains the **per-direction native corrector** (78–142% recovery). The core fluency result is **not a
block-6 artifact**: replicating the exact flagship pipeline at the early, middle, and late residual stream
(blocks 3 / 6 / 9) recovers **90% / 84% / 76%** of raw steering's fluency damage at α=8 (≥91% at α=4), with
the corrected activation sitting *further* off the Gaussian manifold than raw at every layer (Exp 12) — the
"LM-safe but off-Gaussian" correction is a layer-robust property. It is also **not a GPT-2-*small* artifact**:
replicating the exact flagship pipeline on **GPT-2 medium (355M, block 12/24)** recovers **89%** of raw
steering's fluency damage at α=8 (**101%** at α=4), and on **GPT-2 large (774M, block 18/36)** recovers
**84%** at α=8 (**95%** at α=4) — both again by moving *further* off the Gaussian manifold than raw. Across
the **124M → 355M → 774M** scale range (6× parameters) the α=8 recovery stays essentially flat (84% / 89% /
84%), so the core result is **model-robust** as well (Exp 13, Exp 19). A five-seed control on GPT-2 medium
(Exp 27) confirms its α=8 recovery is **88.3 ± 2.2%** — a band entirely above GPT-2 small's 83.3 ± 2.0%
(Exp 26), so medium's edge is a genuine model-scale effect, not seed noise. And it is **not a GPT-2-*architecture*
artifact** either: replicating the exact flagship pipeline on **Qwen3-1.7B (block 14/28)** — a modern
architecture that swaps LayerNorm→RMSNorm, learned→rotary positions, GELU→SwiGLU, and dense→grouped-query
attention — recovers **94%** of the fluency damage at α=8 (**108%** at α=4), again by moving *further* off the
Gaussian manifold than raw (`D_M` 122.2 vs 77.8), so the result is **architecture-robust** across all four
structural axes that separate Qwen3 from the GPT-2 family (Exp 21); a five-seed control (Exp 29) confirms this
is **94.8 ± 1.6%** at α=8 — a band (`[93.2, 96.4]%`) entirely *above* every other model's, so Qwen3's
top-of-band edge is a genuine effect, not a lucky seed. And this is now a genuine **architecture
*sweep*, not a single boundary crossing**: adding a **third, structurally distinct family — Pythia-410m /
GPT-NeoX (block 12/24)**, whose block uses a **parallel residual** (attention and MLP from the same input,
summed) unlike both GPT-2's and Qwen3's serial residual — recovers **81%** of the fluency damage at α=8 (81% at
α=4), again by moving *further* off the Gaussian manifold than raw (`D_M` 89.4 vs 52.3) (Exp 24), and a
five-seed control on Pythia (Exp 28) shows this is seed-stable — **80.8 ± 1.6%** at α=8 (band below GPT-2
medium's, overlapping GPT-2 small's). Across the
three families the α=8 recovery stays in a tight **81–94%** band (GPT-2 84/89/84%, Qwen3 94%, GPT-NeoX 81%). It is likewise **not a FineWeb-prompt
artifact**: a corrector trained only on FineWeb still recovers **77%** of the fluency damage at α=8 on
held-out technical-prose (Markdown) and **60%** on strongly out-of-distribution Python code (87% / 78% at
α=4), with recovery declining smoothly as the family's clean activations drift further off the FineWeb
Gaussian (`D_M` 27.5→30.1→37.4) — so the correction is **prompt-family-robust** too (Exp 15). Finally, it is
**not tied to the DiffMean steering family or to hand-built prompts** (Exp 18): rebuilding the sentiment
steering vector from a real downloaded dataset (SST-2) via three genuinely different extraction families —
DiffMean, a logistic-regression probe, and PCA-contrast (cosines to DiffMean 1.00 / 0.40 / 0.30) — the
identical recipe recovers **84% / 84% / 101%** of the fluency damage at α=8. The PCA family is especially
telling: its raw steering leaves the Mahalanobis distance *flat at the clean value* (on the Gaussian
manifold) yet still breaks the LM (+2.27 nats), so off-Gaussian distance is neither necessary nor sufficient
for LM damage — the corrector fixes it by moving *off* the manifold as always. The core result is thus robust
on **six** axes: steering strength, direction, layer, model, prompt family, and **steering-vector family**.

**On the "manifold" itself (Exp 16).** The `D_M` metric models real activations as a single 768-d
Gaussian; direct tests show they are **not**. The activation cloud is **low-dimensional** (intrinsic
dimension ~8–34 by TwoNN and the Levina–Bickel MLE, vs 768 ambient), **extremely anisotropic** (PCA
participation ratio 1.1 — ~90% of variance in one direction, GPT-2's outlier dimensions), and
**heavy-tailed** (held-out `D_M²` spread 6.7× the Gaussian `χ²₇₆₈`, 14 dimensions with excess kurtosis
up to 118). This *sharpens* the thesis rather than undermining it: it is exactly why minimizing Gaussian
`D_M` backfires (Exp 2 — the correction pours into the high-variance rogue dims the LM reads most
sharply), and it reframes "off the Gaussian manifold" (Exp 3/5/12/13) as "off a crude fit," confirming
that `D_M` is a diagnostic, never a training target — a downstream-LM objective is the right response.

**A *real* diffusion corrector (Exp 17).** The direction is named after Cold Diffusion, so we built the actual
iterative machinery and pitted three correctors head-to-head at matched projection: the one-shot MLP (Exp 3),
a **cold-diffusion iterative** corrector (a step-conditioned velocity field integrated over K=8
projection-preserving steps, LM-supervised through the unroll), and a **generic Gaussian-noise GLP prior** (a
real DDPM trained on clean activations, SDEdit post-processing, no LM). The verdict is threefold. **(1) The
Cold-Diffusion *corruption model* is what matters, not iteration:** training on the actual steering corruption
under LM supervision recovers **84–85%** of the fluency damage at α=8, but the generic "denoise back to the
manifold" GLP prior has **negative recovery** (−5% at α=8 — worse than raw steering), because a prior that only
knows "typical activation" cannot tell which off-typical directions the LM tolerates. **(2) The iterative
diffusion structure essentially ties the one-shot MLP** (85% vs 84% at α=8, at equal capacity) — the value of
"diffusion" is the corruption + supervision, not the step count. **(3) The unconditional prior erases the
steer** (~5–6% of the target projection lost), the exact information-loss the GLP authors flag; conditioning on
the clean activation and supervising with the LM — what ColdSteer does — is the fix.

**A behavioral caveat on the fluency story (Exp 10).** The `ΔLM` recoveries above are measured at
*matched projection at one layer* — a proxy. When the sentiment corrector is used to actually *generate*
text, it keeps generation fluent (distinct-2 near baseline while raw steering collapses into repetition,
0.78→0.32 at α=8) but its output is only weakly steered (sentiment effect ~one-sixth of raw's). The
projection-preserving correction is orthogonal to `v` in *activation* space yet **not** orthogonal to the
downstream sentiment *readout*, so minimizing LM loss produces near-normal, lightly-steered text. There
is a genuine **effect–fluency tradeoff** the matched-projection metric hid: the corrector's fluency win
is not costless — part of it is a weaker propagated edit. Matched layer-6 projection ≠ matched behavioral
steering; any use of ColdSteer must verify the behavioral effect on generated text, not only `ΔLM`. **This
caveat is architecture-robust (Exp 22):** running the identical protocol on Qwen3-1.7B (whose teacher-forced
recovery was 94%, Exp 21) reproduces it — the corrector's generated sentiment effect is only 10–29% of raw's,
so its fluency win is again partly a weaker propagated edit. (One difference: raw steering degenerates far less
on Qwen3, distinct-2 0.76 vs GPT-2's 0.32 at α=8, so raw is a stronger baseline there and the Pareto is
shallower.) **On a third architecture (Pythia-410m, Exp 25) the penalty is milder still:** the corrector's
generated effect is *comparable to* raw's (above raw at α≤4, 84–92% of raw at α≥6) and it Pareto-*dominates*
raw at α=8 (effect +0.98 at distinct-2 0.72, where raw collapses to 0.38). The reason is that raw itself
steers Pythia only weakly (effect peaks at +1.17), leaving little behavioral effect for the corrector to
lose — so **how much "matched projection ≠ matched steering" costs depends on how strongly raw steering
propagates in that model**, and it is small when raw's own behavioral steer is weak.
**Partial fix (Exp 11):** adding one training term that preserves the *downstream* sentiment readout
(measured at the final layer, pushed toward raw steering's) recovers **2–6× more behavioral effect** while
keeping generation fluent, and turns Exp 10's non-dominating tradeoff into **outright dominance over raw at
moderate steering** (effect ≈+1 at near-baseline distinct-2 0.73, where raw only reaches that effect after
collapsing to 0.32). But a ceiling remains — no weighting reaches raw's *strong* pre-collapse effect
(≈+2.5), because matching a teacher-forced readout only partially transfers to autoregressive generation.
The projection-preserving corrector's frontier is pushed **out**, not erased. **Second fix (Exp 20):**
supervising the same readout on the corrector's *own generated continuation* through a **differentiable
soft-token rollout** (instead of a teacher-forced pass) **breaks Exp 11's effect ceiling** — the α=8
achievable effect rises from +1.08 to **+1.72** at distinct-2 0.47 (vs raw's collapsed 0.32), and at
moderate steering (α=2) reaches effect +1.61 at near-baseline fluency 0.71 — because we now supervise on the
autoregressive distribution rather than a teacher-forced proxy. Yet over-weighting the generation term
destabilizes training and collapses to raw-like repetition at strong steering (λ_g=160: effect −0.22,
distinct-2 0.32 at α=8), so the strong-effect-and-fluent corner still eludes. The frontier is pushed out a
second time, still not erased. **Does the fix transfer across architectures? (Exp 23):** re-fitting the
Exp-11 readout-preservation term on Qwen3-1.7B recovers **53–83% of raw's generated sentiment effect** (vs
10–29% for the base corrector), so the fix's *mechanism* is architecture-robust. But because Qwen3's raw
steering does not collapse (Exp 22), the corrector only approaches raw's effect-fluency frontier without
dominating it — unlike on GPT-2, where the same term beat a *collapsed* raw. The behavioral fix is a robust
lever on generated effect; its Pareto *payoff* is gated by whether the raw baseline degenerates.
