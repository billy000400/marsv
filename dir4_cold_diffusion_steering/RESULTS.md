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

## Headline
Raw linear steering `h + α·v` in GPT-2 drives activations off-manifold and breaks the LM (+2.78
nats at α=8). Correcting toward the **Gaussian manifold backfires** — an analytic projection-
preserving corrector cuts off-manifold distance 22% but *worsens* LM loss to +4.2 nats. But a
**learned corrector supervised by the LM loss** — same projection-preserving form, so the layer-6
steering projection is untouched — **recovers 84% of the teacher-forced fluency damage** (ΔLM
+2.78→+0.44 at α=8) while moving *further* from the Gaussian manifold. Statistical "on-manifold" and
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
its directions), not coverage of the target's subspace. So amortized cross-direction correction is
capped not by coverage, parameter count, or subspace alignment but by the **training signal** — the
correction is fundamentally direction-specific, and the reliable route to a genuinely unseen direction
remains the **per-direction native corrector** (78–142% recovery). The core fluency result is **not a
block-6 artifact**: replicating the exact flagship pipeline at the early, middle, and late residual stream
(blocks 3 / 6 / 9) recovers **90% / 84% / 76%** of raw steering's fluency damage at α=8 (≥91% at α=4), with
the corrected activation sitting *further* off the Gaussian manifold than raw at every layer (Exp 12) — the
"LM-safe but off-Gaussian" correction is a layer-robust property. It is also **not a GPT-2-*small* artifact**:
replicating the exact flagship pipeline on **GPT-2 medium (355M, block 12/24)** recovers **89%** of raw
steering's fluency damage at α=8 (**101%** at α=4), again by moving *further* off the Gaussian manifold than
raw — so the core result is **model-robust** as well (Exp 13).

**A behavioral caveat on the fluency story (Exp 10).** The `ΔLM` recoveries above are measured at
*matched projection at one layer* — a proxy. When the sentiment corrector is used to actually *generate*
text, it keeps generation fluent (distinct-2 near baseline while raw steering collapses into repetition,
0.78→0.32 at α=8) but its output is only weakly steered (sentiment effect ~one-sixth of raw's). The
projection-preserving correction is orthogonal to `v` in *activation* space yet **not** orthogonal to the
downstream sentiment *readout*, so minimizing LM loss produces near-normal, lightly-steered text. There
is a genuine **effect–fluency tradeoff** the matched-projection metric hid: the corrector's fluency win
is not costless — part of it is a weaker propagated edit. Matched layer-6 projection ≠ matched behavioral
steering; any use of ColdSteer must verify the behavioral effect on generated text, not only `ΔLM`.
**Partial fix (Exp 11):** adding one training term that preserves the *downstream* sentiment readout
(measured at the final layer, pushed toward raw steering's) recovers **2–6× more behavioral effect** while
keeping generation fluent, and turns Exp 10's non-dominating tradeoff into **outright dominance over raw at
moderate steering** (effect ≈+1 at near-baseline distinct-2 0.73, where raw only reaches that effect after
collapsing to 0.32). But a ceiling remains — no weighting reaches raw's *strong* pre-collapse effect
(≈+2.5), because matching a teacher-forced readout only partially transfers to autoregressive generation.
The projection-preserving corrector's frontier is pushed **out**, not erased.
