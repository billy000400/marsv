# REPORT — Do plateau transitions correspond to activation-manifold transitions?

> Final, presentable, current-best only (no history — see CHANGELOG.md). Read before rewriting.

## Summary

Interpretability research often pictures a neural network's internal activations as living on a
low-dimensional **manifold**, with distinct behaviors sitting on distinct, *separated* pieces of it. A
striking piece of apparent evidence is a **plateau**: when you interpolate a network's hidden
activations from one example to another, the downstream representation sometimes does not slide
smoothly across. Instead it **stays put near output A, jumps abruptly, then stays put near output B** —
a "plateau → sharp boundary → plateau" shape, as if there were a wall between two stable basins.

If that wall were a real **gap in the data** — two disconnected components of the natural activation
manifold — it would matter a lot for safety: we could localize, monitor, and steer behaviors by their
manifold component, and off-distribution detectors could flag the wall. If instead the wall is only an
artifact of walking in a *straight line* through an empty region the data never visits — while the two
plateaus stay connected by an ordinary high-density data path — then the plateau tells us about the
model's **decision geometry**, not about the shape of the data manifold.

We test this **at the population level** across a depth-4 MNIST classifier — every sufficiently
populated plateau pair, not one hand-picked case — with a **single frozen metric**. We separate two
claims: the **universal** claim (every plateau transition is a manifold-component transition; one
counterexample refutes it) and the weaker **typical-association** claim (plateau pairs are *usually*
more separated than within-plateau controls).

**Both claims fail.** The universal claim is **refuted decisively**: 25 of 45 verified plateau
transitions on the base model connect through the natural activation cloud with *no larger gap than
normal travel inside a single plateau*, and this holds in every well-powered replication (a second seed
and two more architectures). The typical-association claim is **not supported**: the between-plateau
"connection gap" metric sits essentially *on top of* the within-plateau baseline in all four
well-powered models, with overlapping confidence intervals and no consistent direction. Plateaus
reflect the model's decision geometry, **not** a hole in the data manifold. The original digit-9 case
that motivated this direction turns out to be one ordinary counterexample among many.

## Methods

### Data & Model

**Model.** The `image-models` checkpoint `mnist_mlp_d4_w200_relu`: a fully-connected MNIST classifier
784→200→200→200→10 with ReLU activations, trained with MSE to one-hot targets on a 1000-image subset;
test accuracy **85.3%**. For replication we reuse four existing checkpoints trained the same way: a
**second seed** (d4w200, 86.9%) and three architectures — **d3w200** shallower (78.1%), **d4w400**
wider (86.9%), **d5w200** deeper (85.9%).

**Layers.** We interpolate the **first hidden layer L1** (200-dim, post-ReLU) — the "intervention
layer" — and measure downstream behavior at the **last hidden layer L3** (200-dim). "OOD" = out of
distribution; "MST" = minimum spanning tree; "slerp" = spherical linear interpolation (constant angular
velocity along the great circle, magnitude interpolated linearly).

**Natural activation cloud.** The empirical manifold reference is the set of L1 activations of all
**1705** correctly-classified test images (of 2000).

**Plateau regions (defined from output behavior, before any manifold test).** We take the **10 digit
classes** as the class-aligned stable output regions. Endpoints must be correctly classified and pass a
fixed **confidence rule**: output **margin** (top-1 minus top-2 logit) ≥ 0.5. (Softmax is uninformative
here because the net regresses to one-hot targets, so we use the logit margin.) 1604 examples pass,
≥130 per digit. Confidence is an *inclusion rule*, not a reported metric. The previously-studied
**digit-9 A/B** sub-plateau (KMeans with `k=2` on L3 activations of the confident 9s) is included as
**one extra transition, treated exactly like any other pair.** We sample **20 endpoint pairs per region
pair** (seed 0), and the same count for within-plateau controls, so digit frequency cannot dominate.

### Metrics

We use exactly two quantitative objects. The first is only a filter; the second is the sole reported
score.

**Plateau observable `d(t)` — verifies a path is genuinely plateau-to-plateau.** For a slerp path
`x_t` in L1 from endpoint `x_0` (region A) to `x_1` (region B), the normalized downstream distance at
L3 is:

```math
d(t) \;=\; \frac{\lVert h_3(x_t) - h_3(x_0) \rVert}{\lVert h_3(x_t) - h_3(x_0) \rVert \;+\; \lVert h_3(x_t) - h_3(x_1) \rVert}
```

`d(t)` runs from 0 (downstream state at A) to 1 (downstream state at B). A **plateau transition** is a
path that is mostly *flat near an endpoint* with a *sharp jump* between. We **accept** a between-region
path as a verified transition iff its **plateau fraction** — the fraction of `t` with `d(t)<0.2` or
`d(t)>0.8` — is at least 0.5, it starts below 0.2, and it ends above 0.8. This is an inclusion filter;
`d(t)` is never reported as a score.

**Manifold observable `G` — the normalized connection bottleneck (the only reported metric).** Build a
Euclidean **minimum spanning tree** `T` over the natural L1 cloud. For two endpoints `u,v`, the
**bottleneck** `B(u,v)` is the largest edge on their unique path through `T`. Equivalently it is the
minimax over *all* paths in the complete Euclidean graph — the smallest step size at which `u` and `v`
become connected through the sampled natural cloud:

```math
B(u,v) \;=\; \min_{P:\,u\rightsquigarrow v}\ \max_{(p,q)\in P}\ \lVert a_p - a_q \rVert
```

where `a_p` are natural L1 activations and `P` ranges over paths in the cloud. We normalize `B` by the
**within-plateau connection scale** so the number is comparable across regions of different density.
Let `s_r` be the median within-region bottleneck of region `r` (frozen from the within-plateau control
pairs, before any between-plateau result was examined). For a pair with endpoints in regions `i,j`:

```math
G \;=\; \frac{B(u,v)}{\max(s_i,\, s_j)}
```

**How to read `G`:** `G = 1` means "no larger gap than is normally required inside a plateau"; `G > 1`
means an unusually large bridge is needed (candidate manifold-component separation); `G \le 1` means the
two plateaus connect through natural activations as easily as two points *within* one plateau — a
**counterexample** to the universal claim. Higher `G` = more evidence of separation.

### Baselines

**Within-plateau control.** The reference distribution is the set of `G` values for **within-region**
endpoint pairs (endpoints from the *same* digit, same sampling and normalization). By construction its
median is 1. Its spread (median 1.00, 95th percentile 1.39 on the base model) is the natural yardstick:
a between-plateau pair only shows real separation if its `G` sits *clearly above* this control band.

**`G = 1` threshold.** The counterexample threshold follows directly from the frozen within-plateau
normalization — it is *not* tuned after seeing between-plateau results.

### Verdict rules

- **Universal claim refuted** iff at least one well-powered verified plateau pair has `G \le 1` (stable
  under resampling / replication).
- **Typical association supported** iff the between-plateau `G` distribution is consistently shifted
  above the within-plateau control across replications; **not supported** if the distributions overlap
  or the direction is unstable.

## Results

### Population verdict on the base model

Across 45 cross-digit plateau pairs plus the digit-9 sub-plateau (all verified by `d(t)`):

| quantity | value |
|--|--|
| within-plateau `G`: median / 95th pct | 1.00 / 1.39 |
| **between-plateau median `G`** (45 verified pairs) | **0.996**  (95% CI 0.97–1.03) |
| between-plateau median `G`: min – max pair | 0.84 – 1.67 |
| verified pairs with median `G > 1` | 44% (20/45) |
| **counterexamples** (median `G ≤ 1`) | **25 / 45** |
| digit-9 A/B sub-plateau (original case) | **G = 1.00** |

**Universal claim — REFUTED.** 25 of 45 verified plateau transitions connect through the natural cloud
with **no larger bottleneck than normal within-plateau travel** (`G ≤ 1`), including the original
digit-9 sub-plateau. A wall between plateaus is not required.

**Typical-association claim — NOT SUPPORTED.** The between-plateau median `G` (0.996) sits on the
within-plateau baseline (1.00); its bootstrap CI (0.97–1.03) overlaps the within-plateau CI
(0.99–1.02). The per-pair distributions overlap almost completely — 44% of pairs above `G=1`, 56% at or
below — and the single largest bridge required (`G=1.67`, digit 1↔8) barely exceeds the *within*-plateau
95th percentile (1.39). There is no pair separated by a dramatic data hole.

![Per-pair G: between-plateau (red) and within-plateau (green) distributions overlap almost completely, both centered at G=1 (a); median G per plateau pair, ~half each side of the G=1 line, digit-9 sub-plateau (orange) exactly at 1 (b).](plots/population_G.png)

The only pairs needing a modestly larger bridge involve digit **1** (1↔8=1.67, 0↔1=1.57, 1↔3=1.46), an
elongated thin manifold whose *own* internal scale `s_1` is small — inflating the ratio — not a genuine
void. The digit×digit heatmap shows no block structure: no set of digits forms its own component.

![Median normalized bottleneck G for every digit pair (diagonal = within-plateau = 1). Values hug 1; the only mild elevations involve digit 1.](plots/population_heatmap.png)

Representative verified transitions confirm `d(t)` selects genuine flat→jump→flat plateaus, yet even the
largest-`G` pair needs a bridge only ~1.7× the normal within-plateau step:

![Three representative verified plateau-to-plateau d(t) curves (largest-G, digit-9 sub-plateau, smallest-G): all flat-near-A, sharp jump, flat-near-B; all connect at G≈1–1.7.](plots/population_dt.png)

### Replication — second seed and three architectures

The identical frozen pipeline on the existing checkpoints:

| model (test acc) | verified pairs | between-plateau median `G` | % pairs `G>1` | counterexamples (`G≤1`) |
|--|--:|--:|--:|--:|
| base d4w200, seed 0 (85.3%) | 45 | **0.996** | 44% | 25 |
| seed 1 d4w200 (86.9%)       | 45 | **0.925** | 22% | 35 |
| d4w400 wider (86.9%)        | 46 | **0.987** | 43% | 26 |
| d5w200 deeper (85.9%)       | 46 | **0.982** | 30% | 32 |
| d3w200 shallower (78.1%)*   | 1  | 0.982 | 0% | 1 |

\*The shallow net is excluded on **structural grounds** (see below), not merely down-weighted: it
produces few *sharp* plateaus, so only 1 pair passes the `d(t)` accept filter. The four well-powered
models agree: between-plateau median `G` is **0.93–1.00 in every case — never a consistent shift above
the within-plateau baseline of 1.0** — and each finds many counterexamples. In seed 1 the direction even
*reverses* (median 0.925 < 1). Neither claim survives replication.

![Replication across five checkpoints: between-plateau median G (red, 95% CI) sits on the within-plateau baseline (green) for every model (a); in every well-powered model ~half the verified pairs fall each side of G=1 (b).](plots/population_replication.png)

### Is the shallow net just under-sampled? No — it is structurally plateau-poor

Before concluding, we check whether the shallow net's lone verified pair is sampling noise that more
endpoint pairs would cure. It is not. Its downstream distance **ramps rather than plateaus**: the mean
plateau fraction across all 46 region pairs is **0.25** (max 0.43), so **0 of 46** region pairs reach
the 0.5 accept threshold even on average, versus **0.60** (43/46 above 0.5) for the base net. Sampling
**10× more** endpoint pairs (20 → 200 per region pair) yields only **2** verified pairs, not 20 — the
deficit is structural, not statistical. We deliberately do **not** relax the `d(t)` filter to admit
these ramps, because a ramp is not a plateau and scoring `G` on non-plateau paths would not test the
claim. And the 1–2 genuine plateau transitions the shallow net does produce are **all counterexamples**
(`G ≤ 1`; median `G` 0.76–0.98) — consistent with the verdict, never against it. We therefore rest the
verdict on the four well-powered models and exclude the shallow net as an invalid test bed.

![(a) The shallow net's d(t) rarely plateaus: per-region-pair plateau fraction (red) sits far below the 0.5 accept threshold (0/46 pass), while the base net (blue) clears it 43/46 times — structural, not sampling noise. (b) Sampling 10× more endpoint pairs (20→200) does not restore power (≤2 verified pairs), and every plateau it does find has G≤1.](plots/population_shallow_power.png)

## Conclusion

At the population level, across a depth-4 MNIST MLP and four replication checkpoints, **plateau
transitions are not transitions between separate empirical manifold components.**

- **Universal claim — REFUTED:** 25/45 verified plateau pairs on the base model (and 26–35 of 45–46 in
  every well-powered model) connect through the natural activation cloud with `G ≤ 1` — no larger gap
  than normal within-plateau travel. The digit-9 case that first motivated this work is one such
  counterexample (`G = 1.00`).
- **Typical-association claim — NOT SUPPORTED:** the between-plateau median `G` (0.93–1.00) sits on the
  within-plateau baseline (1.00) in all four well-powered models, with overlapping bootstrap CIs and no
  consistent direction. The `G` distributions overlap almost completely.

**What this means.** A sharp plateau marks a place where the model's **decision geometry** changes
abruptly — a straight interpolation briefly leaves the data manifold — but it does **not** mark a hole
in the data manifold. The two plateaus remain connected by an ordinary high-density path. For safety
work this is a caution: **plateaus and low-density interpolations are not reliable evidence that two
behaviors occupy disconnected regions of activation space**; behavior boundaries and data-manifold
components are different things.

**Limitations (stated plainly).** (1) Finite activation samples can *support or undermine* an empirical
component split but **cannot prove true topological disconnection** — a denser sample could always
reveal a bridge, and a sparser one could manufacture a gap; `G` measures empirical connectivity at the
sampled scale, nothing stronger. (2) All models share the same 1000-image MNIST training subset; a
genuinely different dataset would be a separate direction. (3) The mild `G>1` pairs (digit-1) reflect an
elongated manifold's small internal scale, not a genuine void — the metric's ratio form is sensitive to
anisotropic regions, which is why we anchor the verdict on the *distribution* of `G` and its overlap
with the control band rather than on any single pair.
