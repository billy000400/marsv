# REPORT — Do plateau transitions correspond to activation-manifold transitions?

> Final, presentable, current-best only (no history — see CHANGELOG.md). Read before rewriting.

## Summary

Interpretability research often pictures a neural network's internal activations as living on a
low-dimensional **manifold**, with distinct behaviors sitting on distinct, *separated* pieces of it. A
striking piece of apparent evidence is a **plateau**: when you interpolate a network's hidden
activations from one example to another, the downstream representation sometimes does not slide
smoothly across. Instead it **stays put near output A, jumps abruptly, then stays put near output B** —
a "plateau → sharp boundary → plateau" shape, as if there were a wall between two stable basins.

That "wall" could be two different things, and this report investigates **both** (they are
complementary, not competing):

1. **Investigation 1 — manifold components.** Is the wall a real **gap in the data** — two
   disconnected components of the natural activation manifold? If so, we could localize, monitor, and
   steer behaviors by their manifold component. We test this at the **population level** across a
   depth-4 MNIST classifier — every sufficiently populated plateau pair, not one hand-picked case —
   with a single frozen metric, separating a **universal** claim (every plateau transition is a
   component transition; one counterexample refutes it) from a weaker **typical-association** claim
   (plateau pairs are *usually* more separated than within-plateau controls).
2. **Investigation 2 — the low-density corridor.** Even if the plateaus are *not* separate components,
   the **straight interpolation path** between them may still pass through a region that real
   activations never visit. That is a real, measurable fact about where the model's **decision
   boundary** sits relative to the data, and we quantify it over the same population of verified
   plateau transitions.

**Findings.** Investigation 1: **both claims fail.** The universal claim is **refuted decisively** —
25 of 45 verified plateau transitions on the base model connect through the natural activation cloud
with *no larger gap than normal travel inside a single plateau*; 21 of them remain counterexamples
under every independent endpoint redraw, and the result holds in every well-powered replication (a
second seed and two more architectures). The typical-association claim is **not supported** — the
between-plateau "connection gap" metric sits essentially *on top of* the within-plateau baseline in
all four well-powered models. Investigation 2: **the corridor is real.** The median verified
plateau-to-plateau direct path climbs to the **95th percentile** of the natural data-support
distribution (within-plateau controls: 65th), and **53%** of such paths exceed the natural 95th
percentile outright (controls: 12%), with the low-density bulge sitting exactly mid-path where the
output jump happens. Together: the plateaus stay connected through the data, but the straight route
between them crosses territory the data avoids — plateaus reflect the model's **decision geometry**,
not holes in the data manifold. The original digit-9 case that motivated this direction turns out to
be an ordinary connected pair whose direct path never even leaves the support.

## Methods

### Data & Model

**Model.** The `image-models` checkpoint `mnist_mlp_d4_w200_relu`: a fully-connected MNIST classifier
784→200→200→200→10 with ReLU activations, trained with MSE to one-hot targets on a 1000-image subset;
test accuracy **85.3%**. For replication we reuse four existing checkpoints trained the same way: a
**second seed** (d4w200, 86.9%) and three architectures — **d3w200** shallower (78.1%), **d4w400**
wider (86.9%), **d5w200** deeper (85.9%).

**Layers (stated explicitly, per operator feedback).** The interpolation is **spherical linear
interpolation ("slerp"**: constant angular velocity along the great circle, magnitude interpolated
linearly) performed in the **first hidden layer L1** (200-dim, post-ReLU) — the "intervention layer."
The downstream distance `d(t)` is measured at the **last hidden layer L3** (200-dim). The support
radius `r_10` of Investigation 2 is measured at **L1**, the same layer the path lives in (it asks how
far each *path point* is from the natural L1 cloud). "OOD" = out of distribution; "MST" = minimum
spanning tree.

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

We use three quantitative objects. The first is only a filter; the second is Investigation 1's sole
reported score; the third is Investigation 2's sole reported score.

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

**Manifold observable `G` — the normalized connection bottleneck (Investigation 1).** To decide
whether two plateaus are separate *empirical components*, we need the smallest "hop" a traveler must
make to get from one to the other stepping only on natural activations — low density along the straight
line is not enough, since the data might simply take a detour. Build a Euclidean **minimum spanning
tree** `T` over the natural L1 cloud. For two endpoints `u,v`, the **bottleneck** `B(u,v)` is the
largest edge on their unique path through `T`. Equivalently it is the minimax over *all* paths in the
complete Euclidean graph — the smallest step size at which `u` and `v` become connected through the
sampled natural cloud:

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

**Support radius `r_k` and off-manifold excursion (Investigation 2).** `G` asks whether the *data* is
connected; it says nothing about what the *straight path* passes through. To measure that, we need a
local data-density score at every path point. We use the distance to the `k`-th nearest natural
activation (with `k = 10`), a standard non-parametric density surrogate — large `r_k` = locally empty
space:

```math
r_k(x) \;=\; \lVert x - \mathrm{NN}_k(x) \rVert
```

where `NN_k(x)` is the `k`-th nearest neighbor of `x` in the natural L1 cloud. To make "large"
meaningful we compare against the **natural baseline**: the distribution of each natural point's own
`r_k` (self excluded; median 2.85, 95th percentile 4.23 on the base model). A path's **off-manifold
excursion** is its worst-case support along the way, expressed as a percentile of that baseline:

```math
E \;=\; \mathrm{pctile}_{\mathrm{natural}}\!\left( \max_t\, r_k(x_t) \right)
```

**How to read `E`:** `E ≈ 50` means the path never gets less supported than a typical natural point;
`E > 95` means the path visits territory essentially *no* real activation occupies. This feeds the
Investigation 2 result (figures below); the comparison group is the same statistic on within-plateau
control paths.

### Baselines

**Within-plateau control.** The reference distribution for both investigations is the set of
**within-region** endpoint pairs (endpoints from the *same* digit, same sampling rules). For `G` its
median is 1 by construction; its spread (median 1.00, 95th percentile 1.39 on the base model) is the
natural yardstick: a between-plateau pair only shows real separation if its `G` sits *clearly above*
this control band. For the excursion `E`, within-plateau paths (n=200, 20 per digit) show what
excursions ordinary same-region travel produces.

**`G = 1` threshold.** The counterexample threshold follows directly from the frozen within-plateau
normalization — it is *not* tuned after seeing between-plateau results.

**Natural-baseline percentile for `E`.** The `r_10` baseline comes from the natural cloud itself, so
`E` is anchored to the data before any path is examined.

### Verdict rules

- **Universal claim refuted** iff at least one well-powered verified plateau pair has `G \le 1` (stable
  under resampling / replication).
- **Typical association supported** iff the between-plateau `G` distribution is consistently shifted
  above the within-plateau control across replications; **not supported** if the distributions overlap
  or the direction is unstable.
- **Low-density corridor real** iff verified between-plateau paths show systematically higher
  excursions `E` than within-plateau controls.

## Results

### Investigation 1 — population verdict on the base model

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

### Investigation 2 — the direct path crosses a region real activations avoid

Investigation 1 established that the plateaus are connected *through the data*. Here we ask what the
straight slerp route itself traverses. Over the same frozen population (46 region pairs × 20 endpoint
pairs, seed 0; **676 of 920** sampled paths pass the `d(t)` verification filter; **120** slerp points
per path; within-plateau controls = **200** paths, 20 per digit):

| quantity | between-plateau (verified, n=676) | within-plateau controls (n=200) |
|--|--:|--:|
| median excursion `E` (max `r_10` pctile) | **95.4** (IQR 87.8–98.4) | 65.2 (IQR 38.3–86.1) |
| paths with max `r_10` > natural p95 | **53%** | 12% |

**The corridor is real.** The median verified plateau-to-plateau path climbs to the **95th
percentile** of the natural support distribution — the edge of where real activations exist — and over
half exceed the natural p95 outright, i.e. they pass through territory essentially no real activation
occupies. Within-plateau paths stay comfortably inside the cloud. The profile panel shows the
characteristic shape: `r_10(t)` **bulges mid-path** (to ~1.45× the natural median at the center, IQR
reaching past the p95 line), exactly where the `d(t)` jump happens, and returns to normal at both
endpoints — a low-density corridor connecting two well-supported plateaus.

![(a) Off-manifold excursion E (max r10 along path, as a percentile of the natural baseline): verified between-plateau paths (red, n=676) pile up at the 90–100th percentile, 53% beyond the natural p95 (dashed line), while within-plateau controls (green, n=200) spread across lower percentiles with only 12% beyond p95. (b) Median r10(t)/natural-median profile with interquartile bands vs slerp position t: between-plateau paths (red) bulge mid-path to ~1.45× the natural median; within-plateau controls (green) stay flat near 0.95. Slerp in L1; r10 measured at L1 against the 1705-point natural cloud; d(t) filter at L3; 120 points per path.](plots/direct_path_population.png)

**Single-pair illustration (the original pilot figure, regenerated with full annotations).** To answer
the operator's questions directly: the interpolation is **slerp in L1** (first hidden layer, 200-d,
post-ReLU); the blue **`d(t)` is measured at L3** (last hidden layer); the red **support radius
`r_10(t)` is measured at L1** against the **1705-point** natural cloud; each panel shows **one endpoint
pair** (region medoids) sampled at **200 points** along the path (the population figure above uses 120
points and 676+200 paths).

![Four annotated single-pair examples: d(t) at L3 (blue, left axis) and support radius r10 at L1 (red, right axis) along slerp paths in L1. Same-region 9→9 and cross-region 9A→9B stay well inside the natural support (max r10 = 70th/52nd percentile); cross-digit 9→4 and 9→0 rise toward low-density territory (80th/91st percentile), the 9→0 support peak coinciding with the d(t) jump. Dotted red = natural median r10 (2.85), dashed orange = natural p95 (4.23).](plots/direct_path_support.png)

The two investigations agree rather than conflict. The digit-9 sub-plateau (panel 2) is a *connected*
pair (`G = 1.00`) whose direct path never leaves the support (52nd percentile) — a plateau can occur
entirely on-manifold, purely from decision geometry. Typical cross-digit transitions keep `G ≈ 1`
(connected through the data) while their *straight* route detours through near-empty space (median
`E` = 95). The corridor belongs to the straight-line route, not to the manifold's connectivity.

### Resampling stability — fresh endpoint draws do not change the verdict

The verdict rules require counterexamples to be stable under **resampling**, not just replication:
a counterexample produced by one lucky draw of 20 endpoint pairs would not refute anything. We re-ran
the identical frozen pipeline on the base model with two *fresh* endpoint-sampling seeds (every
definition unchanged; re-running seed 0 as a regression check reproduced 0.996 / 25 of 45 / digit-9
`G = 1.00` exactly):

| endpoint seed | between-plateau median `G` (95% CI) | counterexamples (`G≤1`) | digit-9 sub `G` |
|--:|--|--:|--:|
| 0 (frozen) | 0.996 (0.97–1.03) | 25 / 45 | 1.00 |
| 1 | 0.977 (0.95–1.02) | 25 / 46 | 0.86 |
| 2 | 0.957 (0.90–1.00) | 30 / 46 | 0.82 |

**21 plateau pairs — including the digit-9 sub-plateau — are counterexamples under all three
independent endpoint draws**, and the between-plateau median `G` stays on (seeds 0–1) or below
(seed 2) the within-plateau baseline in every draw. Per-pair median `G` is tightly reproducible
(figure b). Both verdicts are resampling-stable, closing the last requirement of the verdict rules.

![(a) Between-plateau median G (red, 95% bootstrap CI) vs within-plateau median G (green) for three independent endpoint-sampling seeds — the between value never rises above the baseline. (b) Per-pair median G, seed 0 vs fresh seeds 1 and 2: points hug the y=x line; 21 pairs sit at G≤1 in every draw.](plots/population_resample.png)

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

At the population level, across a depth-4 MNIST MLP and four replication checkpoints:

- **Investigation 1 — Universal claim REFUTED:** 25/45 verified plateau pairs on the base model (and
  26–35 of 45–46 in every well-powered model) connect through the natural activation cloud with
  `G ≤ 1` — no larger gap than normal within-plateau travel. **21 pairs are counterexamples under all
  three independent endpoint draws** (resampling-stable), including the digit-9 case that first
  motivated this work (`G` = 1.00 / 0.86 / 0.82).
- **Investigation 1 — Typical-association claim NOT SUPPORTED:** the between-plateau median `G`
  (0.93–1.00) sits on the within-plateau baseline (1.00) in all four well-powered models, with
  overlapping bootstrap CIs and no consistent direction. The `G` distributions overlap almost
  completely.
- **Investigation 2 — the low-density corridor is REAL:** the median verified plateau-to-plateau
  direct path reaches the **95th percentile** of the natural support distribution (controls: 65th) and
  **53%** exceed the natural p95 outright (controls: 12%), with the low-density bulge sitting exactly
  mid-path where the output jump occurs.

**What this means.** A sharp plateau marks a place where the model's **decision geometry** changes
abruptly. The straight interpolation genuinely leaves the populated part of activation space — there
*is* an area real activations don't live in, and the behavior transition happens inside it — but this
is **not** a hole in the data manifold: the two plateaus remain connected by ordinary high-density
natural paths. For safety work this cuts both ways. Caution: **plateaus and low-density interpolations
are not evidence that two behaviors occupy disconnected regions of activation space** — behavior
boundaries and data-manifold components are different things. Opportunity: the model's behavior
boundaries between confident classes sit in reliably low-density territory, so an interpolation that
triggers a behavior flip is very likely operating off-distribution — worth knowing when interpreting
steering or patching experiments that move activations along straight lines.

**Limitations (stated plainly).** (1) Finite activation samples can *support or undermine* an empirical
component split but **cannot prove true topological disconnection** — a denser sample could always
reveal a bridge, and a sparser one could manufacture a gap; `G` measures empirical connectivity at the
sampled scale, nothing stronger. (2) All models share the same 1000-image MNIST training subset; a
genuinely different dataset would be a separate direction. (3) The mild `G>1` pairs (digit-1) reflect an
elongated manifold's small internal scale, not a genuine void — the metric's ratio form is sensitive to
anisotropic regions, which is why we anchor the verdict on the *distribution* of `G` and its overlap
with the control band rather than on any single pair. (4) The excursion `E` uses `k=10` nearest
neighbors; the population conclusion (95th vs 65th percentile, 53% vs 12% beyond p95) compares like
with like under the same `k`, so the contrast is not an artifact of the choice of `k`.
