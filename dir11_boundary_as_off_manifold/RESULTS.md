# RESULTS — Do plateau transitions correspond to activation-manifold transitions?

> CURRENT-BEST ONLY. One row per experiment. History lives in CHANGELOG.md. Read before rewriting.

**The safety question.** Interpretability often assumes a network's activations live on a
low-dimensional *manifold*, with distinct behaviors on distinct, separated pieces. A sharp **plateau**
in an interpolation — the downstream representation sits still near output A, jumps abruptly, then
sits still near output B — looks like direct evidence: two basins with a wall between them. If
plateaus really marked disconnected manifold components, we could monitor behaviors by component. We
test this at the population level, across the whole model.

**Two complementary investigations:**

1. **Manifold components** — do the two plateaus lie on different empirical components of the natural
   activation cloud? (Universal claim: *every* plateau transition is a component transition; one
   counterexample refutes it. Typical-association claim: plateau pairs are *usually* more separated
   than within-plateau controls.)
2. **The low-density corridor** — does the straight interpolation path pass through a region real
   activations don't live in?

**Answer in one line: the plateaus are connected — no separate components — yet the straight path
between them really does cross an empty region. The "wall" belongs to the straight-line route and the
model's decision geometry, not to the manifold's topology.**

## Setup (frozen before looking at any between-plateau result)
- **Model:** the `image-models` MNIST MLP `mnist_mlp_d4_w200_relu` (784→200→200→200→10, ReLU, MSE to
  one-hot, test acc **85.3%**), plus four existing replication checkpoints (a second seed and three
  architectures).
- **Layers:** the spherical interpolation (slerp) is done in the **first hidden layer L1** (200-dim,
  post-ReLU); the downstream curve `d(t)` is measured at the **last hidden layer L3**; the support
  radius `r_10` (Investigation 2) is measured at **L1**, the layer the path lives in.
- **Natural cloud:** L1 activations of all **1705** correctly-classified test images (of 2000).
- **Plateau regions:** the **10 digit classes**, restricted to correct examples with output margin
  (top-1 − top-2 logit) **≥ 0.5** (a fixed confidence rule; 1604 pass, ≥130 per digit). The
  previously-studied **digit-9 A/B** sub-plateau is included as one extra pair, treated like any
  other.
- **Sampling:** 20 endpoint pairs per region pair (seed 0), same count for within-plateau controls.

## The three quantities (full definitions and equations in REPORT.md Methods)

**`d(t)` — plateau checker, a filter only.** A path is a **verified transition** iff it spends ≥ half
its length flat near an endpoint (`d<0.2` or `d>0.8`), starts below 0.2, and ends above 0.8. Never
reported as a score.

**`G` — Investigation 1's score.** Build a Euclidean **minimum spanning tree (MST)** — the
shortest-total-length network joining all 1705 natural activations. For endpoints `u,v`, the
bottleneck `B(u,v)` = largest edge on their MST path = the single biggest hop any route through the
data must take. `G` = `B` divided by the within-plateau hop scale `max(s_i, s_j)` (medians frozen from
controls). **Read:** `G = 1` — no bigger hop than normal inside-plateau travel; `G > 1` — unusually
big hop needed (separation); `G ≤ 1` — a counterexample to the universal claim.

![Toy 2-D schematic. (a) The MST joins all points with the shortest total edge length. (b) The largest edge (red) on the MST path from u to v is the bottleneck B — the one hop no route can avoid.](plots/mst_explainer.png)

**`E` — Investigation 2's score.** `r_10(x_t)` = distance from path point `x_t` to its 10th-nearest
natural activation (large = locally empty). `E` = the path's max `r_10`, as a percentile of natural
points' own `r_10` (median 2.85, p95 4.23). **Read:** `E ≈ 50` — never less supported than a typical
real activation; `E > 95` — visits near-empty territory.

Plot conventions: red = between-plateau, green = within-plateau control.

## Investigation 1 — population verdict (base model, 45 cross-digit pairs + digit-9 sub-plateau)

| quantity | value |
|--|--|
| within-plateau `G`: median / p95 | 1.00 / 1.39 |
| **between-plateau median `G`** (over 45 verified pairs) | **0.996**  (95% CI 0.97–1.03) |
| between-plateau median `G`: min – max pair | 0.84 – 1.67 |
| verified pairs with median `G > 1` | **44%** (20 / 45) |
| **counterexamples** (verified pairs with median `G ≤ 1`) | **25 / 45** |
| digit-9 A/B sub-plateau (the original case) | **G = 1.00** (a counterexample) |

**Universal claim: REFUTED.** 25 of 45 verified plateau transitions have `G ≤ 1`, including the
original digit-9 sub-plateau. A wall between plateaus is not required.

**Typical-association claim: NOT SUPPORTED.** The between-plateau median `G` (0.996) sits on the
within-plateau baseline (1.00); the CIs overlap (0.97–1.03 vs 0.99–1.02). The largest bridge any pair
needs (`G = 1.67`, digit 1↔8) barely exceeds the *within*-plateau p95 (1.39). No pair is separated by
a dramatic data hole.

![Panel (a): per-pair G histograms — red (between) and green (within) overlap almost completely. Panel (b): median G per pair, about half each side of the dashed G=1 line; digit-9 sub-plateau (orange) exactly at 1.](plots/population_G.png)

The pairs with modestly larger `G` all involve digit **1** (1↔8 = 1.67, 0↔1 = 1.57, 1↔3 = 1.46).
Digit 1's cloud is elongated and thin, so its own internal scale `s_1` is small — the ratio is
inflated, not the gap. The heatmap (median `G` per digit pair, diagonal = 1) shows no block structure.

![Digit-by-digit heatmap of median G. Values hug 1; only digit-1 pairs are mildly elevated.](plots/population_heatmap.png)

Representative verified `d(t)` curves (largest-`G` pair, digit-9 sub-plateau, smallest-`G` pair; x =
interpolation position `t`, y = `d(t)`) confirm the filter selects genuine flat→jump→flat paths:

![Three representative verified d(t) curves, all plateau-jump-plateau.](plots/population_dt.png)

### Why `G` divides by `max(s_i, s_j)` — and what the alternatives show

A journey from region `i` to region `j` travels through both, so hops up to `max(s_i, s_j)` occur on
its within-region legs regardless of the boundary — the sparser region demands them internally. A
smaller denominator (`min`, `mean`, global) flags a pair as "separated" merely because its regions
differ in density. Sensitivity check on the identical frozen paths (the `max` row reproduced the
published numbers exactly):

| denominator | between median `G` (95% CI) | counterexamples | digit-9 sub `G` |
|--|--|--:|--:|
| **`max(s_i,s_j)` — frozen** | **0.996** (0.97–1.03) | **25/45** | 1.00 |
| `min(s_i,s_j)` | 1.274 (1.21–1.34) | 2/45 | 1.26 |
| `mean(s_i,s_j)` | 1.117 (1.06–1.17) | 5/45 | 1.12 |
| global `s` (= 2.46) | 1.093 (1.06–1.11) | 7/45 | 1.21 |

The shifts under the alternatives are density-mismatch arithmetic, not a boundary gap: per pair,
`B / max(s_i,s_j)` has quartiles 0.945–1.059 — the bottleneck almost exactly *equals* the sparser
region's internal scale — so `B/min` etc. exceed 1 exactly when the regions' densities differ.
`min`-normalized `G` correlates 0.68 with the pair's scale-asymmetry ratio (frozen `max`-`G`: −0.09),
and its 14 elevated pairs all involve the three densest regions (digits 1, 8, 0). Decisively: the
biggest forced hop is a genuine boundary crossing (edge joining two different digits' points) on only
**27%** of the 663 verified between-plateau journeys — 73% of the time it lies inside one digit's
cloud (controls: 4%). Verdicts unchanged; even under the harshest `min` denominator two counterexample
pairs remain (4–5, 4–7).

![Panel (a): between-plateau median G under four denominators; only the frozen max sits on the within baseline, and all shifts stay below the within-plateau p95. Panel (b): min-normalized per-pair G grows with scale asymmetry while the frozen max-G stays flat at 1.](plots/normalization_check.png)

## Investigation 2 — the direct path crosses a region real activations avoid

Same frozen population: 46 region pairs × 20 endpoint pairs (seed 0); 676 of 920 paths pass the `d(t)`
filter; 120 slerp points per path; controls = 200 within-plateau paths.

| quantity | between-plateau (verified, n=676) | within-plateau controls (n=200) |
|--|--:|--:|
| median excursion `E` | **95.4** (IQR 87.8–98.4) | 65.2 (IQR 38.3–86.1) |
| paths whose max `r_10` exceeds the natural p95 | **53%** | 12% |

**The corridor is real.** The median verified path climbs to the 95th percentile of natural support;
over half go beyond the natural p95 outright. Controls stay inside the cloud. The support radius
bulges mid-path — to ~1.45× the natural median, exactly where the `d(t)` jump happens — and returns to
normal at both endpoints. Panel (a): histogram of `E` per path. Panel (b): median `r_10(t)` relative
to the natural median, with IQR bands, vs slerp position `t`.

![Panel (a): between-plateau paths (red) pile up at the 90-100th percentile of E, 53% beyond the dashed p95 line; controls (green) spread lower. Panel (b): the red profile bulges mid-path to ~1.45x; the green control stays flat at ~0.95.](plots/direct_path_population.png)

**Single-pair illustration** (the original pilot figure, regenerated, fully annotated). Each panel:
one endpoint pair (region medoids), 200 points per path; `d(t)` at L3 (blue, left axis); `r_10` at L1
(red, right axis) vs the 1705-point cloud; dotted red = natural median (2.85), dashed orange = p95
(4.23).

![Four single-pair examples: same-region 9-9 and sub-plateau 9A-9B stay inside natural support (max r10 at 70th/52nd pctile); cross-digit 9-4 and 9-0 rise toward low density (80th/91st), the 9-0 peak coinciding with the d(t) jump.](plots/direct_path_support.png)

The investigations agree: the digit-9 sub-plateau is *connected* (`G = 1.00`) and its direct path
never leaves the support (`E` = 52) — a plateau can arise entirely on-manifold from decision geometry.
Typical cross-digit transitions keep `G ≈ 1` while their straight route detours through near-empty
space.

### Is the corridor where the wrongly classified images live? — No

The natural cloud deliberately uses only **correct** test images (1705 of 2000). Do the **295
wrong-image activations** live in the corridor? Three tests on the identical frozen paths (base
model; a seed-0 re-run reproduced `E` = 95.4 / 65.2 exactly):

| test | result |
|--|--|
| (A) wrong activations' own support vs the correct cloud | median **74th** pctile (IQR 56–88), 10% beyond p95 — far short of corridor points (95.4, 53%) |
| (B) `E` recomputed against the augmented cloud (all 2000 activations, own baseline) | between-plateau median **95.4 → 95.2**; controls 65.2 → 62.6 — the corridor does not fill |
| (C) corridor points' support *among the wrong cloud itself* (its own 295-pt baseline) | median **92nd** pctile — corridor points are strangers to the wrong cloud too (controls' max points: 90th) |

Wrong-image activations sit in the moderately-thin *edge* of the correct cloud — as expected for
borderline examples — but not in the corridor: they are real images, and the corridor is not populated
by any image, right or wrong. The along-path distance to the wrong cloud shows **no mid-path dip**
(a dip would mean wrong activations congregate where the output jumps); it stays flat at ~1.25× the
wrong cloud's own median support while the distance to the correct cloud bulges to ~1.45×.

![Panel A: wrong-image activations (purple) spread over mid percentiles, median 74; corridor excursions (red) pile up at 90-100. Panel B: along verified between-plateau paths, r10 to the wrong cloud (purple) stays flat with no mid-path dip while r10 to the correct cloud (red) bulges. Panel C: E histograms vs correct-only and augmented clouds coincide (median 95 both).](plots/wrong_class_corridor.png)

## Resampling stability — the verdict does not depend on the endpoint draw

Two fresh endpoint-sampling seeds, everything else frozen (seed 0 re-run reproduced the published
numbers exactly):

| endpoint seed | between-plateau median `G` (95% CI) | counterexamples (`G≤1`) | digit-9 sub `G` |
|--:|--|--:|--:|
| 0 (frozen) | 0.996 (0.97–1.03) | 25 / 45 | 1.00 |
| 1 | 0.977 (0.95–1.02) | 25 / 46 | 0.86 |
| 2 | 0.957 (0.90–1.00) | 30 / 46 | 0.82 |

**21 plateau pairs — including the digit-9 sub-plateau — are counterexamples under all three draws**;
the between-plateau median never rises above the baseline. Panel (a): median `G` (with 95% CI) per
seed. Panel (b): per-pair median `G`, seed 0 vs fresh seeds.

![Panel (a): red between-plateau median G never rises above the green baseline in any seed. Panel (b): per-pair G hugs the y=x line across draws.](plots/population_resample.png)

## Replication — second seed and three architectures

Identical frozen pipeline on the existing checkpoints (`experiments/population_manifold.py`):

| model (test acc) | verified pairs | between-plateau median `G` | % pairs `G>1` | counterexamples (`G≤1`) |
|--|--:|--:|--:|--:|
| base d4w200, seed 0 (85.3%) | 45 | **0.996** | 44% | 25 |
| seed 1 d4w200 (86.9%)       | 45 | **0.925** | 22% | 35 |
| d4w400 wider (86.9%)        | 46 | **0.987** | 43% | 26 |
| d5w200 deeper (85.9%)       | 46 | **0.982** | 30% | 32 |
| d3w200 shallower (78.1%)*   | 1  | 0.982 | 0% | 1 |

\*Excluded on structural grounds (next section). The four well-powered models agree: between-plateau
median `G` = 0.93–1.00, never consistently above the baseline; each finds many counterexamples; in
seed 1 the direction even reverses. Neither claim survives replication.

![Panel (a): between-plateau median G (red, 95% CI) sits on the green baseline for every model. Panel (b): about half of each well-powered model's pairs fall each side of G=1.](plots/population_replication.png)

### Why the shallow net can't be powered up — and why that doesn't change the verdict

Its deficit is structural, not sampling noise. Its `d(t)` **ramps rather than plateaus**: mean plateau
fraction 0.25 (max 0.43) across all 46 region pairs, 0/46 reaching the 0.5 accept threshold even on
average, vs 0.60 (43/46) for the base net. Sampling 10× more endpoint pairs (20 → 200) leaves only 2
verified pairs. We do not relax the `d(t)` filter — a ramp is not a plateau. Its 1–2 genuine plateaus
are all counterexamples (`G ≤ 1`, median 0.76–0.98), consistent with the other models.

![Panel (a): the shallow net's per-region-pair plateau fractions (red) sit far below the dashed 0.5 accept line; the base net (blue) clears it 43/46 times. Panel (b): 10x more sampling does not restore power (still ≤2 verified pairs).](plots/population_shallow_power.png)

## Headline
**The plateaus are connected; the straight path between them is not where the data lives.**
- **Universal claim REFUTED:** 25/45 verified pairs on the base model (26–35 in every well-powered
  model) have `G ≤ 1`; **21 pairs are counterexamples under all three endpoint draws**, including the
  original digit-9 case (`G` = 1.00 / 0.86 / 0.82).
- **Typical-association claim NOT SUPPORTED:** between-plateau median `G` (0.93–1.00) sits on the
  within-plateau baseline in all four well-powered models; CIs overlap; no consistent direction.
- **The low-density corridor is REAL:** median verified direct path reaches `E` = 95.4 (controls:
  65.2); 53% exceed the natural p95 (controls: 12%); the bulge sits exactly mid-path. It is **not**
  where the wrongly-classified images live: adding their 295 activations leaves `E` at 95.2, and
  corridor points sit at the 92nd pctile of the wrong cloud's own support.

The plateau reflects the model's **decision geometry**, not a hole in the data manifold.
**Limitation:** finite samples can support or undermine *empirical* component separation but cannot
prove true *topological* disconnection; all models share the same 1000-image MNIST training subset;
the mild `G>1` pairs (digit 1) reflect an elongated cloud's small internal scale, not a genuine void.
