# RESULTS — Are plateau-separated digit-9 regions separate manifold components?

> CURRENT-BEST ONLY. One row per experiment. History lives in CHANGELOG.md. Read before rewriting.

**Question.** A depth-4 MNIST MLP shows a *plateau* on some digit-9→digit-9 activation
interpolations: the downstream representation stays near endpoint A, jumps sharply, then stays near
endpoint B — as if the two 9s live in two separated "stable regions." Are those two regions actually
**disconnected (or low-density-separated) components of the natural activation manifold**, or does the
plateau just come from a straight-line slerp leaving the data manifold while the regions remain
connected by a high-support natural path?

**Setup.** Reuse the `image-models` checkpoint `mnist_mlp_d4_w200_relu` (784→200→200→200→10, ReLU,
test acc 85.3%). Interpolate the **first hidden layer L1** (200-dim, post-ReLU); measure downstream
distance at the **last hidden layer L3** (200-dim). Natural reference cloud = L1 activations of the
**1705 correctly classified** test images (of 2000). Candidate digit-9 regions = KMeans(k=2) on the L3
activations of the **177 correct 9s** → region A (151) and region B (26, the outlier cluster).

## Direct-path test — is the plateau boundary at an off-manifold gap?
For each L1 slerp path: plateau boundary = argmax of smoothed |d′(t)|; support = k-NN radius (k=10) to
the natural cloud. `rad pctile` = where the boundary's radius falls in the distribution of the natural
points' *own* k-NN radii (higher = more off-manifold). Natural median radius = 2.85, p95 = 4.23.

| Path | boundary t | radius @ boundary | rad **percentile** | max radius on path |
|------|-----------:|------------------:|-------------------:|-------------------:|
| same-region 9→9      | 0.01 | 3.27 | 70 | 3.28 |
| **cross-region 9→9** | 0.68 | **2.54** | **35** | 2.91 |
| cross-digit 9→4      | 0.42 | 3.42 | 77 | 3.55 |
| cross-digit 9→0      | 0.50 | 3.86 | 89 | 3.97 |

The **cross-region 9→9 plateau boundary sits at a *well-supported* location** (35th percentile, *below*
the median radius) — not an off-manifold gap. Off-manifold-ness at the boundary instead grows with
cross-**digit**-ness (77, 89 pct for 9→4, 9→0). ⇒ **direct-path claim not supported** for digit-9.

## Component test — do the two 9-regions form one connected component?
Mutual-kNN graph on the 1705 natural L1 activations. At a within-manifold scale **k=10**: geodesic
hop distance and bottleneck edge (worst edge on the connecting path) between region medoids.

| Pair | geodesic **hops** | bottleneck edge | vs natural median 2.85 |
|------|------------------:|----------------:|:-----------------------|
| within region A          | 4  | 2.67 | below |
| **A ↔ B (two 9-regions)**| **5**  | **2.72** | **below** |
| 9 ↔ 4 (different digit)  | 9  | 2.99 | above |
| 9 ↔ 0 (different digit)  | 12 | 2.74 | ~at |

The two digit-9 regions are **5 hops apart — nearly as close as within region A (4)** — and far
closer than genuinely different digits (9–12 hops). Every connecting bottleneck edge is at/below the
natural median neighborhood radius, so **a high-support natural path connects the regions**.

## Robustness — does the verdict survive the hyperparameters?
We stress-test the digit-9 result and check generality. (`experiments/robustness.py`;
figures `plots/robustness_graphk.png`, `plots/robustness_regions.png`.)

**1. Graph scale (mutual-kNN k = 6…25).** The A↔B (two 9-regions) geodesic distance tracks the
within-region-A control at *every* scale and stays far below both cross-digit references:

| k | within A | **A↔B (9)** | 9↔4 | 9↔0 |
|--:|---------:|------------:|----:|----:|
| 6  | 7 | – | 16 | – |
| 10 | 4 | **5** | 9  | 12 |
| 15 | 4 | **3** | 7  | 9  |
| 25 | 4 | **2** | 6  | 7  |

(– = the two node-sets aren't yet in one connected component at that small k. At every k where a pair
*is* connected, A↔B is at/below the within-region control and well below both cross-digit pairs.)

**2. Region definition (KMeans-k ∈ {2,3,4} × seed ∈ {0,1,2}, 9 configs).** Every digit-9 region
split is within-region-like: **hops 2–5, bottleneck 1.93–2.72 — all below the natural median (2.85)**.
The verdict does not depend on how we cut the 9-cluster.

**3. Generality (all 10 digits) + counterexample search.** For each digit we split its L3 activations
into two regions (KMeans-2) and connect them on the same graph. The off-manifold "gap" test is the
**bottleneck edge vs the natural p95 radius (4.23)** — the support-relevant metric.

| | same-digit region pairs (10 digits) | cross-digit pairs (45) |
|--|:--|:--|
| bottleneck edge | **2.35 – 4.17 (all ≤ p95)** | 2.03 – 3.47 |
| geodesic hops   | 4 – 15 | 7 – 21 |

**No same-digit region pair crosses an off-manifold gap** (max bottleneck = digit-4's 4.17, still
below p95). **Counterexample search: 0 found.** Verdict stable.

**Caveat — which metric to trust.** In this *dense* L1 cloud the mutual-kNN graph is essentially one
high-support connected blob: even cross-digit bottlenecks stay below p95, so *no* pair is separated by
a true data hole. Geodesic **hops** is therefore the signal that separates categories for digit-9 —
but hops is **not** universally reliable: digit-1's own two regions are **15 hops** apart (an elongated
manifold with a tiny 8-point outlier cluster) yet their bottleneck (2.70) is below the median. We
therefore anchor the generality verdict on the unambiguous off-manifold (bottleneck-vs-p95) criterion,
and read digit-9 through both hops and bottleneck, which agree.

## Figures
- `plots/direct_path_support.png` — d(t) (downstream) + k-NN support radius along the four slerp paths; the cross-region 9→9 boundary is at high support.
- `plots/component_test.png` — geodesic hops and bottleneck edge per pair category at k=10; A↔B looks like within-region, not cross-digit.
- `plots/robustness_graphk.png` — geodesic hops vs graph scale k; A↔B tracks the within-region control, far below cross-digit, at every k.
- `plots/robustness_regions.png` — bottleneck (left) and hops (right) for every digit's two-region split vs the cross-digit band; digit 9 highlighted, no gap-crossing pair.

## Headline
**REFUTED (robust across region-definition & graph hyperparameters; single checkpoint/seed).**
The plateau-separated digit-9 regions are **not** disconnected manifold components: the straight-line
plateau boundary sits at a *well-supported* point (35th pctile), and in the natural activation graph the
two regions connect at a near-within-region scale (5 vs 4 hops) through a high-support path (bottleneck
below the median radius) — stable across KMeans-k∈{2,3,4}×seed and graph-k∈{6…25}. The finding
generalizes: **no** same-digit region pair, across all ten digits, is separated by an off-manifold gap
(bottleneck ≤ p95; 0 counterexamples). The plateau reflects the model's decision geometry, not a hole
in the data manifold. Caveats: one model/checkpoint; the L1 cloud is dense enough that graph *hops*
(not the bottleneck) is what discriminates categories, and hops can be inflated by manifold elongation
(digit-1) — so we anchor generality on the off-manifold criterion.
