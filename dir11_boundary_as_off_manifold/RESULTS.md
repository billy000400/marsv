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
distance at the **last hidden layer L3**. Natural reference cloud = L1 activations of the **1705
correctly classified** test images (of 2000). Candidate digit-9 regions = KMeans(k=2) on the L3
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
natural median neighborhood radius, so **a high-support natural path connects the regions**. (Robustness:
in the mutual-kNN graph *every* digit pair already merges into one component at the minimal k=3, so a
saturating "merge-k" is non-discriminative; the fixed-k=10 hop/bottleneck comparison above is the
discriminative measure.)

## Figures
- `plots/direct_path_support.png` — d(t) (downstream) + k-NN support radius along the four slerp paths; the cross-region 9→9 boundary is at high support.
- `plots/component_test.png` — geodesic hops and bottleneck edge per pair category at k=10; A↔B looks like within-region, not cross-digit.

## Headline
**REFUTED (preliminary, digit-9).** The plateau-separated digit-9 regions are **not** disconnected
manifold components: in the natural activation graph they connect at a near-within-region scale (5 vs 4
hops) through a **high-support** path (bottleneck below the median neighborhood radius), and the
straight-line plateau boundary sits at a *well-supported* point, not an off-manifold gap. The plateau
reflects the model's decision geometry, not a hole in the data manifold. Caveats: one model/seed, one
region pair, k-means k=2, finite sample — S2/S3 robustness still pending.
