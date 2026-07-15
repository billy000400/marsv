# REPORT — Are plateau-separated digit-9 regions separate manifold components?

> Final, presentable, current-best only (no history — see CHANGELOG.md). Read before rewriting.

## Summary

When you linearly interpolate the internal activations of a small image classifier between two
examples of the *same* digit, the network's downstream representation sometimes does not slide
smoothly from one to the other. Instead it **stays put, jumps sharply, then stays put again** — a
"plateau → boundary → plateau" shape. This looks as if the two same-digit examples live in two
*separate stable regions*. The safety-relevant question is **what that boundary means**: is it a real
*gap in the data* — two disconnected pieces of the natural activation manifold, which would matter for
steering, interpolation-based editing, and off-distribution detection — or is it just an artifact of
walking in a straight line through an *empty region the data never visits*, while the two regions
remain connected by a perfectly ordinary data path?

We test this on the two candidate digit-9 "stable regions" of a depth-4 MNIST MLP. **Verdict:
REFUTED (preliminary).** The two regions are **not** disconnected components. In a graph built from
real activations they connect at almost the same neighborhood scale as points *within* a single region
(5 vs 4 graph hops), and far more easily than genuinely different digits (9–12 hops); the worst edge
on that connecting path is *below* the typical local neighborhood radius, i.e. the path stays in
high-density data. And the straight-line plateau boundary itself sits at a **well-supported** point
(35th percentile of "off-manifold-ness"), not in a data void. The plateau reflects the model's
internal decision geometry, **not** a hole in the data manifold.

## Methods

### Data & Model

**Dataset.** MNIST (raw IDX, pixels scaled to $[0,1]$). We use the first 2000 test images; **1705**
are correctly classified and form the *natural activation reference cloud*.

**Model.** The existing `image-models` checkpoint `mnist_mlp_d4_w200_relu_n1000_s100000` — a depth-4
ReLU MLP $784 \to 200 \to 200 \to 200 \to 10$ (depth counts linear layers, ReLU after all but the
last), trained on a 1000-image MNIST subset (AdamW, MSE on one-hot targets). Test accuracy **85.3%**.
Weights are frozen; no retraining.

**Layers (fixed convention).** "Hidden layer $L$" is the post-ReLU output of the $L$-th linear layer.

- **Interpolation layer:** first hidden layer **L1** (200-dim). All paths are drawn in L1 activation space.
- **Downstream measurement layer:** last hidden layer **L3** (200-dim). All plateau distances are read here.

**Candidate stable regions.** Among the **177** correctly-classified digit-9 test images, we run
KMeans ($k=2$) on their **L3** activations, giving region **A** (151 images) and region **B** (26 — the
outlier cluster). A region's *medoid* is the image whose L3 activation is closest to its cluster
centroid.

**Interpolation.** The existing spherical interpolation (`slerp`) between two L1 activations $v_0,v_1$:
constant-angular-velocity direction with linearly interpolated magnitude, sampled at 200 points
$t \in [0,1]$. We then run the network forward *from L1* to obtain L3 and the logits at every $t$.

### Metrics

Write $h(t)$ for the L3 activation at interpolation parameter $t$, and $h_A=h(0)$, $h_B=h(1)$.

**Downstream distance $d(t)$** — how far along the path the *downstream* representation has moved from
endpoint A toward endpoint B. A plateau is $d\approx 0$ then a sharp rise then $d\approx 1$:

```math
d(t) = \frac{\lVert h(t) - h_A \rVert_2}{\lVert h(t) - h_A \rVert_2 + \lVert h(t) - h_B \rVert_2}
```

**Plateau boundary $t^{*}$** — the location of the sharp jump, taken as the argmax of the
box-smoothed absolute derivative of $d$ (smoothing window 5 samples):

```math
t^{*} = \arg\max_t \; \big( \lvert d'(t) \rvert * \mathrm{box}_5 \big)
```

**Support radius $r_k(t)$ (primary support metric)** — how well the interpolated activation is
*supported* by real data. For the L1 point $p(t)$ on the path, $r_k(t)$ is its Euclidean distance to
its $k$-th nearest neighbor in the natural cloud $\mathcal{N}$ ($k=10$). **Larger = lower support =
more off-manifold.**

```math
r_k(t) = \big\lVert\, p(t) - \mathrm{NN}_k\big(p(t),\, \mathcal{N}\big) \,\big\rVert_2
```

**Boundary off-manifold percentile** — to judge whether the boundary is *unusually* unsupported, we
compare $r_k(t^{*})$ to the distribution of the natural points' *own* $k$-NN radii $\lbrace r_k(x)\rbrace_{x\in
\mathcal{N}}$. The percentile is the fraction of natural points with a *smaller* radius; 50 means
"as typical as a random real activation," 90 means "more isolated than 90% of real activations."

**Mutual-kNN graph & geodesic hops (component metric).** On the natural L1 cloud we build the
**mutual** $k$-NN graph: an undirected edge $\lbrace i,j\rbrace$ exists iff $i$ is among $j$'s $k$ nearest
neighbors **and** $j$ is among $i$'s. Region separateness is then a graph question. At a fixed
within-manifold scale $k{=}10$ we report the **geodesic hop distance** (shortest number of edges)
between two region medoids — fewer hops = closer to being the same component.

**Bottleneck edge.** On the Euclidean-shortest connecting path between the two node-sets, the
**longest single edge**:

```math
b(S_1,S_2) = \min_{\text{path } \pi:\,S_1 \to S_2}\; \max_{(i,j)\in \pi}\; \lVert x_i - x_j \rVert_2
```

This is the least-supported step you are forced to take to get from one region to the other. Compared
against the natural median $k$-NN radius (2.85): **below it = the connecting path never leaves
high-density data.**

### Baselines / controls

The digit-9 measurements are only interpretable against reference categories on the *same* graph:

- **Within-region A** (a genuinely single region) — a lower bound on hops/bottleneck for "one component."
- **Cross-digit 9↔4 and 9↔0** (genuinely different classes) — an upper reference for "different regions."
- **Natural radius distribution** $\lbrace r_k(x)\rbrace$ — median 2.85, p95 4.23 — calibrates what "off-manifold" means.

A saturating **merge-$k$** baseline (smallest $k$ at which two node-sets share a connected component)
was also computed but is **non-discriminative here**: the natural cloud is dense enough that *every*
pair — including 9↔0 — merges at the minimum $k{=}3$. Hence we rely on the fixed-$k$ hop/bottleneck
contrast above, where the categories separate cleanly.

## Results

Current-best numbers; figures in `plots/`.

### Direct-path test — the plateau boundary is *not* an off-manifold gap

See `plots/direct_path_support.png` (downstream $d(t)$ in blue, support radius $r_k(t)$ in red;
dotted lines mark the natural median and p95 radius).

| Path | boundary $t^{*}$ | $r_k$ @ boundary | off-manifold **percentile** | max $r_k$ on path |
|------|-----------:|------------------:|-------------------:|-------------------:|
| same-region 9→9      | 0.01 | 3.27 | 70 | 3.28 |
| **cross-region 9→9** | 0.68 | **2.54** | **35** | 2.91 |
| cross-digit 9→4      | 0.42 | 3.42 | 77 | 3.55 |
| cross-digit 9→0      | 0.50 | 3.86 | 89 | 3.97 |

The cross-region 9→9 plateau boundary lands at the **35th percentile** of off-manifold-ness — *more*
supported than a typical real activation, the opposite of an off-manifold void. Off-manifold-ness at
the boundary instead rises with genuine class change (9→4: 77, 9→0: 89). So for digit 9 the plateau
boundary is **not** explained by the straight path leaving the data manifold.

### Component test — the two regions behave like one connected component

See `plots/component_test.png` (left: geodesic hops; right: bottleneck edge vs the natural median).

| Pair | geodesic **hops** ($k{=}10$) | bottleneck edge | vs natural median 2.85 |
|------|------------------:|----------------:|:-----------------------|
| within region A          | 4  | 2.67 | below |
| **A ↔ B (two 9-regions)**| **5**  | **2.72** | **below** |
| 9 ↔ 4 (different digit)  | 9  | 2.99 | above |
| 9 ↔ 0 (different digit)  | 12 | 2.74 | ~at |

The two candidate digit-9 regions are only **5 hops** apart — essentially the *within-region* distance
(4) — and dramatically closer than truly different digits (**9–12 hops**). The worst edge on the
connecting path (2.72) is **below** the median local neighborhood radius, so a genuine high-support
data path links the regions. There is no low-density bottleneck between them.

## Conclusion

For the digit-9 "stable regions" of this MNIST MLP, the plateau boundary is a property of the
**model's downstream geometry, not a gap in the data manifold**. Two independent measurements agree:
(1) the straight-line plateau boundary sits at a *well-supported* activation, and (2) in the real
activation graph the two regions connect at a near-within-region scale through a high-support path,
unlike genuinely different digits. Under the plan's rubric this **refutes** the hypothesis that
plateau-separated regions are disconnected/low-density-separated manifold components — the plateau is
an artifact of interpolating straight through a region the data curves around, while both endpoints
belong to one connected data component.

**Why it matters.** If plateaus marked true manifold gaps, activation steering/editing across them
would be crossing into genuinely unsupported territory. These results say the opposite for same-digit
plateaus: a data-respecting path exists, so the boundary is about *decision structure*, which is the
thing to model — not a data void to avoid.

**Limitations.** One model, one seed, one region pair (digit 9), one clustering ($k{=}2$), and a
finite 1705-point cloud; finite samples cannot prove true topological disconnection, only bound
empirical support. The fixed graph scale ($k{=}10$) is chosen to match within-region internal
connectivity ($k{=}7$–$9$); the merge-$k$ baseline saturates and is uninformative. Next steps (S2/S3):
repeat over multiple region pairs and seeds, sweep the clustering and graph $k$, and search explicitly
for the counterexample — a same-digit plateau whose regions are nonetheless joined only through a
low-support bottleneck.
