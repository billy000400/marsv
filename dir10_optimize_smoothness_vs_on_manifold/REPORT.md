# REPORT — Does combined path smoothness recover the weekday activation manifold?

> Final, presentable, current-best only (history in CHANGELOG.md). Read before rewriting.

## Summary

Large language models represent a small, structured task (here: "what weekday is *k* days
after *X*?") as a low-dimensional geometric object in activation space — for the seven weekdays,
the class centroids trace out a smooth, roughly circular loop. A recent paper (Wurgaft et al.,
*Manifold Steering*, arXiv:2605.05115v1) fits that loop with a **periodic cubic spline** and calls
it the weekday *activation manifold*. This raises a mechanistic question with a safety motive: if we
did not already know that manifold, could a **generic smoothness principle** reconstruct it? If a
simple "move smoothly, in activations and in behavior" objective reproduced the fitted manifold, the
manifold would carry little extra information; if it does not, the fitted manifold encodes real
model-specific geometry that smoothness alone misses — which matters for anyone hoping to *steer*
along such manifolds.

We test one concrete version. Fixing the two endpoint centroids of an adjacent weekday pair, we
optimize an activation path to minimize a weighted sum of **activation-space kinetic energy**
(how much the path wiggles in activation space) and **downstream behavior-space kinetic energy**
(how much the induced output distribution wiggles), sweeping the weight $\lambda$. We then measure how
close each optimized path lands to the paper's fitted spline.

**Verdict (Tuesday→Wednesday pilot):** **No.** Across five $\lambda$ values, an output-only extreme,
and three initializations, no optimized path lands near the fitted spline. Recovery *monotonically
worsens* as $\lambda$ grows; the straight chord ($\lambda=0$) is the closest optimized path and is
still far off. Crucially, the fitted spline is itself **dominated in both energies** — it has higher
activation *and* higher behavior kinetic energy than the chord — so this smoothness objective can
never prefer it. Generic combined kinetic smoothness does not explain this fitted manifold; it points
away from it.

## Methods

### Data & Model

**Model.** Llama 3.1 8B **base** (`meta-llama/Llama-3.1-8B`), bfloat16. Because our shared GPU
budget (7.2 GB) is smaller than the 16 GB bf16 model, weights are split across a capped GPU slice and
CPU RAM for one-time activation collection; the numerically identical bf16 precision is used
throughout. For the optimization sweep only the layer-28→31 tail (four transformer blocks + final
RMSNorm + unembedding) is resident on the GPU.

**Task & prompts.** The 49 weekday-addition prompts, template exactly

```
Q: What day is {k} days after {entity}?\nA:
```

with `entity` ∈ {Monday…Sunday} and `k` ∈ {one…seven}; ground truth wraps cyclically mod 7 — so
there are **exactly 7 prompt sequences per ground-truth weekday** (49 = 7 weekdays × 7 increments),
each averaged into one centroid. Behavior energies are averaged over $N_{\mathrm{base}}=16$ base
prompts (a fixed seeded subset of the 49).

**Activation site.** Residual stream at **layer 28** (`hidden_states[28]`, the 28th block's output),
at the final answer-predicting token position (the answer token is *not* appended). PCA is fit over
the 49 activations. *Requested "PCA-64" is impossible with 49 points (subspace rank ≤ 48); we retain
all 48 non-degenerate components. The first-32 optimization subspace and the PCA-32 recovery metric
are unaffected.* Seven weekday **centroids** are the mean PCA projection grouped by **ground-truth
answer**. The **activation manifold** reference is a **periodic cubic spline** through the seven
centroids (Appendix A.3).

*How representative are the first 2–3 PCs?* Only weakly: PC1–PC2 capture **31.4%** of the layer-28
activation variance and PC1–PC3 **43.6%**; **18** PCs are needed for 90% and 32 reach 98.1% (scree
figure in Results). The 2-D PCA scatter is therefore illustrative only — the weekday geometry is
genuinely high-dimensional, so all recovery/energy conclusions are computed in the **PCA-32**
optimization subspace, never in the 2-D picture.

**Behavior representation.** Full-vocab softmax at the answer position; probability mass for each
weekday's tokenizer-valid spelling variants is summed into one bin (2–3 token IDs each), and all
remaining mass into an `other` bin — an 8-bin distribution, mapped to **Hellinger coordinates** via
element-wise square root (Appendix A.2).

### Path parameterization & intervention

A path is a **natural cubic spline through 10 control points** at uniform positions on $[0,1]$; the
first and last control points are pinned to the two endpoint weekday centroids' first-32 PCA
coordinates, and the interior 8 are optimized. The path is evaluated at $N_{\mathrm{wp}}=11$ uniform
waypoints. Control-point→waypoint mapping is a fixed linear operator, so endpoints stay exact.

To read the downstream behavior of a waypoint $w$ (a point in the first-32 PCA subspace), we overwrite
each base prompt's own first-32 PCA coordinates at the answer token with $w$, keeping its higher PCA
components and orthogonal residual fixed, and run the tail. Equivalently the injected activation is

```math
a^{\mathrm{inj}}_p = a_p + V_{32}^{\top}\big(w - V_{32}(a_p - \mu)\big)
```

where $a_p$ is base prompt $p$'s answer-token layer-28 activation, $\mu$ the PCA mean, and $V_{32}$ the
first-32 PCA components. Behavior energy is averaged over $N_{\mathrm{base}}=16$ base prompts (a fixed
seeded sample of the 49).

### Metrics

**Activation kinetic energy** — how much the path moves in the first-32 PCA subspace; the
finite-difference discretization of $\int_0^1 \lVert x'(t)\rVert^2 dt$ over the waypoints $W_i$:

```math
E_{\mathrm{act}}(W) = \frac{1}{\Delta t}\sum_{i=1}^{N_{\mathrm{wp}}-1}\lVert W_{i+1}-W_i\rVert^2,
\qquad \Delta t = \tfrac{1}{N_{\mathrm{wp}}-1}
```

**Behavior kinetic energy** — the same energy applied to the induced 8-bin behavior curve in
Hellinger coordinates $H_i^{(p)} = \sqrt{q_i^{(p)}}$ (with $q_i^{(p)}$ the 8-bin distribution at
waypoint $i$ for base prompt $p$), averaged over base prompts:

```math
E_{\mathrm{out}}(W) = \frac{1}{N_{\mathrm{base}}}\sum_{p}\frac{1}{\Delta t}
\sum_{i=1}^{N_{\mathrm{wp}}-1}\big\lVert H_{i+1}^{(p)}-H_i^{(p)}\big\rVert^2
```

Both energies are **lower = smoother**. We normalize each by its value on the linear chord,
$\tilde E = E / E(\text{linear})$, and minimize

```math
\mathcal{L}(W) = \tilde E_{\mathrm{act}}(W) + \lambda\,\tilde E_{\mathrm{out}}(W)
```

with L-BFGS (strong-Wolfe line search) over the 8 interior control points, up to 30 outer steps
(≤ 4 inner iterations each), early-stopping on relative loss change. Coarse grid
$\lambda \in \lbrace 0, 0.1, 1, 10, 100 \rbrace$ plus an **output-only** baseline that minimizes
$\tilde E_{\mathrm{out}}$ alone.

**Activation-manifold recovery** (primary outcome) — how close the optimized path lands to the fitted
manifold. We densely sample the shorter periodic-spline arc between the two endpoint centroids (first-32
coords), $S$, and report the mean nearest-point distance from the path's waypoints:

```math
R(W) = \frac{1}{N_{\mathrm{wp}}}\sum_{i}\min_{s\in S}\lVert W_i - s\rVert
```

**Lower $R$ = closer to the fitted manifold.** *The paper's Appendix-A.9 SVD common-subspace recovery
score could not be reproduced faithfully (the appendix's exact variance threshold/convention is not
available to this run); we report the transparent PCA-32 nearest-spline distance instead and flag this
as a deviation.*

**Downstream displacement $d(t)$** (diagnostic) — mean over base prompts of the Hellinger distance
between the behavior at waypoint $t$ and at the start:

```math
d(t) = \frac{1}{N_{\mathrm{base}}}\sum_p \frac{1}{\sqrt{2}}\big\lVert H_t^{(p)} - H_0^{(p)}\big\rVert
```

*Note: the direction's plan references a `d(t)` from `slerp_relative_distance.py`, which does not
exist in this repository; we define $d(t)$ as above and use it only as a downstream diagnostic, never
as a substitute for activation-space path distance.*

**Plateau coordinate $p(t)$** (diagnostic, added per operator request) — a normalized downstream
progress metric that is 0 at the start behavior and 1 at the end behavior, so a *plateau* (behavior
lingering near an endpoint before switching) shows as a flat stretch. With $h(t)=H_t^{(p)}$ the
Hellinger-coordinate behavior at waypoint $t$ and $h_A,h_B$ the start/end endpoint behaviors:

```math
p(t) = \frac{1}{N_{\mathrm{base}}}\sum_p
\frac{\lVert H_t^{(p)}-H_0^{(p)}\rVert}{\lVert H_t^{(p)}-H_0^{(p)}\rVert + \lVert H_t^{(p)}-H_{N-1}^{(p)}\rVert}
```

### Baselines / reference paths

- **Linear chord** — the straight line between the two centroids (the $\lambda=0$ minimizer; the
  $E_{\mathrm{act}}$ global optimum). Normalizer denominator for both energies.
- **Output-only** — minimizes $\tilde E_{\mathrm{out}}$ alone (the $\lambda\to\infty$ extreme).
- **Centroid spline** — the fitted periodic-spline arc itself, sampled at the waypoints; the target
  we ask the objective to reconstruct.

## Results

**Setup validation (S2).** The paper-consistent weekday setup reproduces cleanly: task accuracy
**0.939** (46/49), mean weekday probability mass **0.743**, mean `other` mass **0.257**. The seven
ground-truth centroids are well separated (adjacent L2 spacing 8.5–11.8 in PCA-48), so the periodic
spline is well-posed. In the PCA scatter below, the **star (★) markers are the seven ground-truth
weekday centroids** (mean of each weekday's 7 prompts), the small dots are the 49 individual prompt
activations, and the gray curve is the fitted periodic spline.

![PCA (PC1-PC2) of the 49 layer-28 weekday activations. Dots = individual prompts (7 per weekday); stars = the 7 ground-truth centroids; gray line = fitted periodic cubic spline.](plots/s2_pca_weekday_manifold.png)

The scree plot confirms 2–3 PCs are *not* representative (PC1–2 ≈ 31%, PC1–3 ≈ 44%; 18 PCs for 90%),
which is why every quantitative result uses the PCA-32 subspace rather than this 2-D view.

![Explained variance vs. number of principal components. Bars = per-PC variance; red curve = cumulative. PC1-2 reach only ~31%, PC1-3 ~44%.](plots/s2_pca_cumvar.png)

**Sanity checks (S4).** $\lambda=0$ recovers the linear chord for every seed (endpoint error 0,
$E_{\mathrm{act}}$ at its global minimum 88.8, recovery 0.961). Output-only reduces $E_{\mathrm{out}}$
below the chord (0.930 vs 1.026). All paths keep the two endpoints exactly fixed.

**Activation-manifold recovery (primary).** Recovery $R$ (mean nearest-spline distance, PCA-32;
lower = closer; adjacent centroids are ~9.6 apart):

| init | $\lambda{=}0$ | $\lambda{=}0.1$ | $\lambda{=}1$ | $\lambda{=}10$ | $\lambda{=}100$ | output-only |
|---|---|---|---|---|---|---|
| linear       | **0.961** | 0.965 | 0.986 | 1.020 | 1.022 | 1.023 |
| perturbed #1 | 0.961 | 0.975 | 1.029 | 1.099 | 1.390 | 1.400 |
| perturbed #2 | 0.961 | 0.978 | 1.067 | 1.146 | 1.408 | 1.425 |
| **centroid spline (target)** | | | | | | **0.004** |

Recovery worsens monotonically with $\lambda$ for every seed — the opposite of the working hypothesis.
No intermediate $\lambda$ improves on the extremes, and the best optimized path (the chord) still sits
~0.96 from the spline while the spline is 0.004 from itself: a ~250× gap
(`plots/s4_recovery_vs_lambda.png`). From perturbed starts the high-$\lambda$ and output-only paths
diverge wildly in activation space ($E_{\mathrm{act}}\approx 306$–$313$ vs $93$ from the linear start,
at nearly identical $E_{\mathrm{out}}\approx 0.94$): the behavior objective has a broad flat minimum
and does **not determine the activation path**.

![Manifold-recovery distance vs λ for all three seeds. Optimized paths near 1.0; the centroid-spline target near 0 — a ~250x gap no λ closes.](plots/s4_recovery_vs_lambda.png)

**Energy trade-off.** Raw energies (lower = smoother):

| path | $E_{\mathrm{act}}$ | $E_{\mathrm{out}}$ |
|---|---|---|
| linear chord ($\lambda{=}0$) | 88.8 | 1.026 |
| $\lambda{=}1$ | 89.8 | 0.957 |
| output-only (linear init) | 92.6 | 0.930 |
| **centroid spline (target)** | **104.9** | **1.118** |

The fitted spline occupies the **worst corner** of the trade-off plane — the highest activation KE
*and* the highest behavior KE (`plots/s4_energy_tradeoff.png`). It is Pareto-dominated by the chord in
both energies. The optimized paths trace the true lower-left frontier; the reference we were asked to
reconstruct is nowhere near it.

![E_act vs E_out for Tuesday→Wednesday. Optimized paths trace the lower-left frontier; the fitted centroid spline (star) sits alone in the dominated top-right corner.](plots/s4_energy_tradeoff.png)

**Downstream diagnostic.** $d(t)$ rises smoothly along every path; higher $\lambda$ flattens it
modestly, confirming the objective does act on downstream behavior — but, as the recovery result
shows, a flatter behavior curve does **not** imply an on-manifold activation path.

![Downstream displacement d(t) = mean Hellinger distance from the start behavior along each path.](plots/s4_dt_curves.png)

**Plateau metric (operator request).** The normalized plateau coordinate $p(t)$ rises smoothly and
nearly monotonically from 0 to 1 along *every* path — there is **no sharp plateau**. Decisively, the
**centroid spline's $p(t)$ curve is essentially identical to the linear chord's** (both ≈ 0.59 at the
midpoint), so downstream behavior progresses the same way along the fitted manifold and along the
trivial straight line. Behavior does not single out the on-manifold path. Higher-$\lambda$ /
output-only paths are only slightly flatter (midpoint ≈ 0.51–0.53), the expected mild effect of
penalizing behavior kinetic energy.

![Plateau coordinate p(t) for Tuesday→Wednesday. All paths rise smoothly 0→1 with no plateau; the centroid spline overlaps the linear chord almost exactly.](plots/s7_plateau_metric.png)

The illustrative PCA geometry of the paths (conclusions use the PCA-32 metric, not this 2-D view):

![PCA (PC1-PC2) view of the chord, output-only path, fitted spline, and λ-paths.](plots/s4_pca_geometry.png)

**Generalization to all seven adjacent pairs.** Repeating the coarse grid + output-only (linear init)
for every adjacent weekday pair gives the identical pattern:

![Recovery for all 7 adjacent pairs: optimized paths (≈ chord) far above the centroid-spline target (≈0).](plots/s6_allpairs_recovery.png)


| pair | linear chord = best optimized | centroid spline | spline dominated in both energies |
|---|---|---|---|
| Mon→Tue | 0.998 | 0.004 | yes |
| Tue→Wed | 0.961 | 0.004 | yes |
| Wed→Thu | 0.908 | 0.004 | yes |
| Thu→Fri | 0.998 | 0.004 | yes |
| Fri→Sat | 1.031 | 0.004 | yes |
| Sat→Sun | 0.945 | 0.005 | yes |
| Sun→Mon | 1.075 | 0.004 | yes |
| **mean** | **0.988** | **0.004** | **7/7** |

For every pair the best optimized path over the whole $\lambda$ grid is *exactly* the linear chord (no
$\lambda$ improves recovery), the spline target is ~235× closer to itself, and the spline is
Pareto-dominated in both energies. The negative result is robust, not a Tuesday→Wednesday artifact.
The per-pair energy trade-off makes the domination visual: for all 7 pairs the optimized-path family
(chord → λ grid → output-only) traces the lower-left frontier while every fitted spline (★) sits alone
in the dominated top-right corner.

![E_act vs E_out for all 7 pairs: optimized families on the lower-left frontier; every fitted spline (star) in the dominated top-right corner.](plots/s6_allpairs_energy_tradeoff.png)

## Conclusion

For all seven adjacent weekday pairs, a generic "move smoothly in activations and in behavior"
objective does not reconstruct the paper's fitted activation manifold. Three findings, each on its own
sufficient to reject the hypothesis:

1. **Wrong direction.** Recovery worsens monotonically as the behavior-smoothness weight $\lambda$
   increases; the straight chord is the closest optimized path and is still far from the spline.
2. **The target is the least smooth path.** The fitted spline is Pareto-dominated in *both* kinetic
   energies, so no non-negative weighting of these two terms could ever select it.
3. **The behavior objective is underdetermined.** Its broad flat minimum lets the activation path
   wander by init; downstream behavior does not uniquely fix the activation trajectory.

Interpreted through the plan's decision rule, this is the "centroid spline is dominated in both
energies" outcome: these two kinetic terms do not explain why that reference path should be preferred.
The safety-relevant implication is that the fitted weekday manifold encodes model-specific geometry
that a generic smoothness prior does not capture — so steering methods that assume "smooth = natural"
would not, by that assumption alone, stay on this manifold. A fourth, independent check reinforces
this: the normalized downstream plateau coordinate $p(t)$ progresses *identically* along the fitted
spline and the trivial straight chord, so downstream behavior cannot be what distinguishes the
manifold either.

**Limitations.** (1) The three-seed initialization study was run for the Tuesday→Wednesday pilot; the
seven-pair generalization uses the linear init only (its best-over-λ already equals the chord, so an
intermediate λ cannot help regardless of seed). (2) The primary recovery metric is the transparent
PCA-32 nearest-spline distance; the
paper's Appendix-A.9 SVD common-subspace recovery score was not reproducible from available materials.
(3) The direction's referenced $d(t)$ helper (`slerp_relative_distance.py`) is absent from the repo, so
$d(t)$ is defined here from first principles and used only as a diagnostic. (4) Shared-GPU limits force
the 16 GB bf16 model to be split GPU/CPU for collection (precision unchanged); "PCA-64" is capped at
the 48 non-degenerate components available from 49 activations (optimization/recovery unaffected).
(5) Behavior energy is normalized per initialization, so cross-seed *normalized* losses are not
directly comparable — but all recovery and raw-energy conclusions use init-independent quantities.
