# JOURNAL — Are plateau-separated digit-9 regions separate manifold components?

Append-only working log. One entry per iteration: what I did, what I learned, the revised next step,
and a final line `On track? <yes/no> — <stage, % done, blocker>`.

---

## 2026-07-15 — iter 1: full pipeline + first verdict

**Did.** Located the base `image-models` code at `/network/mars-plateaus-image` (no local
`image-models` branch; wrapper auto-commits, so I did NOT run git — I import the base repo via
sys.path instead of checking out a branch). Inspected `src/mnist.py` (MLP depth-4 w200, hidden layer
convention, `forward_from`), `src/paths.py` (slerp), `scripts/image/interpolate_digits.py`
(d(t) definition). Wrote `experiments/analyze_manifold.py` (CPU, torch threads=2) that: (1) builds the
natural L1 cloud from 1705 correctly-classified test images; (2) defines two digit-9 regions via
KMeans(k=2) on L3 activations (A=151, B=26); (3) runs 4 slerp paths (same-region, cross-region 9→9,
9→4, 9→0) computing d(t), plateau boundary=argmax smoothed|d'|, k-NN support radius; (4) mutual-kNN
graph with merge-k, fixed-k=10 geodesic hops + bottleneck edge. Wrote RESULTS.md, REPORT.md
(math verified via GitHub API: 4/4 display-math, 0 pre-math, no inline `\{` hazards), CHANGELOG.

**Learned.** (a) Direct-path claim FAILS for digit-9: cross-region 9→9 boundary sits at 35th-pctile
support (well inside the manifold), while off-manifold-ness rises with genuine class change (77/89 for
9→4/9→0). (b) Component claim REFUTED: the two 9-regions are 5 graph hops apart (~within-region's 4)
vs 9–12 for different digits, and the connecting bottleneck edge (2.72) is below the natural median
radius (2.85) → high-support path exists. (c) merge-k saturates at k=3 for ALL pairs (dense cloud) —
non-discriminative; the fixed-k hop/bottleneck contrast is the discriminative measure. Assumption
logged: chose k=10 to match within-region internal-connect scale (k=7–9); rejected merge-k as primary.

**Next.** S2/S3 robustness: repeat over multiple region pairs & seeds, sweep KMeans-k and graph-k,
and explicitly hunt the counterexample (a same-digit plateau whose regions connect only via a
low-support bottleneck). Also add region B↔B same-region control and confirm on a second digit.

On track? yes — S1+partial S2/S3 done (~55%), verdict REFUTED (preliminary); robustness sweeps pending.

---

## 2026-07-15 — iter 2: robustness pass + metric correction

**Did.** Wrote `experiments/robustness.py` (reuses base loading + mutual-kNN machinery): (1) graph-k
sweep k=6–25; (2) digit-9 region-definition sweep KMeans-k∈{2,3,4}×seed∈{0,1,2}; (3) all-10-digit
generality (each digit → two regions via KMeans-2, connect on the graph); (4) explicit counterexample
search. Produced `plots/robustness_graphk.png`, `plots/robustness_regions.png`, `results/robustness.json`.
Curated RESULTS.md + REPORT.md (added Robustness sections, upgraded verdict, math re-verified 4/4 via
GitHub API, no inline hazards). CHANGELOG appended.

**Learned.** (a) Digit-9 REFUTED is rock-solid: across all 9 region-def configs hops 2–5 / bottleneck
1.93–2.72 (all < median 2.85); across graph-k A↔B tracks the within-region control and stays far below
cross-digit at every scale. (b) **Metric correction:** my first counterexample rule (hops≥7) flagged 4
digits — but those flags were spurious. In this *dense* L1 cloud even cross-digit bottlenecks stay
below p95 (max 3.47), so no pair crosses a real gap; hops is what separates digit-9 categories, but
hops is contaminated by manifold shape — **digit-1's own two regions are 15 hops apart yet bottleneck
2.70 (< median)** because digit-1 is a long thin manifold with a tiny 8-pt outlier cluster. Switched
the counterexample criterion to the plan's support/gap definition (bottleneck > p95) → **0
counterexamples**, verdict stable. Generality claim now anchored on the off-manifold criterion; hops
kept as a secondary distance measure with the elongation caveat stated. Assumption logged: p95 as the
"off-manifold gap" threshold (rejected: hops-band overlap, which is not support-relevant).

**Next.** Optional S3 hardening: a second trained model/seed to confirm the plateau=decision-geometry
reading transfers (the one remaining real limitation), and fold the direct-path off-manifold-percentile
test into the multi-region/multi-digit sweep. Otherwise the question is answered → candidate for STOP.

On track? yes — S2/S3 done (~85%), verdict REFUTED and now robust across analysis hyperparameters;
only cross-model/seed generalization remains.

---

## 2026-07-15 — iter 3: cross-model confirmation (final S3 hardening)

**Did.** Wrote `experiments/cross_model.py` (trains on GPU under memory-fraction 0.225, analyzes on
CPU). Trained a second depth-4/width-200/ReLU MLP from scratch, seed 1, 100k AdamW/MSE steps, 1000-image
subset → test acc 86.9%, saved `results/mnist_mlp_d4_w200_relu_n1000_seed1.pt`. Re-ran all three
decisive tests (direct-path boundary percentile, component hops/bottleneck at k=10, all-10-digit
counterexample search) on both model 2 and — as a control — model 1. Produced `plots/cross_model.png`,
`results/cross_model.json`. Curated RESULTS.md (new Cross-model section + upgraded headline) and
REPORT.md (Summary + Methods + new Results subsection + Limitations); math re-verified 4/4 display,
0 pre-math, 0 inline hazards via GitHub API. CHANGELOG appended.

**Learned.** The REFUTED verdict transfers cleanly to an independently trained model. Model 2:
cross-region 9→9 plateau boundary at the **53rd** support percentile (right at its median — not a
data hole), while cross-digit 9→0 is at 88th; the two digit-9 regions connect at **exactly the
within-region hop distance (4 = 4)** through a bottleneck (1.54) below the natural median (1.95);
0 counterexamples across all ten digits. Re-running model 1 through the identical code reproduced the
originally reported numbers exactly (good regression check). One nuance: model 2's cross-region
argmax-|d'| boundary lands at t≈0.99 (near endpoint B) but its support percentile (53) is still the
decisive number and is well-supported. The plateau-is-decision-geometry reading is a property of the
phenomenon, not one checkpoint. The remaining limitation is now architecture/dataset, not seed.

**Next.** Question fully answered per PLAN success criterion (direct-path + component + connectivity +
robustness + cross-model, clear REFUTED verdict). Writing STOP.

On track? yes — S1/S2/S3 complete (~100%); verdict REFUTED, robust across analysis hyperparameters and a
second independent training seed; finalizing and STOP.
