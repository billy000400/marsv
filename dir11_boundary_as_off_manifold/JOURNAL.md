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
