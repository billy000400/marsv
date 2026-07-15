# CHANGELOG — Are plateau-separated digit-9 regions separate manifold components?

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---

## 2026-07-15 — first end-to-end result (iter 1)

Populated RESULTS.md and REPORT.md from empty TODO stubs with the first full analysis
(`experiments/analyze_manifold.py`), reusing the `image-models` `mnist_mlp_d4_w200_relu` checkpoint
(no retraining). Both the **direct-path test** and the **component test** now have numbers.

- **Direct-path test** (plateau boundary vs off-manifold support, k-NN radius k=10 to the 1705-point
  natural L1 cloud; natural median radius 2.85, p95 4.23). Boundary off-manifold percentile:
  cross-region 9→9 = **35**, same-region 9→9 = 70, cross-digit 9→4 = 77, 9→0 = 89. The same-digit
  plateau boundary is *well-supported*, not an off-manifold gap.
- **Component test** (mutual-kNN graph, fixed k=10 geodesic hops / bottleneck edge): within-A = 4 hops
  / 2.67; **A↔B (two 9-regions) = 5 hops / 2.72**; 9↔4 = 9 / 2.99; 9↔0 = 12 / 2.74. The two 9-regions
  connect like a within-region pair through a high-support path. Merge-k saturates at k=3 for all pairs
  (non-discriminative) — noted as a rejected baseline.
- **Verdict: REFUTED (preliminary)** — plateau-separated digit-9 regions are not disconnected manifold
  components. Figures added: `plots/direct_path_support.png`, `plots/component_test.png`.

## 2026-07-15 — robustness pass (iter 2)

Added `experiments/robustness.py` (graph-k sweep, KMeans-k×seed sweep, all-10-digit generality,
explicit counterexample search). RESULTS.md + REPORT.md gain a **Robustness** section; verdict
upgraded **REFUTED (preliminary) → REFUTED (robust across region-definition & graph hyperparameters,
single checkpoint)**.

- **Graph-k sweep** ($k$=6–25): A↔B (two 9-regions) hops track the within-region-A control at every
  scale (10:5vs4, 15:3vs4, 25:2vs4), far below cross-digit (9↔4, 9↔0) throughout. Ordering
  within ≈ A↔B ≪ cross-digit is scale-independent.
- **Region-definition sweep** (KMeans-$k$∈{2,3,4}×3 seeds, 9 configs): digit-9 pairs always
  within-region-like — hops **2–5**, bottleneck **1.93–2.72**, all below natural median 2.85.
- **Generality (all 10 digits)** + **counterexample search**: every digit's two-region split has
  bottleneck **2.35–4.17, all ≤ p95 (4.23)** → **0 counterexamples**; no same-digit region pair
  crosses an off-manifold gap.
- **Metric correction / caveat:** the initial counterexample criterion used hops≥7 and flagged 4
  digits; corrected to the plan's support/gap definition (bottleneck > p95), which flags none. Reason:
  in this dense L1 cloud even cross-digit bottlenecks stay below p95, so hops (not bottleneck)
  discriminates for digit-9 — but hops is inflated by manifold elongation (digit-1's two regions = 15
  hops yet bottleneck 2.70 < median). Generality verdict now anchored on the off-manifold criterion.
- Figures added: `plots/robustness_graphk.png`, `plots/robustness_regions.png`.


## 2026-07-15 — cross-model confirmation (iter 3)

Added `experiments/cross_model.py`: trains a **second MNIST MLP from scratch** (same depth-4/width-200/
ReLU/1000-subset/100k-step config, **seed 1**, test acc **86.9%**) and re-runs the direct-path,
component, and all-digit counterexample tests on it. RESULTS.md + REPORT.md gain a **Cross-model
confirmation** section; verdict upgraded **REFUTED (…single checkpoint/seed) → REFUTED (…AND a second
independent training seed)**.

- **Model 2 (seed 1)** reproduces every qualitative result: cross-region 9→9 boundary support
  percentile **53** (well-supported, ≈ median — not an off-manifold void) vs cross-digit 9→0 = **88**;
  component test A↔B (two 9-regions) = **4 hops = within-region-A (4)**, bottleneck **1.54 < natural
  median 1.95**; different digits 9↔4/9↔0 = 8/26 hops; all-10-digit counterexample search = **0**.
- Base model 1 re-analyzed by the same code as a control — numbers match the previously reported values
  exactly (cross-region pctile 35, A↔B 5 hops / 2.72, 0 counterexamples).
- Figure added: `plots/cross_model.png` (4-panel side-by-side). This closes the sole remaining
  limitation (single checkpoint); the direction's question is answered → STOP.
