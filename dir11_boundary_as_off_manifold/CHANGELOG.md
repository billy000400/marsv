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

