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

## 2026-07-15 — architecture generalization (iter 4)

Added `experiments/arch_generalize.py`: trains **three additional MNIST MLPs from scratch** with
different shapes (**d3w200** shallower / **d4w400** wider / **d5w200** deeper, new seeds) and re-runs the
three decisive tests via the reused, architecture-agnostic `analyze()` (refactored `cross_model.py`
under a `main()`/`__main__` guard and made its paths mount-agnostic `/network`↔`/workspace` so it is
importable). RESULTS.md + REPORT.md gain an **Architecture generalization** section; verdict upgraded
**REFUTED (…second seed) → REFUTED (…second seed AND four architectures)**. Also converted the REPORT.md
figure references from bare `See plots/…` prose (which does NOT render) to embedded `![…](…)` images, and
embedded the RESULTS.md Figures list likewise (rule 12).

- **All 4 architectures → 0 same-digit counterexamples** (no same-digit region pair separated by an
  off-manifold gap, bottleneck ≤ p95). The direct cross-region 9→9 path's **max radius ≤ own p95 in every
  net** (stays on-manifold): base 2.91≤4.23, d3w200 0.71≤0.93, d4w400 4.72≤7.08, d5w200 6.50≤6.50 —
  whereas cross-digit 9→0 *exceeds* p95 in 2/4 (d3w200 1.16>0.93, d4w400 7.21>7.08), and in d3w200 the
  9↔0 pair is fully disconnected at k=10 (a real data gap) while the two 9-regions stay connected.
- **Honest caveat recorded:** the single-point `argmax|d′|` boundary *percentile* is NOT
  architecture-robust — in d5w200 it lands at a sparse path endpoint (t≈0.99) reading 94th pctile even
  though the whole path stays ≤ p95. Verdict re-anchored on the off-manifold-gap (max-radius / bottleneck
  vs p95) criterion, which is architecture-robust; boundary-percentile and raw hops flagged as fragile
  summaries. Limitation narrowed from "architecture and dataset" → **dataset only**.
- Figure added: `plots/arch_generalize.png` (direct-path support, component hops, all-digit
  counterexample search across the 4 architectures).

## 2026-07-15 — population-level reframe with the normalized bottleneck G (iter 5)

Reframed the deliverables from the digit-9 case study to the PLAN's population-level question ("do
plateau transitions correspond to activation-manifold transitions across the whole model?") using the
single frozen metric **G = MST bottleneck / within-plateau scale**. Added
`experiments/population_manifold.py` (Euclidean MST over the natural L1 cloud; per-pair bottleneck via
max-edge on the MST path; `d(t)` plateau-accept filter; 20 endpoint pairs per region pair over all 45
cross-digit pairs + the digit-9 A/B sub-plateau + 10 within-plateau controls; run on the base model plus
the four existing replication checkpoints). RESULTS.md and REPORT.md fully rewritten around G; the old
digit-9-specific hops/percentile narrative (mutual-kNN hops, argmax-|d'| boundary percentile) is retired
from the deliverables per the PLAN's metric budget (history preserved here).

- **Frozen definitions:** plateau regions = 10 digit classes (confident-correct, output margin ≥ 0.5);
  natural cloud = 1705 correct L1 activations; `d(t)` accept = plateau fraction ≥ 0.5 with start<0.2,
  end>0.8; `s_r` = median within-region MST bottleneck; `G = B / max(s_i,s_j)`; 20 pairs/region-pair,
  seed 0.
- **Base model (d4w200, seed 0):** within-plateau `G` median 1.00 / p95 1.39; **between-plateau median
  `G` = 0.996** (95% CI 0.97–1.03), range 0.84–1.67; **25/45 counterexamples** (`G ≤ 1`); digit-9 sub
  `G = 1.00`.
- **Verdicts (supersede the old digit-9 "REFUTED"):**
  - **Universal claim → REFUTED decisively** (25/45 verified pairs with `G ≤ 1`; the digit-9 case is
    one of them). Old framing had reported digit-9 as a single candidate counterexample; now shown to be
    typical.
  - **Typical-association claim → NOT SUPPORTED** (between-plateau median `G` sits on the within-plateau
    baseline in all 4 well-powered models; CIs overlap; seed-1 direction reverses to 0.925).
- **Replication:** seed 1 median `G` 0.925 (35/45 cx), d4w400 0.987 (26/46 cx), d5w200 0.982 (32/46 cx);
  d3w200 shallower under-powered (1 verified pair — few sharp plateaus) and down-weighted.
- Figures added: `plots/population_G.png`, `plots/population_heatmap.png`, `plots/population_dt.png`,
  `plots/population_replication.png` (embedded in both deliverables). REPORT math re-verified via the
  GitHub API: 3/3 display-math, 0 pre-math, 0 inline hazards; 0 bare plot paths in either file.
