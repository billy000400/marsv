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

## 2026-07-15 — shallow-net power restoration + structural exclusion (iter 6)

Closed the one open S3 item: whether the shallow net d3w200's single verified pair is sampling noise
that more endpoint pairs would fix. Parametrized `analyze(..., n_pairs=)` in
`experiments/population_manifold.py` (base + other models unchanged at the frozen 20 pairs) and re-ran
d3w200 at 20/60/120/200 endpoint pairs per region pair.

- **Result: power cannot be restored — the deficit is structural, not statistical.** The shallow net's
  d(t) *ramps* rather than plateaus: mean plateau fraction 0.25 (max 0.43) across all 46 region pairs,
  **0/46** reach the 0.5 accept threshold even on average, vs base 0.60 (43/46). Sampling 10× more
  pairs (20→200) still yields only **2** verified pairs, not 20.
- We do NOT relax the frozen d(t) filter (a ramp is not a plateau; scoring G on it would not test the
  claim). The 1–2 genuine plateau transitions it does produce are **all counterexamples** (`G ≤ 1`,
  median 0.76–0.98) — consistent with the verdict, never against it.
- RESULTS.md + REPORT.md: the d3w200 replication footnote changed from "under-powered, we do not weight
  it" → **excluded on structural grounds** (invalid plateau test bed), with a new subsection and figure.
  Population verdict and all headline numbers are UNCHANGED (base between-G 0.996, 25/45 counterexamples;
  four well-powered models 0.93–1.00).
- Figure added: `plots/population_shallow_power.png` (embedded in both deliverables). Math re-verified
  via GitHub API: 3/3 display-math, 0 pre-math, 0 inline hazards; 0 bare plot paths in either file.
- S3 and the PLAN success criterion are now fully met (both claims answered, replicated, shallow-net
  power question resolved). Writing STOP.

## 2026-07-16 — endpoint-resampling stability check (iter 7)

Re-entered with a fresh wall-clock budget and no STOP file on disk (iter 6's STOP did not persist /
was cleared by the wrapper relaunch). The one verdict-rule requirement not yet *explicitly* tested was
stability under **resampling** (all endpoint sampling had used seed 0; bootstrap CIs only resample the
seed-0 measurements). Added `experiments/resample_check.py` (re-runs the frozen pipeline on the base
model at endpoint-sampling seeds 0/1/2; seed 0 doubles as a regression check) and parametrized
`analyze(..., sample_seed=)` in `experiments/population_manifold.py`, wrapping its run-all block in a
`_main()` guard so `analyze()` is importable (same pattern as `cross_model.py`, iter 4).

- **Regression check passed:** seed 0 reproduces the published numbers exactly (between-G median
  0.996, 25/45 counterexamples, digit-9 G = 1.00).
- **Fresh draws (seeds 1, 2):** between-G median 0.977 (CI 0.95–1.02) and 0.957 (CI 0.90–1.00);
  counterexamples 25/46 and 30/46; digit-9 sub G = 0.86 and 0.82.
- **21 plateau pairs — including 9A-9B — are counterexamples (G ≤ 1) under ALL three endpoint
  draws**; 45 pairs verified in all three seeds; per-pair median G hugs y=x across draws.
- RESULTS.md + REPORT.md: added a "Resampling stability" section with the table + new embedded figure
  `plots/population_resample.png`; headline/conclusion universal-claim bullets now cite the 21
  resampling-stable counterexamples (digit-9 G = 1.00/0.86/0.82). All other numbers unchanged.
- Verified: GitHub API 3/3 display-math, 0 pre-math, 0 inline hazards; 0 bare plot paths; 6 embedded
  figures in each deliverable. New artifact: `results/resample_check.json`. Verdicts UNCHANGED and now
  explicitly resampling-stable → re-writing STOP.

## 2026-07-16 — operator feedback: both investigations + annotated direct-path figures (iter 8)

Addressed `human_feedback_0716.txt` (renamed `.addressed.md`): the operator asked to (1) keep BOTH
investigations — the manifold-component verdict AND the "low-density region real activations don't
live in" finding (retired from the deliverables in the iter-5 reframe) — (2) revive
`plots/direct_path_support.png`, and (3) state in the figure/REPORT.md which layer the interpolation
is done in, which layer `d` is measured at, and how many samples the plot uses.

- Added `experiments/direct_path_offmanifold.py`: regenerates `plots/direct_path_support.png` with
  full annotations (slerp in L1, 200-d post-ReLU; d(t) at L3; support radius r_10 at L1 vs the
  1705-point natural cloud; 200 points per path; 1 endpoint pair per panel) AND elevates the
  direct-path finding to **population level** with the frozen sampling (46 region pairs × 20 endpoint
  pairs, seed 0, d(t) accept filter): per-path off-manifold excursion E = max r_10 along path as a
  percentile of the natural baseline.
- **New population result (Investigation 2):** verified between-plateau direct paths (n=676) have
  median excursion at the **95.4th percentile** (IQR 87.8–98.4) of the natural support distribution
  and **53%** exceed the natural p95; within-plateau controls (n=200): 65.2 (38.3–86.1), 12%. The
  r_10(t) profile bulges mid-path (~1.45× natural median) exactly where the d(t) jump occurs — a
  genuine low-density corridor. Single-pair panels: 9A→9B stays at the 52nd pctile (on-manifold),
  9→0 reaches the 91st.
- RESULTS.md + REPORT.md restructured as **two complementary investigations** (components: both
  claims still fail, all numbers unchanged; corridor: real). New figures embedded in both:
  `plots/direct_path_support.png` (annotated), `plots/direct_path_population.png`. Headline/Summary/
  Conclusion updated to the combined statement: "the plateaus are connected; the straight path between
  them is not where the data lives." New artifact: `results/direct_path.json`.
- Verified: GitHub API 5/5 display-math (added r_k and E equations), 0 pre-math, 0 inline hazards;
  0 bare plot paths; 8 embedded figures per deliverable.

## 2026-07-16 — operator feedback: readability rewrite + MST explainer figure (iter 9)

Addressed `human_feedback_07161625.txt` (renamed `.addressed.md`): (1) explain the minimum spanning
tree; (2) shorten/simplify the Summary; (3) break up overloaded sentences; (4) reduce terminology with
a "How to read the plots" box; (5) cut repetition and shorten figure captions. No numbers changed —
this is a presentation rewrite of both deliverables; every result, table, and verdict is identical.

- **MST explained plainly:** REPORT.md Methods gains a stepping-stones paragraph (spanning tree →
  minimum spanning tree → why the largest MST-path edge is the smallest unavoidable hop between two
  points) plus a new toy 2-D schematic `plots/mst_explainer.png`
  (`experiments/mst_explainer_fig.py`), embedded in both deliverables.
- **Summary rewritten:** now motivation → three principal findings → one-line verdict; pair counts,
  CIs, seeds, and the shallow-net resolution moved entirely to Results.
- **Core vocabulary reduced to `d(t)` (filter), `G`, `E`** via a "How to read the plots" box in
  REPORT.md Methods (with plot color conventions); RESULTS.md metric section condensed to the same
  three quantities with a pointer to REPORT.md for the equations. Terms like "plateau fraction" are
  now described inline where used rather than treated as standalone vocabulary.
- **Captions shortened** to 1–3 lines each; axis/legend definitions moved into the adjacent prose
  (still satisfying the every-figure-readable rule). Repetition between Summary/Results/Conclusion
  removed — each finding is stated fully once in Results; Summary and Conclusion reference it briefly.
- Note: feedback point 5 quotes phrases ("gradual emergence", "late boundary movement", "after
  accuracy saturates") and line numbers that do not occur in this direction's REPORT.md — they appear
  to reference another direction's report. The general asks (dedupe, shorten captions) were applied
  here regardless.
- Verified: GitHub API 5/5 display-math in REPORT.md, 0 pre-math, 0 inline hazards; 0 bare plot
  paths; 9 embedded figures per deliverable (8 prior + `mst_explainer.png`).

## 2026-07-17 — operator feedback: is the corridor the wrongly-classified images' home? (iter 10)

Addressed `human_feedback_1.txt` (renamed `.addressed.md`): the operator noted the natural cloud uses
only correctly-classified test images and asked whether the low-density corridor corresponds to the
wrongly-classified images. Added `experiments/wrong_class_corridor.py` (base model, identical frozen
paths/seed as `direct_path_offmanifold.py`; seed-0 regression check reproduced the published E =
95.4 / 65.2 exactly). Three tests, all answering **no**:

- **(A) Where do the 295 wrong activations sit?** Median support percentile **74** (IQR 56–88) vs the
  correct cloud, 10% beyond p95 — the thin edge of the cloud (borderline examples), far short of
  corridor points (95.4, 53%).
- **(B) Does adding them fill the corridor?** E recomputed against the augmented 2000-point cloud
  (own baseline): between-plateau median **95.4 → 95.2**, controls 65.2 → 62.6 — no fill.
- **(C) Are corridor points at home among wrong activations?** Corridor points (per-path argmax r_10)
  sit at the **92nd** percentile of the wrong cloud's own self-support baseline (control paths' worst
  points: 90th) — no closer to wrong activations than ordinary travel; the along-path distance to the
  wrong cloud is flat (~1.25×), no mid-path dip.
- RESULTS.md + REPORT.md gain a "corridor is not the wrong images' home" subsection under
  Investigation 2 (plus a Methods paragraph on the three reference clouds/baselines; no new metric —
  reuses r_10/E percentile machinery); corridor bullets in Headline/Summary/Conclusion note the
  result. New artifacts: `plots/wrong_class_corridor.png` (embedded in both deliverables),
  `results/wrong_class_corridor.json`.
- Verified: GitHub API 5/5 display-math, 0 pre-math, 0 inline hazards; 0 bare plot paths; 10 embedded
  figures per deliverable. All prior numbers/verdicts unchanged.
