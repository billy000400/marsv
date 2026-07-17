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

---

## 2026-07-15 — iter 4: architecture generalization (final limitation closed)

**Did.** Found the base repo had moved `/network/mars-plateaus-image` → `/workspace/mars-plateaus-image`
(no `/network` mount this run), so made `cross_model.py` path-agnostic (BASE detection, `HERE` from
`__file__`) and wrapped its run/plot block in `main()` under `if __name__=='__main__':` so `analyze()` is
importable. Wrote `experiments/arch_generalize.py`: trains 3 new-architecture MLPs from scratch
(d3w200 shallower seed2 / d4w400 wider seed3 / d5w200 deeper seed4, ~127s each on GPU under mem-frac
0.225) + base d4w200 reference, re-runs direct-path / component / all-digit counterexample tests via the
reused `analyze()`. Produced `plots/arch_generalize.png`, `results/arch_generalize.json`. Curated
RESULTS.md + REPORT.md (new Architecture-generalization sections; verdict → "…AND four architectures";
narrowed limitation to dataset-only) and — while curating — fixed a standing rule-12 violation: REPORT.md
figures were bare `See plots/…` prose (don't render); converted all 6 refs in both files to embedded
`![…](…)` images. Math re-verified via GitHub API: REPORT 4/4 display-math, 0 pre-math, 0 inline
hazards; both files 6/6 plot paths embedded, 0 bare.

**Learned.** The REFUTED verdict is **architecture-robust on its decisive test**: 0 same-digit
counterexamples in all 4 architectures, and the direct cross-region 9→9 path's max radius stays ≤ that
model's own p95 in every net (never leaves the manifold), while cross-digit 9→0 exceeds p95 in 2/4 and is
even fully *disconnected* at k=10 in the shallow net — exactly the same-vs-different asymmetry the verdict
predicts. **Important honesty finding:** the two *summary* metrics I'd been quoting are architecture-
fragile — (a) the single-point `argmax|d′|` boundary percentile lands at a sparse path *endpoint* in the
deeper net (t≈0.99 → 94th pctile) despite the whole path being ≤ p95, and (b) hop ordering scrambles under
manifold elongation (d4w400: within-region A = 9 hops > A↔B = 5). So I re-anchored the verdict explicitly
on the off-manifold-gap (max-radius / bottleneck vs p95) criterion and down-weighted boundary-percentile
and raw hops to "fragile summaries" in both deliverables. Assumption logged: chose max-radius-vs-own-p95
as the per-architecture direct-path statistic (rejected: cross-architecture percentile comparison, which
is not commensurable since each net has a different radius scale).

**Next.** Question fully answered per PLAN success criterion and now robust across analysis
hyperparameters, a second seed, AND four architectures; only the shared 1000-image MNIST *dataset*
remains fixed (a genuinely new dataset would be a new direction). No unaddressed feedback files. Writing
STOP.

On track? yes — S1/S2/S3 complete (~100%); verdict REFUTED, robust across analysis hyperparameters, a
second training seed, and four architectures; finalizing and STOP.

---

## 2026-07-15 — iter 5: population-level reframe + frozen G metric (S1→S3 core)

**Did.** The PLAN was reframed (digit-9 = one case, not the organizing question) around a single frozen
manifold metric **G = MST bottleneck / within-plateau scale**. Implemented it end-to-end in
`experiments/population_manifold.py`: (1) natural L1 cloud = 1705 correct test activations; (2) plateau
regions = 10 digit classes, endpoints filtered by output margin (top1−top2 logit) ≥ 0.5 — softmax is
useless here since the net is MSE-to-one-hot (max softmax < 0.5 for everything), so I froze margin as
the confidence rule (≥130/class retained); (3) Euclidean MST over the cloud, bottleneck B(u,v) = max
edge on the unique MST path via memoized DFS; (4) `d(t)` accept filter (plateau fraction ≥ 0.5,
start<0.2, end>0.8) to verify genuine plateau-to-plateau paths; (5) `s_r` = median within-region
bottleneck (frozen before between-results), `G = B/max(s_i,s_j)`; (6) 20 pairs per pair over all 45
cross-digit pairs + digit-9 A/B sub-plateau + 10 within controls; (7) ran base + the 4 existing
replication checkpoints (reused, no retraining). Rewrote RESULTS.md + REPORT.md around G, embedded 4 new
figures, retired the old hops/percentile metrics from the deliverables (kept in CHANGELOG). Math
verified via GitHub API (3/3 display, 0 pre-math, 0 inline hazards; 0 bare plot paths).

**Learned.** Both PLAN claims **fail**, and this is a cleaner, stronger statement than the old digit-9
"REFUTED". (a) **Universal claim REFUTED decisively:** base model has **25/45 verified plateau pairs
with median G ≤ 1** — they connect through natural activations with no larger gap than normal
within-plateau travel; the original digit-9 sub-plateau is just one of them (G = 1.00). (b)
**Typical-association NOT SUPPORTED:** between-plateau median G = 0.996 sits exactly on the within-plateau
baseline (1.00), bootstrap CIs overlap (0.97–1.03 vs 0.99–1.02), 44% of pairs above 1 / 56% below — no
shift. Replication is consistent: seed1 0.925 (direction even reverses), d4w400 0.987, d5w200 0.982; the
only elevated pairs involve digit-1 (elongated thin manifold → small internal scale s_1 inflates the
ratio), max G = 1.67, barely above the within-plateau p95 (1.39). **Honest coverage gap:** d3w200
shallower yields only 1 verified pair — its downstream distance ramps rather than plateaus, so the d(t)
filter rejects almost everything; I down-weighted it and said so. Assumptions logged: margin≥0.5 as the
confidence rule (rejected softmax — degenerate under MSE training); G normalized by max(s_i,s_j)
(rejected a single global scale — less faithful to "no larger than normal *inside* a plateau"); anchored
verdict on the G *distribution* vs the control band rather than any single pair (rejected per-pair
hard-threshold counting as the headline, since the ratio is anisotropy-sensitive).

**Next.** The population question is answered per the success criterion (both claims have a clear
verdict, replicated across a second seed and two more architectures). Remaining nice-to-haves, not
blockers: (i) restore statistical power for the shallow net (relax/adapt the d(t) sharpness filter or
sample more pairs) so all 5 models are well-powered; (ii) a k-sensitivity note that MST bottleneck is
parameter-free (no k), which the PLAN flags as a plus over the retired kNN graph. If next iter confirms
(i) doesn't change the verdict, write STOP. No unaddressed feedback files.

Metric check: I added ONE reported quantity (G) and removed several (hops, boundary percentile,
merge-k, nearest-natural distance). G is required because it is the PLAN's designated manifold observable
and directly decides both the universal (G≤1 counterexample) and typical-association (distribution shift)
verdicts. d(t) remains a filter, not a reported score.

On track? yes — S1/S2/S3 core complete (~90%), both claims answered (universal REFUTED, typical NOT
SUPPORTED) and replicated across a second seed + two architectures; only shallow-net power + final STOP
remain.

---

## 2026-07-15 — iter 6: shallow-net power resolved (structural), verdict unchanged → STOP

**Did.** Addressed the sole remaining S3 next-step: restore statistical power for the shallow net
d3w200 (1 verified pair at 20 endpoint pairs) and confirm the verdict holds. Parametrized
`analyze(..., n_pairs=)` (3-line change; base + other 4 models stay at the frozen 20 pairs) and re-ran
d3w200 at n_pairs = 20/60/120/200. Added `plots/population_shallow_power.png`, recorded a
`shallow_power` block in `results/population.json`, and curated both deliverables (footnote + new
subsection + embedded figure). Verified: GitHub API 3/3 display-math, 0 pre-math, 0 inline hazards; both
files 0 bare plot paths, all 4 figures embedded.

**Learned.** Sampling MORE pairs does **not** restore power — the deficit is **structural, not
statistical**. The shallow net's d(t) *ramps* rather than plateaus: mean plateau fraction 0.25 (max
0.43) across all 46 region pairs, **0/46** reach the 0.5 accept threshold even on average, vs base 0.60
(43/46 above 0.5). At 10× sampling (200 pairs/region-pair) it still has only 2 verified pairs. I
deliberately did NOT relax the frozen d(t) filter — a ramp is not a plateau, and scoring G on non-plateau
paths would not answer the question. The 1–2 genuine plateaus it does produce are **all counterexamples**
(G ≤ 1, median 0.76–0.98), so its sparse evidence agrees with the four well-powered models. Correct
resolution: exclude the shallow net as an invalid plateau test bed (stronger, more honest than the old
"under-powered, down-weighted"), verdict UNCHANGED. Assumption logged: chose structural exclusion +
report the plateau-fraction diagnostic (rejected relaxing PLATEAU_FRAC, which would admit ramps and
answer a different question; rejected forcing 20 pairs by sampling, which empirically fails).

**Next.** PLAN success criterion fully met: both claims have a clear verdict (universal REFUTED, typical
NOT SUPPORTED), replicated across a second seed and two more architectures, and the shallow-net power
question is resolved. No unaddressed human_feedback*/REVIEW files. Writing STOP.

Metric check: I added NO new reported metric. `plateau_frac` is the existing d(t) inclusion filter's
internal quantity, surfaced once as a diagnostic — it changes the decision (shallow net moves from
"down-weighted for noise" to "excluded on structural grounds" and shows more sampling cannot help). G
remains the sole reported score; d(t) remains a filter.

On track? yes — S1/S2/S3 complete (~100%); both claims answered and replicated, shallow-net power
question resolved as structural, deliverables curated + verified. Writing STOP.

---

## 2026-07-16 — iter 7: endpoint-resampling stability (last verdict-rule box ticked) → STOP

**Did.** Re-entered to find NO STOP file on disk despite iter 6 ending with "Writing STOP" (fresh
240-min wrapper session; either the file never persisted or the relaunch cleared it — no unaddressed
feedback files in this direction, though; checked). Rather than blindly re-writing STOP, used the
fresh budget for the one verdict-rule clause never explicitly tested: "stable under resampling" —
every prior run drew endpoint pairs with seed 0 only, and the bootstrap CI resamples seed-0
measurements rather than drawing fresh endpoints. Parametrized `analyze(..., sample_seed=)` and
wrapped `population_manifold.py`'s run-all block in a `_main()` guard (import-safe, same pattern as
`cross_model.py`); wrote `experiments/resample_check.py` running the frozen pipeline on the base model
at endpoint seeds 0/1/2 (19 s total). Added the table + `plots/population_resample.png` to both
deliverables as a "Resampling stability" section, updated the universal-claim bullets, re-verified
rendering (3/3 display math, 0 pre-math, 0 inline hazards, 0 bare plot paths, 6 embedded figures per
file). CHANGELOG appended.

**Learned.** The verdict is resampling-stable, decisively: seed 0 re-run reproduced published numbers
exactly (0.996, 25/45, digit-9 G=1.00 — good regression check of the guard refactor too); fresh seeds
give between-G medians 0.977 / 0.957 (never above the within baseline — seed 2 even sits below with CI
0.90–1.00), counterexample counts 25/46 and 30/46, and **21 pairs including the digit-9 sub-plateau
are G ≤ 1 under all three draws** (digit-9: 1.00/0.86/0.82). The universal-claim refutation no longer
rests on any single endpoint draw. Assumption logged: 3 seeds × the frozen 20 pairs/region-pair is
enough to demonstrate draw-stability given the per-pair scatter hugs y=x (rejected: a large seed sweep,
which would burn budget without any plausible verdict change; rejected: increasing n_pairs, which
tests a different axis already covered by bootstrap CIs).

**Next.** Nothing — every PLAN verdict-rule clause is now explicitly satisfied (counterexamples stable
under resampling AND replication; typical-association overlap stable across models and draws). No
unaddressed human_feedback*/REVIEW files. Writing STOP (again).

Metric check: I added NO new reported metric — the resampling table reports the same G statistics
under fresh draws; it exists because the PLAN's universal-claim verdict rule explicitly requires
resampling stability ("stable under resampling / model replication"), which no prior result tested.

On track? yes — S1/S2/S3 complete (100%); both claims answered (universal REFUTED, typical NOT
SUPPORTED), now explicitly stable under endpoint resampling and replicated across seeds/architectures.
STOP written.

---

## 2026-07-16 — iter 8: operator feedback — both investigations, annotated direct-path figure → STOP

**Did.** FEEDBACK FIRST: found unaddressed `human_feedback_0716.txt` (no STOP on disk). It asked to
(1) report BOTH investigations — the component verdict AND the "low-density region real activations
don't live in" finding that the iter-5 reframe had retired; (2) revive the iter-1
`direct_path_support.png` the operator liked; (3) state the interpolation layer, the layer where d is
measured, and the sample counts, in the figure or REPORT.md. Wrote
`experiments/direct_path_offmanifold.py` (reuses the frozen population conventions by importing
slerp/MARGIN/N_PAIRS/PLATEAU_FRAC from `population_manifold.py`): regenerated the 4-panel digit-9
figure with every requested annotation written on it (slerp in L1 200-d post-ReLU; d(t) at L3; support
r_10 at L1 vs the 1705-pt natural cloud; 200 pts/path; 1 endpoint pair per panel = region medoids),
and — so the corridor claim isn't anecdotal — ran the direct-path test over the whole frozen
population (46 region pairs × 20 endpoint pairs, seed 0, d(t) accept filter): per-path excursion
E = percentile of max r_10 vs the natural baseline. New `plots/direct_path_population.png`,
`results/direct_path.json`. Restructured RESULTS.md + REPORT.md as two complementary investigations
(all Investigation-1 numbers unchanged), embedded both figures in both files, answered the layer/
sample questions verbatim in REPORT.md Methods AND in the figure captions/annotations. Renamed the
feedback file `.addressed.md`. Verified rendering: 5/5 display math (added r_k and E equations),
0 pre-math, 0 inline hazards, 0 bare plot paths, 8 embedded figures per file.

**Learned.** The operator's instinct was right — the corridor finding is strong at population level,
not just in the pilot panels: verified between-plateau direct paths have median excursion at the
**95.4th pctile** of natural support (53% beyond p95) vs 65.2 (12%) for within-plateau controls, and
the r_10(t) profile bulges mid-path exactly where d(t) jumps. It also *sharpens* the headline rather
than contradicting it: the digit-9 sub-plateau pair is connected (G=1.00) AND its direct path never
leaves the support (52nd pctile) — a plateau can arise purely from decision geometry on-manifold —
while typical cross-digit transitions are connected through the data (G≈1) yet their straight route
crosses near-empty space. Combined one-liner now in both deliverables: "the plateaus are connected;
the straight path between them is not where the data lives." Assumption logged: interpreted the
feedback's "which layer you are measuring the d" as covering both the d(t) curve (L3) and the support
distance (L1) and answered both explicitly (rejected: answering only d(t), which would leave the red
curve's layer ambiguous). Used max-r_10-percentile as the excursion statistic per the iter-4 lesson
(single-point boundary percentile is architecture-fragile; max-along-path is robust).

**Next.** Feedback fully addressed (file renamed); plan remains complete and both investigations now
live in the deliverables. Zero unaddressed feedback files → writing STOP.

Metric check: I added ONE reported quantity (off-manifold excursion E, with its r_k support radius)
because the operator feedback explicitly requested this investigation be included; it decides the new
Investigation-2 verdict (corridor real vs not). It is anchored to the same frozen sampling and d(t)
filter as G, and the fragile alternatives the PLAN retired (boundary-point percentile, hop counts)
stay retired.

On track? yes — plan complete (100%), feedback addressed, both investigations reported and verified;
STOP written.

---

## 2026-07-16 — iter 9: readability feedback (MST explainer, simpler Summary, shorter captions) → STOP

**Did.** FEEDBACK FIRST: found unaddressed `human_feedback_07161625.txt` (and no STOP on disk — the
wrapper relaunch cleared it again). Five asks, all presentation-level: (1) explain the minimum
spanning tree; (2) shorten/simplify the Summary to motivation + three findings + verdict; (3) one
claim per sentence; (4) reduce terminology, add a "How to read the plots" box distinguishing the
primary metric from secondary controls; (5) remove repetition and shorten captions. Rewrote REPORT.md
and RESULTS.md accordingly (all numbers/tables/verdicts unchanged): new plain-words MST paragraph
(stepping stones → cheapest skeleton → largest MST-path edge = smallest unavoidable hop) plus a toy
2-D schematic `plots/mst_explainer.png` (new `experiments/mst_explainer_fig.py`, deterministic,
scipy MST, headless Agg); Summary cut to ~1/2 length with all counts/CIs moved to Results; core
vocabulary reduced to d(t)/G/E in a quoted "How to read the plots" box (colors + reference lines
included); captions cut to 1–3 lines with axis/legend definitions moved to adjacent prose (rule-12
readability preserved); Conclusion de-duplicated to brief references. Renamed the feedback file
`.addressed.md`, appended CHANGELOG.

**Learned.** Two of feedback point 5's quoted phrases and all four quoted line numbers ("gradual
emergence", "late boundary movement", "after accuracy saturates"; lines 190/194/202/219) do not occur
in this direction's REPORT.md — they look like they reference dir12's training-dynamics report.
Assumption logged: applied the *general* principles of point 5 (dedupe findings, shorten captions)
here rather than hunting phrases that don't exist; rejected ignoring point 5 entirely (the captions
here genuinely were 3–6 lines and repetitive). RESULTS.md now carries 0 display equations by design —
definitions live once in REPORT.md Methods (rule 8 applies to REPORT.md; duplication was part of the
terminology overload the operator flagged). Verified rendering: REPORT 5/5 display math, 0 pre-math,
0 inline hazards, 0 bare plot paths, 9 embedded figures per deliverable.

**Next.** Feedback fully addressed (file renamed); plan complete; zero unaddressed feedback files →
writing STOP.

Metric check: I added NO reported quantity. The only new artifact is an explanatory schematic
(`mst_explainer.png`, toy 2-D, labeled "schematic, not data") requested implicitly by "explain MST";
it reports no numbers and changes no verdict.

On track? yes — plan complete (100%), readability feedback addressed, deliverables re-verified;
STOP written.

---

## 2026-07-17 — iter 10: operator feedback — corridor vs wrongly-classified images → STOP

**Did.** FEEDBACK FIRST: found unaddressed `human_feedback_1.txt` (and no STOP on disk — cleared by
the wrapper relaunch again): "the natural activation cloud is only correctly classified test images —
does the low-density corridor correspond to the wrongly classified images?" Wrote
`experiments/wrong_class_corridor.py` (imports the frozen slerp/MARGIN/N_PAIRS/PLATEAU_FRAC from
`population_manifold.py`, replicates `direct_path_offmanifold.py`'s exact seed-0 sampling so the
paths are identical; seed-0 regression check reproduced E = 95.4/65.2 to the digit). Three tests on
the base model: (A) the 295 wrong-image L1 activations' own r_10 support vs the correct cloud; (B)
per-path excursion E recomputed against the augmented 2000-pt cloud with its own baseline; (C) each
verified path's corridor point (argmax r_10 vs correct cloud) scored against the wrong cloud's own
295-pt self-baseline. New figure `plots/wrong_class_corridor.png` + `results/wrong_class_corridor.json`;
new subsection in both deliverables + Methods paragraph (reference clouds/baselines); renamed the
feedback file `.addressed.md`; CHANGELOG appended; rendering re-verified (5/5 display math, 0
pre-math, 0 hazards, 0 bare paths, 10 embedded figures per file).

**Learned.** Clean NO on all three axes: wrong activations sit at the moderately-thin EDGE of the
correct cloud (median 74th pctile, 10% > p95 — sensible for borderline examples) but nowhere near
corridor territory (95.4, 53%); adding them to the cloud leaves the corridor untouched (E 95.4→95.2
between, 65.2→62.6 within); and corridor points are strangers to the wrong cloud too (92nd pctile of
its own baseline, vs 90th for control paths' worst points — no differential proximity). The along-path
distance-to-wrong-cloud profile is flat (~1.25x its median) with NO mid-path dip, while
distance-to-correct bulges to ~1.45x. Interpretation: wrong images are still real images, so their
activations hug the data manifold; the corridor is populated by no image at all. First figure draft
titled panel B "the bulge is there for both clouds" — corrected after looking at it: the wrong-cloud
profile is flat, not bulging; the decisive feature is the ABSENCE of a dip. Assumption logged: scored
proximity-to-wrong-cloud against the wrong cloud's own self-baseline (fair despite ~6x sparsity;
rejected raw distances, which would conflate sparsity with distance; rejected pooling wrong+correct
percentiles, which hides the question). Kept it base-model-only — the corridor result being probed is
itself a base-model population result; replication of a negative control across checkpoints would be
scope creep.

**Next.** Feedback fully addressed (file renamed); plan complete; zero unaddressed feedback files →
writing STOP.

Metric check: I added NO new reported metric — tests A–C reuse the existing r_10/E percentile
machinery against three reference clouds; the three numbers exist solely to answer the operator's
question (decision: corridor = wrong-image home, yes/no → NO) and live in one subsection.

On track? yes — plan complete (100%), operator feedback addressed with a clear negative answer,
deliverables curated + re-verified; STOP written.

---

## 2026-07-17 — iter 11: operator feedback — why max(s_i, s_j)? → STOP

**Did.** FEEDBACK FIRST: found unaddressed `human_feedback_2.txt` (no STOP on disk — cleared by the
wrapper relaunch as usual): "motivation of why including a max(si, sj), why max? is it a true
reflection at the boundary?" Wrote `experiments/normalization_check.py`: replicates the frozen
seed-0 sampling stream of `population_manifold.analyze` byte-for-byte (same rng call order: within
scales for regions 0–9, then the 45 cross-digit pairs, then 9A/9B sub-scales and pair) while
recording each pair's RAW bottleneck B and — new — the identity of the actual bottleneck EDGE via an
edge-tracking DFS over the MST. Recomputed all verdict quantities under four denominators
(max/min/mean/global) and added a normalization-free boundary diagnostic (is the biggest forced hop a
cross-class edge?). Regression check passed exactly (max variant: 0.996, 25/45, digit-9 1.00). Added a
"Why max?" Methods paragraph + a "Is max the right yardstick?" Results subsection + figure
(`plots/normalization_check.png`) to REPORT.md, a compact mirror subsection to RESULTS.md, updated the
Limitations pointer, renamed the feedback file `.addressed.md`, appended CHANGELOG. Re-verified
rendering (5/5 display math, 0 pre-math, 0 inline hazards, 0 bare paths, 11 embedded figures/file).

**Learned.** The operator's question hits a genuine sensitivity, and answering it honestly made the
verdict stronger, not weaker. (a) Under min/mean/global the between-plateau median G rises to
1.274/1.117/1.093 with CIs above 1 — so the typical-association direction DOES flip under alternative
normalizations; hiding that would have been spin. (b) But the movement is fully explained by one
normalization-free fact: per pair, B/max(s_i,s_j) has quartiles 0.945–1.059 — the between-pair
bottleneck almost exactly EQUALS the sparser region's internal scale, so dividing by anything smaller
mechanically exceeds 1 whenever densities differ. min-G correlates 0.68 with scale asymmetry (max-G:
−0.09) and its 14 elevated pairs all involve the three densest digit clouds (1, 8, 0) — it scores
density mismatch, not the boundary. (c) Best new evidence: the bottleneck edge is a genuine boundary
crossing on only 27% of 663 verified journeys (73% inside one digit's cloud; controls 4%) — "is it a
true reflection at the boundary?" answers itself: the hardest hop usually isn't at the boundary, which
is exactly why no denominator choice can conjure a wall. Assumption logged: kept the frozen max as
primary (it implements the PLAN's "no larger gap than is normally required inside a plateau" and was
frozen before between results; rejected switching to mean/global post hoc — that would be tuning the
threshold after seeing results, explicitly forbidden by the PLAN's verdict rules; rejected reporting
max only — the sensitivity is real and is now stated in both deliverables).

**Next.** Feedback fully addressed (file renamed); plan complete; zero unaddressed feedback files →
writing STOP.

Metric check: I added NO new reported score — the subsection re-reports the SAME G under alternative
denominators plus one diagnostic fraction (bottleneck-edge-at-boundary %). It exists solely to answer
the operator's question (decision: does the verdict depend on the max choice? — no, and the frozen
choice is now motivated); G with max(s_i,s_j) remains the sole primary metric.

On track? yes — plan complete (100%), operator feedback addressed with motivation + stress test,
verdicts unchanged and now normalization-audited; STOP written.
