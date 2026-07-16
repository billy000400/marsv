# PLAN — Direction: Do plateau transitions correspond to activation-manifold transitions?

> Working folder: `dir11_boundary_as_off_manifold`. Agent REWRITES "Current status"/"Next step" + ticks stages each
> iteration. Disk (PLAN/JOURNAL/RESULTS/CHANGELOG + ../BUDGET.md + ../CLAUDE.md) is the only memory.

## Success criterion (definition of "done")

`RESULTS.md` and `REPORT.md` give a clear population-level answer to:

> When an activation path moves from one stable output plateau to another, do the two plateaus lie on different empirical components of the natural activation manifold?

The digit-9 pair already studied is one case, not the organizing question. Analyze all sufficiently
populated plateau regions in the model and all plateau pairs for which reproducible plateau-to-plateau
paths can be constructed.

The report must distinguish two claims:

1. **Universal claim:** every plateau transition implies a transition between empirical manifold
   components. A single reliable counterexample refutes this claim.
2. **Typical-association claim:** plateau pairs are usually more separated in the natural activation
   cloud than within-plateau controls. This requires a comparison across the full set of plateau pairs.

Use only two quantitative objects in the main analysis:

1. **Plateau observable — `d(t)`:** the existing normalized downstream-distance curve, used only to
   verify that a sampled path genuinely begins in one stable region and ends in another. Do not add a
   second plateauness score unless `d(t)` demonstrably cannot make this decision.
2. **Manifold observable — normalized connection bottleneck `G`:** build a minimum spanning tree (MST)
   over natural activations at the intervention layer. For two endpoints, let `B` be the largest edge on
   their MST path; this is the smallest step size needed to connect them through the sampled natural
   activation cloud. Normalize `B` by the corresponding within-plateau connection scale, so `G = 1`
   means “no larger gap than is normally required inside a plateau” and `G > 1` means an unusually large
   bridge is required. Define the exact within-plateau normalization once, before examining between-
   plateau results, and keep it fixed.

The final deliverable contains:

- one main figure comparing `G` for between-plateau pairs against within-plateau controls;
- one compact pairwise table or heatmap showing which plateau pairs do or do not exhibit a manifold gap;
- at most one small illustrative `d(t)` panel showing representative verified transitions;
- a direct verdict on both the universal and typical-association claims;
- a clearly labeled limitation that finite activation samples can support or undermine empirical
  component separation, but cannot prove true topological disconnection.

Null, mixed, and negative results are COMPLETE if the general question is answered. When done, the loop
writes an empty `STOP` file.

## Fallback (if time runs short)

On the existing base checkpoint:

1. define the class-aligned stable output regions using correctly classified, high-confidence MNIST
   examples;
2. test every sufficiently populated pair of regions rather than selecting digit 9;
3. sample at least 10 endpoint pairs per between-plateau pair and matched within-plateau controls;
4. verify the between-plateau paths with `d(t)`;
5. build one natural-activation MST and report the normalized bottleneck `G`.

Produce the main comparison figure, the pairwise summary, a preliminary general verdict, and `STOP`.
Do not spend fallback time adding metrics or running another architecture.

## Setup (fixed)

- **Base code:** build directly upon the `image-models` branch of `FranciscoHS/mars-plateaus` and reuse
  the existing `dir11` analysis wherever it is correct.
- **Development branch:** continue the existing direction branch if available; do not start a separate
  implementation merely because the scientific question has been reframed.
- **Primary model:** the existing four-layer ReLU MNIST MLP.
- **Data:** correctly classified natural MNIST examples. Apply one fixed confidence threshold to remove
  ambiguous endpoints; confidence is an inclusion rule, not a reported metric.
- **Intervention layer:** the same first hidden layer used in the existing plateau experiments.
- **Downstream measurement layer:** the same final hidden layer used for the existing `d(t)` analysis.
- **Direct paths:** use the existing spherical interpolation implementation only to verify plateau
  transitions. A direct interpolation leaving the activation cloud is not evidence that the endpoint
  plateaus are different manifold components.
- **Plateau-region definition:** determine stable regions from downstream/output behavior before looking
  at intervention-layer manifold connectivity. Start with the expected class-aligned plateaus; include
  reproducible within-class sub-plateaus if the output-side evidence supports them. Do not use the
  intervention-layer graph to define the groups that the same graph will later test.
- **Coverage:** include every stable region with at least 30 eligible natural examples and every pair
  that passes the fixed plateau-transition verification rule. Sample the same number of endpoint pairs
  per region pair (target: 20) so digit frequency does not dominate the result.
- **Controls:** use within-plateau endpoint pairs sampled from the same regions and with the same endpoint
  selection rules. These define the normal connection scale.
- **Primary manifold test:** Euclidean MST over the natural intervention-layer activation cloud. If an
  approximate kNN graph is required for memory, verify once that increasing `k` no longer changes the
  MST bottlenecks; do not turn the `k` sweep into a headline result.
- **Replication:** after the base-model analysis is complete, reuse the already-trained second seed and
  architecture checkpoints for confirmation. Do not train more models unless the existing checkpoints
  are unusable.
- **Shared limits in `../BUDGET.md`; operator rules in `../CLAUDE.md` — read both every iteration.**
- **Deliverable hygiene (see CLAUDE.md):** RESULTS.md/REPORT.md = current-best only, no history;
  CHANGELOG.md = the history.
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, flax** — they break the
  CUDA build.

### Metric and readability budget

The main report is limited to `d(t)` and `G`. A quantity does not belong in `REPORT.md` merely because
the code can compute it. Before adding anything else, write in `JOURNAL.md` what decision it changes;
if it changes no decision, omit it.

Remove the following from the main analysis because they are redundant, fragile, or do not answer the
component question:

- nearest-natural-activation distance;
- kNN radius along the direct interpolation;
- graph hop count;
- the percentile or exact location of `argmax |d'(t)|`;
- alignment between a single derivative peak and a single support peak;
- multiple clustering-quality scores;
- raw bottleneck distances when the normalized `G` is already shown;
- PCA/UMAP separation as evidence of manifold components;
- separate plots for every graph hyperparameter, seed, or architecture.

Robustness settings may be logged and summarized in one compact table. Promote a robustness result into
the main text only if it changes the verdict.

## Stages (checklist)

- [x] S1 — Reframe the existing work and freeze the minimal measurement procedure
  - Inspect the current `dir11` scripts, `RESULTS.md`, `REPORT.md`, figures, and completed digit-9 runs.
  - Record which existing outputs can be reused and which only answer the old digit-9-specific question.
  - Treat the prior digit-9 result as a pilot/candidate counterexample, not as the final population-level
    answer.
  - Define, before testing manifold connectivity:
    - the output-side plateau-region assignment;
    - the fixed `d(t)` rule for accepting a path as plateau-to-plateau;
    - the within-plateau normalization used in `G`;
    - endpoint sampling count and random seed.
  - Remove weak metrics from the main report and figures. Preserve historical results in `CHANGELOG.md`
    rather than carrying them into the current-best narrative.
  - If an old `STOP` marks the digit-9-specific question as complete, remove it when this reframed work
    begins.

- [x] S2 — Build a balanced set of plateau transitions across the model
  - Collect natural first-hidden-layer activations and downstream outputs for all eligible MNIST examples.
  - Identify all sufficiently populated stable output regions without privileging any digit.
  - Enumerate all candidate region pairs and sample the same number of endpoint pairs from each.
  - Compute `d(t)` for the sampled direct paths and retain only pairs that satisfy the predeclared
    plateau-transition rule.
  - Sample matched within-plateau controls from every included region.
  - Report coverage plainly: number of stable regions, number of possible region pairs, number passing
    the plateau-transition rule, and exclusions with reasons. These are dataset counts, not new metrics.
  - Include the previously studied digit-9 pair in this table exactly like any other pair.

- [x] S3 — Test empirical manifold-component correspondence and write the verdict
  - Build the natural-activation MST and compute `G` for all verified between-plateau pairs and matched
    within-plateau controls.
  - Search explicitly for counterexamples to the universal claim: verified plateau transitions with
    `G <= 1`, meaning that the plateaus can be connected through natural activations without a larger
    gap than normal within-plateau travel.
  - Compare the full between-plateau and within-plateau `G` distributions to assess the typical-
    association claim. Report the distribution and uncertainty without adding a collection of effect-
    size metrics.
  - Repeat the same frozen analysis on the existing independent seed and architecture checkpoints.
    Summarize replication in one table; investigate only verdict-changing failures.
  - Write `REPORT.md` around the general claim, with the digit-9 case mentioned only if it is a useful
    representative example or counterexample.
  - Save only the current-best figures to `plots/`, update `RESULTS.md` and `CHANGELOG.md`, and create
    `STOP` once both claims have a clear verdict.

## Verdict rules

- **Universal claim supported:** every well-powered verified plateau pair has `G > 1`, and this remains
  true under resampling and the existing model replications.
- **Universal claim refuted:** at least one reproducible verified plateau pair has `G <= 1` across
  resampling/model replication. Do not let a majority trend hide this counterexample.
- **Typical association supported:** the between-plateau `G` distribution is consistently shifted above
  within-plateau controls across model replications.
- **Typical association not supported:** the distributions substantially overlap or the direction is not
  stable across replications.
- **Mixed:** separation is real for some plateau pairs but depends strongly on class pair, seed, or
  architecture. State this directly instead of forcing a binary conclusion.

The threshold `G = 1` must follow from the frozen within-plateau normalization, not be tuned after seeing
between-plateau results.

## Out of scope (do NOT)

- Do not make the two digit-9 regions the primary experimental object.
- Do not infer the general answer from one same-digit pair or a few hand-selected cross-digit paths.
- Do not optimize a smooth activation path; this direction tests connectivity through natural
  activations.
- Do not equate a direct interpolation leaving the activation cloud with distinct manifold components.
- Do not claim that low density, clustering, PCA, or UMAP alone proves component separation.
- Do not revive removed metrics for visual variety.
- Do not run the LLM lambda-sweep experiments or switch datasets/models before the MNIST population-level
  result is complete.
- Do not refactor unrelated parts of the `image-models` branch.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with:

`On track? <yes/no> — <stage, % done, blocker if any>`

Also answer:

`Metric check: Did I add a reported quantity? If yes, what verdict-changing decision requires it?`

## Current status

**COMPLETE (iter 9) — all feedback addressed, STOP written.** Iter 9 addressed readability feedback
(`human_feedback_07161625.txt` → `.addressed.md`): plain-words MST explanation + toy schematic
(`plots/mst_explainer.png`), Summary shortened to motivation + three findings + verdict, core
vocabulary reduced to d(t)/G/E with a "How to read the plots" box, captions shortened, repetition
removed — all numbers/verdicts unchanged. Iter 8 addressed operator feedback
(`human_feedback_0716.txt` → `.addressed.md`): the deliverables now report **both investigations** —
(1) the manifold-component verdict (unchanged) and (2) the low-density-corridor finding
(`experiments/direct_path_offmanifold.py`): verified between-plateau direct paths reach a median
**95.4th percentile** of natural support (53% beyond p95) vs 65.2 (12%) for within-plateau controls;
`direct_path_support.png` regenerated with the requested layer/sample annotations (slerp in L1, d(t)
at L3, r_10 at L1 vs the 1705-pt cloud, 200 pts/path) and the answers stated in REPORT.md Methods. All stages done; both PLAN claims have a clear, replicated,
**resampling-stable** verdict (iter 7 re-ran the frozen pipeline at endpoint seeds 0/1/2: 21 pairs
incl. digit-9 are counterexamples under every draw; between-G median 0.957–0.996, never above the
within baseline) and the shallow-net power item is resolved. S1–S2 complete and S3 core
done (iter 5). The reframed population analysis is implemented in
`experiments/population_manifold.py` with all definitions frozen: plateau regions = 10 digit classes
(confident-correct, output margin ≥ 0.5), natural cloud = 1705 correct L1 activations, `d(t)` accept =
plateau fraction ≥ 0.5, `G = MST-bottleneck / max(within-region medians)`, 20 pairs/region-pair (seed 0).
Ran base + the four existing replication checkpoints (no retraining).

**Both PLAN claims answered (supersedes the old digit-9 "REFUTED"):**
- **Universal claim → REFUTED decisively:** 25/45 verified plateau pairs on the base model have median
  `G ≤ 1` (and 26–35/45–46 in every well-powered model); the digit-9 case is one ordinary counterexample
  (`G = 1.00`).
- **Typical-association claim → NOT SUPPORTED:** between-plateau median `G` (0.93–1.00) sits on the
  within-plateau baseline (1.00) in all four well-powered models; bootstrap CIs overlap; seed-1 direction
  reverses. Distributions overlap almost completely.

Replicated across a second seed (d4w200) and two architectures (d4w400, d5w200). The shallow d3w200 is
under-powered (only 1 pair passes the `d(t)` sharpness filter) and is down-weighted, stated honestly.

## Next step

None — direction complete and all feedback addressed (zero unaddressed feedback files; iter 9 handled
the 2026-07-16 readability review). Every verdict-rule clause is explicitly satisfied: counterexamples are
stable under **resampling** (iter 7: endpoint seeds 0/1/2 → 21 pairs incl. digit-9 have G ≤ 1 in every
draw; seed-0 regression check reproduced published numbers exactly) AND under model replication (second
seed + two architectures); shallow-net power resolved as structural (iter 6, d3w200 excluded as an
invalid plateau test bed — its d(t) ramps, 0/46 region pairs reach the accept threshold even at 10×
sampling; its 1–2 genuine plateaus are all counterexamples). Both claims answered (universal REFUTED,
typical NOT SUPPORTED). Deliverables curated to current-best and verified; no unaddressed feedback
files. STOP written.