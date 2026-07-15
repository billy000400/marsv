# PLAN — Direction: How plateau/stable regions evolve during training in the MNIST MLP

> Working folder: `plateau_during_training`. Agent REWRITES "Current status"/"Next step" + ticks stages each
> iteration. Disk (PLAN/JOURNAL/RESULTS/CHANGELOG + ../BUDGET.md + ../CLAUDE.md) is the only memory.

## Priority and decision framing

This is a **bounded validation study**, not the highest-upside direction. The expected result is already strong:
as training progresses, approximately ten stable regions should emerge, one for each confident model prediction,
while low-confidence points remain outside those stable regions. This is qualitatively consistent with the robust
partition/region-migration picture in *Deep Networks Always Grok and Here is Why*.

The experiment is worth doing only if it cheaply establishes **when** the regions appear and whether their
evolution contains a nontrivial split, merge, lag, or transient extra region. If the expected monotonic picture is
confirmed, finish the report and stop; do not expand this into a large sweep.

## Research question

How do plateau-bearing stable regions evolve over training in the existing 4-layer ReLU MNIST MLP?

Specifically:

1. Does plateau strength increase with prediction confidence rather than correctness?
2. Does the number of validated stable regions converge to approximately ten, one per predicted digit?
3. Do stable regions arise monotonically, or do persistent splits/merges/transient extra regions occur?
4. Does plateau emergence coincide with training interpolation, test confidence/generalization, or the later
   robustness/region-migration phase described in the grokking paper?

## Expected result and surprise criteria

### Expected / low-surprise outcome

- Plateau strength grows during training as confidence grows.
- Confident-correct and confident-wrong examples both show plateaus; uncertain examples do not.
- Validated stable-region count approaches approximately ten and the regions are organized by **predicted**
  class, not true class.
- Any early extra clusters are small, low-confidence, unstable under resampling, or fail the plateau test.

This outcome is a complete result. Report it clearly and stop.

### Result worth escalating

Escalate only if at least one of the following is replicated across at least two seeds:

- a predicted class contains two or more plateau-validated regions that persist for at least two adjacent
  checkpoints;
- plateau-validated regions split or merge non-monotonically rather than simply becoming more pronounced;
- strong plateaus appear in low-confidence examples, or plateau strength tracks correctness after controlling
  for confidence;
- plateau emergence substantially precedes or lags confidence/generalization/robustness rather than moving with
  them;
- plateau strength weakens late in training while confidence continues to rise.

Transient agglomerative-clustering artifacts alone are **not** a surprise. They must also pass the plateau test.

## Success criterion (definition of "done")

`RESULTS.md` and `REPORT.md` give a reproducible, current-best answer to the four research questions above.
Completion requires:

- one primary run plus two confirmation seeds using the existing MNIST MLP setup;
- a fixed checkpoint sweep from initialization to 100,000 optimization steps, with denser coverage near any
  observed transition;
- training/test loss, accuracy, confidence, and (if already available in the branch) adversarial accuracy at each
  checkpoint;
- plateau response curves and a scalar plateau-contrast trajectory, broken down by confidence and correctness;
- a plateau-validated stable-region count and cluster composition at each checkpoint;
- a compact split/merge analysis using the same held-out examples across checkpoints;
- figures saved under `plots/`, raw summary tables saved under `results/`, and every reported metric defined in
  `REPORT.md` Methods;
- a direct verdict: **expected monotonic emergence**, **replicated surprise**, or **inconclusive**, with limitations.

Null/expected results are COMPLETE. When the answer is documented, write an empty `STOP` file.

## Fallback (if time runs short)

Use one seed and six checkpoints: initialization, early training, first near-zero training error, an intermediate
post-interpolation point, late training, and 100,000 steps. Reuse the existing final-checkpoint measurement
protocol; report class/confidence-conditioned response curves, plateau contrast, and the number of predicted
classes with a validated plateau. Skip full cluster-lineage analysis. Produce `REPORT.md`, state that the result is
single-seed, and write `STOP`. The wrapper reserves the last 20 minutes to finalize + STOP.

## Setup (fixed)

- Build on the existing `image-models` branch and reuse its current training and plateau-analysis code rather
  than creating a parallel implementation.
- Model: 4-layer ReLU MLP, hidden width 200.
- Data: the same fixed 1,000-sample MNIST training subset and the same train/test split used in the existing
  reproduction.
- Training: batch size 200, 100,000 optimization steps, and the exact optimizer, learning rate, weight decay,
  initialization, preprocessing, and evaluation conventions already used by the branch. Record all values in
  `REPORT.md`; do not silently substitute defaults.
- Seeds: one primary seed and two confirmation seeds. Keep the dataset subset fixed across seeds unless the
  existing experiment defines the subset from the seed; document whichever convention is already in use.
- Intervention point: first hidden-layer post-ReLU activation.
- Primary downstream measurement: L2 displacement at the last hidden-layer activation, matching the current
  experiment. Logit-space displacement is a secondary robustness check.
- Evaluation set: choose one fixed, class-balanced held-out set before the sweep and preserve example IDs across
  all checkpoints. Do not filter to correctly classified examples before the primary analysis.
- Use the same perturbation directions for all checkpoints within a seed and the same radius grid across seeds.
- Include norm-and-sparsity-matched random first-layer activations as the negative control at every checkpoint.
- **Shared limits in `../BUDGET.md`; operator rules in `../CLAUDE.md` — read both every iteration.**
- **Deliverable hygiene (see CLAUDE.md):** RESULTS.md/REPORT.md = current-best only, no history; CHANGELOG.md
  = the history.
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, flax** — they break the CUDA
  build.

## Operational definitions (lock before running the sweep)

Let `h1(x)` be the first hidden-layer post-ReLU activation and let `G_t` be the checkpoint-`t` mapping from
`h1` to the last hidden layer. For relative perturbation radius `rho` and random unit direction `u`, use the
cross-checkpoint response

`R_t(x, rho) = median_u ||G_t(h1 + rho * ||h1|| * u) - G_t(h1)||_2 / (||G_t(h1)||_2 + eps)`.

Also retain the branch's existing absolute-distance plot so the final checkpoint can be compared directly with
the current result.

Before examining intermediate checkpoints, freeze:

- the perturbation-radius grid and number of directions;
- the small-radius interval used for scalar summaries;
- confidence bins (use absolute probability bins, not per-checkpoint quantiles);
- clustering preprocessing, distance metrics, candidate cluster counts, minimum cluster size, and bootstrap
  procedure.

Primary scalar metric:

`plateau_contrast = 1 - AUC(response_data) / AUC(response_matched_random)`

over the frozen small-radius interval. Larger positive values mean stronger suppression near natural activations
relative to the matched-random control. Report bootstrap confidence intervals over examples. A descriptive knee
or breakpoint radius may also be reported, but return `NA` when a two-segment fit is not better supported than a
single smooth curve; do not force a knee into every checkpoint.

Candidate regions are obtained by average-linkage agglomerative clustering of last-hidden-layer activations on
the fixed evaluation set. Check both cosine and Euclidean distance, as in the existing analysis. A cluster counts
as a **validated stable region** only if it:

1. contains at least 20 evaluation examples;
2. has at least 90% purity in the model's predicted label; and
3. has a positive plateau contrast whose 95% bootstrap interval excludes zero.

Select the candidate cluster count without labels (for example, silhouette over `k=2..15`) and report sensitivity
to the two distance metrics. Use true labels, predicted labels, correctness, and confidence only to interpret the
result after clustering.

## Stages (checklist)

- [x] **S1 — Audit and reproduce the endpoint.** Read `../BUDGET.md` and `../CLAUDE.md`; map the existing
  training, checkpoint, activation-hook, perturbation, clustering, and plotting code. Record exact conventions in
  `JOURNAL.md`. Reproduce the current 100,000-step findings: stable class clusters show plateaus, the uncertain
  mixed cluster does not, and confident-wrong examples behave like members of the predicted stable region. Do
  not launch the checkpoint sweep until this endpoint agrees qualitatively with the existing figures.
- [x] **S2 — Lock protocol and generate checkpoints.** Freeze the evaluation examples, perturbation directions,
  radius grid, metric definitions, confidence bins, and clustering choices. Train the primary seed and save at
  steps `0, 10, 30, 100, 300, 1k, 3k, 10k, 20k, 30k, 50k, 75k, 100k`; add event-aligned checkpoints around
  the first near-zero training error or any sharp change in confidence/robustness. Then run two confirmation
  seeds with the same schedule. Save lightweight state dicts and a manifest; do not commit bulky redundant
  artifacts.
- [x] **S3 — Measure plateau emergence.** At every checkpoint compute response curves for all held-out examples,
  the matched-random control, and absolute confidence/correctness groups: confident-correct, uncertain-correct,
  confident-wrong, and uncertain-wrong. Save raw per-example/per-radius summaries and plot (a) representative
  early/middle/late response curves and (b) plateau contrast versus optimization step with seed uncertainty.
  Overlay loss, accuracy, mean confidence, and adversarial accuracy only if it is already available cheaply.
- [ ] **S4 — Count and track stable regions.** Cluster last-hidden activations independently at each checkpoint;
  validate every candidate cluster with its prediction purity and plateau contrast. Plot validated region count,
  cluster confidence, and prediction composition through training. Because the same examples are reused, align
  adjacent checkpoints by maximum membership overlap and produce a compact transition heatmap/table marking
  births, deaths, splits, and merges. Any claimed extra region must survive bootstrap/distance-metric checks and
  appear in at least two adjacent checkpoints.
- [ ] **S5 — Verdict, cleanup, and stop.** Compare the trajectories with the preregistered expected result and
  surprise criteria. `REPORT.md` must distinguish plateau/stable regions from the paper's spline/linear regions:
  similar timing is evidence of association, not identity. Keep only current-best figures in `RESULTS.md` and
  `REPORT.md`, move iteration history to `CHANGELOG.md`, document limitations, and write empty `STOP`.

## Required figures

1. `plots/training_dynamics.*` — loss, accuracy, confidence, and optional adversarial accuracy versus log step,
   with key checkpoint events marked.
2. `plots/plateau_curves_by_stage.*` — matched early/middle/late response curves for confidence/correctness
   groups plus matched-random controls.
3. `plots/plateau_contrast_and_region_count.*` — plateau contrast and validated stable-region count versus step,
   with three-seed uncertainty.
4. `plots/region_composition_and_lineage.*` — predicted-label/confidence composition and the minimal split/merge
   view needed to support the verdict.

Each figure must be generated by a named script, saved to `plots/`, cited from `REPORT.md`, and defined in the
Methods/caption. Avoid grids of dozens of unreadable checkpoint panels.

## Interpretation guardrails

- The paper's spline/linear regions tile input space according to ReLU activation patterns. Our stable regions
  are empirical basins defined by downstream insensitivity plus cluster coherence. Do not call them the same
  object.
- A cluster is not automatically a plateau, and a plateau is not automatically a disconnected manifold
  component.
- Do not condition the primary analysis on correct classification; doing so would hide the key confident-wrong
  result.
- Do not compare absolute hidden-space distances across checkpoints without normalization; activation scales
  change during training.
- Do not infer births/splits/merges from changing cluster IDs. Align the same examples by membership overlap.
- Report sample counts for every confidence/correctness group. If a group is too small, mark it underpowered
  instead of smoothing it away.

## Out of scope (do NOT)

- Scaling to a larger ResNet.
- Testing the small GPT/transformer from the grokking paper.
- Input-noise sweeps or the error-correction hypothesis.
- Activation steering interventions.
- Testing whether stable regions are disconnected activation-manifold components.
- Reimplementing SplineCam or the paper's full local-complexity analysis from scratch. If a compatible metric is
  already available, log it as a secondary timing reference only.
- Architecture, width, depth, optimizer, dataset-size, or weight-decay sweeps.
- Turning an expected result into a broad mechanistic claim about LLMs.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with: `On track? <yes/no> — <stage, % done, blocker if any>`.

If the result matches the expected monotonic picture, the correct next action is to finish `REPORT.md` and
`STOP`, not to add experiments.

## Current status

**Iter 1 done (3 seeds).** Trained d4/w200 MNIST MLP for seeds 0/1/2 with 13 log-spaced checkpoints;
ran the frozen plateau protocol at each. Result replicates: plateau contrast rises 0.42→0.80 while test
accuracy declines after its step-~300 peak (plateau lags generalization); validated stable-region count
converges to 10 by step ~300 in every seed; confident-wrong plateaus strongly (confidence, not
correctness). The step-10k dip is seed-0-only → seed noise, not a real transient. Verdict: **expected
monotonic emergence, replicated**. RESULTS.md/REPORT.md are current-best with 4 embedded figures.
Remaining: required figure #4 (region composition + membership-overlap split/merge lineage), then STOP.

For reference, the existing final checkpoint already showed the main endpoint behavior:

- the 4-layer width-200 ReLU MLP trained on 1,000 MNIST samples exhibits plateaus when first-hidden-layer
  natural activations are perturbed and last-hidden-layer displacement is measured;
- norm-and-sparsity-matched random activations do not show the same plateau;
- agglomerative clustering finds ten class-dominant clusters plus an uncertain mixed cluster rather than a
  convincing eleventh stable region;
- the mixed cluster lacks a plateau;
- confident-wrong examples show a plateau, supporting confidence rather than correctness as the key correlate;
- within-proposed-region perturbations move downstream much less than cross-region perturbations.

What is missing is the checkpoint trajectory that establishes when these properties arise and whether region
membership evolves monotonically.

## Next step

Add required figure #4 `region_composition_and_lineage`: (a) per-checkpoint predicted-label / confidence
composition of validated clusters, and (b) a compact membership-overlap heatmap aligning the same eval
examples across adjacent checkpoints (births/deaths/splits/merges) for seed 0. Compositions are already
in `results/sweep_seed*.json`; the lineage needs per-checkpoint cluster labels saved for the fixed eval
set. Embed the figure in RESULTS.md + REPORT.md, note that region membership evolves monotonically
(no replicated split/merge), then finish REPORT.md and write empty `STOP` (no unaddressed feedback).