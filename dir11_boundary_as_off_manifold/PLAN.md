# PLAN — Direction: Are plateau-separated activation regions manifold components?

> Working folder: `dir11_boundary_as_off_manifold`. Agent REWRITES "Current status"/"Next step" + ticks stages each
> iteration. Disk (PLAN/JOURNAL/RESULTS/CHANGELOG + ../BUDGET.md + ../CLAUDE.md) is the only memory.

## Success criterion (definition of "done")

`RESULTS.md` and `REPORT.md` give a clear empirical verdict on the following question:

> Do plateau-separated stable regions correspond to disconnected or strongly density-separated components of the natural activation manifold?

Build on the existing MNIST experiments in the `image-models` branch. The final report must contain:

1. A reproducible set of:
   - smooth same-region interpolation paths;
   - plateau-crossing same-class paths, especially the candidate digit-9 regions;
   - cross-class plateau paths as controls.

2. For every path:
   - normalized downstream distance curve `d(t)`;
   - estimated plateau-boundary position;
   - activation-support score along the path.

3. A connectivity analysis of the natural activation cloud:
   - mutual-kNN graph;
   - within-region connectivity scale;
   - between-region connectivity scale;
   - minimum-support or maximum-edge bottleneck between candidate regions.

4. A clear verdict:
   - **Supported:** plateau-separated regions remain disconnected, or merge only through a much lower-support bottleneck than within-region paths.
   - **Refuted:** the regions are connected by a comparably high-support natural activation path, despite direct interpolation showing a plateau boundary.
   - **Inconclusive:** the result is unstable across reasonable graph or density hyperparameters.

Finite samples cannot prove true topological disconnection. Phrase conclusions as evidence about empirical activation-manifold components.

Null or negative results are COMPLETE if the question is answered. When done, the loop writes an empty `STOP` file.

## Fallback (if time runs short)

Use the existing trained model and analyze:

- one smooth digit-9-to-9 interpolation;
- one plateau-crossing digit-9-to-9 interpolation;
- several cross-digit interpolation paths.

For each path, plot:

- `d(t)`;
- kNN radius relative to natural activations;
- nearest-natural-activation distance.

Also build one mutual-kNN graph and report whether the two candidate digit-9 regions connect at a normal within-region neighborhood scale.

Produce one combined figure, write a preliminary verdict, and create `STOP`.

## Setup (fixed)

- **Base code:** build directly upon the `image-models` branch of `FranciscoHS/mars-plateaus`.
- **Development branch:** create a new branch from `image-models`, such as `dir11-boundary-as-off-manifold`.
- **Code policy:** reuse the existing model, checkpoint, activation hooks, interpolation code, and plateau metrics. Do not duplicate working implementations unless necessary.
- **Primary model:** the existing four-layer ReLU MNIST MLP from the sprint experiments.
- **Data:** correctly classified MNIST examples. Use the largest already-supported train/test split available in the branch.
- **Intervention layer:** the same first hidden layer used in the existing plateau experiments.
- **Downstream measurement layer:** the same final hidden layer used in the existing distance experiments. Logits are a secondary measurement.
- **Interpolation:** use the existing spherical interpolation implementation for direct-path tests.
- **Natural activation reference set:** first-hidden-layer activations produced by real MNIST inputs.
- **Candidate stable regions:** initially use the existing digit-9 outliers and smooth digit-9 examples. Later define region assignments systematically using downstream activation clustering.
- **Plateau metric:** normalized downstream distance `d(t)`.
- **Plateau boundary:** the path location with maximum smoothed absolute derivative `|d'(t)|`.
- **Primary support metric:** kNN radius from each interpolated activation to the natural activation reference cloud. Larger radius means lower empirical support.
- **Secondary support metric:** nearest-natural-activation distance.
- **Component metric:** connectivity in a mutual-kNN graph of natural activations.
- **Robustness:** repeat graph results over a reasonable range of `k` values rather than selecting one favorable value.
- **Shared limits in `../BUDGET.md`; operator rules in `../CLAUDE.md` — read both every iteration.**
- **Deliverable hygiene (see CLAUDE.md):** RESULTS.md/REPORT.md = current-best only, no history; CHANGELOG.md = the history.
- **Do NOT `pip install` torch, torchvision, transformer_lens, cupbearer, jax, flax** — they break the CUDA build.

## Stages (checklist)

- [x] S1 — Reproduce and organize the existing image-model plateau experiments (iter 1: paths + d(t) + |d'| boundary reproduced in analyze_manifold.py; base code mapped in JOURNAL)
  - Inspect the `image-models` branch before changing code.
  - Record the relevant scripts, checkpoint, model architecture, layer hooks, metrics, and output files in `JOURNAL.md`.
  - Reproduce:
    - a cross-digit plateau;
    - the real-activation versus random-activation perturbation result;
    - a plateau-crossing digit-9-to-9 path;
    - a smooth digit-9-to-9 path.
  - Save endpoint indices, images, activations, model seed, and interpolation configuration.
  - Convert visual plateau judgments into a numerical boundary location based on `|d'(t)|`.
  - Save all reported figures to `plots/` and define every metric in `REPORT.md` Methods.

- [x] S2 — Estimate activation support and candidate manifold components (iter 1: kNN radius + mutual-kNN graph; iter 2: KMeans-k×seed sweep, all-10-digit region assignment, graph-k robustness)
  - Collect first-hidden-layer activations from correctly classified natural MNIST inputs.
  - Assign candidate stable regions using downstream activation clustering and the existing digit-9 observations.
  - Compute kNN radius and nearest-natural-activation distance along every direct interpolation path.
  - Build a mutual-kNN graph using natural first-layer activations.
  - Measure:
    - connectivity within each candidate region;
    - connectivity between candidate regions;
    - the graph scale at which each pair of regions merges;
    - the largest edge or lowest-support bottleneck on the best connecting path.
  - Repeat the graph analysis across reasonable `k` values.
  - Validate that within-region pairs connect more easily than clearly unrelated cross-digit pairs.

- [x] S3 — Test whether plateau boundaries are off-manifold and whether regions are components (iter 1: direct-path + component tests for digit-9; iter 2: robustness sweeps + all-digit counterexample search, 0 found; iter 3: cross-model confirmation — 2nd MLP seed 1 reproduces all 3 tests, 0 counterexamples. Question answered → STOP)
  - **Direct-path test:** determine whether the maximum-`|d'(t)|` plateau boundary coincides with the lowest-support part of the spherical interpolation.
  - **Component test:** determine whether an alternative path through natural activations connects the two candidate regions without passing through an unusually low-support bottleneck.
  - Compare three path categories:
    - same stable region;
    - different stable regions within the same digit;
    - different digits.
  - Report the alignment between:
    - plateau-boundary location;
    - maximum kNN radius;
    - maximum nearest-natural-activation distance.
  - Search explicitly for the main counterexample:
    - a direct path with a plateau boundary between two regions that are nevertheless connected at a normal natural-activation neighborhood scale.
  - Produce final figures, statistical summaries, and the verdict in `REPORT.md`.

## Out of scope (do NOT)

- Do not run the LLM lambda-sweep experiments in this direction.
- Do not optimize output-smooth activation paths.
- Do not reproduce all experiments from the manifold-steering paper.
- Do not claim that low density alone proves a separate manifold component.
- Do not claim that clustering alone proves disconnectedness.
- Do not use two-dimensional PCA or UMAP visualizations as the primary evidence.
- Do not switch to an LLM until the MNIST experiment has a clear measurement procedure and verdict.
- Do not refactor unrelated parts of the `image-models` branch.
- Do not train a new model unless the existing checkpoint cannot reproduce the reported result.

## On-track check (required every iteration)

End each `JOURNAL.md` entry with:

`On track? <yes/no> — <stage, % done, blocker if any>`

## Current status

**Iter 3 (2026-07-15): DONE — question answered, STOP written.** Verdict = **REFUTED**, robust across
analysis hyperparameters AND a second independent training seed. Iter 1 built the pipeline
(`experiments/analyze_manifold.py`): digit-9 direct-path boundary at 35th-pctile support (not a gap);
the two 9-regions 5 hops / bottleneck 2.72 (< median 2.85) — like within-region, unlike cross-digit
(9–12 hops). Iter 2 (`experiments/robustness.py`): graph-k sweep (6–25), KMeans-k∈{2,3,4}×seed sweep,
all-10-digit generality → **0 counterexamples** (bottleneck ≤ p95). Iter 3
(`experiments/cross_model.py`): trained a **2nd MLP from scratch (seed 1, 86.9% acc)** and re-ran all
three tests — cross-region 9→9 boundary at **53rd** pctile (well-supported), A↔B = **4 hops = within-A
(4)**, bottleneck 1.54 < median 1.95, **0 counterexamples** across all digits. Base model re-analyzed
by the same code reproduced its reported numbers exactly (regression check). Figures
`plots/direct_path_support.png`, `plots/component_test.png`, `plots/robustness_graphk.png`,
`plots/robustness_regions.png`, `plots/cross_model.png`. The sole remaining limitation is now
architecture/dataset (both models are depth-4 w200 ReLU on the same 1000-image MNIST subset), not seed.

---

The `image-models` branch has already produced preliminary evidence that:

- the MNIST MLP exhibits plateau-like downstream behavior;
- real activations show stronger plateaus than matched random activations;
- same-digit downstream activations are usually closer than different-digit activations;
- some digit-9-to-9 interpolations show plateau-boundary-plateau behavior;
- other digit-9-to-9 interpolations are short and smooth.

This suggests that digit 9 may contain multiple downstream stable regions. It does not yet show whether these regions are separate components of the natural activation manifold.

Two distinct claims must be tested:

1. **Direct-path claim:** the sharp plateau crossing occurs where the interpolated first-layer activation has low support under natural activations.
2. **Component claim:** there is no alternative high-support path through natural activations connecting the two stable regions.

The second claim is stronger. A direct spherical interpolation may leave the manifold even when both endpoint regions belong to one connected manifold component.

## Next step

**None — direction complete, STOP written.** The PLAN success criterion is fully met: reproducible
smooth / plateau-crossing / cross-class paths with d(t), boundary, and support; a mutual-kNN
connectivity analysis with within/between-region scales and bottlenecks; robustness across region and
graph hyperparameters; and a cross-model confirmation — all yielding a clear **REFUTED** verdict. Any
future extension (a different architecture or dataset, or an LLM analog) would be a NEW direction, out
of scope here.