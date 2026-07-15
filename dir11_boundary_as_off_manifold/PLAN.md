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

- [~] S3 — Test whether plateau boundaries are off-manifold and whether regions are components (iter 1: direct-path + component tests for digit-9; iter 2: robustness sweeps + all-digit counterexample search, 0 found. Remaining: 2nd trained model/seed to confirm cross-model transfer)
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

**Iter 2 (2026-07-15): robustness pass done; verdict = REFUTED, now robust across analysis
hyperparameters (single checkpoint).** Iter 1 built the pipeline (`experiments/analyze_manifold.py`):
digit-9 direct-path boundary at 35th-pctile support (not a gap); the two 9-regions 5 hops / bottleneck
2.72 (< median 2.85) — like within-region, unlike cross-digit (9–12 hops). Iter 2
(`experiments/robustness.py`) stress-tested it: graph-k sweep (6–25) — A↔B tracks the within-region
control at every scale, far below cross-digit; KMeans-k∈{2,3,4}×seed sweep — 9-regions always
hops 2–5 / bottleneck 1.93–2.72 (all < median); all-10-digit generality — every same-digit region
pair's bottleneck ≤ p95 (2.35–4.17 vs threshold 4.23) → **0 counterexamples**. Key correction: hops is
contaminated by manifold elongation (digit-1 = 15 within-digit hops yet high-support), so the
generality verdict is anchored on the off-manifold (bottleneck-vs-p95) criterion. Figures
`plots/robustness_graphk.png`, `plots/robustness_regions.png`. **Remaining:** a 2nd trained model/seed
for cross-model transfer (the one open limitation) — otherwise the question is answered.

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

Cross-model confirmation (final S3 hardening): retrain / load a second MNIST MLP (different seed or
width) and re-run the digit-9 direct-path + component tests + all-digit counterexample search, to check
the plateau=decision-geometry (not data-hole) reading transfers beyond one checkpoint — the only
remaining real limitation. Reuse `experiments/robustness.py` (parameterize the checkpoint path). If a
second model reproduces 0 counterexamples, the question is fully answered → write STOP. If no second
checkpoint is available and time is short, the question is already answered for the given model;
finalize and STOP per the fallback.