# PLAN — Token distance vs. transition width

> Working folder: `dir27_interp_dist_vs_plateau_width`.
> Read `../BUDGET.md` and `../CLAUDE.md` every iteration. Keep the experiment minimal.

## Question

For GPT-2 Large, does the L2 distance between two token representations at the interpolation layer relate to how sharply the downstream representation transitions between them?

Use:

> "The house was big"

as endpoint A, and replace the final token ` big` with every other token in the GPT-2 vocabulary to form endpoint B.

Also ask a second, descriptive question:

> Do many tokens have similar final-layer transition widths, suggesting grouped behavior?

No sophisticated statistics are needed. The main evidence should be visually obvious from plots.

## Success criterion

The experiment is done when we have:

1. For every usable GPT-2 vocabulary token B:
   - token ID / decoded token
   - L2 distance from ` big` to B at the interpolation layer
   - transition width at a middle layer
   - transition width at the final layer

2. Clear plots answering:
   - Does interpolation-layer L2 distance visibly relate to transition width?
   - Does the distribution of final-layer transition widths show obvious groups / modes?

3. A few representative interpolation curves confirming that the measured widths correspond to visibly different transition shapes.

A null result is a complete result.

## Setup — fixed

- Model: GPT-2 Large.
- Base sequence: `"The house was big"`.
- Keep all prefix token IDs fixed.
- Construct endpoint B by directly replacing the final token ID with another vocabulary token.
  - Do **not** decode and re-tokenize the prompt.
  - This ensures A and B differ only in the final token.
- Endpoint A is always the token ` big`.
- Sweep all other usable vocabulary tokens.
- Exclude only tokens that cannot be meaningfully passed through the existing pipeline, e.g. special tokens if necessary.

### Interpolation layer

Use the same hidden-space interpolation convention as the existing plateau experiments:

- interpolate the last-position hidden state at the existing default interpolation layer (`layer 0`);
- keep the rest of the sequence unchanged.

For token B:

\[
D_B = ||h_A - h_B||_2
\]

where `h_A` and `h_B` are the endpoint representations at the interpolation layer.

Interpolate:

\[
h(t) = (1-t)h_A + t h_B,\qquad t\in[0,1].
\]

Use a sufficiently fine, fixed grid of `t` values for all tokens. Start with 101 evenly spaced points.

### Readout layers

Measure the trajectory at two downstream layers:

- middle: block 18
- final: final GPT-2 Large transformer block

At each readout layer `l`, define:

\[
y_l(t) =
\frac{||h_l(t)-h_l(0)||_2}
     {||h_l(1)-h_l(0)||_2}.
\]

Thus:

- `y = 0` corresponds to the A endpoint
- `y = 1` corresponds to the B endpoint

Define:

\[
W_l = t_{0.9} - t_{0.1}
\]

where `t_0.1` and `t_0.9` are the locations where the curve crosses `y=0.1` and `y=0.9`.

Use simple linear interpolation between neighboring sampled `t` points to estimate the crossings.

Do not smooth or fit the curves.

If a trajectory is strongly non-monotonic or crosses a threshold multiple times, flag it rather than introducing a complicated correction.

## Stages

- [x] **S1 — Verify the measurement**
  - Run ` big` → a small set of ~20 diverse tokens.
  - Plot `y(t)` at the middle and final layers.
  - Confirm that the 0.1 → 0.9 width visually matches the curves.
  - Confirm endpoint normalization gives approximately `y(0)=0`, `y(1)=1`.

- [x] **S2 — Vocabulary sweep**
  - Run ` big` → every other usable GPT-2 token.
  - Save one row per token containing:
    - token ID
    - decoded token
    - interpolation-layer L2 distance `D`
    - middle-layer width `W_mid`
    - final-layer width `W_final`
    - simple flag for non-monotonic / unusual trajectories

- [x] **S3 — Distance vs. transition width**
  Produce two primary scatter plots:

  **Figure 1a**
  - x: interpolation-layer L2 distance `D`
  - y: middle-layer transition width `W_mid`

  **Figure 1b**
  - x: interpolation-layer L2 distance `D`
  - y: final-layer transition width `W_final`

  Plot every token as one point with transparency.

  The question is simply whether there is an obvious visual relationship:
  - larger distance → wider transition?
  - larger distance → narrower transition?
  - or essentially no visible relationship?

  Do not add Pearson/Spearman correlations, hypothesis tests, regression models, etc. in this first pass.

- [x] **S4 — Look for grouped final-layer behavior**
  Make two simple views of `W_final`:

  **Figure 2a — Histogram**
  - x: final-layer transition width
  - y: number of tokens

  Look for obvious peaks or separated groups.

  **Figure 2b — Sorted widths**
  - sort tokens by `W_final`
  - x: token rank
  - y: `W_final`

  This should make shelves / clusters of similar widths visually apparent if they exist.

  If obvious groups appear, inspect them descriptively:
  - choose several representative tokens from each visible group;
  - show their decoded token strings;
  - do not run k-means or another clustering algorithm.

- [x] **S5 — Representative curves**
  Pick a small number of tokens from visually different width regimes, and several tokens from any apparent same-width group.

  Plot their final-layer `y(t)` curves together.

  The goal is to visually verify that:
  - narrow-width tokens really transition sharply;
  - wide-width tokens really transition gradually;
  - tokens assigned to the same apparent group have genuinely similar transition shapes.

## Main figures

Keep the final report centered on four intuitive plots:

1. **L2 distance vs. middle-layer transition width**
2. **L2 distance vs. final-layer transition width**
3. **Histogram + sorted view of final-layer widths**
4. **A few representative `y(t)` curves**

The report should be understandable from these plots without statistical machinery.

## Questions to answer in REPORT.md

### Finding 1 — Distance

Does the distance between ` big` and another token at the interpolation layer visibly predict the width of the downstream transition?

Describe only what is visible.

### Finding 2 — Depth

Is the relationship different at the middle layer versus the final layer?

### Finding 3 — Grouping

Do final-layer transition widths form obvious groups or modes?

If yes, give example tokens from the visible groups and show representative curves.

Do not claim that a group is semantic unless the token examples clearly support that interpretation.

## Out of scope

Do **not**:

- test multiple prompts;
- test multiple models;
- change the interpolation layer;
- introduce cosine distance or other distance metrics;
- run clustering algorithms;
- run correlation significance tests;
- fit regression models;
- create new plateau metrics;
- explain the mechanism behind any grouping.

Those can be follow-ups only if the basic plots reveal something interesting.

## Fallback

If the full vocabulary sweep is too expensive:

1. finish the pipeline on a few thousand uniformly sampled vocabulary tokens;
2. produce all four plots;
3. verify the result is interpretable;
4. only then scale to the full vocabulary.

Do not replace the basic experiment with a more complicated analysis.

## Current status

COMPLETE. All stages S1–S5 done; success criterion met. 50,256 tokens in `results/sweep.csv`;
Figures in `plots/fig1_distance_vs_width.png`, `plots/fig2_final_width_distribution.png`,
`plots/fig3_representative_curves.png`; REPORT.md and RESULTS.md written, render checks pass.
Answer: D does not visibly predict W in the bulk; final layer sharper than block 18; W_final is one
broad peak with a tail, the only tight group being 64 rarely seen tokens at D≈5.17, W≈0.33.

## Next step

None — direction finished (STOP written).
