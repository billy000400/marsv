# RESULTS — Do continuous MNIST targets reduce activation plateaus?

> CURRENT-BEST ONLY. One row per experiment. No history (that lives in CHANGELOG.md).
> Full write-up with metric definitions and baselines: **REPORT.md**.

## Headline

**Yes — strongly.** Two 4-layer MLPs differing *only* in their target (10-way digit label vs a
continuous 49-value reconstruction of the clean image, downsampled to 7x7) — same corrupted MNIST
inputs, bit-identical initial weights in all three shared layers, same batch order, optimizer and
30,000 steps, MSE loss both. **Each model is probed at its best-validation-loss checkpoint**
(classifier: step 7,500 / 16,200 / 14,400 for seeds 0/1/2; regressor: step 29,800 in all seeds).
Under the identical first-hidden-layer SLERP probe the regressor's representation moves
**4.3-5.9x closer to a constant-rate transition** than the classifier's, on **89 of 90 digit pairs
at every layer**, across 3 seeds. Probing the final step-30,000 weights instead changes nothing.

## Metrics

Definitions, equations and baselines: REPORT.md § Methods. In brief — for each fixed test-image pair
we interpolate the first-hidden activation (101-point SLERP), propagate, and record
$d_\ell(\alpha)=\lVert h_\ell(\alpha)-h_\ell(0)\rVert / \lVert h_\ell(1)-h_\ell(0)\rVert$.
**Linearity deviation** LD $= \mathrm{mean}_\alpha |d_\ell(\alpha)-\alpha|$ (0 = perfectly even
transition, lower is smoother). **Max normalized jump** MJ $= 100\max_i|d_\ell(\alpha_{i+1})-d_\ell(\alpha_i)|$
(1 = constant rate, higher = a cliff). 90 fixed cross-digit pairs, seeds 0/1/2, 95% percentile
bootstrap CI over pairs (10,000 resamples).

### Main comparison — linearity deviation at the best-val checkpoint (lower = smoother)

| layer | classifier | regressor | paired diff (clf − reg) [95% CI] | ratio | pairs regressor smoother |
|---|---|---|---|---|---|
| hidden 2 | 0.1182 | 0.0263 | **0.0918** [0.0804, 0.1036] | 4.5x | 89 / 90 |
| hidden 3 | 0.1295 | 0.0304 | **0.0990** [0.0874, 0.1107] | 4.3x | 89 / 90 |
| output   | 0.1792 | 0.0304 | **0.1487** [0.1395, 0.1580] | 5.9x | 89 / 90 |

### Plateau-and-cliff shape — max normalized jump (1 = constant rate)

| layer | classifier | regressor | paired diff [95% CI] | ratio |
|---|---|---|---|---|
| hidden 2 | 2.01 | 1.14 | **0.87** [0.77, 0.97] | 1.8x |
| hidden 3 | 2.31 | 1.21 | **1.11** [0.99, 1.23] | 1.9x |
| output   | 6.35 | 1.21 | **5.13** [4.82, 5.45] | 5.2x |

### Control — probing the final step-30,000 weights instead

| layer | clf best-val | clf step 30k | reg best-val | reg step 30k | paired diff at step 30k [95% CI] |
|---|---|---|---|---|---|
| hidden 2 | 0.1182 | 0.1253 | 0.0263 | 0.0263 | 0.0990 [0.0868, 0.1119] |
| hidden 3 | 0.1295 | 0.1345 | 0.0304 | 0.0304 | 0.1041 [0.0927, 0.1161] |
| output   | 0.1792 | 0.1801 | 0.0304 | 0.0304 | 0.1497 [0.1403, 0.1590] |

### Robustness — dir12's alternative normalization $d^{\text{frac}}$

| layer | classifier | regressor | paired diff [95% CI] |
|---|---|---|---|
| hidden 2 | 0.0842 | 0.0221 | 0.0621 [0.0568, 0.0675] |
| hidden 3 | 0.0923 | 0.0238 | 0.0685 [0.0621, 0.0754] |
| output   | 0.1617 | 0.0256 | 0.1361 [0.1256, 0.1466] |

### Training quality (both models adequate; metrics at the probed best-val checkpoint)

| model | test metric | seeds 0/1/2 | reference |
|---|---|---|---|
| classifier | test accuracy (10k test set) | 96.34 / 96.39 / 96.37% | — |
| classifier | test accuracy (dir12's 2k pool) | 95.00 / 94.50 / 94.65% | dir12 same-pool clean-input MNIST MLP: 97.9-98.1% (ours sees $\sigma=0.3$ noise) |
| classifier | val-loss min step | 7,500 / 16,200 / 14,400 | val loss then rises 5.7 / 0.8 / 1.6% → mild overfitting ✓ |
| regressor | val-loss min step | 29,800 / 29,800 / 29,800 | val loss flattens, never rises → converged, no overfitting |
| regressor | test MSE per pixel | 0.001124 / 0.001114 / 0.001112 | mean-target baseline 0.02907 (26.0x worse); pooled-corrupted-input baseline 0.01292 (11.6x worse) ✓ |

## Figures

Both models trained adequately, and Figure 1 also shows which weights the probe uses — the
validation-loss minimum, marked on each curve:

![Train and validation MSE against training step for the classifier and the regressor, plus a rescaled validation-loss panel](plots/training_curves.png)

**Figure 1.** Training adequacy and checkpoint selection, 3 seeds per panel. x: training step
(0-30,000). y (left, middle): MSE per output unit, log scale — solid = train loss (10k-image train
subset), dashed = validation loss (test images 2,000-9,999), down-triangle = validation minimum (the
probed checkpoint). y (right): each model's validation loss divided by its own minimum, linear scale;
circles = classifier minima, square = regressor minimum.

The regressor's smoothness would be meaningless if it had learned a degenerate shortcut, so we check
its outputs directly:

![Three rows of ten digit images: corrupted 28x28 inputs, 7x7 regressor outputs, 7x7 clean targets](plots/reconstructions.png)

**Figure 2.** The regressor really denoises. Columns: first test image of each digit 0-9. Rows:
corrupted 28x28 input / 49-value model output as 7x7 / clean 7x7 target. Grayscale range $[0,1]$
throughout.

What the two objectives do, seen through each model's own output along one path:

![Two rows of eleven panels along the 6-to-7 path: regressor 7x7 outputs morphing, classifier logit bars snapping between digits](plots/path_reconstructions.png)

**Figure 3.** Same interpolation path, both models (seed 0, pair $6\to7$). Columns: 11 evenly spaced
$\alpha$ (value above each column). Top: regressor output as a 7x7 image — the 6 continuously
deforms into a 7. Bottom: classifier's 10 raw logits as bars (x: digit 0-9, y: logit, range
$[-0.4,1.2]$); the digit below is the arg-max. The classifier holds a confident "6", collapses to a
low-confidence smear in the middle, then snaps to a confident "7" — flat regions separated by jumps.

Individual transitions behind the aggregate:

![Grid of d(alpha) curves for four digit transitions across three layers, classifier versus regressor](plots/hand_selected_curves.png)

**Figure 4.** $d(\alpha)$ on four hand-selected transitions, all 3 seeds drawn. Rows: $6\to7$,
$3\to5$, $0\to1$, $4\to9$. Columns: hidden 2, hidden 3, output. x: $\alpha$; y: $d(\alpha)$. Solid =
classifier, dashed = regressor, dotted = straight transition $d=\alpha$. The regressor tracks the
dotted line in every panel; the classifier plateaus then lurches, most sharply at the output layer.

Averaged over the whole frozen 90-pair set:

![Mean d(alpha) over 90 pairs and 3 seeds for both models at three layers, with interquartile bands](plots/mean_curves.png)

**Figure 5.** Mean transition shape over 90 pairs x 3 seeds. x: $\alpha$; y: $d(\alpha)$. Circles/
solid = classifier, squares/dashed = regressor, dotted = straight transition; bands (hatched `//`
classifier, `\\` regressor) are the interquartile range over all 270 pair-seed curves. The
classifier's output layer shows the classic S: flat, steep, flat.

The headline aggregate, with uncertainty:

![Two panels of paired classifier-minus-regressor differences with bootstrap confidence intervals, per layer](plots/paired_difference.png)

**Figure 6.** Paired difference (classifier − regressor) per layer. x: layer. y: difference in
linearity deviation (left) and max normalized jump (right); positive = classifier less smooth.
Circles with error bars = seed-averaged mean with 95% bootstrap CI over the 90 pairs; small triangles
= individual seed means; dashed line = no difference.

Is the gap universal or driven by a few extreme transitions?

![Scatter of per-pair regressor deviation against classifier deviation at three layers, nearly all points below the diagonal](plots/per_pair_scatter.png)

**Figure 7.** One point per digit pair (90 points, seed-averaged). x: classifier linearity deviation;
y: regressor linearity deviation, same scale. Dashed line = equal smoothness. 89 of 90 points fall
below the diagonal at all three layers.

Finally, the check that the checkpoint rule is not doing the work:

![Mean d(alpha) at three layers comparing each model at its best-validation checkpoint and at step 30,000](plots/checkpoint_control.png)

**Figure 8.** x: $\alpha$; y: mean $d(\alpha)$ over 90 pairs x 3 seeds. Circles/solid = classifier at
its best-val checkpoint; up-triangles/dash-dot = classifier at step 30,000; squares/dashed =
regressor at its best-val checkpoint; diamonds/dash-dot-dot = regressor at step 30,000; dotted =
straight transition. Each model's two curves nearly coincide, and the classifier pair stays far from
the regressor pair — the gap is a target-type effect, not a checkpoint or overtraining artifact.

## Caveats

Continuous supervision **reduces** plateaus, it does not abolish them (regressor LD 0.026 → 0.030
grows with depth; MJ 1.14-1.21 > 1.0). Matching step count exactly means the two models reach their
best-validation checkpoints at different steps (classifier 7.5k-16.2k, regressor 29.8k), and the
regressor never overfits; the checkpoint control (Figure 8) shows this does not drive the result. The
output-layer comparison is 10-d vs 49-d; the hidden-layer results (identical 200-d spaces, identical
initial weights) carry the conclusion. Interpolated activations are not guaranteed to be on the data
manifold. One architecture, one dataset.
