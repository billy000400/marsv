# RESULTS — Do continuous low-to-high-resolution targets reduce activation plateaus?

> CURRENT-BEST ONLY. One row per experiment. No history (that lives in CHANGELOG.md).
> Full write-up with metric definitions, baselines and figure captions: **REPORT.md**.

## Headline

**Yes — robustly positive.** Two 4-layer MLPs get the *identical* 49-value input (a clean MNIST
digit average-pooled 4x4 to 7x7) and differ *only* in their target: a 10-way digit label versus the
original clean 28x28 image. The image target requires predicting spatial detail that the 7x7 input
does not contain — the removed detail carries **39.1%** of the image's pixel energy — and the
predictor genuinely learns it (**R² detail recovery 0.660**, versus 0.195 for bicubic upsampling and
0.165 for a privileged digit-template that is given the true label). Under an identical
first-hidden-layer SLERP probe the predictor's representation still moves **4.9x closer to a
constant-rate transition** than the classifier's at hidden layer 3 — on **90 of 90** digit pairs, at
every layer, on every seed, under both metrics and both normalizations. Continuous supervision
reduces plateaus even when the continuous task *adds* predicted detail rather than discarding it.

## Task adequacy — both gates pass

| gate | requirement | measured | pass |
|---|---|---|---|
| classifier | ≥ 95% top-1 on the untouched pool | 95.8 / 96.3 / 96.8% (seeds 0/1/2) | yes |
| predictor beats block repeat + bicubic | paired 95% CI of the MSE gap excludes 0 | all four CIs strictly positive | yes |
| detail recovery | lower 95% bound of R²detail > 0 | **0.654** | yes |

### Low-to-high task quality on the untouched pool (test[:2000], best-val checkpoints)

Lower MSE is better; higher R²detail is better (0 = no detail beyond a block-constant image).

| predictor of `y` | full-image MSE | removed-detail MSE | R²detail [95% CI] | low-res consistency MSE |
|---|---|---|---|---|
| mean training image | 0.0640 | 0.0375 | 0.065 [0.062, 0.069] | 2.7e-02 |
| block repeat `U(z)` | 0.0401 | 0.0401 | 0.000 [-0.000, 0.000] | 9.1e-16 |
| bicubic 7x7→28x28 | 0.0346 | 0.0323 | 0.195 [0.191, 0.199] | 2.3e-03 |
| digit template (privileged) | 0.0335 | 0.0335 | 0.165 [0.158, 0.172] | 3.0e-16 |
| **trained predictor** | **0.0137** | **0.0136** | **0.660 [0.654, 0.666]** | 1.1e-04 |

The predictor beats the *privileged* digit-template diagnostic by 0.0198 detail-MSE
[0.0195, 0.0202], so the recovered detail is instance-specific, not a per-digit prototype.

## Main comparison — linearity deviation LD at the best-val checkpoint (lower = smoother)

90 fixed cross-digit pairs, seeds 0/1/2, 95% percentile bootstrap CI over pairs (10,000 resamples).

| layer | classifier | predictor | paired diff (clf − pre) [95% CI] | ratio | pairs predictor smoother |
|---|---|---|---|---|---|
| hidden 2 | 0.1187 | 0.0219 | **0.0968** [0.0873, 0.1063] | 5.4x | 90 / 90 |
| hidden 3 | 0.1677 | 0.0342 | **0.1335** [0.1197, 0.1474] | 4.9x | 90 / 90 |
| output   | 0.1858 | 0.0455 | **0.1403** [0.1300, 0.1510] | 4.1x | 90 / 90 |

Per-seed paired differences at hidden layer 3: 0.1373 / 0.1381 / 0.1251 — every seed positive.

## Plateau-and-cliff shape — max normalized jump MJ (1 = perfectly constant rate)

| layer | classifier | predictor | paired diff [95% CI] | ratio |
|---|---|---|---|---|
| hidden 2 | 2.32 | 1.13 | **1.19** [1.08, 1.31] | 2.1x |
| hidden 3 | 2.86 | 1.24 | **1.62** [1.44, 1.80] | 2.3x |
| output   | 6.42 | 1.38 | **5.04** [4.62, 5.47] | 4.6x |

MJ agrees with LD in sign and significance at every layer, so the verdict is **robust positive**,
not mixed.

## Robustness controls

| control | hidden 2 | hidden 3 | output |
|---|---|---|---|
| alternative fraction normalization, LD diff [95% CI] | 0.0626 [0.0575, 0.0679] | 0.0730 [0.0665, 0.0796] | 0.1196 [0.1094, 0.1306] |
| final step-30,000 checkpoints, LD diff [95% CI] | 0.1008 [0.0910, 0.1108] | 0.1409 [0.1264, 0.1558] | 0.1440 [0.1338, 0.1543] |

Protocol checks: operator identities `D(U(z))=z`, `D(P(y))=0`, `UD+P=I` hold to ≤2.4e-07; endpoint
reproduction error ≤1.4e-06 (tolerance 1e-04); every rerun bit-identical.

## Figures

Each figure is motivated and captioned in **REPORT.md**.

### Figure 1 — the target contains detail the input does not

![MNIST digit, its 7x7 pooled input, block repetition, bicubic upsample and the removed detail](plots/data_audit.png)

**Figure 1.** Rows top-to-bottom: clean target `y`; the only model input `z=D(y)` (7x7); block
repetition `U(z)`; bicubic upsample; removed detail `r=P(y)`. Columns are five test digits.
Grayscale rows use a fixed [0,1] scale; the detail row uses a diverging blue↔red scale over
[-0.5, 0.5] with white at zero. The detail row is not blank: it holds 39.1% of the pixel energy.

### Figure 2 — both models train stably; probed checkpoints marked

![Training and validation loss curves for both models across three seeds](plots/training_curves.png)

**Figure 2.** x: training step; y: MSE per output unit (log scale in the first two panels). Left =
classifier, middle = predictor (solid = train, dashed = validation, triangle = the probed
best-validation checkpoint). Right: validation loss divided by its own minimum, classifier (solid,
circles) vs predictor (dashed, squares).

### Figure 3 — the predictor really super-resolves

![Panel comparing input, bicubic, digit template, prediction, target and detail residuals](plots/superres_panel.png)

**Figure 3.** One column per digit 0–9 from the untouched pool. Rows: input `z` shown as `U(z)`;
bicubic; privileged digit template; predictor output; target; predicted detail `P(ŷ)`; true detail
`P(y)`. The predictor's strokes are sharp and match the specific instance, unlike the blurry
bicubic and the generic template.

### Figure 4 — quantified against every frozen baseline

![Bar chart of full MSE, detail MSE and R2 detail for four baselines and the model](plots/baseline_bars.png)

**Figure 4.** Bars with 95% bootstrap intervals over the 2,000 pool images. Left: full-image MSE per
pixel; middle: removed-detail MSE per pixel (both lower = better); right: R²detail (higher = better,
0 = no detail recovered). Bars are distinguished by hatch as well as hue.

### Figure 5 — hand-selected transitions

![d(alpha) curves for four hand-selected digit transitions](plots/hand_selected_curves.png)

**Figure 5.** x: interpolation position α; y: endpoint-relative path coordinate `d(α)`. Rows are the
preregistered transitions 6→7, 3→5, 0→1, 4→9; columns are hidden layer 2, hidden layer 3, output.
Solid = classifier, dashed = predictor, three seeds each; dotted diagonal = a perfectly constant-rate
path. The classifier's S-shaped curves are the plateau-and-cliff signature.

### Figure 6 — the effect over all 90 pairs

![Mean d(alpha) curves over 90 pairs for both models at three layers](plots/mean_curves.png)

**Figure 6.** x: α; y: mean `d(α)` over 90 pairs and 3 seeds; shaded band with hatching = the
interquartile range over pairs and seeds. Solid/circles = classifier, dashed/squares = predictor,
dotted = straight line. The predictor tracks the diagonal at every layer.

### Figure 7 — paired effect with uncertainty

![Paired classifier-minus-predictor differences for both metrics at three layers](plots/paired_difference.png)

**Figure 7.** x: layer; y: paired classifier − predictor difference (left: LD; right: MJ). Circles
with bars = seed-averaged mean and 95% bootstrap CI over pairs; triangles = the three individual
seed means. Every interval lies above the dashed zero line.

### Figure 8 — the effect is not driven by a few pairs

![Per-pair scatter of classifier vs predictor linearity deviation](plots/per_pair_scatter.png)

**Figure 8.** x: classifier LD; y: predictor LD, one point per pair (seed-averaged), per layer.
Dashed line = equality. All 90 points fall below it at every layer.

### Figure 9 — what the models output along a path

![Predictor image outputs, detail component and classifier outputs along the 6-to-7 path](plots/path_predictions.png)

**Figure 9.** The 6→7 path at 11 evenly spaced α (seed 0). Top: predictor output `ŷ(α)` as a 28x28
image ([0,1] grayscale). Middle: its removed-detail component `P(ŷ(α))` (diverging scale,
[-0.5, 0.5]). Bottom: the classifier's 10 raw outputs, x = digit class, with the argmax printed
below. The image morphs continuously while the classifier's decision snaps 6 → 5 → 7.

### Figure 10 — the verdict does not depend on the checkpoint

![Mean d(alpha) curves at the best-validation and final checkpoints](plots/checkpoint_control.png)

**Figure 10.** x: α; y: mean `d(α)` over 90 pairs and 3 seeds, at each layer. Four series:
classifier and predictor, each at its best-validation checkpoint and at final step 30,000,
distinguished by line style and marker. The best-val and final curves nearly coincide.
