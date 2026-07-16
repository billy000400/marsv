# RESULTS — Animating plateau formation through training (MNIST MLP)

> CURRENT-BEST ONLY. History lives in CHANGELOG.md. Full definitions in REPORT.md.
> Model: 4-layer ReLU MLP (784→200→200→200→10), 1,000-sample MNIST subset, AdamW
> (lr 1e-3, wd 0.01), MSE on one-hot, batch 200, 100k steps. Primary protocol: 50-point
> norm-rescaled SLERP between the post-ReLU first-hidden activations $h_1$ of two fixed test
> images, patched at $h_1$ and propagated; $d(\alpha)$ = relative L2 distance of the
> propagated **logits** to the two endpoint outputs (logit-space in every figure unless
> labeled otherwise). 55 fixed pairs (45 cross-class, 10 within-class), all endpoints from
> the **first 2,000 test images** (operator feedback 07161151). Seed 0: 205 checkpoints
> (steps 0, 10, 30, 100, 300, then every 500 to 100k) plus two deterministic dense reruns
> (every 5 steps 0–1,000; every 50 steps 82,000–82,500), both bit-exact vs the movie records;
> seeds 1–2: 56 checkpoints (every 2,000). All 317 movie records verified complete by
> `experiments/manifest_check.py`.

## Headline

**Plateaus form gradually and keep sharpening (and moving) long after test accuracy has
stabilized.** At initialization $d(\alpha)$ is the featureless diagonal — no plateaus. Test
accuracy saturates at ~0.88 by step ~70–120 (train accuracy 1.0 at step 145), but
plateau–boundary–plateau structure matures over tens of thousands of steps: the plateau
fraction (share of path points with $d<0.1$ or $d>0.9$; diagonal baseline ≈ 0.20) rises from
~0.20 at init to ~0.34 (step 100), ~0.4 (step 10k), and 0.54–0.61 at 100k across all three
seeds, with no sudden global transition. By 100k, 22–29 of 45 cross-class pairs show a
textbook plateau→boundary→plateau curve, many as multi-step staircases through third-class
regions. Late in training the plateaus persist but their **boundaries keep relocating
abruptly**: the largest movie jump (pair 5→6, steps 82,000→82,500, seed 0) is a complete
boundary flip that the 50-step-resolution rerun resolves to ~150 steps. Within-class controls
stay boundary-free (8/10 pairs keep one predicted class along the whole path; the 2 exceptions
have a genuinely misclassified endpoint). The radial-perturbation control replicates the
timing: plateau contrast vs matched-random activations rises 0.42→0.80 between step 100 and
100k while test accuracy declines.

**Early phase (linear time, every 5 steps, 0–1,000):** the diagonal deforms within the first
tens of steps; curves flicker rapidly while the loss falls fastest (first ~150–200 steps),
then settle into stable soft sigmoids. Plateau fraction: 0.19 (step 0) → 0.27 (25) → 0.34
(100) → ~0.37 (200–1,000, nearly frozen) — the early phase builds the soft structure, the
flattening into true plateaus happens over the following tens of thousands of steps.

## Plateau fraction and test accuracy (seeds 0 / 1 / 2)

| step | plateau fraction (s0 / s1 / s2) | test acc on first 2,000 (s0 / s1 / s2) |
|-----:|:-------------------------------:|:--------------------------------------:|
|       0 | 0.19 / 0.21 / 0.20 | 0.089 / 0.097 / 0.109 |
|      10 | 0.25 / 0.26 / 0.25 | 0.595 / 0.654 / 0.594 |
|     100 | 0.34 / 0.34 / 0.33 | 0.878 / 0.891 / 0.877 |
|   1,000 | 0.37 / 0.37 / 0.35 | 0.881 / 0.891 / 0.886 |
|  10,000 | 0.38 / 0.42 / 0.39 | 0.873 / 0.896 / 0.889 |
|  20,000 | 0.53 / 0.41 / 0.41 | 0.865 / 0.871 / 0.882 |
|  50,000 | 0.47 / 0.65 / 0.64 | 0.854 / 0.873 / 0.869 |
| 100,000 | 0.56 / 0.54 / 0.61 | 0.848 / 0.869 / 0.858 |

Plateau fraction = mean over the 45 cross-class pairs of the fraction of the 50 path points
with $d<0.1$ or $d>0.9$ (the one curve-derived summary; a straight diagonal scores ≈ 0.20, a
perfect two-plateau step function scores 1). Protocol checks: patched $\alpha=0/1$ outputs
reproduce the unpatched endpoint outputs to 3.7e-4; the vectorized SLERP matches the branch
`slerp_path` to 9.5e-7; both dense reruns reproduce the movie records bit-exactly.

## Figures

All curve/heatmap figures: x = interpolation $\alpha$ (0 = image A, 1 = image B), $d$ =
logit-space relative endpoint distance; squares under curves = predicted class.

![Main animation (seed 0, 205 frames, step in title): logit-space d(alpha) for ten fixed cross-class pairs; insets: train/test accuracy + confidence (top) and train/test loss (bottom, log y) vs step (log x).](plots/plateau_evolution.gif)

![Early-phase animation (seed 0, steps 0-1,000, one frame per 5 steps, LINEAR time): the diagonal deforms within tens of steps, flickers while loss falls fastest, then settles into soft sigmoids.](plots/plateau_evolution_early.gif)

![Early-phase heatmap: d(alpha) (color, blue=0 red=1) vs alpha (x) and step (y, linear, one row per 5 steps) for the ten pairs + two within-class controls.](plots/plateau_early_heatmap.png)

![Selected main-animation frames (rows: steps 0, 100, 1,000, 20,000, 100,000): diagonal at init, soft sigmoids by a few hundred steps, plateau-boundary-plateau staircases by tens of thousands.](plots/frames_selected_steps.png)

![Full-run heatmap: d(alpha) (color) vs alpha (x) and checkpoint (y, rows 0,10,30,100,300 then every 500): plateaus consolidate gradually; boundary positions keep shifting late; within-class pairs (right) stay boundary-free.](plots/plateau_training_heatmap.png)

![Layerwise d(alpha) at h2, h3 and logits for early/middle/late checkpoints: successive layers sharpen the same transition (only figure not purely logit-space).](plots/layerwise_selected_steps.png)

![Seed comparison: plateau fraction vs step for seeds 0-2 (left) and all 45 cross-pair curves overlaid at matched steps (right): gradual sharpening, consistent across seeds.](plots/seed_comparison.png)

![Dense 50-step zoom into the largest late movie jump (pair 5 to 6, seed 0, steps 82,000-82,500): the boundary flip completes within ~150 steps.](plots/dense_zoom.png)

![Training context (seed 0): train/test accuracy and confidence (mean max raw output) vs step; test accuracy saturates by step ~100 and drifts slightly down late.](plots/training_context.png)

## Perturbation control (secondary; 13 log-spaced checkpoints, 3 seeds)

Local-robustness control consistent with the interpolation movie: plateau contrast of natural
$h_1$ activations vs norm-and-sparsity-matched random activations rises 0.42 (step 100) →
0.80 (step 100k) while the validated stable-region count converges to 10 (one per predicted
digit) by step ~300; contrast tracks confidence, not correctness (confident-wrong 0.73 vs
confident-correct 0.85 vs uncertain 0.49 at 100k).

![Perturbation control: plateau contrast vs matched-random activations keeps rising after test accuracy saturates (left; mean of 3 seeds, band = min-max); validated stable-region count converges to 10 (right).](plots/plateau_contrast_and_region_count.png)
