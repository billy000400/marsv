# RESULTS — Animating plateau formation through training (MNIST MLP)

> CURRENT-BEST ONLY. History lives in CHANGELOG.md. Full definitions in REPORT.md.
> **Model:** 4-layer ReLU MLP (784→200→200→200→10), AdamW (initial lr 1e-3, wd 0.01), MSE on
> one-hot, batch 200. **Primary runs (this report's focus):** trained from verified untrained
> step-0 initializations on **all 60,000 MNIST training images**, shuffled without replacement
> each epoch (300 steps/epoch), 30,000 steps = 100 epochs, with a `ReduceLROnPlateau(factor 0.5,
> patience 100 on the full-train loss, rel threshold $10^{-4}$, min LR $10^{-8}$)` schedule
> chosen by a scheduler search so the loss converges smoothly (operator feedback
> human_feedback_1). Seeds 0/1/2 (104/25/25 checkpoints) plus a CE-loss seed-0 rerun (104
> checkpoints). **Reference (one dedicated size section):** the SAME initializations trained on a
> fixed 1,000-image subset, 100k steps, same schedule; seeds 0–2 + CE. **Protocol:** 50-point
> norm-rescaled SLERP between the post-ReLU first-hidden activations $h_1$ of two fixed test
> images, patched at $h_1$ and propagated; $d(\alpha)$ = relative L2 distance of the propagated
> **logits** to the two endpoint outputs. 55 fixed pairs (45 cross-class, 10 within-class) plus a
> frozen bank of **50 unfiltered 3→5 pairs** (rank-$i$ test 3 with rank-$i$ test 5, $i<50$,
> chosen before viewing any result). All endpoints and metrics use the **first 2,000 test images**
> (feedback 07161151). All 258 primary 60k records + 201 early-zoom records manifest-verified.

## Headline

**Activation plateaus are entirely learned, form early, and — with enough data — keep sharpening
for thousands of steps of smoothly converging training before freezing.** At initialization
$d(\alpha)$ is the featureless diagonal. Training the full 60,000-image MNIST set, the plateau
fraction (PF = share of path points with $d<0.1$ or $d>0.9$; diagonal floor ≈ 0.20) rises 0.19
(step 0) → 0.35 (100) → 0.43 (300) → 0.62 (1,500) → ~0.67 (6,000) and freezes there (final
**0.674 / 0.663 / 0.668** across seeds; peak 0.674 at step ~27,000; late curve motion
$7.6\times10^{-4}$ per 300-step gap), while the loss converges smoothly (final full-train loss
$2.6\times10^{-5}$, spike-free) and test accuracy reaches **0.9775 / 0.9795 / 0.9785**. The loss
picks the coordinates: under cross-entropy the logit-space curves stay near-diagonal (PF 0.25 ≈
floor) but the softmax **probabilities** develop the sharpest plateaus of the study (PF 0.90 at
30,000). Within-class controls stay boundary-free (9/10 per seed; every exception has a
genuinely misclassified endpoint), and **0/90** cross-pair endpoints are misclassified at step
30,000.

**Training-set size sets the ceiling (dedicated section).** The *same initializations* trained on
a fixed 1,000-image subset under the *same schedule* freeze at PF ≈ 0.37 within ~300 steps — the
moment the subset is memorized — and never sharpen again (late curve motion $5.6\times10^{-7}$).
The 60k runs sail past 0.37 and roughly double it. So the converged-PF ceiling of the small-data
model is a **small-data effect, not a property of convergence.** The 3-vs-5 pair that was the 1k
model's single hardest (worst AUROC of 45, 0.977) is near-perfectly separated by the full-data
model (AUROC **0.9993**, rank 4/45; confusion 6% → 0.8%) — pair difficulty was a small-data
effect too.

**3→5 sub-plateau verdict.** On the frozen 50-path 3→5 bank the full-data model mainly fixes the
*endpoints*: **49/50** paths have both endpoints predicted correctly at step 30,000 (36/50 for
the 1k model at the matched step; both 0/50 at step 0, so the preregistered "correct at step 0"
subset is empty). The original 3→5 path simplifies from `2 | 3 | 5` (1k, misclassified "3"
endpoint) to **`3 | 5` in all three full-data seeds** — an **endpoint correction plus removal of
a third-class detour, not a merge**: mean segment count 1.98 (60k) vs 1.90 (1k), third-class
detours 0% vs 2% at the final checkpoint, and **no** path in the smooth 60k run has a
repeated-class RLE (the only merge-capable pattern) at any checkpoint. These are statements about
the measured 1-D paths, not global class-region topology.

## Plateau fraction and test accuracy (full-60k, primary)

| step | PF logit (60k s0) | PF (1k s0, matched) | test acc (60k s0) | test acc (1k s0) |
|-----:|:---:|:---:|:---:|:---:|
|      0 | 0.19 | 0.19 | 0.089 | 0.089 |
|    100 | 0.35 | 0.34 | 0.917 | 0.878 |
|    300 | 0.43 | 0.37 | 0.960 | 0.879 |
|  1,500 | 0.62 | 0.37 | 0.978 | 0.880 |
|  3,000 | 0.66 | 0.37 | 0.980 | 0.880 |
|  6,000 | 0.67 | 0.37 | 0.979 | 0.881 |
| 30,000 | **0.674** | 0.37 | 0.978 | 0.881 |

Final PF across 60k seeds: **0.674 / 0.663 / 0.668**; final test acc 0.9775 / 0.9795 / 0.9785;
0/90 cross-pair endpoints misclassified per seed. CE seed 0 (logit / probability space): 0.19/0.18
(step 0), 0.25/0.55 (100), 0.26/0.67 (300), 0.26/0.80 (1,500), **0.25/0.90 (30,000)**. Step-0
weights verified bit-identical to the reference initializations (untrained); first-epoch shuffling
verified to use each of the 60,000 indices exactly once. Chosen schedule (seed 0): LR halves 8×
between steps 1,402 and 20,844 to $3.9\times10^{-6}$; full-train loss decays $10^{-1}\to
2.6\times10^{-5}$ spike-free (largest transient 1.56× the running minimum, tail range 1.07).

## Figures

All curve/heatmap figures: x = interpolation $\alpha$ (0 = image A, 1 = image B), $d$ =
logit-space relative endpoint distance unless labeled otherwise; squares under curves = predicted
class (tab10 colors, digits 0–9). Comparison figures: full-60k green, 1k reference blue. Figure
numbers match REPORT.md.

![Figure 1. Converged training, all four full-60k runs (x: step, log): per-step full-train (60k) loss (left, log y, all smooth), LR cascade 1e-3 to ~4e-6 (middle, log y), test accuracy on first 2,000 (right). Solid: MSE seeds 0-2; dashed red: CE seed 0.](plots/smooth_convergence_60k.png)

![Figure 2. Training context (60k seed 0): train acc, test acc, and confidence (mean max raw output) vs step (log x); test acc saturates at ~0.978 by step ~1,500.](plots/training_context_60k.png)

![Figure 3. Main animation (60k seed 0, 104 frames, step in title): logit-space d(alpha) for ten fixed cross-class pairs; insets: accuracy + confidence (top) and train/test loss (bottom, log y, smooth) vs step (log x).](plots/plateau_evolution_60k.gif)

![Figure 4. Selected main-animation frames (rows: steps 0, 100, 1,500, 10,500, 30,000): diagonal at init, sharp plateau-boundary-plateau staircases by a few thousand steps, then frozen — the last two rows are nearly identical.](plots/frames_selected_steps_60k.png)

![Figure 5. Early-phase animation (60k seed 0, steps 0-1,000, one frame per 5 steps, LINEAR time): the diagonal deforms within tens of steps, flickers while loss falls fastest, then keeps sharpening (PF still climbing at step 1,000, unlike the 1k regime). Bit-exact for the scheduled run (first LR cut at step 1,402).](plots/plateau_evolution_early_60k.gif)

![Figure 6. Early-phase heatmap (60k seed 0): d(alpha) (color, blue=0 red=1) vs alpha (x) and step (y, linear, one row per 5 steps, 0-1,000) for the ten pairs + two within-class controls.](plots/plateau_early_heatmap_60k.png)

![Figure 7. Full-run heatmap (60k seed 0): d(alpha) (color) vs alpha (x) and checkpoint (y, rows 0,10,30,100 then every 300): structure laid down early, boundaries sharpen then run vertically (frozen); within-class pairs (right) stay boundary-free.](plots/plateau_training_heatmap_60k.png)

![Figure 8. Layerwise d(alpha) at h2, h3 and logits for steps 100, 3,000, 30,000 (60k seed 0): successive layers sharpen the same transition (only figure showing d at hidden layers).](plots/layerwise_selected_steps_60k.png)

![Figure 9. Seed comparison (60k): plateau fraction vs step for seeds 0-2 (left) and all 45 cross-pair curves overlaid at matched steps (right): identical gradual-then-frozen shape in all seeds.](plots/seed_comparison_60k.png)

## MSE vs cross-entropy (60k seed 0, identical init/data/batches/schedule)

![Figure 10. MSE vs CE: train/test loss (top, log y, smooth), accuracy + CE confidence (bottom left), and plateau fraction in logit vs probability space (bottom right): CE has near-diagonal logit curves but sharp probability-space plateaus (PF 0.90).](plots/mse_vs_ce_training_60k.png)

![Figure 11. CE run (60k seed 0), selected steps (rows 0, 100, 1,500, 10,500, 30,000): d(alpha) in logit space (blue) stays near the diagonal while probability space (red) develops sharp plateau-boundary-plateau curves; squares: predicted class.](plots/frames_selected_steps_ce_prob_60k.png)

![Figure 12. CE animation (60k seed 0, 104 frames): logit-space d(alpha) stays near-diagonal throughout training while predicted classes (squares) still switch between discrete regions; insets: accuracy + confidence (max softmax prob) and CE loss (smooth).](plots/plateau_evolution_60k_ce.gif)

## Pairwise AUROC: 3 vs 5 is no longer the hardest pair (60k seed 0, step 30k)

![Figure 13. Pairwise AUROC matrix and ranking: with 60k data 3v5 reaches AUROC 0.9993 (rank 4/45 from worst; worst pair now 4v9 at 0.9975), 1-AUROC vs curve mid-fraction scatter, and the clean 3->5 curve at step 30k (segments 3 then 5, both endpoints correct). On the 1k model 3v5 was rank 1/45 (0.977) — the difficulty was a small-data effect.](plots/pairwise_auc_60k.png)

## The frozen 50-path 3→5 bank (60k)

![Figure 14. Animation of the frozen 50-path 3->5 bank through 60k training (x: alpha, y: logit d; thin green: 50 paths; thick red: the original 3->5 pair; right panel: mean segment count, third-class detour fraction, and endpoints-correct fraction vs step, current step marked).](plots/full_mnist_3v5_training.gif)

## The effect of training-set size (1,000-image reference)

![Figure 15. Training context, 60k vs 1k. Left: test accuracy vs step (log x) for 60k seeds 0-2 (greens/cyans) vs the 1k reference (blue), plus 60k seed-0 confidence (mean max raw output, dashed purple); 60k runs reach 0.976-0.979 vs 0.881. Right: train/test MSE loss (log y) — the 60k train loss converges smoothly to 2.6e-5.](plots/full_mnist_training_context.png)

![Figure 16. Synchronized side-by-side animation, frames aligned by optimizer step (25 common steps, step in title): logit d(alpha) for five preregistered pairs, 1k run (top, blue) vs full-60k run (bottom, green); squares: predicted class. Insets: test accuracy and train loss of both runs with the current step marked. The 60k curves keep sharpening long after the 1k curves freeze.](plots/full_vs_1k_evolution.gif)

![Figure 17. Aligned static frames, all ten preregistered pairs (row pairs: steps 0, 300, 3,000, 30,000; within each: 1k blue over 60k green): by step 3,000 the 60k curves are sharp plateau-boundary-plateau staircases while the 1k curves remain frozen soft sigmoids.](plots/full_vs_1k_frames.png)

![Figure 18. Frozen 50-path 3->5 bank, 60k vs 1k at matched steps. Top left: the original 3->5 pair at step 30,000 (60k green: clean 3|5 plateau pair; 1k blue: 2|3|5 staircase with misclassified "3" endpoint; pred squares below). Top right: all 50 paths at 30,000. Bottom left: run-length segment count (mean, IQR) vs step. Bottom right: third-class detour fraction (solid) and endpoints-correct fraction (dashed) vs step - the dominant difference is endpoint correction (49/50 vs 36/50).](plots/full_mnist_3v5_summary.png)
