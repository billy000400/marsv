# RESULTS — Animating plateau formation through training (MNIST MLP)

> CURRENT-BEST ONLY. History lives in CHANGELOG.md. Full definitions in REPORT.md.
> Model: 4-layer ReLU MLP (784→200→200→200→10), AdamW (initial lr 1e-3, wd 0.01), MSE on
> one-hot, batch 200. **Reference runs:** 1,000-sample MNIST subset, 100k steps,
> ReduceLROnPlateau (factor 0.5, patience 100 on the per-step full-train loss, rel threshold
> $10^{-4}$, min LR $10^{-8}$) — chosen by an explicit scheduler search so the loss converges
> smoothly (operator feedback 07161834); seeds 0–2 plus a CE-loss (cross-entropy) seed-0 rerun.
> **Full-data extension (reopened plan):** the SAME untrained step-0 initializations trained
> from scratch on all 60,000 MNIST training images, shuffled without replacement each epoch
> (300 steps/epoch), cosine LR $10^{-3}\to10^{-6}$ over 30,000 steps; seed 0 at 104 checkpoints
> (0, 10, 30, 100, then every 300), seeds 1–2 at 25 fallback checkpoints. Primary protocol: 50-point
> norm-rescaled SLERP between the post-ReLU first-hidden activations $h_1$ of two fixed test
> images, patched at $h_1$ and propagated; $d(\alpha)$ = relative L2 distance of the propagated
> **logits** to the two endpoint outputs. 55 fixed pairs (45 cross-class, 10 within-class) plus,
> for the extension, a frozen bank of **50 unfiltered 3→5 pairs** (rank-$i$ test 3 with rank-$i$
> test 5, $i<50$, selected before viewing any full-data result); all endpoints from the
> **first 2,000 test images** (feedback 07161151). All records manifest-verified (520 reference
> + 154 extension + 16 re-evaluations of the 1k model on the 105-pair bank).

## Headline

**Plateau structure is entirely learned and forms early, but what happens next depends on the
data, not on convergence per se.** At initialization $d(\alpha)$ is the featureless diagonal.
On the fixed 1,000-example subset the plateau fraction (PF = share of path points with $d<0.1$
or $d>0.9$; diagonal floor ≈ 0.20) rises 0.19 → 0.34 (step 100) → 0.37 (step 300) and then
freezes for the rest of converged training in all three MSE seeds (late curve motion
$M = 5.6\times10^{-7}$ per 500-step gap). **Training the SAME initializations from scratch on
all 60,000 images breaks that ceiling: PF reaches 0.43 by step 300, 0.61 by 1,500, peaks at
0.73 (step ~8,400) and ends at 0.64–0.69 across seeds — while the loss still converges smoothly
(final train loss $2.3\times10^{-7}$, test acc 0.979, late curve motion $3\times10^{-4}$).**
The converged-1k PF ceiling of ~0.37 is therefore a small-data effect, not a property of
convergence: with 60× more data the network keeps carving sharp logit-space plateaus for
thousands of steps while fitting. Under CE the same protocol read in **probability space**
gives sharp plateaus on the 1k subset too (PF$_{\rm prob}$ 0.863 at 100k) while CE logit space
stays near-diagonal (0.24). Within-class controls stay boundary-free in all runs (9/10 typical;
every exception has a genuinely misclassified endpoint). 3 vs 5 is the hardest pair (worst
AUROC of 45 under both losses: 0.9772 MSE / 0.9697 CE).

**3→5 sub-plateau verdict (extension).** On the frozen 50-path 3→5 bank, the full-data model
mainly fixes the *endpoints*: 49/50 paths have both endpoints predicted correctly at step
30,000 vs 36/50 for the 1k model at the matched step (both start at 0/50 — no path had correct
endpoints at step 0, so that preregistered subset is empty). The original 3→5 path simplifies
from `2 | 3 | 5` (1k, misclassified "3" endpoint) to **`3 | 5` in all three full-data seeds** —
an **endpoint correction plus removal of the third-class detour, not a merge**: third-class
detours are rare in both runs (4% of paths at 30k for 60k-training vs 2% for 1k), mean segment
count is ~2 in both (2.04 vs 1.90), and no path in either run retains two non-adjacent
same-class segments at the final checkpoint (2/50 show one transiently mid-training under 60k).
These are statements about the measured 1-D paths, not about global class-region topology.

## Plateau fraction and test accuracy

Converged 1k reference (seeds 0/1/2, MSE): PF 0.19/0.21/0.20 (step 0) → 0.34/0.34/0.33 (100) →
0.37/0.37/0.35 (300) → 0.37/0.37/0.35 (100,000, frozen from ~300 on); test acc plateaus at
0.881/0.893/0.885. CE seed 0, logit/probability space: 0.19/0.19 (0), 0.25/0.58 (100),
0.26/0.79 (1k), 0.24/0.863 (100k). Full table history: CHANGELOG 2026-07-17 iter 8.

Full-60k from-scratch (seed 0; seeds 1–2 confirm at fallback steps):

| step | PF logit (60k s0) | PF (1k s0, matched step) | test acc (60k s0) | test acc (1k s0) |
|-----:|:---:|:---:|:---:|:---:|
|      0 | 0.19 | 0.19 | 0.089 | 0.089 |
|    100 | 0.35 | 0.34 | 0.918 | 0.878 |
|    300 | 0.43 | 0.37 | 0.960 | 0.879 |
|  1,500 | 0.61 | 0.37 | 0.973 | 0.880 |
|  3,000 | 0.69 | 0.37 | 0.978 | 0.880 |
|  8,400 | **0.73 (peak)** | 0.37 | 0.978 | 0.881 |
| 15,000 | 0.71 | 0.37 | 0.978 | 0.881 |
| 30,000 | 0.64 | 0.37 | 0.979 | 0.881 |

Final PF across 60k seeds: 0.64 / 0.69 / 0.65; final test acc 0.979 / 0.976 / 0.977. Step-0
weights verified bit-identical to the reference runs' initializations (and thus untrained);
first-epoch shuffling verified to use each of the 60,000 indices exactly once. The prescribed
cosine schedule is smooth on full data: full-train loss at checkpoints decays $10^{-1}\to
2.3\times10^{-7}$ with transients ≤17× the running minimum (vs $10^{5}$–$10^{6}$ for
constant/cosine on the 1k subset), none after step ~21k; last-10-checkpoint range 1.32.

## Figures

All curve/heatmap figures: x = interpolation $\alpha$ (0 = image A, 1 = image B), $d$ =
logit-space relative endpoint distance unless labeled otherwise; squares under curves =
predicted class (tab10 colors, digits 0–9). Figure numbers match REPORT.md.

![Figure 1. Converged training, all four 1k reference runs (x: step, log): per-step full-train loss (left, log y, all smooth), LR cascade 1e-3 to 1.5e-8 (middle, log y), test accuracy on first 2,000 (right). Solid: MSE seeds 0-2; dashed red: CE seed 0.](plots/smooth_convergence.png)

![Figure 2. Training context (1k seed 0): train acc, test acc, and confidence (mean max raw output) vs step (log x); test acc pinned at 0.881 from step ~300.](plots/training_context_pl_f0.5_p100.png)

![Figure 3. Main animation (1k seed 0, 205 frames, step in title): logit-space d(alpha) for ten fixed cross-class pairs; insets: accuracy + confidence (top) and train/test loss (bottom, log y, smooth) vs step (log x).](plots/plateau_evolution_pl_f0.5_p100.gif)

![Figure 4. Selected main-animation frames (rows: steps 0, 100, 1,000, 20,000, 100,000): diagonal at init, soft sigmoids by a few hundred steps, then frozen — the last three rows are nearly identical.](plots/frames_selected_steps_pl_f0.5_p100.png)

![Figure 5. Early-phase animation (1k seed 0, steps 0-1,000, one frame per 5 steps, LINEAR time): the diagonal deforms within tens of steps, flickers while loss falls fastest, then settles into soft sigmoids. Bit-exact for the scheduled run (first LR cut at step 1,375).](plots/plateau_evolution_early.gif)

![Figure 6. Early-phase heatmap: d(alpha) (color, blue=0 red=1) vs alpha (x) and step (y, linear, one row per 5 steps) for the ten pairs + two within-class controls.](plots/plateau_early_heatmap.png)

![Figure 7. Full-run heatmap (1k seed 0): d(alpha) (color) vs alpha (x) and checkpoint (y, rows 0,10,30,100,300 then every 500): structure laid down in the bottom sliver, boundaries perfectly vertical (frozen) for 100k steps; within-class pairs (right) stay boundary-free.](plots/plateau_training_heatmap_pl_f0.5_p100.png)

![Figure 8. Layerwise d(alpha) at h2, h3 and logits for early/middle/late checkpoints: successive layers sharpen the same transition (only figure showing d at hidden layers).](plots/layerwise_selected_steps_pl_f0.5_p100.png)

![Figure 9. Seed comparison (1k): plateau fraction vs step for seeds 0-2 (left) and all 45 cross-pair curves overlaid at matched steps (right): identical gradual-then-frozen shape in all seeds.](plots/seed_comparison_pl_f0.5_p100.png)

## MSE vs cross-entropy (1k seed 0, identical init/data/batches/schedule)

![Figure 10. MSE vs CE: train/test loss (top, log y, smooth; dashed line = first train-acc-1.0 checkpoint — loss keeps falling under both losses), accuracy + CE confidence (bottom left), and plateau fraction in logit vs probability space (bottom right): CE has near-diagonal logit curves but sharp probability-space plateaus.](plots/mse_vs_ce_training_pl_f0.5_p100.png)

![Figure 11. CE run, selected steps (rows 0, 100, 1,000, 20,000, 100,000): d(alpha) in logit space (blue) stays near the diagonal while probability space (red) develops sharp plateau-boundary-plateau curves; squares: predicted class.](plots/frames_selected_steps_ce_prob_pl_f0.5_p100.png)

![Figure 12. CE animation (1k seed 0, 205 frames): logit-space d(alpha) stays near-diagonal throughout training while predicted classes (squares) still switch between discrete regions; insets: accuracy + confidence (max softmax prob) and CE loss (smooth).](plots/plateau_evolution_ce_pl_f0.5_p100.gif)

## Pairwise AUROC: 3 vs 5 (converged 1k seed 0, step 100k)

![Figure 13. Pairwise AUROC matrix and ranking (3v5 worst at 0.9772; CE: 0.9697, also worst), 1-AUROC vs curve mid-fraction scatter (weak correlation, Spearman -0.30), and the annotated 3->5 curve: segments predicted 2 / 3 / 5 with a misclassified "3" endpoint.](plots/pairwise_auc_pl_f0.5_p100.png)

## Full-data extension: same initialization, all 60,000 images

![Figure 14. Full-60k training context. Left: test accuracy vs step (log x) for 60k seeds 0-2 (greens/cyans) vs the 1k reference (blue), plus 60k seed-0 confidence (mean max raw output, dashed purple); 60k runs reach 0.976-0.979 vs 0.881. Right: train/test MSE loss (log y) — the 60k train loss converges smoothly to 2.3e-7.](plots/full_mnist_training_context.png)

![Figure 15. Synchronized side-by-side animation, frames aligned by optimizer step (25 common steps, step in title): logit d(alpha) for five preregistered pairs, 1k run (top, blue) vs full-60k run (bottom, green); squares: predicted class. Insets: test accuracy and train loss of both runs with the current step marked. The 60k curves keep sharpening long after the 1k curves freeze.](plots/full_vs_1k_evolution.gif)

![Figure 16. Aligned static frames, all ten preregistered pairs (row pairs: steps 0, 300, 3,000, 30,000; within each: 1k blue over 60k green): by step 3,000 the 60k curves are sharp plateau-boundary-plateau staircases while the 1k curves remain the frozen soft sigmoids.](plots/full_vs_1k_frames.png)

![Figure 17. Animation of the frozen 50-path 3->5 bank through full-60k training (x: alpha, y: logit d; thin green: 50 paths; thick red: the original 3->5 pair; right panel: mean segment count, third-class detour fraction, and endpoints-correct fraction vs step, current step marked).](plots/full_mnist_3v5_training.gif)

![Figure 18. Frozen 50-path 3->5 bank, full-60k vs 1k at matched steps. Top left: the original 3->5 pair at step 30,000 (60k green: clean 3-to-5 plateau pair; 1k blue: 2|3|5 staircase with misclassified "3" endpoint; pred squares below). Top right: all 50 paths at 30,000. Bottom left: run-length segment count (mean, IQR) vs step. Bottom right: third-class detour fraction (solid) and endpoints-correct fraction (dashed) vs step - the dominant difference is endpoint correction (49/50 vs 36/50).](plots/full_mnist_3v5_summary.png)
