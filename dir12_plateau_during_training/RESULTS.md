# RESULTS — Animating plateau formation through training (MNIST MLP)

> CURRENT-BEST ONLY. History lives in CHANGELOG.md. Full definitions in REPORT.md.
> Model: 4-layer ReLU MLP (784→200→200→200→10), 1,000-sample MNIST subset, AdamW
> (initial lr 1e-3, wd 0.01), MSE on one-hot, batch 200, 100k steps; plus a CE-loss
> (cross-entropy) rerun of seed 0 with identical init/data/batch order. **All primary runs use
> ReduceLROnPlateau (factor 0.5, patience 100 steps on the per-step full-train loss, rel
> threshold $10^{-4}$, min LR $10^{-8}$)** — chosen by an explicit scheduler search so that the
> training loss decreases smoothly and converges (operator feedback 07161834; only smoothly
> converged runs are shown). Primary protocol: 50-point norm-rescaled SLERP between the
> post-ReLU first-hidden activations $h_1$ of two fixed test images, patched at $h_1$ and
> propagated; $d(\alpha)$ = relative L2 distance of the propagated **logits** to the two
> endpoint outputs (logit-space in every figure unless labeled otherwise; the CE figures also
> show $d$ on the softmax probabilities). 55 fixed pairs (45 cross-class, 10 within-class), all
> endpoints from the **first 2,000 test images** (feedback 07161151). Seed 0: 205 checkpoints
> (steps 0, 10, 30, 100, 300, then every 500 to 100k); seeds 1–2: 55 checkpoints (every 2,000);
> CE seed 0: full 205-checkpoint schedule. All 520 records manifest-verified. The early-phase
> zoom (every 5 steps, 0–1,000, linear time) is bit-exact for these runs (first LR cut at step
> 1,375, verified).

## Headline

**Plateau structure is entirely learned, forms in the first few hundred steps, and converged
training freezes it — while the training loss decides *where* plateaus live, not *whether* they
exist.** At initialization $d(\alpha)$ is the featureless diagonal. The plateau fraction (PF =
share of path points with $d<0.1$ or $d>0.9$; diagonal floor ≈ 0.20) rises 0.19 → 0.34 (step
100) → 0.37 (step 300) and then stays at 0.35–0.37 for the remaining 99,700 steps in all three
MSE seeds; after the LR cascade the curves are frozen (late curve motion $M = 5.6\times10^{-7}$
per 500-step gap vs $2.4\times10^{-2}$ at constant LR — no late boundary flips). Under MSE the
converged plateaus are *soft* logit-space sigmoids; under CE the same protocol read in
**probability space** gives the only textbook-sharp plateau→boundary→plateau curves of the
project: PF$_{\rm prob}$ 0.58 (step 100) → 0.79 (1k) → 0.85 (10k) → **0.863 (100k)** while CE
logit space stays near-diagonal (PF 0.24). Decision regions along the path are
piecewise-constant under both losses. Within-class controls stay boundary-free (9/10, 9/10,
8/10 across seeds; every exception has a genuinely misclassified endpoint). 3 vs 5 is the
hardest pair (worst AUROC of 45 under both losses: 0.9772 MSE / 0.9697 CE), and its odd curve
traces to a misclassified "3" endpoint plus a detour through the 3-region.

**Scheduler search (feedback 07161834):** among constant LR, cosine annealing, and three
ReduceLROnPlateau settings (seed 0, identical init/data/batches), only the loss-adaptive
schedules are smooth (loss never exceeds 2× its running minimum; constant and cosine spike by
factors up to $10^{5}$–$10^{6}$ over most of training). Winner: **factor 0.5 / patience 100** —
smooth, converged (loss range over the last 5k steps: 0.6%), final train loss $8.4\times10^{-9}$
(≈ the constant run's $4.0\times10^{-9}$ floor, 350× below patience-10's $2.9\times10^{-6}$),
and the best test accuracy of any run (0.8815 vs 0.8475 constant). Every converged schedule
freezes logit-space PF at its LR-collapse value (~0.37); only the never-converging constant-LR
run sharpens further (0.556 at 100k) — its plots are omitted per operator preference (kept on
disk in `plots/`; numbers in `results/lr_scheduler_search.json`).

## Plateau fraction and test accuracy (converged runs, seeds 0 / 1 / 2, MSE)

| step | plateau fraction (s0 / s1 / s2) | test acc on first 2,000 (s0 / s1 / s2) |
|-----:|:-------------------------------:|:--------------------------------------:|
|       0 | 0.19 / 0.21 / 0.20 | 0.089 / 0.097 / 0.109 |
|      10 | 0.25 / 0.26 / 0.25 | 0.595 / 0.654 / 0.594 |
|     100 | 0.34 / 0.34 / 0.33 | 0.878 / 0.890 / 0.877 |
|     300 | 0.37 / 0.37 / 0.35 | 0.879 / 0.891 / 0.885 |
|   2,000 | 0.37 / 0.37 / 0.35 | 0.880 / 0.893 / 0.887 |
|  10,000 | 0.36 / 0.37 / 0.35 | 0.882 / 0.893 / 0.885 |
|  50,000 | 0.37 / 0.37 / 0.35 | 0.881 / 0.893 / 0.885 |
| 100,000 | 0.37 / 0.37 / 0.35 | 0.881 / 0.893 / 0.885 |

CE seed 0 for comparison, logit / probability space: 0.19/0.19 (step 0), 0.25/0.58 (100),
0.26/0.79 (1,000), 0.26/0.85 (10,000), 0.24/0.863 (100,000). LR cascades ($10^{-3}$ →
$1.5\times10^{-8}$, 16 halvings): MSE seeds at steps 1,375–25,824 / 1,415–29,030 /
1,175–31,465; CE at 15,941–17,650. Protocol checks: patched $\alpha=0/1$ outputs reproduce the
unpatched endpoint outputs to 3.7e-4; the vectorized SLERP matches the branch `slerp_path` to
9.5e-7; scheduled and constant runs are bit-identical before the first LR cut (verified through
step 1,000).

## Figures

All curve/heatmap figures: x = interpolation $\alpha$ (0 = image A, 1 = image B), $d$ =
logit-space relative endpoint distance unless labeled otherwise; squares under curves =
predicted class. Figure numbers match REPORT.md.

![Figure 1. Converged training, all four primary runs (x: step, log): per-step full-train loss (left, log y, all smooth), LR cascade 1e-3 to 1.5e-8 (middle, log y), test accuracy on first 2,000 (right). Solid: MSE seeds 0-2; dashed red: CE seed 0.](plots/smooth_convergence.png)

![Figure 2. Training context (seed 0): train acc, test acc, and confidence (mean max raw output) vs step (log x); test acc pinned at 0.881 from step ~300.](plots/training_context_pl_f0.5_p100.png)

![Figure 3. Main animation (seed 0, 205 frames, step in title): logit-space d(alpha) for ten fixed cross-class pairs; insets: accuracy + confidence (top) and train/test loss (bottom, log y, smooth) vs step (log x).](plots/plateau_evolution_pl_f0.5_p100.gif)

![Figure 4. Selected main-animation frames (rows: steps 0, 100, 1,000, 20,000, 100,000): diagonal at init, soft sigmoids by a few hundred steps, then frozen — the last three rows are nearly identical.](plots/frames_selected_steps_pl_f0.5_p100.png)

![Figure 5. Early-phase animation (seed 0, steps 0-1,000, one frame per 5 steps, LINEAR time): the diagonal deforms within tens of steps, flickers while loss falls fastest, then settles into soft sigmoids. Bit-exact for the scheduled run (first LR cut at step 1,375).](plots/plateau_evolution_early.gif)

![Figure 6. Early-phase heatmap: d(alpha) (color, blue=0 red=1) vs alpha (x) and step (y, linear, one row per 5 steps) for the ten pairs + two within-class controls.](plots/plateau_early_heatmap.png)

![Figure 7. Full-run heatmap: d(alpha) (color) vs alpha (x) and checkpoint (y, rows 0,10,30,100,300 then every 500): structure laid down in the bottom sliver, boundaries perfectly vertical (frozen) for 100k steps; within-class pairs (right) stay boundary-free.](plots/plateau_training_heatmap_pl_f0.5_p100.png)

![Figure 8. Layerwise d(alpha) at h2, h3 and logits for early/middle/late checkpoints: successive layers sharpen the same transition (only figure showing d at hidden layers).](plots/layerwise_selected_steps_pl_f0.5_p100.png)

![Figure 9. Seed comparison: plateau fraction vs step for seeds 0-2 (left) and all 45 cross-pair curves overlaid at matched steps (right): identical gradual-then-frozen shape in all seeds.](plots/seed_comparison_pl_f0.5_p100.png)

## MSE vs cross-entropy (seed 0, identical init/data/batches/schedule)

![Figure 10. MSE vs CE: train/test loss (top, log y, smooth; dashed line = first train-acc-1.0 checkpoint — loss keeps falling under both losses), accuracy + CE confidence (bottom left), and plateau fraction in logit vs probability space (bottom right): CE has near-diagonal logit curves but sharp probability-space plateaus.](plots/mse_vs_ce_training_pl_f0.5_p100.png)

![Figure 11. CE run, selected steps (rows 0, 100, 1,000, 20,000, 100,000): d(alpha) in logit space (blue) stays near the diagonal while probability space (red) develops sharp plateau-boundary-plateau curves; squares: predicted class.](plots/frames_selected_steps_ce_prob_pl_f0.5_p100.png)

![Figure 12. CE animation (seed 0, 205 frames): logit-space d(alpha) stays near-diagonal throughout training while predicted classes (squares) still switch between discrete regions; insets: accuracy + confidence (max softmax prob) and CE loss (smooth).](plots/plateau_evolution_ce_pl_f0.5_p100.gif)

## Pairwise AUROC: 3 vs 5 (converged seed 0, step 100k)

![Figure 13. Pairwise AUROC matrix and ranking (3v5 worst at 0.9772; CE: 0.9697, also worst), 1-AUROC vs curve mid-fraction scatter (weak correlation, Spearman -0.30), and the annotated 3->5 curve: segments predicted 2 / 3 / 5 with a misclassified "3" endpoint.](plots/pairwise_auc_pl_f0.5_p100.png)
