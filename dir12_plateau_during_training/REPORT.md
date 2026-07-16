# Animating plateau formation through training in an MNIST MLP

## Summary

**The question.** Take two inputs, interpolate between their internal activations, and watch the
model's output. In trained networks the output often stays glued to one endpoint's output, then
snaps to the other across a narrow **boundary**. This *plateau → boundary → plateau* shape is an
"activation plateau" (Shinkle & Heimersheim, *Activation Plateaus: Where and How They Emerge*).
Plateaus mean the model organizes its internal states into discrete regions of near-constant
behavior. For safety this matters twice. Plateaus make behavior stable: small internal
perturbations, including imperfect steering vectors, do nothing. And boundaries concentrate all
the change: a tiny nudge across one flips the output. We ask *when during training* this
discreteness appears.

**What we did.** We trained a small ReLU MLP on MNIST for 100,000 steps and saved hundreds of
checkpoints. At every checkpoint we ran the identical activation-interpolation experiment on
fixed image pairs and rendered the result as one frame of a movie.

**Findings.**

1. **No plateaus at initialization.** The interpolation curve of the random network is a
   featureless diagonal.
2. **Plateaus form gradually, and pairs do not synchronize.** Soft structure appears in the same
   few hundred steps in which accuracy rises; genuinely flat plateaus with sharp boundaries take
   tens of thousands of further steps to mature. There is no sudden global transition.
3. **The structure keeps changing long after test accuracy stabilizes.** Plateaus keep
   sharpening, and even past step 80,000 their boundaries relocate in fast ~150-step events.

**Verdict: plateaus emerge gradually, do not synchronize across pairs, and keep developing long
after test accuracy has stopped improving.**

## Methods

### Data, model, training

**Data.** MNIST, pixels in $[0,1]$, flattened to 784 dimensions. Training uses a fixed
1,000-image subset (drawn by `torch.randint` after `torch.manual_seed(seed)`, so the subset is
seed-determined). All evaluation — the interpolation endpoints and every test metric — uses only
the **first 2,000 of the 10,000 test images** (per operator feedback 07161151).

**Model & training.** 4-layer ReLU MLP, 784→200→200→200→10, with a ReLU after every linear
layer except the last. "Hidden layer $L$" means the post-ReLU output of the $L$-th linear layer,
so $h_1, h_2, h_3$ are 200-dimensional and $h_3$ is the last hidden layer. Optimizer: AdamW
(learning rate $10^{-3}$, weight decay 0.01). Loss: mean squared error (MSE) to one-hot targets.
Batch 200, 100,000 steps. This reproduces the training setup of *Deep Networks Always Grok*
(arXiv:2402.15555) used throughout this branch.

**Accuracy, confidence, and loss.** These three curves appear in the training-context figure and
in the animation insets. Let $f(x;\theta_t)\in\mathbb{R}^{10}$ be the logits at checkpoint $t$
and $y_n$ the true label of image $x_n$. Accuracy is the fraction of correct argmax predictions:

```math
\mathrm{acc}(t) = \frac{1}{N}\sum_{n=1}^{N}\mathbf{1}\Bigl[\arg\max_i f_i(x_n;\theta_t) = y_n\Bigr],
```

with $N=1{,}000$ (the training subset) for train accuracy and $N=2{,}000$ (the first 2,000 test
images) for test accuracy. Loss is the training objective itself, the MSE to the one-hot target
$e_{y_n}$:

```math
\mathcal{L}(t) = \frac{1}{10\,N}\sum_{n=1}^{N}\bigl\lVert f(x_n;\theta_t) - e_{y_n}\bigr\rVert_2^2
```

(the extra factor 10 is because `MSELoss` averages over the 10 output entries as well as over
images). Confidence needs care here: MSE-to-one-hot training drives the target logit toward 1
rather than toward $+\infty$, so softmax probabilities saturate near 0.23 for every image and
carry no information. We therefore define confidence as the **maximum raw output**, averaged
over images:

```math
\mathrm{conf}(t) = \frac{1}{N}\sum_{n=1}^{N}\max_i f_i(x_n;\theta_t).
```

Read it as: near 1 = the model puts a full-strength one-hot answer on some class; near 0 = no
class is asserted.

**Checkpoints.** Seed 0 is the primary run. It saves checkpoints at steps 0, 10, 30, 100, 300,
then every 500 up to 100,000 — 205 in total. Seeds 1 and 2 are confirmation runs with 56
checkpoints each (same early steps, then every 2,000). Every checkpoint stores the model weights
and a self-contained record of the protocol below: the distance curves, per-point logits,
predictions and softmax probabilities, and the endpoint activations at every hidden layer.
`experiments/manifest_check.py` verifies every expected file and field; all 317 records pass.
Training is deterministic given the seed, and a from-scratch rerun reproduced the movie's
records **bit-exactly**. We exploit that determinism twice, rerunning with extra recording
density where the movie is too coarse: every 50 steps inside 82,000–82,500 (the largest late
change) and every 5 steps from 0 to 1,000 (the early phase, on a linear time axis).

### The frozen interpolation protocol

**Pair bank (fixed before any results were seen).** One pair for each unordered pair of distinct
digits (45 cross-class pairs) plus one same-digit pair per digit (10 within-class controls). For
cross pair $(a,b)$ with $a<b$ we take the rank-$b$ test image of class $a$ and the rank-$a$ test
image of class $b$ (ranks in test-set order); within-class pairs use ranks 10 and 11. All
indices land within the first 233 test images. Pairs were never replaced after seeing results.
The animations show a fixed subset of ten cross-class pairs chosen by digit identity in advance
— (0,1), (2,3), (4,5), (6,7), (8,9), (0,8), (1,7), (3,5), (4,9), (2,6) — with every digit
appearing exactly twice; all 55 pairs enter the saved records and the summary statistics.

**Interpolation.** For each pair we run both images through the checkpointed model and take
their post-ReLU first-hidden activations $h_1^A, h_1^B$. Straight linear interpolation would
shrink the vector's norm in the middle (the midpoint of two nearly-orthogonal vectors is much
shorter than either), pushing the interpolant off-distribution for a reason unrelated to
plateaus. Following the post's `slerp_rescale` convention (and this branch's `slerp_path`), we
instead rotate the direction at constant angular speed and interpolate the norm linearly
("spherical interpolation"). With $u_A = h_1^A/\lVert h_1^A\rVert$,
$u_B = h_1^B/\lVert h_1^B\rVert$, and $\theta = \arccos(u_A \cdot u_B)$:

```math
h_1(\alpha) = \Bigl[(1-\alpha)\,\lVert h_1^A\rVert + \alpha\,\lVert h_1^B\rVert\Bigr]\;
\frac{\sin\bigl((1-\alpha)\theta\bigr)\,u_A + \sin(\alpha\theta)\,u_B}{\sin\theta},
\qquad \alpha \in \{0, \tfrac{1}{49}, \dots, 1\}.
```

We use 50 evenly spaced $\alpha$ values including both endpoints. Both sine coefficients are
non-negative, so the interpolant of two non-negative (post-ReLU) vectors stays non-negative — it
remains a valid $h_1$. Each $h_1(\alpha)$ is **patched** in at hidden layer 1 and propagated
through the rest of the network, recording $h_2$, $h_3$, and the logits.

**Relative endpoint distance $d(\alpha)$ — the primary metric.** It answers: is the output stuck
to one endpoint, or morphing smoothly? Raw distances are not comparable across checkpoints
(activation scales grow during training), so, following the post, we measure where the
propagated activation $x(\alpha)$ sits *between* the two endpoint outputs $x(0), x(1)$:

```math
d(\alpha) = \frac{\lVert x(\alpha) - x(0)\rVert_2}
{\lVert x(\alpha) - x(0)\rVert_2 + \lVert x(\alpha) - x(1)\rVert_2 + 10^{-10}}
```

$d$ runs from 0 (output equals endpoint $A$'s output) to 1 (equals endpoint $B$'s); the
$10^{-10}$ only guards the $\alpha=0$ division. **Unless a figure says otherwise, $d(\alpha)$ is
computed on the logits** — the closest analogue of the post's final-layer measurement; the
layerwise figure additionally shows $d$ at $h_2$ and $h_3$, which is saved at every checkpoint
too. Sanity checks: the patched $\alpha=0/1$ outputs reproduce the unpatched endpoint outputs
(max deviation 3.7e-4, float16 storage rounding); the vectorized interpolation matches the
reference `slerp_path` to 9.5e-7.

**Predicted class along the path.** For each of the 50 points we record the argmax of the
logits, shown as colored squares under each animation curve. This reveals *staircase* structure:
paths that pass through a third class's region on the way from $A$ to $B$.

**Plateau fraction — the one summary number.** Comparing emergence timing across three seeds
needs one number per checkpoint; the raw curves stay the primary evidence and no per-curve
"is it a plateau" threshold is imposed on them. We use the fraction of path points sitting near
either endpoint's output, averaged over the 45 cross-class pairs:

```math
\mathrm{PF}(t) = \frac{1}{45 \cdot 50} \sum_{p=1}^{45} \sum_{k=1}^{50}
\mathbf{1}\bigl[\,d_{t,p}(\alpha_k) < 0.1 \ \lor\ d_{t,p}(\alpha_k) > 0.9\,\bigr]
```

Reading it: the diagonal (no plateau) scores ≈ 0.20 — that is the floor, not zero, because the
diagonal itself spends its first and last tenth within 0.1 of an endpoint. A perfect two-plateau
step function scores 1.

### Baselines

**Initialization (step 0)** — the built-in baseline of the movie: whatever the curves show at
step 0 is what random networks produce (empirically: the diagonal). Any departure from it is
learned.

**Diagonal reference** $d(\alpha)=\alpha$ — the fully smooth, structure-free response, drawn
dotted in every curve figure. Its plateau fraction (≈ 0.20) is the floor for PF.

**Within-class control pairs** (same digit) — a path between two activations of the same class
should stay inside one region and cross no boundary. They calibrate what "no structure" looks
like under the identical protocol.

**Matched-random activations** (secondary control only) — in the radial-perturbation experiment,
each natural $h_1$ is compared to a random vector with the same L2 norm and the same number of
positive entries (post-ReLU sparsity). Flatness beyond what scale and sparsity mechanically
produce is then attributable to learned structure.

### How to read the plots

> **All curve figures share one format.** X-axis: interpolation position $\alpha$ from 0 (image
> $A$) to 1 (image $B$). Y-axis: relative endpoint distance $d(\alpha)$, from 0 (output = $A$'s
> output) to 1 (output = $B$'s output), **computed on the logits unless labeled otherwise**. The
> gray dotted diagonal is the no-structure reference. A plateau → boundary → plateau curve hugs
> 0, jumps across a narrow $\alpha$ interval, and hugs 1. Colored squares under a curve give the
> predicted digit at each path point (matplotlib `tab10` colors, digit 0–9). Heatmaps show the
> same $d(\alpha)$ as color (blue 0 → red 1) with $\alpha$ on x and training step on y.
>
> **Primary metric:** $d(\alpha)$ at the logits. **Summary number:** plateau fraction PF.
> **Secondary only** (final Results subsection): plateau contrast and stable-region count from
> the radial-perturbation control.

## Results

**Training context.** Train accuracy hits 1.0 at step 145 — the 1,000-image subset is memorized.
Test accuracy (first 2,000 test images) reaches its ~0.88 plateau by step ~70–120, peaks at
0.885, then drifts slowly down to 0.85–0.87 by 100k with visible late oscillations. Train loss
keeps decaying toward zero long after that; test loss is flat at ~0.02 from step ~200 onward.
Confidence keeps climbing after accuracy saturates. So almost all plateau development below
happens in the *post-generalization* phase.

![Training context (seed 0): train accuracy (blue), test accuracy (red), and confidence = mean max raw output (purple) vs training step (log-scale x).](plots/training_context.png)

**The movie: plateaus grow out of a featureless diagonal.** One frame per checkpoint, ten fixed
pairs, with the training step in the title. The insets track accuracy/confidence (top) and
train/test loss (bottom) with the current step marked.

![Main animation (seed 0, 205 frames): logit-space d(alpha) vs alpha for the ten preregistered pairs; squares: predicted class along the path. Insets: accuracy and confidence (top), train/test loss (bottom, log y), both vs step (log x).](plots/plateau_evolution.gif)

Static frames for reading without playback:

![Selected frames of the main animation (rows: steps 0, 100, 1,000, 20,000, 100,000): logit-space d(alpha) vs alpha for the ten pairs. The diagonal at step 0 becomes soft sigmoids by a few hundred steps and sharp staircases by tens of thousands.](plots/frames_selected_steps.png)

**The early phase on a linear time axis.** The main movie's schedule (every 500 steps) is
dominated by late training and compresses the beginning. To watch training start, we reran seed
0 deterministically and recorded **every 5 steps from 0 to 1,000** — 201 frames on a linear time
axis, covering the entire rise of accuracy and several hundred steps of flat accuracy after it
(bit-exact match to the main records at all 7 overlapping steps). The movie shows the diagonal
deforming within the first tens of steps; curves wobble rapidly while the loss falls fastest
(roughly the first 150–200 steps), then settle into stable soft sigmoids. The plateau fraction
tells the same story: 0.19 at step 0, 0.34 by step 100, then nearly frozen at ~0.37 through step
1,000 — far below its final 0.54–0.61. The early phase creates the soft structure and the
class layout of the path; the actual flattening into plateaus happens over the following tens of
thousands of steps.

![Early-phase animation (seed 0, steps 0–1,000, one frame per 5 steps, linear time): logit-space d(alpha) vs alpha for the ten pairs; insets as in the main animation but with a linear step axis.](plots/plateau_evolution_early.gif)

![Early-phase heatmap: logit-space d(alpha) (color, blue 0 to red 1) vs alpha (x) and training step (y, linear, one row per 5 steps, 0–1,000) for the ten pairs and two within-class controls. Rapid flicker below ~200 steps, then stable soft structure.](plots/plateau_early_heatmap.png)

**Gradual consolidation, wandering boundaries.** The full-run heatmap shows both effects:
plateaus (saturated blue/red) expand gradually over tens of thousands of steps, and the white
boundary stripe keeps shifting position throughout training, including large late relocations.
Within-class pairs (rightmost panels) never develop a comparable two-plateau structure.

![Full-run heatmap: logit-space d(alpha) (color) vs alpha (x) and checkpoint (y; rows 0,10,30,100,300 then every 500 steps) for the ten pairs and two within-class controls.](plots/plateau_training_heatmap.png)

**Successive layers sharpen the same transition.** At a fixed checkpoint the transition gets
sharper the deeper you measure: $d$ at $h_2$ is smoothest, at $h_3$ sharper, at the logits
sharpest. Early in training all layers are near-diagonal; late in training the deep layers have
hard plateaus while $h_2$ still changes smoothly. The discreteness is built up depth-wise, not
inherited from layer 1. This is the only figure where $d$ is shown at layers other than the
logits.

![Layerwise d(alpha) vs alpha at h2 (green), h3 (orange), and logits (blue), for pair 0-1 (solid) and the mean over all 45 cross pairs (dashed), at steps 100, 5,000, 100,000.](plots/layerwise_selected_steps.png)

**Consistent across seeds, with no synchronized transition.** Plateau fraction rises from the
~0.20 floor through ~0.34 (step 100) and ~0.4 (step 10k) to 0.54–0.61 (step 100k) in all three
seeds — the same gradual shape with no common jump time. The late per-checkpoint fluctuations
are boundary relocations, not protocol noise (the records are exact). At 100k, 22–29 of the 45
cross-class pairs show a textbook plateau→boundary→plateau curve, many as multi-step staircases
through third-class regions. Per-seed numbers are tabulated in RESULTS.md.

![Seed comparison. Left: plateau fraction (y) vs training step (x, log) for seeds 0-2. Right: all 45 cross-pair logit-space d(alpha) curves overlaid at steps 0, 100, 1,000, 20,000, 100,000 (columns) for each seed (rows).](plots/seed_comparison.png)

**Late boundary flips are fast but not instantaneous.** The largest adjacent-frame change in the
movie (pair 5→6, between steps 82,000 and 82,500, seed 0) was rerun deterministically with
records every 50 steps. The boundary sits near $\alpha\approx0.95$ until step ~82,300, then
sweeps to $\alpha\approx0.1$ by step ~82,450 — a ~150-step relocation, at a point where the
model has held train accuracy 1.0 for ~82,000 steps.

![Dense zoom: logit-space d(alpha) vs alpha for pair 5-6 at every 50 steps from 82,000 (dark) to 82,500 (yellow), seed 0.](plots/dense_zoom.png)

**Endpoint and control checks.** At step 100k, 9 of the 90 cross-pair endpoints are
misclassified — consistent with 0.85 test accuracy (endpoints were fixed in advance, never
filtered). Within-class controls behave as predicted: 8/10 keep a single predicted class along
the whole path, and the two exceptions (2→2, 7→7) each contain one endpoint the model genuinely
misclassifies, so those paths really do cross a decision boundary.

**Perturbation control (secondary).** An earlier study in this direction probed the same
checkpoints from a different angle. It perturbed natural $h_1$ activations radially and compared
the output response to matched-random activations (13 log-spaced checkpoints, 3 seeds). Its
**plateau contrast** — flatness of natural activations relative to the matched-random control; 0
= no learned flatness, →1 = perfectly flat — rises from 0.42 at step 100 to 0.80 at step 100k,
while test accuracy declines. Its validated **stable-region count** converges to 10, one region
per predicted digit, by step ~300. Flatness tracks confidence, not correctness: at 100k the
contrast is 0.73 for confident-wrong points vs 0.85 for confident-correct vs 0.49 for uncertain.
Full definitions live in `experiments/analyze_sweep.py`; these two quantities appear nowhere
else in this report. The timing agrees with the interpolation movie: local robustness also lags
generalization.

![Perturbation control: plateau contrast (left, y) and validated stable-region count (right, y) vs training step (x, log; mean of 3 seeds, band = seed min-max).](plots/plateau_contrast_and_region_count.png)

## Conclusion

Activation plateaus in this MLP are entirely learned, and learned slowly. They are absent at
initialization, take soft form during the few hundred steps in which the network fits the data,
and only become genuinely flat regions with sharp, staircase boundaries over tens of thousands
of post-generalization steps — while boundaries continue to relocate in fast ~150-step events
even past step 80,000. The within-class controls, the layerwise sharpening, and the
matched-random perturbation control corroborate the same picture: training first solves the
task, then keeps carving the representation into increasingly discrete, confidently-held regions
— including around confidently *wrong* predictions.

**Limitations.** One architecture (depth-4, width-200 MLP), one dataset (1,000-image MNIST
subset, MSE-on-one-hot training), three seeds. Confirmation seeds use 2,000-step checkpoint
spacing rather than 500. The plateau fraction depends on its 0.1/0.9 margins (used only as a
cross-seed timing summary; all raw curves are saved and shown). Endpoints come from test images
the model may misclassify — deliberate, but a few "cross-class" paths therefore connect regions
of the same predicted class. The dense zooms (early phase, one late flip) cover seed 0 only.
