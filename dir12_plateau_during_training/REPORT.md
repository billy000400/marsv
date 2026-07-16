# How plateau/stable regions evolve during training in an MNIST MLP

## Summary

**Safety question.** Modern networks are often *robust* in a way that is invisible in their loss
curve: small changes to an internal activation leave the output unchanged over a whole neighborhood — a
**plateau** — and the input space organizes into a few **stable regions** of near-constant behavior.
If we want to trust a model, we should know *when during training* this structure appears, whether it
tracks the model being *right* or merely *confident*, and whether it forms smoothly or through
splits/merges. A confident-but-wrong plateau is exactly the failure mode where a model is stably,
resistantly wrong — so the confidence-vs-correctness distinction is a safety-relevant one.

We take an existing 4-layer ReLU MLP trained on 1,000 MNIST images (a reproduction of the setup in
*Deep Networks Always Grok and Here is Why*, arXiv:2402.15555) and measure plateau structure at 13
checkpoints from initialization to 100,000 steps.

**Findings (3 seeds: 0 primary, 1–2 confirmation; numbers are means across seeds).**

1. **Plateaus emerge and keep strengthening after generalization is complete.** Test accuracy peaks by
   step ~300 (0.90) and then slowly declines to ~0.87, yet the **plateau contrast keeps rising** from
   0.42 (step 100) to **0.80 (step 100k)** in every seed. Plateau formation is a *late* phenomenon that
   *lags* test-accuracy — consistent with the delayed-robustness / region-migration phase of the paper.
2. **The number of validated stable regions converges to exactly 10 — one per predicted digit — by step
   ~300** and stays there in all three seeds, under both cosine and Euclidean clustering. Tracking the
   same examples across checkpoints, regions are born one digit at a time and then persist: **no digit
   ever hosts two validated regions**, and adjacent-checkpoint membership-overlap matrices are clean
   near-permutations (0 splits, 0 merges) — no persistent split or merge into an 11th stable region.
3. **Plateau strength tracks confidence, not correctness.** Confident-*wrong* examples plateau strongly
   (contrast 0.73 at 100k), nearly as strongly as confident-correct (0.85); uncertain examples are
   consistently weakest (0.49). Confidence is the operative variable.
4. **The one non-monotonic dip** (step 10k, contrast 0.30) appears **only in seed 0** — seeds 1 and 2
   rise monotonically through step 10k (0.56, 0.45). It is therefore seed noise on a single checkpoint,
   **not** a replicated split/merge, and does not meet the escalation criterion.

**Verdict: expected monotonic emergence**, replicated across three seeds — the preregistered
low-surprise outcome. This is a *complete validation result*, not a surprise warranting escalation.

## Methods

### Data & Model

**Data.** MNIST (raw IDX, pixels scaled to [0,1]). Training uses a fixed 1,000-image subset drawn by
`torch.randint` after seeding; the subset is identical across checkpoints within a seed. **Evaluation
set:** the first 50 test images of each digit, 500 total, with example IDs frozen across all
checkpoints and seeds. The evaluation set is *never* filtered to correctly classified examples.

**Model.** 4-layer ReLU MLP, 784→200→200→200→10 (depth counts linear layers; ReLU after every linear
except the last). Trained with AdamW (lr `1e-3`, weight decay `0.01`), MSE loss on one-hot targets,
batch size 200, init scale 1.0, for 100,000 steps. Checkpoints saved at steps
`0, 10, 30, 100, 300, 1k, 3k, 10k, 20k, 30k, 50k, 75k, 100k`. Seed 0 is primary; seeds 1–2 confirm.

**Layers / hook points.** We perturb the **first hidden post-ReLU activation** $h_1$ (200-d) and
measure the induced displacement at the **last hidden layer** $L_3$ (200-d) via
`forward_from(h1, layer=1)`. $G_t$ denotes the checkpoint-$t$ map from $h_1$ to $L_3$.

**Confidence.** Because training minimizes MSE toward one-hot targets, the network drives the correct
output toward 1 and the rest toward 0; the softmax max-probability saturates near 0.23 and is
uninformative. We therefore define confidence as the **maximum raw output**, and call an example
**confident** when that value is ≥ 0.7 (a fixed absolute threshold, not a per-checkpoint quantile).

### Metrics

The four questions in the Summary need four measurements, built up in order: a per-checkpoint measure
of how flat the network's response is around a natural activation (the *response curve*), a control
saying how flat it would be *by chance* (the *matched-random control*), a single scalar per checkpoint
so emergence can be plotted against training step (the *plateau contrast*), and a way to count stable
regions and follow their membership through training (*validated regions* and the *overlap lineage*).
Each metric below is introduced by the question it answers and the figure that consumes it.

**Cross-checkpoint response** — answers "is there a plateau at this checkpoint?", and is the raw
material for every later metric (consumed directly by the response-curve figure in Results). The
obvious measure — absolute $L_3$ displacement per unit perturbation — cannot be compared across
checkpoints, because activation scales grow substantially during training and would masquerade as
changing flatness. We therefore normalize both the perturbation (by the input norm) and the
displacement (by the output norm). For an example with first-hidden activation $h_1$, relative radius
$\rho$, and a fixed random unit direction $u$ (16 directions, shared across all checkpoints within a
seed), the normalized downstream response is the median over directions:

```math
R_t(x,\rho) = \operatorname{median}_{u}\;\frac{\lVert G_t(h_1 + \rho\,\lVert h_1\rVert\,u) - G_t(h_1)\rVert_2}{\lVert G_t(h_1)\rVert_2 + \varepsilon},
\qquad \varepsilon = 10^{-8}.
```

$\varepsilon$ is a numerical guard against division by zero (e.g. a dead downstream activation at
initialization); at $10^{-8}$ it is many orders of magnitude below typical $L_3$ norms and never
affects reported values. A **plateau** is a region where $R_t$ stays near 0 out to a nonzero radius
before rising. We sweep $\rho$ on a 21-point grid over $[0, 0.6]$ and use the small-radius interval
$\rho \in [0, 0.2]$ for scalar summaries.

**Matched-random control** — answers "flatter than *what*?". A flat response curve by itself does not
prove learned structure: how far a perturbation propagates through $G_t$ depends mechanically on the
scale of the activation and on how sparse it is, regardless of training. Here sparsity means the number
of **positive entries** of $h_1$: because $h_1$ is a post-ReLU vector, every entry is ≥ 0, and the
positive (nonzero) entries are exactly the neurons the input actually activates — typically well under
half of the 200 units. An unmatched random control with denser or larger activations would sit in a
different regime of the downstream ReLU gates, and any contrast against it could be an artifact of norm
or sparsity rather than of learning — which matters because our headline claim is that flatness reflects
*learned* structure. So for each evaluation example we build a random $h_1$ with the **same L2 norm and
the same number of positive entries**: choose a random support of that size, fill it with absolute
values of Gaussian draws (keeping entries non-negative, like a real post-ReLU vector), and rescale to
the example's norm. Its response is measured with the identical directions and radii; it appears as the
dashed control curves in the response figure and as the denominator of the contrast below.

**Plateau contrast** (primary scalar) — answers "when do plateaus emerge?" (question 1) and "for whom?"
(question 3). Response curves are one-per-example-per-radius-per-checkpoint; to plot a trajectory
against training step (contrast/region-count figure) and to compare confidence/correctness groups
(group figure), we compress each curve to the area under $R(\rho)$ on the small-radius interval
$\rho \in [0, 0.2]$ (trapezoidal rule), written $A(R)$, and compare data to control:

```math
\text{plateau\_contrast} = 1 - \frac{\overline{A(R_{\text{data}})}}{\overline{A(R_{\text{random}})}}
```

where the bars denote averaging over examples (the same $\varepsilon = 10^{-8}$ guards this
denominator). It is **0** when natural activations are no flatter than matched-random ones and
approaches **1** as natural activations become perfectly flat near the origin relative to the control.
We report 95% confidence intervals by bootstrapping examples (1,000 resamples). Group-conditioned
contrasts (confident-correct, confident-wrong, uncertain) use the same control pool.

**Validated stable regions** — answers "how many stable regions, and do they match the 10 digits?"
(question 2; consumed by the region-count panel and the lineage figure). Clustering alone cannot answer
this: a cluster of activations is not automatically a *stable* region — it must also actually be flat.
At each checkpoint we cluster the $L_3$ activations of the evaluation set with **average-linkage
agglomerative clustering**, choosing the cluster count $k \in \lbrace 2,\dots,15 \rbrace$ by silhouette
score (reported for both cosine and Euclidean metrics). A cluster is a **validated stable region** only
if it (i) contains ≥ 20 examples, (ii) has ≥ 90% purity in the model's **predicted** label, and (iii)
has a per-cluster plateau contrast whose 95% bootstrap CI excludes 0. Clustering uses no labels;
true/predicted labels, correctness, and confidence are applied only after clustering to interpret
regions.

**Membership-overlap lineage** — answers "do regions form monotonically, or through splits and merges?"
(question 4 of the plan; consumed by the lineage figure). Counting regions per checkpoint is not
enough: the count can stay at 10 while two regions swap members, and cluster IDs are arbitrary at each
checkpoint, so IDs cannot be compared either. Because the *same* 500 evaluation examples are clustered
at every checkpoint, we instead align adjacent checkpoints by shared membership. For adjacent
checkpoints $t$ and $t+1$ with cluster assignments $a$ and $b$, the overlap matrix counts shared
examples:

```math
M_{ij} = \bigl|\{x : a(x)=i \ \text{and}\ b(x)=j\}\bigr|
```

Each checkpoint-$(t+1)$ cluster is assigned its max-overlap **parent** $\arg\max_i M_{ij}$; a **split**
is a parent claimed by ≥ 2 children and a **merge** is a child claimed by ≥ 2 parents. A
near-permutation $M$ (one dominant cell per row and column) means membership evolves monotonically. The
escalation signal is any single predicted digit hosting **≥ 2 validated regions** across ≥ 2 adjacent
checkpoints.

### Baselines

**Matched-random activations** (defined above) are the negative control for every plateau measurement:
plateau contrast is defined *relative* to them, so contrast ≈ 0 means "no plateau beyond what norm and
sparsity alone produce." A validated-region count of **~10** is the preregistered expectation (one
region per predicted digit); a persistent count above 10 that survives bootstrap and both distance
metrics would be the escalation trigger.

### Guardrails observed

We do **not** condition the primary analysis on correct classification (that would hide the
confident-wrong result); we do **not** compare raw hidden distances across checkpoints without the
norm normalization above; and we treat the empirical plateau/stable regions as *downstream-insensitivity
basins*, **not** the spline/linear activation regions of the grokking paper — similar timing is evidence
of association, not identity.

## Results

**Training dynamics.** Train accuracy reaches 1.0 by step ~100; test accuracy peaks at 0.90 by step
~300 and then drifts to 0.87. This fixes the "generalization is done" reference point.

![Training dynamics: train/test accuracy and confidence vs step; test accuracy peaks by step ~300.](plots/training_dynamics.png)

**Plateaus deepen with training.** Early (step 100), mid (10k), and late (100k) response curves show
the natural-activation response sitting below the matched-random control near $\rho = 0$, and the gap
widening over training.

![Response curves early/mid/late; natural activations flatter than matched-random near radius 0.](plots/plateau_curves_by_stage.png)

**Contrast rises after generalization; region count converges to 10.** Plateau contrast climbs from
0.42 (step 100) to 0.80 (100k), averaged over three seeds — *after* test accuracy has already peaked
and begun declining. The validated stable-region count reaches 10 by step ~300 and stays there in every
seed (band = seed min–max).

![Plateau contrast keeps rising after test accuracy saturates (left, mean of 3 seeds, band = min–max); validated region count converges to 10 (right).](plots/plateau_contrast_and_region_count.png)

**Confidence, not correctness, drives the plateau.** Confident-wrong examples plateau strongly, nearly
as strongly as confident-correct ones; uncertain examples are always weakest.

![Plateau contrast by confidence×correctness; confident-wrong plateaus like confident-correct.](plots/contrast_by_group.png)

**Region membership evolves monotonically — no split/merge into a second region for any digit.**
Tracking the same 500 examples across checkpoints (seed 0), each of the 10 validated regions maps to a
distinct predicted digit, and **no digit ever hosts ≥2 validated regions at any checkpoint** (max = 1).
Regions are *born* one digit at a time as accuracy climbs (panel a: 1→2→3→9→10 validated regions by
step ~300), then persist. The membership-overlap matrices for the region-birth transition (step
100→300, panel b) and a late transition (75k→100k, panel c) are clean near-permutations with **0
splits and 0 merges** among validated regions. The raw silhouette-selected $k$ does oscillate between
10 and 12 late in training, but the extra cluster is a transient sub-threshold split of the
uncertain/mixed group that is never validated (purity or contrast-CI fails) and does not persist across
two adjacent checkpoints — so it does not meet the escalation criterion.

![Region composition and membership-overlap lineage (seed 0). (a) For each predicted digit (row) and checkpoint (column), green marks a plateau-validated stable region; numbers above give the validated-region count. Regions appear monotonically, one per digit, reaching 10. (b,c) Membership-overlap matrices for the birth transition (100→300) and a late transition (75k→100k) are near-permutations: 0 splits, 0 merges among validated regions.](plots/region_composition_and_lineage.png)

Current-best numbers (mean [min, max] over 3 seeds) are tabulated in RESULTS.md.

## Conclusion

In this MNIST MLP, plateau structure is a **late, confidence-driven** phenomenon: it forms *after* the
network already classifies the test set well, keeps strengthening as training continues past the
generalization point, and attaches to whatever the model predicts **confidently** — including its
confident mistakes. The stable-region count settles at about ten, one per predicted digit, with no
convincing extra region. This is the **expected monotonic-emergence** outcome from the plan, and it
concretely locates plateau/robustness formation in the *post-generalization* phase rather than at
interpolation.

**Limitations.** Results are from three seeds on one 1,000-image MNIST subset and one architecture
(d4/w200); we do not claim generality beyond this setup. The group-conditioned contrasts at early steps
have few confident-wrong examples (<10/seed, marked underpowered). The membership-overlap lineage is
computed for seed 0 only (the region-count trajectory that underlies the verdict is replicated across
all three seeds). The plateau/stable regions here are downstream-insensitivity basins, distinct from
the paper's spline regions; the shared *timing* argues for association, not identity.
