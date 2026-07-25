# Confident predictions of a digit the network has never seen there

**Direction 12.1 — stable third-class predictions on MNIST activation paths**

> Final, presentable, current-best only (no history — see CHANGELOG.md).

## Summary

If you take the internal activations of a trained image classifier for a real "6" and a real "9" and
slide smoothly from one to the other, the network does not simply switch from "6" to "9". Over a wide
stretch of the way across it confidently answers **"8"** — a digit that is neither endpoint. This
report asks whether that third answer is trustworthy: **when the network says "8" in the middle of an
interpolation, are its internal activations actually similar to the activations that real 8s produce?**

That matters for AI safety because a model's output is often the only signal a monitor gets. If a
network emits a confident, *stable*, reproducible class label in a part of its internal state space
that contains no examples of that class, then the label is not evidence about what the input resembles
— and confidence-based monitoring in that region is measuring nothing.

We ran a complete census: all 45 unordered digit pairs, 100 fixed image pairs each, 50 interpolation
points per path, at three independently trained networks. Three findings:

1. **The phenomenon is real and common.** At the primary network (seed 0), **19 of 45** digit
   transitions produce a *stable* third-class prediction — the same third digit appears as an
   unbroken run in at least half of the 100 image pairs, with a median run of 4–16 of the 50
   interpolation points. This is not a one-pair curiosity: it survives dropping any single pair.
2. **The prediction is not backed by the activations.** Pooling all 14,700 interpolation points that
   sit inside a stable third-class segment, only **2.5%** are inside the real activation region of the
   digit they are predicted to be. The typical segment point is **1.5× to 3.7×** farther from that
   digit's mean activation than 95% of real held-out images of it, and **1.2× to 2.1×** farther than
   the 95th percentile of the *nearest* of all ten digits. These points are outside every digit's
   usual activation spread. The verdict for all 19 transitions is **"prediction only"**, not
   "activation-region match".
3. **The identity of the third digit is a property of the trained network, not of the digits.** Seed 0
   almost always answers **7 or 8**; seed 1 almost always answers **1**; seed 2 answers **2 or 8**.
   Only one transition — **6→9 with third digit 8** — is labelled stable at all three seeds *and*
   names the same third digit at all three. Each network has its own private "default answer" for the
   space between classes.

4. **Later layers do not rescue the prediction either.** Repeating the identical measurement at the
   second and third hidden layers, on the same frozen segments, the fraction of segment points inside
   the predicted digit's region is 2.5% at `h1`, 10.6% at `h2` and 0.2% at `h3`. The apparent `h2`
   improvement is an illusion: at `h2` a segment point falls inside **5.8 of the 10** digit regions at
   once, so "inside z" stops being a discriminating test there — and even at `h2` the predicted digit
   is the *nearest* region for only 11.7% of segment points, no better than picking one of the eight
   non-endpoint digits at random (12.5%).

A separate label, the **sub-plateau** (a flat intermediate shelf in the distance curve), turns out to
be much rarer: 6 of 45 transitions at seed 0, and every one of those 6 is also a stable third-class
case. No transition has a sub-plateau without a stable third-class prediction. The two phenomena are
therefore not the same thing, and third-class prediction is by far the more common of the two.

All four preregistered controls pass, so the negative result in finding 2 is not an artifact of a
broken distance measure.

---

## Methods

### Data & model

**Dataset.** MNIST handwritten digits (28×28 greyscale, flattened to 784 inputs, scaled to [0,1]).

**Model.** A 4-layer fully connected network (multi-layer perceptron, "MLP"),
784 → 200 → 200 → 200 → 10, with a ReLU nonlinearity after every layer except the last. This is
direction 12's model, trained from scratch on the full 60,000-image MNIST training set with AdamW and
a mean-squared-error loss on one-hot targets. We **do not retrain anything**: we load direction 12's
final checkpoints at training step 30,000 for seeds 0, 1 and 2
(`results/full_mnist_from_scratch/seed_{0,1,2}/ckpts/step30000.pt`; SHA-256 recorded in
`results/s1_census.npz` and `results/s1_classification.json`). Seed 0 is the primary network; seeds 1
and 2 are used only to check whether the seed-0 labels replicate.

**Hook point.** The interpolation always happens at **`h1`**, the 200-dimensional post-ReLU output of
the *first* hidden layer. We read `h1` out, build a path between two images' `h1` vectors, write the
path back in at `h1`, and let the remaining three layers run forward to the logits. Results 1–7
compare activations at `h1`; Result 8 repeats the comparison at `h2` and `h3` (the second and third
post-ReLU hidden layers, also 200-dimensional) for the same paths.

**Interpolation path.** For two images with first-layer activations `h_A` and `h_B` we use direction
12's frozen 50-point spherical interpolation ("SLERP"): the *direction* follows the great circle
between the two unit vectors, and the *length* is interpolated linearly. Writing
`u_A = h_A / ||h_A||`, `theta = arccos(u_A · u_B)`, and `t` running over 50 equally spaced values from
0 to 1, the interpolated activation is:

```math
h(\alpha_t) \;=\; \underbrace{\Big[(1-t)\lVert h_A\rVert + t\lVert h_B\rVert\Big]}_{\text{length}}
\cdot
\underbrace{\frac{\sin\big((1-t)\theta\big)\,u_A + \sin\big(t\theta\big)\,u_B}{\sin\theta}}_{\text{direction}}
```

We call the position along the path `alpha`, running from 0 (the first image) to 1 (the second). A
plain straight line would shrink the activation norm in the middle and confound "the network is
confused" with "the activation got small", which is why direction 12 fixed this norm-preserving form
and why we reuse it unchanged.

**Endpoint image pairs (the 100-pair bank).** A single image pair proves nothing about a *digit
transition*, so every number in this report is an aggregate over 100 fixed pairs. For each unordered
digit pair (a, b) with a < b, we pair the rank-*i* class-a test image with the rank-*i* class-b test
image, i = 0…99, where "rank" is the order of appearance inside the **first 2,000 MNIST test images**
(direction 12's frozen endpoint pool). That is 45 transitions × 100 pairs = **4,500 paths**, each with
50 interpolation points, evaluated identically at every seed. The exact index pairs are saved once in
`results/s1_census.npz` and reused for all three seeds. Both endpoint digits are classified correctly
in at least 94% of the 100 pairs for every transition (overall endpoint accuracy 98.7% at seed 0,
97.9% at seed 1, 98.0% at seed 2), and every classification below is reported both on all 100 pairs
and on the correctly-classified-endpoint subset.

**Activation reference sets (used only in Results 4).** For each digit c we summarize what real
class-c activations look like at `h1` using **2,000 real training images** of digit c (mean and
coordinate-wise standard deviation). We then calibrate "how far is normal" on **700 real held-out test
images** of digit c drawn from `test[2000:]` — deliberately disjoint from the endpoint pool
`test[:2000]`, so calibration images are never path endpoints. **Interpolation points are never used
to define a region.**

### Metrics

Four questions drive the four metrics below, in order: *where along the path is the network?*,
*does it name a third digit, reproducibly?*, *does the distance curve flatten in the middle?*, and
*is the named digit's activation region actually there?*

---

**Relative endpoint distance `d(alpha)` — where along the path the network's output has moved.**
Position along the path (`alpha`) is not the same as progress in the network's output. We want a
single number per interpolation point saying "how far from the first endpoint, relative to the total",
measured in the *logit* space (the network's 10 raw output scores) rather than in the input space.
It is 0 at the first endpoint, 1 at the second, and — because it is a ratio — it is comparable across
transitions with very different absolute logit scales:

```math
d(\alpha) \;=\; \frac{\lVert x(\alpha) - x(0)\rVert}{\lVert x(\alpha) - x(0)\rVert + \lVert x(\alpha) - x(1)\rVert}
```

Here `x(alpha)` is the 10-dimensional logit vector produced by the interpolated activation. Reading
it: values near 0 mean the output still looks like the first endpoint; near 1, like the second; a long
flat stretch means the output is *not moving* even though the activation is. This is the curve plotted
in Result 2 and the input to the sub-plateau rule. It is direction 12's definition, unchanged.

---

**Third-class segment, prevalence, and the "stable third-class" label — does the network name a third
digit, reproducibly?** The obvious thing to report — "the network predicted a 7 somewhere on this
path" — is too weak: one stray interpolation point on one image pair would trigger it. So we require
a *contiguous run* and we require it to *recur across the 100 pairs*. A **third-class segment** is a
maximal run of consecutive interpolation points whose argmax prediction is a digit that is neither
endpoint digit. For a candidate third digit z, let `n_z` be the number of the 100 paths on which z
appears as such a run, and let `M_z` be the median, over exactly those paths, of the *longest* z-run
length (in interpolation points, out of 50). Transition (a, b) is labelled **stable third-class** iff:

```math
\exists\, z \notin \lbrace a,b \rbrace \;:\; n_z \ge 50 \quad\text{and}\quad M_z \ge 3
```

The digit z achieving this with the largest `n_z` is the **dominant third digit**. `n_z / 100` is the
**prevalence** and is quoted as a percentage. Reading it: prevalence answers "on what fraction of
image pairs does this happen at all?" and `M_z` answers "when it happens, is it a real stretch of the
path or a single flickering point?". Both thresholds were frozen before any census output was viewed
(recorded in JOURNAL.md), and Result 1 reports a sensitivity table at prevalence 25%, 50% and 75%.

---

**Sub-plateau (shelf) rule — does the output curve flatten in the middle?** A sub-plateau is
visually obvious but needs a number, and a derivative-based rule would need a per-transition threshold
(rejected for exactly that reason; see JOURNAL.md). Instead we apply a flatness rule directly to the
100-pair **mean** curve `m` (50 values, `m[0]` at alpha=0). A **flat run** is a maximal contiguous
index range [i, j] with:

```math
\max_{i \le k \le j} m_k \;-\; \min_{i \le k \le j} m_k \;\le\; 0.05
\qquad\text{and}\qquad
j - i + 1 \;\ge\; 5
```

i.e. the mean curve moves by at most 0.05 across at least 5 of the 50 interpolation points (≥10% of
the path). A flat run counts as a **sub-plateau** only if it is *intermediate* — separated from the
two endpoint plateaus — which we require as:

```math
0.15 \;\le\; \frac{1}{j-i+1}\sum_{k=i}^{j} m_k \;\le\; 0.85
```

Transition (a, b) is labelled **stable sub-plateau** iff at least one flat run is intermediate. We
report the alpha interval and the `d` level of the widest such run. Reading it: this label says the
network's output *pauses* in a state that is neither endpoint, which is a different claim from "it
names a third digit" — a curve can flatten while the argmax stays on an endpoint digit, and a third
digit can win briefly without the curve flattening. Result 3 shows how the two labels relate.

---

**Robustness of a label — is it driven by a few unusual image pairs?** An average over 100 pairs can
still be dominated by outliers, so each seed-0 label is re-derived under resampling. **Leave-one-out
(LOO):** drop each of the 100 pairs in turn, recompute the mean curve (or the prevalence counts), and
re-apply the rule; we report how many of the 100 re-derived labels match. **Bootstrap:** draw 1,000
resamples of the 100 pairs with replacement and report the fraction of resamples whose label matches
the full-sample label. The preregistered bar is LOO = 100/100 and bootstrap ≥ 0.90. This is a
stability check on the aggregate, not a significance test.

---

**Normalized activation distance and the "inside the region" criterion — is the named digit's
activation region actually there?** This is the core question. We need to compare an interpolated
activation to what *real* digit-c activations look like, in the full 200-dimensional `h1` space, with a
scale that is comparable across digits. Raw Euclidean distance will not do: post-ReLU coordinates have
wildly different spreads, and digits differ in how tightly clustered they are, so an unnormalized
distance would just rank digits by cluster size. We therefore z-score each coordinate by that digit's
own spread and take the root-mean-square across coordinates. For an activation `u` and digit c, with
`mu_c` and `s_c` the mean and coordinate-wise standard deviation over the 2,000 real training images
of digit c, and d = 200 coordinates:

```math
D_c(u) \;=\; \sqrt{\frac{1}{d}\sum_{j=1}^{d}\left(\frac{u_j - \mu_{c,j}}{s_{c,j}}\right)^{2}}
```

Because ReLU makes some coordinates identically zero within a class, `s_c,j` is floored at 1% of the
same coordinate's standard deviation pooled over all ten digits, which stops a dead coordinate from
producing an infinite distance.

`D_c` is still in arbitrary units, so we calibrate it against *real* images. Let `q_c` be the 95th
percentile of `D_c` over the 700 held-out real test images of digit c (values 1.45–1.87 across the ten
digits). The reported quantity is the **normalized distance ratio**:

```math
R_c(u) \;=\; \frac{D_c(u)}{q_c}
```

Reading it: `R_c < 1` means "this activation is closer to digit c's mean than 95% of real held-out
digit-c images are" — i.e. it lies inside digit c's usual spread. `R_c = 2` means it is twice as far
from digit c's mean as that 95th-percentile real image. Lower is more digit-c-like.

An interpolation point in a segment predicted as z is counted as **inside the real-z activation
region** only if it satisfies *both* conditions:

```math
R_z(u) < 1 \qquad\text{and}\qquad R_z(u) = \min_{0 \le c \le 9} R_c(u)
```

The second condition matters: a point could be inside a loose digit's spread while still being nearer
to some other digit, which would not support the claim that the network is "seeing a z". We report the
fraction of segment **points** meeting both conditions, and the fraction of **paths** whose segment
has a majority (>50%) of its points meeting both. A transition is called an **activation-region
match** only if most segment points and most paths qualify; **prediction only** if essentially none
do; **mixed** in between. Results 4 and 5 consume these.

---

**Region occupancy — is "inside a region" even a discriminating test at this layer?** The
inside-region criterion is only meaningful if the ten regions are reasonably distinguishable. If a
layer's regions overlap heavily, a point can be inside many of them at once and "inside z" tells you
nothing. So alongside the criterion we count, for each segment point `u`, how many of the ten digit
regions contain it:

```math
N(u) \;=\; \big|\lbrace c : R_c(u) < 1 \rbrace\big|
```

Reading it: `N = 1` is the ideal (the point is inside exactly one digit's region); `N = 0` means it is
outside all ten; `N` near 10 means the test is vacuous at that layer. Result 8 uses this to interpret
the later-layer numbers, and it is the reason a higher "inside" fraction at `h2` is *not* evidence of
a match.

---

**Later-layer follow-up (`h2`, `h3`).** PLAN.md preregisters this as conditional: run it only if
stable third-class segments are predicted as z but are *not* close to real z activations at `h1`.
Result 4 establishes exactly that, so it is unlocked. The follow-up repeats every definition above
unchanged — same 100-pair bank, same frozen Stage-1 segments, same std floor, same held-out
95th-percentile calibration, same inside-region criterion — and changes only the hook point at which
activations are compared. Note the interpolation is still *constructed and patched at `h1`*; `h2` and
`h3` are that path's images under the remaining layers, which is what "does similarity appear only
after later layers?" means. Controls C1 and C3 are recomputed at each layer. `h1` numbers are
recomputed by the same code path so the three layers are strictly comparable.

### Baselines and controls

There is no external baseline method to beat here — the question is whether one measurement (the
prediction) is corroborated by another (the activation region). What the study needs instead is proof
that the region measurement itself works. Four preregistered controls, all reported in Result 6:

**C1 — real held-out images land in their own region.** Take the 700 held-out real test images per
digit (never used to build the means) and score them with the same `R_c`. A working measure puts them
inside their own region and nearest to their own digit. By construction of `q_c`, about 95% should
satisfy `R_own < 1`; the informative part is whether `argmin_c R_c` equals the true label.

**C2 — endpoint-predicted portions land in the endpoint region.** For each stable transition, take the
interpolation points predicted as endpoint digit a (or b) and ask what fraction are inside digit a's
(or b's) region by the same both-conditions criterion. Reported twice: over *all* such points, and
restricted to the **endpoint plateau** (the first six or last six interpolation points, alpha ≤ 0.1 or
≥ 0.9). If even the endpoint plateau failed this, the region measure would be useless.

**C3 — within-digit interpolation stays inside.** For each digit c, build 100 *within-digit* paths
(digit c to digit c) by pairing the rank-i class-c test image with the rank-(i + n/2) class-c test
image inside the endpoint pool, and measure the fraction of all 5,000 interpolation points with
`R_c < 1`. Interpolating between two images of the same digit should not leave that digit's region.

**C4 — reversing a path only reverses alpha.** Build the path from B to A, flip its order, and compare
element-wise to the path from A to B; report the maximum absolute deviation relative to the largest
activation. This confirms that presenting every pair in the low-digit→high-digit orientation loses
nothing.

### Figure conventions

Every figure below encodes each series with a hatch, linestyle or marker *in addition to* its hue, and
no figure uses a red-versus-green contrast; the ten digit classes ride on a light-to-dark sequential
ramp with the digit printed on the plot wherever a band or cell is large enough. Captions therefore
name a series by its non-colour channel ("the dashed triangle line"), never by its colour. This keeps
the figures readable in grayscale and for readers with colour-vision deficiency. The palette and
helpers live in `experiments/cvd_style.py`.

---

## Results

### Result 1 — 19 of 45 transitions have a stable third-class prediction; 6 have a sub-plateau

Applying the two frozen rules to the seed-0 census sorts the 45 transitions into four categories:

| category | count (of 45) |
|---|---|
| both a stable third-class prediction and a stable sub-plateau | 6 |
| stable third-class prediction only | 13 |
| stable sub-plateau only | **0** |
| neither | 26 |

![Ten-by-ten matrix of all 45 digit transitions at seed 0. Rows are endpoint digit a, columns are endpoint digit b; the diagonal is blank because within-digit pairs are not transitions. Each cell's fill and hatch pattern give its category, as spelled out in the legend below the matrix: unhatched pale = neither, diagonal hatch "//" = stable third-class prediction only, back-diagonal hatch "\\" = stable sub-plateau only (this category never occurs), cross hatch "xx" = both. The number printed inside a cell is the dominant third digit z.](plots/s1_transition_matrix.png)

The matrix is symmetric because a transition is an unordered digit pair (control C4 confirms that
reversing a path only reverses alpha, so the two orientations carry the same information).

Two things stand out. First, **every** transition with a sub-plateau also has a stable third-class
prediction — the "sub-plateau only" cell is empty. So on this model the flat shelf in the output curve
never happens without a third digit taking over, but the reverse happens often (13 transitions).
Sub-plateau is the rarer, stricter phenomenon. Second, the dominant third digit is **7 in 13 of the 19
cases and 8 in the other 6** — the third answer is drawn from a tiny set, not spread over the eight
non-endpoint digits.

The 50% prevalence threshold is an organizational choice, so here is its sensitivity: at a 25%
threshold 23 transitions qualify, at 50% 19 qualify, at 75% 7 qualify. The phenomenon does not hinge
on the exact cut. Every one of the 19 labels is unchanged when any single image pair is dropped
(leave-one-out 100/100 for all 45 transitions).

### Result 2 — the mean distance curves, all 45 transitions

![Five-by-nine grid of all 45 mean d(alpha) curves at seed 0. In each panel the x-axis is alpha (0 = first endpoint image, 1 = second endpoint image) and the y-axis is the logit-space relative endpoint distance d(alpha) from 0 to 1; the thick line is the pointwise mean over the 100 image pairs and the shaded band is plus or minus one standard deviation across those pairs. Line style and band hatch encode the category exactly as in the matrix above and in the shared legend: solid unhatched = neither, solid with "//" hatch = stable third-class only, dash-dot with "xx" hatch = both. The panel title carries the same information in words (it names the dominant third digit and its prevalence whenever the transition is stable third-class). The dotted diagonal is the reference line d = alpha, and a horizontal shaded bar marks the detected sub-plateau shelf where one exists.](plots/s1_mean_curves_grid.png)

These mean curves are the primary evidence for the sub-plateau label; the frozen rule is only the
summarizer. The six detected shelves sit at `d` levels of 0.44–0.57 — squarely mid-output — and span
alpha intervals of 0.37–0.59 (0→1), 0.37–0.45 (0→4), 0.45–0.59 (2→9), 0.49–0.59 (3→6), 0.43–0.55
(6→7) and 0.37–0.53 (6→9). All six are near the middle of the path, as a genuine intermediate shelf
should be.

The bootstrap check flags three transitions as fragile rather than robust: **0→4** (sub-plateau label
agrees in only 68.4% of 1,000 resamples), **1→9** (no-sub-plateau label agrees in 63.6%; it is also
the single transition where leave-one-out changed the label, 99/100) and **5→7** (no-sub-plateau,
89.5%, just under the 90% bar). All three sit on the boundary of the flatness rule. We report them as
borderline rather than quietly relabelling them. Every other transition's sub-plateau label clears
both robustness bars, and every third-class label clears leave-one-out.

### Result 3 — the third-class prediction is spread across the 100 pairs, not driven by outliers

![Predicted-class composition across the interpolation for each of the 19 stable third-class transitions at seed 0. In every panel the x-axis is alpha from 0 to 1 and the y-axis is the fraction of the 100 image pairs whose prediction at that alpha is a given digit, stacked to 1. Each band is one predicted digit 0-9 on a light-to-dark sequential ramp running from digit 0 to digit 9 (as in the shared legend at the bottom); every band wide enough to hold it also has its digit printed inside it, so a band can be identified without reading its shade.](plots/s1_class_composition.png)

Reading these: at alpha = 0 essentially all 100 paths predict the first endpoint digit, at alpha = 1
essentially all predict the second, and in between a wedge belonging to a third digit opens up. In the
strongest cases (0→1, 2→9, 6→9, 3→4) the third digit holds a majority of the 100 paths over a wide
band of alpha. In the weaker ones (6→7, 1→5) it is a substantial minority rather than a majority — the
label only requires that the segment occur on ≥50 paths, not that the third digit win the vote at any
single alpha, which is why the third digit's band never reaches 50% in some panels.

![Box plots of per-pair third-class segment width for the 19 stable third-class transitions at seed 0. The x-axis lists the transition and its dominant third digit z; the y-axis is the longest run of consecutive alpha points predicted z on that path, out of the 50 points per path, with 0 meaning that path never predicts z. Each box covers the interquartile range over the 100 image pairs, the horizontal line inside a box is the median, the diamond marker is the mean, whiskers extend to 1.5 times the interquartile range and open circles are individual outlying pairs. The dashed horizontal line across the whole panel is the frozen median-run threshold of 3 points.](plots/s1_segment_widths.png)

This figure is the direct answer to "is this an outlier effect?". For 17 of the 19 transitions the
*lower quartile* of segment width already sits at or above the 3-point threshold, meaning at least 75
of the 100 image pairs produce a segment of at least that width. Median widths run from 4 points
(6→7) to 16 points (6→9) out of 50 — for 6→9 the network answers "8" across nearly a third of the
path on a typical pair.

### Result 4 — the segments are far outside the activation region of the digit they are predicted to be

This is the central measurement. For every stable third-class transition we take each path's longest
third-class segment and score every point in it against all ten real digit regions.

| transition | z | prevalence % | median run | IQR run | shelf (alpha interval @ d level) | fraction of segment points inside real-z region | fraction of paths with majority inside | median R_z on segment | median nearest-digit ratio | z at seed 1 / seed 2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0→1 | 7 | 94 | 15 | 11–19 | 0.39–0.53 @ 0.49 | 0.000 | 0.000 | 3.29 | 1.92 | –/2 |
| 0→3 | 7 | 73 | 9 | 6–13 | — | 0.000 | 0.000 | 2.68 | 1.40 | –/8 |
| 0→4 | 7 | 74 | 14 | 8–16 | 0.37–0.45 @ 0.49 | 0.000 | 0.000 | 2.15 | 1.60 | 1/2 |
| 0→9 | 7 | 67 | 9 | 5–14 | — | 0.017 | 0.015 | 1.74 | 1.40 | 1/8 |
| 1→2 | 7 | 56 | 8 | 5–13 | — | 0.014 | 0.000 | 2.85 | 1.68 | –/– |
| 1→4 | 7 | 65 | 7 | 4–12 | — | 0.000 | 0.000 | 2.91 | 2.09 | –/2 |
| 1→5 | 7 | 53 | 6 | 3–8 | — | 0.000 | 0.000 | 3.71 | 1.83 | –/2 |
| 1→6 | 8 | 74 | 7 | 4–11 | — | 0.000 | 0.000 | 2.24 | 1.92 | –/2 |
| 1→9 | 7 | 85 | 12 | 8–15 | — | 0.001 | 0.000 | 2.49 | 1.95 | –/2 |
| 2→4 | 7 | 76 | 12 | 5–15 | — | 0.033 | 0.000 | 1.87 | 1.46 | 1/– |
| 2→5 | 8 | 68 | 6 | 3–12 | — | 0.050 | 0.015 | 1.78 | 1.23 | 3/– |
| 2→9 | 7 | 84 | 15 | 9–19 | 0.45–0.59 @ 0.44 | **0.185** | **0.143** | 1.50 | 1.32 | 1/8 |
| 3→4 | 7 | 90 | 12 | 7–14 | — | 0.013 | 0.000 | 2.08 | 1.54 | 1/2 |
| 3→6 | 8 | 83 | 10 | 5–16 | 0.49–0.59 @ 0.47 | 0.000 | 0.000 | 2.08 | 1.45 | –/8 |
| 3→9 | 7 | 62 | 8 | 5–13 | — | 0.108 | 0.081 | 1.75 | 1.37 | –/– |
| 4→5 | 7 | 57 | 7 | 3–11 | — | 0.000 | 0.000 | 3.47 | 1.46 | 1/8 |
| 4→6 | 8 | 63 | 7 | 4–10 | — | 0.000 | 0.000 | 2.00 | 1.30 | –/– |
| 6→7 | 8 | 59 | 4 | 2–8 | 0.43–0.55 @ 0.57 | 0.000 | 0.000 | 2.15 | 1.59 | 1/2 |
| 6→9 | 8 | 93 | 16 | 10–20 | 0.37–0.53 @ 0.50 | 0.000 | 0.000 | 2.14 | 1.37 | 8/8 |

"–" in the last column means the transition is not labelled stable third-class at that seed.
"Prevalence" is the percentage of the 100 pairs on which z appears as a contiguous segment; "median
run" and "IQR run" are in interpolation points out of 50. All 19 labels are identical on the
correctly-classified-endpoint subset, so nothing here depends on including the few misclassified
endpoint images.

Pooled over all 19 transitions, **1,376 paths carry a segment, containing 14,700 interpolation points,
of which 2.5% are inside the real region of the predicted digit.** The best single case, 2→9 → "7",
reaches 18.5% of points and 14.3% of paths — still a clear minority. Fifteen of the 19 transitions are
below 5%, and nine are at exactly 0.

![Per-transition summary of activation-region membership at seed 0. The x-axis lists the 19 stable third-class transitions with their dominant third digit z. Bars use the left axis (fraction from 0 to 1): the "//"-hatched bar of each pair is the fraction of third-class segment points inside the real-z region, the "\\"-hatched bar beside it is the fraction of paths whose segment has a majority of points inside. Marker lines use the right axis (normalized distance ratio): the solid line with diamond markers is the median ratio to digit z on the segment, the dashed line with triangle markers the median ratio to the nearest of all ten digits; the dotted horizontal line marks a ratio of 1, the boundary of the usual spread of real images.](plots/s4_region_membership.png)

The two marker lines carry the strongest version of the result. The solid diamond line (distance to
the digit the network *names*) sits at 1.50–3.71, so a typical segment point is one and a half to
nearly four times farther from that digit's mean than the 95th-percentile real image of it. But the
dashed triangle line — the distance to whichever of the ten digits happens to be *nearest* — sits at 1.23–2.09,
also above 1 everywhere. **These points are not merely in the wrong digit's region; they are outside
every digit's region.** Where a nearest digit exists, it is usually not the predicted one: pooled over
all segment points, the nearest real region is digit 5 for 6,108 points, digit 2 for 2,073, and the
predicted digit 7 for only 1,984 of 14,700.

![Distance from the interpolated h1 activation to each real digit activation region, for each of the 19 stable third-class transitions at seed 0. In every panel the x-axis is alpha from 0 to 1 and the y-axis is the normalized distance ratio R_c on a logarithmic scale, averaged over the 100 image pairs. Three lines per panel give the ratio to endpoint digit a (solid, circle markers), to endpoint digit b (dashed, square markers) and to the dominant third digit z (dash-dot, triangle markers); the panel legend names each line and its digit. The dashed horizontal line is a ratio of 1, the held-out 95th percentile for that digit. The shaded vertical band, where present, marks the alpha range in which a majority of the 100 paths predict z.](plots/s4_distance_view.png)

The shape is consistent across all 19 panels: the two endpoint curves cross in the middle, both well
above 1 there, while the third digit's curve (`z`) stays roughly **flat and high** across the whole
path. The network's distance to a real 7 barely changes as you slide from a 0 to a 1 — it is far from
real 7s at every alpha — and yet its output says "7" through the middle. The third digit's curve is
never the lowest of the three in the region where the third-class prediction occurs.

![Two-dimensional principal-component view of the three highest-prevalence stable transitions at seed 0 (0 to 1 with z=7, 6 to 9 with z=8, 3 to 4 with z=7). Both axes are the first two principal components of real h1 activations, fitted only on real images of the two endpoint digits and the third digit; units are arbitrary. Faint small markers are individual real training images, one marker shape per digit — circles for endpoint digit a, squares for endpoint digit b, triangles for the third digit z, as named in each panel's legend. Large stars are the per-digit means, thin faint lines are twelve individual interpolation paths, and cross ("x") markers mark the interpolation points whose prediction is the third digit z.](plots/s4_pca_view.png)

This projection is **visualization only** — the conclusion above comes from the full 200 coordinates.
But it makes the geometry legible: the paths run in a narrow corridor directly between the two
endpoint clouds, and the crosses (points predicted as the third digit) sit in the empty middle
of that corridor, visibly far from the third digit's own cloud and its mean star.

### Result 5 — the verdict for each analyzed transition

Applying the preregistered three-way verdict:

| verdict | transitions |
|---|---|
| **activation-region match** (most segment points and most paths inside real-z region) | **0 of 19** |
| **mixed** | 0 of 19 |
| **prediction only** | **19 of 19** |

No transition comes close to the "match" bar; the best (2→9) is at 18.5% of points. This is a clean
null result for the hypothesis that a stable third-class prediction indicates arrival in the third
digit's activation region at `h1`.

### Result 6 — all four controls pass, so the null result is not a broken ruler

![Three control panels for the activation-region measure at h1. Left (C1): histogram over 7,000 held-out real test images of the normalized distance ratio to their own digit's region; x-axis is the ratio, y-axis is the number of images, dashed line at ratio 1. Middle (C3): bar chart with digit c on the x-axis and, on the y-axis, the fraction of within-digit interpolation points whose ratio to digit c is below 1. Right (C2): histogram of the fraction of endpoint-predicted interpolation points that are inside that endpoint digit's region, with one entry per endpoint per transition (2 x 19 = 38 values); the "//"-hatched bars count only the endpoint plateau (alpha at most 0.1 or at least 0.9) and the "\\"-hatched bars count all endpoint-predicted points, as labelled in the panel legend.](plots/s5_controls.png)

- **C1 passes.** 96.0% of held-out real test images have their own digit as `argmin_c R_c`, 95.0% have
  `R_own < 1` (as designed), and 93.6% satisfy both. Real images land in their own region.
- **C3 passes.** Interpolating between two images of the *same* digit keeps 91.8%–96.5% of points
  inside that digit's region across all ten digits — very close to the 95% that real images achieve.
  So interpolation as such does not push activations out of a region; only *cross-digit* interpolation
  does.
- **C2 passes on the endpoint plateau, with an informative caveat.** Restricted to alpha ≤ 0.1 or
  ≥ 0.9, the median fraction of endpoint-predicted points inside the endpoint region is 0.81, and 33
  of the 38 endpoint/transition cases exceed 0.5. Over *all* endpoint-predicted points the median
  falls to 0.37. The caveat is real and worth stating: **mid-path points predicted as an endpoint
  digit are also often outside that digit's region.** Being far from every real region is a general
  property of the middle of a cross-digit path, not something unique to third-class segments. The
  distinctive claim that survives is narrower but still the point: the third-class prediction adds no
  information about proximity to the named digit — the ratio to z is flat and high everywhere
  (Result 4), so the segment is not where the path comes *closest* to z.
- **C4 passes.** Reversing a path and flipping its order reproduces the forward path to a maximum
  relative deviation of 1.7 × 10⁻⁷ (floating-point noise). Evaluating each digit pair in one
  orientation loses nothing.

The weakest control is C2 at 0→1, where only 19.5%/27.4% of endpoint-plateau points are inside the
endpoint region — 0 and 1 have the tightest and most distinctive activation clusters, and the SLERP
path leaves them quickly. We report it rather than dropping it.

### Result 7 — the third digit's identity is a property of the seed, not of the digit pair

Repeating the frozen Stage-1 evaluation at seeds 1 and 2 — same image pairs, same thresholds, nothing
retuned — gives 12 and 18 stable third-class transitions respectively, against 19 at seed 0. So the
*phenomenon* replicates: a large minority of digit transitions produce a stable third-class prediction
in every network we tested. All three seeds agree on the stable-third-class label for 27 of 45
transitions and on the sub-plateau label for 39 of 45.

The *identity* of the third digit does not replicate at all:

| seed | stable third-class transitions | dominant third digits |
|---|---|---|
| 0 | 19 of 45 | 7 (×13), 8 (×6) |
| 1 | 12 of 45 | 1 (×10), 3 (×1), 8 (×1) |
| 2 | 18 of 45 | 2 (×9), 8 (×9) |

![Cross-seed comparison across all 45 digit transitions. Top panel: stable third-class prediction label, with transitions on the x-axis and the three seeds as rows; a black cell means the label is true at that seed. Middle panel: the same for the stable sub-plateau label. Bottom panel: the dominant third digit z where the transition is labelled stable third-class — the digit is printed in each cell, and the cell shading repeats it on a light-to-dark sequential ramp from digit 0 to digit 9; a blank cell means the transition is not stable at that seed.](plots/s1_seed_agreement.png)

The bottom panel makes the point at a glance: the printed digits in each seed's row repeat one or two
values, and those values differ from row to row. Exactly **one** transition — 6→9 — is labelled stable at all three seeds
*and* names the same third digit (8) at all three, with prevalence 93%, 53% and 93%.

The natural reading is that each trained network develops a private "default class" that wins in the
part of `h1` space between the real digit clusters, and which class that is depends on the random
initialization and data order rather than on any visual similarity among 6, 9 and 8. That is
consistent with Result 4: if the third digit were being chosen because the activations genuinely
resemble that digit, the choice should be reproducible across networks trained on the same data.

### Result 8 — the match does not appear at `h2` or `h3` either

Because the `h1` result is a clean null, the preregistered later-layer follow-up applies. Same frozen
segments, same rules, only the hook point changes:

| hook point | segment points inside the real-z region | fraction with ratio to z below 1 | fraction where z is the *nearest* region | mean number of the ten regions containing the point | median ratio to z (range over transitions) | control C1 | control C3 (mean) |
|---|---|---|---|---|---|---|---|
| `h1` | **2.5%** | 2.4% | 14.3% | **0.08** | 1.50 – 3.71 | 0.936 | 0.946 |
| `h2` | **10.6%** | 78.4% | 11.7% | **5.80** | 0.25 – 4.33 | 0.949 | 0.984 |
| `h3` | **0.2%** | 0.2% | 0.6% | **0.00** | 174.8 – 471.0 | 0.950 | 0.966 |

![Later-layer follow-up over the 19 stable third-class transitions at seed 0, comparing the three hook points h1, h2 and h3. In the two bar panels the three hook points are distinguished by hatch — h1 "//", h2 "\\", h3 "xx" — and named in the legend; in the line panel each hook point has its own marker (h1 circles, h2 squares, h3 triangles). Top panel: x = transition, y = fraction of third-class segment points inside the real-z activation region. Middle panel: x = transition, y = median normalized distance ratio on the segment, log scale; for each hook point the solid line is the distance to the predicted digit z and the dashed line with open markers the distance to the nearest of all ten digits; dotted line at ratio 1. Bottom panel: x = transition, y = the mean number of the ten digit regions that contain a segment point, where 1 would be ideal and 10 means the inside-region test is vacuous; dotted line at 1.](plots/s6_later_layers.png)

Read naively, `h2` looks like a partial rescue: the "inside" fraction rises from 2.5% to 10.6%, and
the median distance to the predicted digit falls from 1.50–3.71 to 0.25–4.33, mostly below 1. The
bottom panel shows why that reading is wrong. At `h2` a segment point lies inside **5.8 of the 10**
digit regions simultaneously (against 0.08 at `h1`), so passing "ratio to z below 1" is close to
automatic there — 78.4% of segment points pass it, and they would pass it for most other digits too.
The discriminating half of the criterion does not improve at all: the predicted digit z is the
*nearest* of the ten regions for 11.7% of segment points at `h2`, against 14.3% at `h1`, and both are
at or below the 12.5% you would get by picking one of the eight non-endpoint digits at random. The
per-digit regions simply become mutually overlapping at `h2` under a mean-and-variance summary, which
is a fact about the summary at that layer, not evidence about the prediction.

`h3` goes the other way, decisively. At the last hidden layer real digit classes are extremely tightly
clustered, so a point off the data has an enormous z-scored distance: the median segment point is
**175–471 times** the held-out 95th percentile away from the predicted digit, and 45–198 times away
from the *nearest* digit. It is inside zero regions. Controls C1 (0.950) and C3 (0.966) still pass at
`h3`, so this is not a broken measurement — real held-out images and within-digit interpolations
remain comfortably inside their own regions at the same layer. The interpolation points are simply
nowhere near any class.

So the answer to the follow-up question is no: the similarity to real z activations does not emerge in
later layers. It is absent at `h1`, undetectable at `h2` because the region test loses its power
there, and emphatically absent at `h3`.

---

## Conclusion

**A stable, reproducible class prediction is not evidence that the network's internal state resembles
that class.** Across a complete census of all 45 MNIST digit transitions with 100 fixed image pairs
each, 19 transitions at the primary network make the same third-class prediction on a majority of
image pairs over a median of 4–16 of the 50 interpolation points. Not one of those 19 is an
activation-region match: pooled over 14,700 segment points, 2.5% are inside the real activation region
of the digit they are predicted to be, and the median point is farther from *every* digit's region
than 95% of real held-out images are from their own. The verdict is "prediction only" in all 19 cases,
and it does not change at the second or third hidden layer: the similarity never appears downstream.

Two secondary findings sharpen this. First, the sub-plateau — the flat intermediate shelf in the
output distance curve that motivated this direction — is much rarer than third-class prediction (6 of
45 versus 19 of 45) and never occurs without it. The two labels are related but not interchangeable,
and the more common phenomenon is the confident wrong answer, not the pause. Second, which digit the
network names is a property of the trained network: 7 or 8 at seed 0, 1 at seed 1, 2 or 8 at seed 2,
with only 6→9 → "8" surviving in all three. A monitor calibrated on one network's "default answer in
between classes" would be calibrated on nothing transferable.

For safety monitoring the practical implication is direct: in the regions of activation space between
real data clusters, the argmax label is generated by whichever decision region happens to extend
there, and it carries no information about what the activation is close to. Any pipeline that treats a
confident class label as a proxy for "the input resembles this class" will be systematically wrong in
exactly the off-data regions where an adversary or a distribution shift puts you. Distance to the real
activation region is a separate measurement and, on this model, it disagrees with the label.

**Limitations.** (i) One model family (a 4-layer 200-wide MLP trained with MSE on MNIST) and three
seeds; nothing here establishes that the same holds for convolutional networks, larger models, or
other datasets. (ii) The activation region is summarized by a per-digit mean and coordinate-wise
variance — a deliberately simple, preregistered summary. It passes all four controls, but a richer
summary could place some segment points inside a region that this one puts outside; the effect would
have to be large, since 15 of 19 transitions are below 5%. (iii) Result 8 shows that at `h2` this
summary's ten regions overlap so heavily (a segment point is inside 5.8 of them on average) that the
inside-region test carries little information at that layer; the conclusion there rests on the
nearest-region half of the criterion, which is at chance. A summary that kept the regions separable at
`h2` would give a sharper test. (iv) Three
transitions (0→4, 1→9, 5→7) have sub-plateau labels that fail the preregistered bootstrap-agreement
bar and should be read as borderline; the third-class labels are unaffected. (v) The bank uses the
first 100 test images of each digit inside a frozen 2,000-image pool, so the 100 pairs are fixed by
construction rather than randomly sampled; leave-one-out and bootstrap check robustness to individual
pairs but not to choosing an entirely different pool.

**Reproducibility.** `experiments/s1_census.py` (census), `experiments/s1_analyze.py` (frozen rules,
Stage-1 figures), `experiments/s3_s4_regions.py` (activation regions, controls, region figures),
`experiments/s6_later_layers.py` (`h2`/`h3` follow-up), `experiments/cvd_style.py` (shared
colour-vision-deficiency-safe figure palette). Numeric outputs: `results/s1_census.npz`,
`results/s1_classification.{json,csv}`, `results/s3_s4_regions.json`,
`results/s6_later_layers.json`. Every result file records the source checkpoint path and its SHA-256.
No model was trained in this direction and no file in direction 12 was modified.
