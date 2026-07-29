# Does a switch-like *continuous* target create activation plateaus?

A controlled sweep from a near-linear to a step-function regression target on MNIST.

> Final, presentable, current-best only (no history — see CHANGELOG.md).

## Summary

**The safety question.** Neural networks trained to classify show **activation plateaus**: as you slide
an input smoothly from one class toward another, the internal representation barely moves for a long
stretch, then lurches across a narrow transition, then goes quiet again. Plateaus matter for AI safety
because they are where a model's internal state stops tracking the input. A monitor reading those
activations sees "nothing changed" while the input has in fact drifted a long way, and interpretability
tools that assume a smooth input-to-representation map are least reliable exactly there. A natural guess
is that plateaus are an artifact of *discrete* supervision — one-hot labels are a step function, so
maybe the network copies that step inward.

**What we did.** We test that guess without ever using a discrete target. The same 4-layer ReLU network
is trained on the same MNIST images to regress one continuous number from image **brightness** $b$, and
a single knob $k$ morphs the target from a straight line to a step function:

```math
y_k(b) = \frac{\tanh\big(k(b-b_0)\big)}{\tanh(0.3k)}, \qquad b_0 = 0.7 .
```

Inputs, architecture, loss, optimizer, schedule and seeds are held fixed; only $k$ changes. We then
slide each held-out image through the brightness range and measure **where along that path the hidden
representation actually moves**.

**What we found — target discreteness is not the mechanism.** Sharpening the target does pull hidden
movement toward the transition, monotonically and much more strongly in deeper layers. But the effect
**saturates early and low**. Our concentration measure runs from $1.0$ (movement spread perfectly evenly
along the path) to $5.0$ (all movement inside the middle 20% of the brightness range). Between $k=0.5$
and $k=20$ the deepest hidden layer climbs from $1.094 \pm 0.010$ to $1.491 \pm 0.068$; from $k=20$ to
$k=320$ — a further 16-fold sharpening, over which the target itself goes from $4.17$ to its maximum
$5.00$ — it does not move at all ($1.483$, $1.468$, $1.451$, $1.458$).

**The decisive control.** One could object that the network simply never learned the step. So we
retrained the whole grid on 10x more data, where it does learn it: at $k=320$ the network's **output** is
itself nearly a step (concentration $4.13$ of a possible $5.00$; only 0.5% of its total change happens in
the outer 40% of the brightness range, versus 40% for a uniform curve). Its **deepest hidden layer** is
nonetheless still nearly uniform — concentration $1.659 \pm 0.168$, with 28% of its movement still out on
the flanks. The network can compute a switch at its output while keeping a representation that slides
smoothly the whole way.

**Verdict: a switch-like continuous target is not sufficient to produce classification-style activation
plateaus.** It nudges the representation in that direction — measurably, and more strongly the deeper you
look — but plateau-transition-plateau structure never appears, at any sharpness, at either training-set
size.

**How to read this.** This is a clean negative result about *one* candidate cause. It does not say
plateaus in classifiers are unexplained — only that "the target has few discrete values" is not by itself
the mechanism. Cross-entropy loss, a softmax head, multi-class competition, and scale far beyond this
model all remain live candidates, and are outside this experiment's scope.

## Methods

### Data & Model

**Dataset.** MNIST handwritten digits (28x28 grayscale, flattened to 784 values in $[0,1]$). Digit
labels are **never** a learning target; they are used only to make every split digit-balanced (equal
numbers of each digit). Each image $x$ is first rescaled to unit Euclidean length, then rescaled again to
a sampled *brightness* $b$:

```math
\tilde{x} = \frac{x}{\lVert x \rVert_2 + \epsilon}, \qquad x_b = b\,\tilde{x}, \qquad b \sim U(0.4,\,1.0)
```

Because $\lVert \tilde{x} \rVert_2 \approx 1$, the input's Euclidean norm *is* the brightness:
$\lVert x_b \rVert_2 = b$ to within $1.8 \times 10^{-7}$ (verified numerically over all splits). Digit
identity and handwriting style are nuisance variation the model must ignore. Every image gets one fixed
brightness, drawn once per seed and **shared by all ten $k$ settings**, so the ten models see
bit-identical inputs and differ only in the scalar they are asked to predict.

**Splits.** Training set: 1000 digit-balanced images from MNIST-train`[:50000]`. Validation: 2000
digit-balanced images from MNIST-train`[50000:]`. Probe: 100 digit-balanced images from the MNIST **test**
set, held fixed across seeds.

**Model and hook point.** A 4-layer fully-connected ReLU network, $784 \to 200 \to 200 \to 200 \to 1$ —
this project's canonical MNIST plateau architecture with the 10-way head replaced by a single scalar
output. We record **post-ReLU activations at all three hidden layers** (layer 1 = shallowest, layer 3 =
deepest; 200 units each). "Layer $l$" always means this post-ReLU hook point.

**Training.** Mean squared error, AdamW (learning rate $10^{-3}$, weight decay $0.01$), batch size 200,
2000 epochs, with a global cosine learning-rate decay. Every hyperparameter is shared by all ten $k$
values and all three seeds (0, 1, 2); weight initialization and batch order are identical across $k$
within a seed. Nothing is tuned per $k$. Both the final checkpoint and the minimum-validation-loss
checkpoint are saved; the **final** checkpoint is the primary analysis and the other is a robustness
check only.

**Robustness grid.** The identical experiment is repeated with 10,000 training images and 200 epochs
(same number of optimizer steps). This is the *secondary* grid — it is what removes the "the model never
learned the step" confound in Results §5.

### The target family, and why $k$ runs to 320

The denominator $\tanh(0.3k)$ pins the endpoint range at $[-1,1]$ for every $k$, so target *amplitude*
cannot confound the comparison: sharpness is the only thing that varies. We use ten settings,

```math
k \in \lbrace 0.5,\ 1,\ 2,\ 5,\ 10,\ 20,\ 40,\ 80,\ 160,\ 320 \rbrace .
```

The first five span near-linear to visibly sigmoid. The upper five matter because the interesting regime
is where the target is genuinely *discrete-like*. Measuring the transition width as the brightness
interval over which $y_k$ goes from $-0.9$ to $+0.9$: it is $0.29$ at $k=10$ — half the entire brightness
range, not a switch at all — but $0.0092$ at $k=160$ and $0.0046$ at $k=320$. The brightness probe grid
has spacing $0.003$, so at $k=320$ the target is a **step function at the resolution at which we measure**
and its concentration score sits exactly at the theoretical maximum. Testing the hypothesis only on
$k \le 10$ would test it on a family that never actually becomes discrete.

### Probe: where along the brightness path does the representation move?

For each of the 100 held-out probe images we hold the image fixed and sweep brightness over **201 evenly
spaced values in $[0.4, 1.0]$**, recording the prediction and all three hidden activations at each point.
This traces one path per image through activation space, and "is there a plateau?" becomes a question
about how that path's arc length is distributed along the sweep.

To answer it we need a measure invariant to how *large* the activations are — a layer with big weights
moves more in absolute terms without being any more plateau-like — so we normalize each path by its own
total travel. **Local movement** between adjacent brightness points, and its normalized version:

```math
m_l(b_i) = \lVert h_l(b_{i+1}) - h_l(b_i) \rVert_2 , \qquad
s_l(b_i) = \frac{m_l(b_i)}{\sum_j m_l(b_j) + \epsilon} .
```

$s_l(b_i)$ is the *share* of layer $l$'s total path length spent crossing step $i$; it sums to 1 over the
200 steps. A perfectly uniform path has $s_l = 1/200 = 0.005$ everywhere. Plateau-transition-plateau
structure means $s_l$ near zero at both ends and a tall spike near $b_0$. Figure 4 plots $s_l(b)$ directly.

**Concentration gain — the headline metric.** We need one number per curve to plot against $k$. The
natural question is "what share of the path's movement happens inside the target's transition region?",
so we sum $s_l$ over the central 20% of the brightness range, $b \in [0.64, 0.76]$ (40 of the 200 steps):

```math
C_l(k) = \sum_{b_i \in [0.64,\,0.76]} s_l(b_i), \qquad \Gamma_l(k) = \frac{C_l(k)}{0.2} .
```

Dividing by $0.2$ makes the number self-calibrating: **$\Gamma = 1$ means movement is spread perfectly
uniformly, $\Gamma > 1$ means it is concentrated near the transition, and $\Gamma = 5$ is the maximum**
(all movement inside the window, none outside). Higher is more plateau-like. Figures 5a, 6a and 8d report
$\Gamma$, and it is the quantity the verdict turns on.

*A note on notation.* This report writes the concentration gain as $\Gamma$ rather than the $R$ used in
the original plan, because $R$ with a subscript is easily misread as the coefficient of determination
$R^2$. **In this report $R^2$ always means goodness of fit and nothing else**, and layer indices only ever
appear as subscripts on $\Gamma$, $\Phi$, $s$, $C$ and $h$.

**Flank share — because $\Gamma$ alone can be fooled.** A curve can raise $\Gamma$ by growing a modest
bump in the middle while still moving plenty everywhere else; that is not a plateau. The plateau claim is
specifically that movement **away from** the transition goes to zero. So we also measure the share of
movement left in the outer 40% of the range:

```math
\Phi_l(k) = \sum_{b_i < 0.52} s_l(b_i) \;+\; \sum_{b_i \ge 0.88} s_l(b_i) .
```

**$\Phi = 0.4$ is the uniform value and $\Phi \to 0$ is a true plateau**; here *lower* is more
plateau-like. Figure 5b and Table 2 report $\Phi$. Taken together $\Gamma$ and $\Phi$ are what make the
verdict falsifiable: the hypothesis needs *both* a rising $\Gamma$ and a collapsing $\Phi$, and we find
only the first, weakly.

**Sweep fit $R^2$ — a validity check, not a result.** Concentration scores mean nothing if the model
never learned the map, and "but did it learn the switch?" is exactly the objection the large-$k$ runs must
answer. Over the probe sweep, with the average taken across probe images, brightness points and seeds:

```math
R^2 = 1 - \frac{\overline{\big(\hat y(b) - y_k(b)\big)^2}}{\mathrm{Var}\big(y_k(b)\big)} .
```

$R^2 = 1$ is a perfect fit; $R^2 = 0$ is no better than predicting the target's mean. Figures 3 and 6b and
Tables 3–4 report it.

**Training-adequacy ratio.** The plan requires reported models to be trained to a smooth minimum with
slight validation overfitting, so that results are not artifacts of an unconverged or wildly overfit
network. With $V(e)$ the validation loss at epoch $e$ and $E$ the final epoch:

```math
\rho_{\mathrm{val}} = \frac{V(E)}{\min_e V(e)} .
```

The pre-registered gate is $\rho_{\mathrm{val}} \le 1.2$ with the minimum occurring before the last epoch,
plus a training loss ending within 5% of its own minimum. Figure 2 and Table 4 report this.

### Baselines

Every number in Results is read against these references. All are computed on the same 201-point sweep and
the same windows, so they are directly comparable to the hidden-layer scores.

**Uniform-movement baseline** — the "no plateau at all" null. A path whose movement is spread evenly over
brightness scores exactly

```math
\Gamma = 1 , \qquad \Phi = 0.4 .
```

Drawn as a dotted horizontal line in every $\Gamma$ / $\Phi$ figure.

**Maximum baseline** — the "perfect plateau" ceiling, reached when all movement falls inside the central
window:

```math
\Gamma_{\max} = 5 , \qquad \Phi_{\min} = 0 .
```

Drawn as a dashed horizontal line in Figures 5a, 6a and 8d. Because both extremes are pinned, we can
state results as a **percentage of the way from uniform to a perfect plateau**, $(\Gamma - 1)/4$.

**Target reference** — the same two metrics computed on the target curve itself, replacing $m_l(b_i)$ with

```math
m_{\mathrm{tgt}}(b_i) = \big| y_k(b_{i+1}) - y_k(b_i) \big| .
```

This is the score the representation would get *if it tracked the target exactly*, and it says how sharp
the target really is at each $k$. Drawn as a dark dotted line with star markers.

**Model-output reference** — the same two metrics computed on the network's own prediction curve
$\hat y(b)$, replacing $m_l(b_i)$ with

```math
m_{\mathrm{out}}(b_i) = \big| \hat y(b_{i+1}) - \hat y(b_i) \big| .
```

This separates two very different failures: *the network never learned the switch* (this reference stays
near 1) versus *the network learned the switch but its representation does not plateau* (this reference
climbs while the hidden layers do not). It is the pivot of the whole argument. Drawn as a dash-dot line
with triangle markers.

Two further controls come from the design rather than a formula. The **near-linear target** $k=0.5$ fixes
how much movement structure a ReLU network shows under pure input rescaling with no target sharpness at
all. **Layer 1**, the shallowest hidden layer, should show the smallest effect if the phenomenon is
genuinely a deep-representation one.

### Uncertainty

All primary numbers are means over 100 held-out probe images and 3 seeds. Reported $\pm$ is a **95%
confidence interval across the 3 seeds**, $1.96 \cdot \mathrm{SD}/\sqrt{3}$, which measures model-to-model
variation; probe images are fixed across seeds by construction, so this interval is not contaminated by
image sampling. Spread across *images* is reported separately as a standard deviation where it matters,
and is substantially larger than the seed interval.

## Results

### 1. The target family really does become a step function

The manipulation has to work before anything downstream means much, and the whole point of extending to
$k=320$ is that the earlier five-setting grid never reached a genuinely discrete target. Figure 1 shows
the ten targets; the zoom panel is where the sharpest five separate from each other.

![ten tanh target curves of increasing sharpness, plus a zoom on the transition](plots/target_functions.png)

**Figure 1.** x: brightness $b$; y: target value $y_k(b)$. Ten series, one per sharpness $k$, ordered dark
blue ($k=0.5$) to yellow ($k=320$) with a distinct line style per series. Dotted vertical line = transition
centre $b_0 = 0.7$; grey band = the central window $[0.64, 0.76]$ used by the concentration metric.
**(a)** the full brightness range: $k=0.5$ is a straight line, $k \ge 80$ is visually a step. **(b)** zoom
on $b \in [0.60, 0.80]$, where the five sharpest targets are still distinguishable from one another.
Amplitude is normalized to $[-1,1]$ for every curve, so sharpness is the only difference between them.

Measured by the target reference, sharpness rises from $\Gamma = 1.01$ at $k=0.5$ to the theoretical
maximum $\Gamma = 5.00$ at $k \ge 80$, while the target's flank share falls from $\Phi = 0.397$ (uniform)
to exactly $0$: beyond $k=40$ the target does not change *at all* outside the middle 20% of the range.

### 2. Every reported model is adequately trained

A plateau claim from a badly-trained network means nothing, so we check the pre-registered adequacy gate
before reading any concentration score.

![train and validation loss curves for the ten k settings, log scale](plots/training_curves.png)

**Figure 2.** Left — x: epoch; y: training MSE on the full 1000-image training set (log scale). Right —
x: epoch; y: validation MSE on the 2000 held-out validation images (log scale); open circle marks each
curve's minimum. Ten series ordered by $k$ with the same colour and line-style scheme as Figure 1 (seed 0
shown; the other two seeds are visually identical). Every run ends at a smooth floor slightly past a
validation minimum. The vertical ordering of the validation curves is the difficulty gradient: sharper
targets are harder to generalize.

All 30 runs (10 values of $k$ x 3 seeds) pass the gate: the validation minimum occurs before the last
epoch, the final validation loss is 1.7–10.9% above it (gate: $\le$ 20%), and training loss ends within
0.1% of its own minimum (gate: $\le$ 5%). One caveat we flag rather than hide: at $k=160$ and $k=320$ the
validation minimum arrives very early (mean epoch 15 of 2000), because those targets are essentially
unlearnable in detail from 1000 images, so the curve is flat-then-slightly-rising from the start. The
gate's numeric conditions still hold, but the minimum-validation checkpoint at those two settings is a
barely-trained network — which matters only for §6.

### 3. The models fit the smooth targets well and the step targets poorly — at 1000 images

This is the validity check that the large-$k$ extension makes essential.

![target vs mean prediction across brightness, one panel per k](plots/prediction_sweeps.png)

**Figure 3.** x: brightness $b$; y: value on the shared target/prediction scale. One panel per $k$. Dark
dotted = true target $y_k(b)$; coloured solid or dashed = mean prediction over 100 held-out images and 3
seeds, coloured and styled by $k$ as in Figure 1; shaded band = $\pm 1$ SD across probe images. Grey band =
the central window. Sweep $R^2$ appears in each panel title. The predictions for $k \ge 40$ are all much
the same soft sigmoid no matter how sharp the target gets.

Sweep $R^2$ is flat at $0.90$ for $k \le 5$ and then falls steadily to $0.612$ at $k=320$ (Table 4). With
only 1000 training images the network cannot place a step precisely enough to generalize, and its mean
prediction stops sharpening past $k \approx 40$. That is a real limit of the primary grid, and §5 removes
it.

### 4. Activation movement concentrates a little, then stops — the main result

This is the direct test: given a target that is a step function, does the deepest layer develop
plateau-transition-plateau structure?

![normalized movement vs brightness for the target and the three hidden layers](plots/activation_movement_by_k.png)

**Figure 4.** x: brightness $b$; y: normalized local movement $s(b)$, the share of the path's total travel
spent at that brightness. Panels left to right: the target's own normalized $|\Delta y_k|$ (log y-axis,
because it spans many orders of magnitude), then hidden layers 1, 2 and 3 (deepest) on linear axes. Ten
series per panel, coloured and styled by $k$ as in Figure 1; shaded bands = 95% CI across the 3 seeds;
dotted horizontal line = the uniform value $0.005$; grey band = the central window. **The y-scales differ
per panel**: layer 1 spans only $0.0042$–$0.0052$ around a uniform value of $0.005$, while the target panel
spans many decades. In layer 3 the curves for $k \ge 10$ lie almost on top of one another — the shape stops
responding to $k$ long before the target stops sharpening.

Reducing each curve to one number makes both halves of the verdict visible at once.

![concentration gain and flank share versus k, all layers](plots/concentration_vs_k.png)

**Figure 5.** x (both panels): target sharpness $k$, log scale. Five series in each panel: target reference
(dark dotted, star), model output (orange dash-dot, inverted triangle), and hidden layers 1 / 2 / 3 (blue
solid circle / vermillion dashed square / pink dash-dot triangle). Error bars = 95% CI across 3 seeds.
**(a)** y: concentration gain $\Gamma$, higher = more plateau-like; lower dotted line = uniform baseline
$\Gamma = 1$, upper dashed line = maximum $\Gamma = 5$. **(b)** y: flank share $\Phi$, **lower** = more
plateau-like; dotted line = uniform baseline $\Phi = 0.4$. The target reaches both extremes; the hidden
layers reach neither.

**Table 1 — concentration gain $\Gamma$** (1.0 = uniform movement, 5.0 = maximum possible). Primary grid,
1000 training images, final checkpoint, mean over 100 held-out images and 3 seeds; $\pm$ = 95% CI across
seeds.

| $k$ | target curve | model output | hidden layer 1 | hidden layer 2 | hidden layer 3 (deepest) |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 1.01 | 1.12 | 1.007 ± 0.001 | 1.034 ± 0.006 | 1.094 ± 0.010 |
| 1   | 1.03 | 1.14 | 1.007 ± 0.001 | 1.036 ± 0.010 | 1.105 ± 0.016 |
| 2   | 1.11 | 1.18 | 1.007 ± 0.001 | 1.044 ± 0.007 | 1.137 ± 0.011 |
| 5   | 1.61 | 1.42 | 1.011 ± 0.003 | 1.092 ± 0.013 | 1.326 ± 0.015 |
| 10  | 2.70 | 1.80 | 1.014 ± 0.003 | 1.130 ± 0.004 | 1.455 ± 0.036 |
| 20  | 4.17 | 2.11 | 1.017 ± 0.003 | 1.137 ± 0.013 | 1.491 ± 0.068 |
| 40  | 4.92 | 2.20 | 1.019 ± 0.003 | 1.125 ± 0.005 | 1.483 ± 0.128 |
| 80  | 5.00 | 2.25 | 1.025 ± 0.008 | 1.120 ± 0.001 | 1.468 ± 0.212 |
| 160 | 5.00 | 2.17 | 1.020 ± 0.007 | 1.110 ± 0.019 | 1.451 ± 0.205 |
| 320 | 5.00 | 2.29 | 1.021 ± 0.004 | 1.114 ± 0.008 | 1.458 ± 0.189 |

**Table 2 — flank share $\Phi$** (0.4 = uniform movement, 0 = perfect plateau; lower is more plateau-like).
Same runs and same averaging as Table 1.

| $k$ | target curve | model output | hidden layer 1 | hidden layer 2 | hidden layer 3 (deepest) |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.397 | 0.344 | 0.397 | 0.384 | 0.356 ± 0.004 |
| 1   | 0.389 | 0.339 | 0.397 | 0.384 | 0.353 ± 0.005 |
| 2   | 0.357 | 0.323 | 0.397 | 0.381 | 0.342 ± 0.005 |
| 5   | 0.209 | 0.249 | 0.395 | 0.364 | 0.283 ± 0.005 |
| 10  | 0.048 | 0.151 | 0.394 | 0.357 | 0.265 ± 0.016 |
| 20  | 0.002 | 0.098 | 0.393 | 0.358 | 0.269 ± 0.018 |
| 40  | 0.000 | 0.086 | 0.392 | 0.361 | 0.272 ± 0.031 |
| 80  | 0.000 | 0.081 | 0.389 | 0.360 | 0.282 ± 0.036 |
| 160 | 0.000 | 0.082 | 0.391 | 0.363 | 0.279 ± 0.034 |
| 320 | 0.000 | 0.071 | 0.391 | 0.363 | 0.283 ± 0.030 |

Reading the two tables together:

- **The positive half.** Concentration does rise with target sharpness, and the effect grows with depth
  exactly as hypothesized: across the full sweep layer 1 barely moves ($1.007 \to 1.021$), layer 2 rises
  modestly ($1.034 \to 1.114$), layer 3 rises most ($1.094 \to 1.458$). Over the range where the rise
  happens ($k = 0.5 \to 20$) it is monotone and seed-consistent.
- **The negative half, and it is decisive.** Past $k=20$ the deepest layer stops responding —
  $1.491$, $1.483$, $1.468$, $1.451$, $1.458$ across a 16x increase in target sharpness, every value inside
  every other's confidence interval — while the target reference climbs from $4.17$ to its $5.00$ ceiling.
  Layer 3 ends at 11% of the distance from uniform to maximum. The flank share says it more bluntly:
  $\Phi_3$ bottoms out at $0.265$ around $k=10$ and then *rises slightly* to $0.283$, never getting below
  two-thirds of the uniform value, while the target's $\Phi$ is exactly $0$. **At no $k$ does any hidden
  layer show plateau-transition-plateau structure.**
- **This is a saturation, not a slow climb.** Extending the grid is what made this visible: on $k \le 10$
  alone the curve looks like it might keep rising — precisely the ambiguity that motivated adding the five
  larger settings.

Variation across probe *images* is much larger than across seeds, and grows with $k$: the image-level SD of
$\Gamma_3$ is $0.036$ at $k=0.5$ and $0.507$ at $k=320$. Individual images differ a lot in how sharply
their representation turns; the average still does not plateau.

### 5. The ceiling is not a failure to fit — the decisive control

The obvious objection to §4 is that at $k=320$ the 1000-image model reaches only $R^2 = 0.61$, so perhaps
the representation looks flat merely because the *function* was never learned. To test that we retrained
the identical grid on 10,000 training images with the same number of optimizer steps.

![two panels: concentration gain saturating at both training-set sizes, and sweep fit quality](plots/saturation_and_fit.png)

**Figure 6.** x (both panels): target sharpness $k$, log scale. Solid lines with filled markers = 1000
training images (primary); dashed lines with open markers = 10,000 training images. **(a)** y:
concentration gain $\Gamma$. Series: target reference (dark dotted, star), model output (orange, inverted
triangle) and hidden layer 3 (pink, triangle), each drawn at both training-set sizes; lower dotted line =
uniform baseline $\Gamma = 1$, upper dashed line = maximum $\Gamma = 5$. More data lifts the **output**
almost to the target, but layer 3 still flattens out below $\Gamma = 2$. **(b)** y: sweep $R^2$ (blue,
circles), the fit-quality check that panel (a)'s interpretation depends on; dotted line = perfect fit.

**Table 3 — 10,000-image robustness grid.** Final checkpoint, mean over 100 held-out images and 3 seeds;
$\pm$ = 95% CI across seeds. Concentration gain $\Gamma$ and flank share $\Phi$ as defined in Methods.

| $k$ | $\Gamma$ target | $\Gamma$ model output | $\Gamma$ hidden layer 3 | $\Phi$ model output | $\Phi$ hidden layer 3 | sweep $R^2$ |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 1.01 | 1.06 | 1.045 ± 0.004 | 0.370 | 0.377 | 0.981 |
| 1   | 1.03 | 1.07 | 1.055 ± 0.001 | 0.365 | 0.372 | 0.984 |
| 2   | 1.11 | 1.14 | 1.109 ± 0.002 | 0.337 | 0.353 | 0.985 |
| 5   | 1.61 | 1.57 | 1.419 ± 0.016 | 0.213 | 0.263 | 0.985 |
| 10  | 2.70 | 2.42 | 1.823 ± 0.222 | 0.066 | 0.204 | 0.978 |
| 20  | 4.17 | 3.41 | 1.913 ± 0.173 | 0.010 | 0.239 | 0.959 |
| 40  | 4.92 | 3.91 | 2.024 ± 0.504 | 0.003 | 0.242 | 0.929 |
| 80  | 5.00 | 4.00 | 1.670 ± 0.091 | 0.003 | 0.284 | 0.893 |
| 160 | 5.00 | 4.06 | 1.665 ± 0.114 | 0.003 | 0.281 | 0.867 |
| 320 | 5.00 | 4.13 | 1.659 ± 0.168 | 0.005 | 0.279 | 0.848 |

The objection is answered, and the answer strengthens the verdict:

- **The network does learn the step.** At $k=320$, sweep $R^2 = 0.848$ and the output's own flank share is
  $0.005$: the prediction curve changes by only half a percent of its total variation outside the middle
  20% of the brightness range, against $0.4$ for a uniform curve. For practical purposes the output *is* a
  switch — $\Gamma_{\mathrm{out}} = 4.13$ of a maximum $5.00$, i.e. 78% of the way from uniform to a perfect
  plateau.
- **The representation still does not plateau.** Those same models' deepest hidden layer sits at
  $\Gamma_3 = 1.659 \pm 0.168$ and $\Phi_3 = 0.279$ — only 16% of the way to the maximum, with 28% of its
  movement still on the flanks (uniform is 40%). Layer 1 is at $1.038$, essentially uniform.
- **More data raises the effect but does not change its shape.** $\Gamma_3$ peaks at $2.024$ ($k=40$) and
  then *falls back* to about $1.66$ for the three step-like settings — a non-monotonic response we report as
  observed rather than smoothing over. Either way it saturates far below the target, so the primary grid's
  numbers are a lower bound on effect size and the verdict is identical at both training-set sizes.

The gap between the output column and the layer-3 column in Table 3 is the cleanest statement of the
finding: **at $k=320$ the network's output is 78% of the way to a perfect plateau while its deepest hidden
layer is 16% of the way.** A switch-like function, computed by a representation that slides smoothly.

This grid is the *secondary* one because it fails the pre-registered adequacy gate in spirit: with 10x the
data the models show essentially no validation overfitting ($\rho_{\mathrm{val}} \approx 1.005$ at every
$k$), and the plan requires slight overfitting for the primary result. We report it in full because it
removes the fitting confound that would otherwise weaken §4.

### 6. Neither the checkpoint choice nor the training diagnostics change anything

![concentration gain for the final versus minimum-validation-loss checkpoint, all layers](plots/checkpoint_robustness.png)

**Figure 7.** x: target sharpness $k$, log scale; y: concentration gain $\Gamma_l(k)$; dotted line = uniform
baseline $\Gamma = 1$. Six series: hidden layers 1 / 2 / 3 (blue circle / vermillion square / pink triangle),
each drawn twice — solid with filled markers = final checkpoint, dashed with open markers =
minimum-validation-loss checkpoint. Error bars = 95% CI across 3 seeds. The two checkpoint choices are
indistinguishable wherever the minimum-validation checkpoint is a trained model.

For every $k \le 80$ the two checkpoints agree to within $0.035$ in $\Gamma_3$ (for example $1.455$ vs
$1.451$ at $k=10$). At $k=160$ and $k=320$ the minimum-validation checkpoint is the epoch-15 network flagged
in §2 — barely trained — and reads $1.336$ and $1.331$ against the final checkpoint's $1.451$ and $1.458$;
that difference reflects the checkpoint being untrained, not a dependence of the conclusion on checkpoint
choice. No conclusion in this report changes if the other checkpoint is read.

**Table 4 — fit and training diagnostics, primary grid.** Mean over 3 seeds. $\rho_{\mathrm{val}}$ is the
adequacy ratio (gate: $\le 1.2$); "val-min epoch" is where the validation minimum falls within the 2000
epochs.

| $k$ | sweep $R^2$ | final validation MSE | $\rho_{\mathrm{val}}$ | val-min epoch |
|---:|---:|---:|---:|---:|
| 0.5 | 0.899 | 0.031 | 1.058 | 547 |
| 1   | 0.900 | 0.031 | 1.058 | 648 |
| 2   | 0.897 | 0.034 | 1.056 | 583 |
| 5   | 0.900 | 0.048 | 1.079 | 663 |
| 10  | 0.862 | 0.088 | 1.064 | 548 |
| 20  | 0.789 | 0.174 | 1.040 | 438 |
| 40  | 0.722 | 0.251 | 1.041 | 1050 |
| 80  | 0.663 | 0.311 | 1.048 | 340 |
| 160 | 0.622 | 0.343 | 1.062 | 15 |
| 320 | 0.612 | 0.369 | 1.094 | 15 |

### 7. The whole experiment in one figure

![four-panel summary: targets, predictions, deepest-layer movement, concentration gain](plots/main_summary.png)

**Figure 8.** All panels use the ten-$k$ colour and line-style scheme of Figure 1; grey bands in (a)–(c) =
the central window $[0.64, 0.76]$. **(a)** x: brightness $b$, y: target $y_k(b)$ — the manipulation, running
from a straight line to a step. **(b)** x: $b$, y: mean prediction $\hat y(b)$ over probe images and seeds —
the primary-grid models answer every sharp target with much the same soft sigmoid. **(c)** x: $b$, y:
normalized movement $s_3(b)$ in the deepest hidden layer, bands = 95% CI across seeds, dotted line = uniform
value $0.005$ — a shallow bump, not a plateau, and its shape stops changing past $k \approx 10$. **(d)** x:
$k$ (log scale), y: concentration gain $\Gamma_l(k)$ for the target reference (dark dotted, star) and hidden
layers 1–3 (blue circle / vermillion square / pink triangle); lower dotted line = uniform baseline
$\Gamma = 1$, upper dashed line = maximum $\Gamma = 5$. The gap between the star curve and the layer curves
in (d) is the result.

## Conclusion

**Verdict against the pre-registered decision rule.** The rule called the hypothesis supported only if
(i) target concentration rises with $k$, (ii) activation concentration rises consistently with it, (iii) the
effect strengthens with depth, and (iv) the sharpest model shows *low movement away from* $b_0$. Conditions
(i)–(iii) hold. **Condition (iv) fails, and by a wide margin**: at the step-function targets the deepest
layer keeps 27–28% of its movement out on the flanks against the target's 0%, and its concentration gain is
stuck near $1.5$ (primary grid) or $1.7$ (10,000-image control) against a maximum of $5.0$. Extending the
grid to $k=320$ converted what had looked like a slowly rising trend into a clear saturation.

**So: a switch-like continuous target is not sufficient to produce classification-style activation
plateaus.** The most informative single comparison is the 10,000-image control at $k=320$, where the
network's output is 78% of the way from uniform to a perfect switch while its deepest hidden layer is 16% of
the way. A network can compute an almost-discrete function through a representation that still slides
smoothly along the input direction. Whatever produces plateaus in classifiers, target discreteness alone is
not the mechanism.

**Limitations.** (1) One architecture (4-layer, width-200 MLP), one input direction (global brightness), one
dataset. Plateaus are usually reported in much larger models, and this result does not transfer
automatically. (2) One target family (normalized $\tanh$); a different route to discreteness — quantized
targets, a discrete latent bottleneck, or a classification head — could behave differently, and this
experiment says nothing about them. (3) Even the near-linear control $k=0.5$ is not perfectly uniform
($\Gamma_3 = 1.094$, and $s_3(b)$ peaks around $b \approx 0.6$), which is a baseline property of ReLU
networks under input rescaling rather than a target effect; we therefore read *changes* in $\Gamma$ and
$\Phi$ with $k$, not their absolute values. (4) The concentration window is fixed at the middle 20% of the
brightness range and centred on the true $b_0$, so a model whose transition sits slightly off-centre is
scored down — which if anything biases *toward* rejecting the hypothesis, and the primary-grid predictions
do sit slightly right of $b_0$ (Figure 3). (5) The primary grid under-fits the sharpest targets; the
10,000-image grid fixes that but shows no validation overfitting, so neither grid is ideal on every axis.
They agree, which is the reassuring part. (6) We do not claim this explains classification plateaus, and we
do not claim the activation path is a manifold in any technical sense.

**What would come next.** The natural follow-up is to reintroduce the ingredients we deliberately removed,
one at a time — a softmax head, cross-entropy loss, multi-class competition — and find which one, if any,
drives $\Phi_3$ toward zero. That would identify the actual mechanism rather than eliminating a candidate,
and it is a direct extension of this setup.

---

*Reproduce:* `python3 experiments/s1_dataset.py` (dataset checks), then
`python3 experiments/train.py --epochs 2000 --seeds 0 1 2 --tag main --save_ckpt`, then
`python3 experiments/analyze.py`, then `python3 experiments/plots.py`. The 10,000-image grid uses
`--epochs 200 --n_train 10000 --tag n10k --ckpt_prefix ckpt10k`, analyzed with
`CKPT_PREFIX=ckpt10k ANALYSIS_OUT=analysis_n10k.json`.
