# Does a switch-like *continuous* target create activation plateaus?

> Final, presentable, current-best only (no history — see CHANGELOG.md).

## Summary

Neural networks trained to classify show **activation plateaus**: as you slide an input smoothly from
one class toward another, the network's internal representation barely moves for a long stretch, then
lurches across a narrow transition, then goes quiet again. Plateaus matter for AI safety because they
are where a model's internal state stops tracking the input: a monitor reading those activations sees
"nothing changed" while the input has in fact drifted a long way, and interpretability tools that
assume a smooth input→representation map are least reliable exactly there. A natural guess is that
plateaus are an artifact of the *discrete* supervision signal — one-hot labels are a step function, so
maybe the network copies that step inward.

This experiment tests that guess without ever using a discrete target. We train the same 4-layer MLP on
the same MNIST images to predict one continuous number from image **brightness**, and we vary a single
knob $k$ that morphs the target from a near-straight line ($k=0.5$) to a near-binary switch ($k=10$).
Everything else — inputs, architecture, loss, optimizer, schedule, seeds — is held fixed.

**Result: partial support, with a clear ceiling.** Activation movement does become more concentrated
near the target's transition as $k$ grows, the effect is monotone in $k$, grows with depth, and is
consistent across 3 seeds — the deepest hidden layer's concentration ratio rises from
$R_3 = 1.094 \pm 0.010$ at $k=0.5$ to $R_3 = 1.455 \pm 0.036$ at $k=10$ (95% CI across seeds; 1.0 means
perfectly uniform). But it is nowhere near a plateau. Over the same range the target's own
concentration rises to 2.70, and the fraction of representation movement left in the outer 40% of the
brightness range falls only from 0.356 to 0.265 in the deepest layer, versus 0.397 → 0.048 for the
target. The network *declines to be a switch*: even when trained on a near-binary target it fits it
with a visibly softer curve, and its hidden layers keep moving steadily everywhere.

That softening is partly a data limit, and correcting it does not rescue the plateau. Retraining the
whole grid on 10× more images makes the models fit the switch almost exactly at their output
(concentration 2.42 vs the target's 2.70) and lifts the deepest layer to $R_3 = 1.823 \pm 0.222$ — but
that layer still leaves 0.204 of its movement in the flanks, half the uniform baseline and four times
the target's 0.048. The gap between output sharpness and representation sharpness persists.

**Verdict: a switch-like continuous target is not sufficient to produce classification-style activation
plateaus.** It nudges the representation in that direction — measurably, monotonically, and more
strongly the deeper you look — but the plateau-transition-plateau structure does not appear. Target
sharpness alone does not explain plateaus.

## Methods

### Data & Model

**Dataset.** MNIST handwritten digits (28×28 grayscale, flattened to 784 values in $[0,1]$). Digit
labels are **never** a learning target; they are used only to make every split digit-balanced (equal
numbers of each digit). Each image $x$ is first rescaled to unit Euclidean length, then rescaled again
to a sampled *brightness* $b$:

```math
\tilde{x} = \frac{x}{\lVert x \rVert_2 + \epsilon}, \qquad x_b = b\,\tilde{x}, \qquad b \sim U(0.4,\,1.0)
```

Because $\lVert \tilde{x} \rVert_2 \approx 1$, the input's Euclidean norm *is* the brightness:
$\lVert x_b \rVert_2 = b$ to within $1.8\times10^{-7}$ (verified numerically over all splits). Digit
identity and handwriting style are nuisance variation the model must ignore. Every image gets one fixed
brightness, drawn once per seed and **shared by all five $k$ settings**, so the five models see
bit-identical inputs and differ only in the scalar they are asked to predict.

**Splits.** Training set: 1000 digit-balanced images from MNIST-train[:50000]. Validation set: 2000
digit-balanced images from MNIST-train[50000:]. Probe set for all plateau measurements: 100
digit-balanced images from the MNIST **test** set (10 per digit), never seen in training or validation,
and identical across all $k$ and all seeds. The 1000-image training set is inherited from this
project's canonical MNIST plateau model, and is what makes the mild validation overfitting required by
the pre-registered training-adequacy check possible at all; a 10× larger training set is reported as a
robustness check in Result 5.

**Target family.** One knob, $k$, controls how switch-like the target is. With the transition centred
at $b_0 = 0.7$:

```math
y_k(b) = \frac{\tanh\!\big(k\,(b-b_0)\big)}{\tanh(0.3\,k)}
```

The denominator normalizes the endpoint range to $[-1,1]$ for every $k$, so target *amplitude* is not a
confound — only *sharpness* changes. We use $k \in \{0.5, 1, 2, 5, 10\}$: $k=0.5$ is nearly linear,
$k=10$ is continuous but nearly a binary switch (Figure 1). The target is a function of brightness
alone, so the model must extract $b$ from the image and apply a scalar map.

![five tanh target curves of increasing sharpness](plots/target_functions.png)

**Figure 1.** The five continuous targets. x: brightness $b \in [0.4, 1.0]$; y: target value $y_k(b)$,
normalized to endpoint range $[-1,1]$ for every $k$. Series are the five sharpness settings $k$,
distinguished by both color and line style (solid $k$=0.5, dashed $k$=1, dash-dot $k$=2, dotted $k$=5,
dash-dot-dot $k$=10). The vertical dotted line is the transition centre $b_0 = 0.7$; the grey band is
the central window $[0.64, 0.76]$ used by the concentration score below. Only sharpness changes across
$k$ — every curve runs from $-1$ to $+1$.

**Model.** The project's canonical MNIST plateau network: a 4-layer ReLU MLP,
784 → 200 → 200 → 200 → 1, ReLU after every linear layer except the last. The only change from the
classification version is the output head (10 units → 1). "Hidden layer $l$" ($l \in \{1,2,3\}$) means
the **post-ReLU output of the $l$-th linear layer**, a 200-dimensional vector; layer 3 is the deepest.
These are the hook points for every activation measurement below.

**Training.** Mean-squared-error loss; AdamW (lr $10^{-3}$, weight decay 0.01); batch size 200; 2000
epochs (10,000 steps) with a cosine learning-rate decay to zero. Identical for all five $k$ and all
three seeds — no per-$k$ tuning. For a given seed, weight initialization and batch order are identical
across $k$. Three seeds (0, 1, 2) vary initialization, batch order, and the brightness assignments. We
save both the final checkpoint (used for all primary results) and the minimum-validation-loss
checkpoint (used only for the robustness check in Result 5).

### Metrics

Before trusting any representation measurement we have to know the models are trained comparably and
actually learned the map. Those are the first two metrics; the last three are the plateau measurements.

**Training-adequacy ratios** — a plateau claim from an undertrained or wildly overfit model is
worthless, so training is pre-registered as adequate only if validation loss bottoms out *before* the
end and then rises only mildly, and training loss ends at a smooth minimum rather than mid-oscillation.
With $L_{\text{val}}(e)$ the validation MSE at epoch $e$ and $E$ the final epoch:

```math
\rho_{\text{val}} = \frac{L_{\text{val}}(E)}{\min_e L_{\text{val}}(e)} \le 1.2,
\qquad
\rho_{\text{train}} = \frac{L_{\text{train}}(E)}{\min_e L_{\text{train}}(e)} \approx 1
```

Read them as "how far above its own best does training end". Result 1 and Figure 3 report these.

**Sweep $R^2$** — the concentration scores below are only interesting if the model actually learned
$y_k$, and a model that ignored the input entirely would still produce some movement pattern. This is
the ordinary coefficient of determination of the model's prediction against the true target, computed
over the full brightness sweep on the held-out probe images ($N=100$ images $\times$ $S=201$ brightness
values), where $\mathrm{Var}[y_k]$ is the variance of the target over the sweep grid:

```math
R^2 = 1 - \frac{\frac{1}{NS}\sum_{i,j}\big(\hat{y}(x_{b_j}^{(i)}) - y_k(b_j)\big)^2}{\mathrm{Var}\big[y_k\big]}
```

1.0 is a perfect fit, 0.0 is no better than predicting the target's mean. Result 2 and Figure 4 report it.

**Normalized local activation movement $s_l(b)$** — this is the core measurement. For each held-out
image we walk brightness across a grid of $S=201$ evenly spaced values $b_1 < \dots < b_S$ spanning
$[0.4, 1.0]$, and ask *where along that walk* the representation changes. The raw step size is the
Euclidean distance between consecutive activation vectors,

```math
m_l(b_i) = \big\lVert h_l(b_{i+1}) - h_l(b_i) \big\rVert_2 ,
```

but raw distances are not comparable across models, layers, or images — a model whose activations are
simply larger moves more everywhere. So we normalize each image's path to total length 1:

```math
s_l(b_i) = \frac{m_l(b_i)}{\sum_{j} m_l(b_j) + \epsilon}
```

$s_l$ is therefore a *distribution over brightness* saying what share of the representation's total
travel happens at each point. Perfectly uniform movement gives $s_l = 1/200 = 0.005$ everywhere; a
plateau-transition-plateau structure gives a curve near zero at both ends with a spike near $b_0$.
Curves are averaged over the 100 probe images and then over seeds. Result 3 and Figure 5 report it.

**Concentration ratio $R_l(k)$** — the one number summarizing "how much of the movement happens at the
transition". Take the middle 20% of the brightness range, $[0.64, 0.76]$ (exactly 40 of the 200
movement points), and sum the normalized movement inside it:

```math
C_l(k) = \sum_{b_i \in [0.64,\,0.76)} s_l(b_i), \qquad R_l(k) = \frac{C_l(k)}{0.2}
```

Dividing by 0.2 makes the scale readable: $R = 1$ means movement is uniform, $R = 2$ means the central
20% of brightness absorbs twice its fair share of the representation's travel, higher is more
plateau-like. This is the headline metric — the hypothesis predicts $R_l(k)$ rises with $k$. Result 4
and Figure 6a report it.

**Flank movement fraction $F_l(k)$** — $R$ alone can be misleading: a curve can gain a modest central
bump while still moving briskly everywhere else, which is *not* a plateau. So we also measure what is
left in the flanks — the outer 40% of the range, $b < 0.52$ or $b \ge 0.88$:

```math
F_l(k) = \sum_{b_i \,<\, 0.52} s_l(b_i) \;+\; \sum_{b_i \,\ge\, 0.88} s_l(b_i)
```

Uniform movement gives $F = 0.4$; a genuine plateau at both ends drives $F \to 0$. Lower is more
plateau-like. This is the metric that separates "somewhat more concentrated" from "actually flat", and
it carries the negative half of the verdict. Result 4 and Figure 6b report it.

### Baselines

There is no competing method to beat here — the question is whether a measured curve differs from
*no structure*, and how it compares to the sharpness it was trained on. Three reference lines:

**Uniform-movement baseline** — what a representation that changes at a perfectly constant rate along
the brightness path would score. It follows directly from the normalization: $s_l(b_i) = 1/200$ for all
$i$, hence

```math
R_l^{\text{unif}} = 1, \qquad F_l^{\text{unif}} = 0.4
```

Every panel of Figure 6 draws this as a dotted grey line. Deviation from it is the effect.

**Target-curve reference** — the sharpness the model was *asked* for, obtained by running the identical
pipeline on the target function itself instead of on activations:

```math
s^{\text{tgt}}(b_i) = \frac{|y_k(b_{i+1}) - y_k(b_i)|}{\sum_j |y_k(b_{j+1}) - y_k(b_j)|}
```

then scoring it with the same $C/0.2$ and flank sums. This is the ceiling: it says how concentrated the
supervision signal is at each $k$, so we can see how much of the target's sharpness the representation
actually inherits.

**Prediction-curve reference** — the same pipeline applied to the model's own *output* sweep
$\hat{y}(b)$ rather than to a hidden layer. This sits between the two and separates two very different
explanations of a weak effect: if the output is sharp but hidden layers are not, the network has
learned the switch and confined it to the last linear map; if the output itself is soft, the network
never learned the switch at all.

## Results

All numbers are the final checkpoint, averaged over 100 held-out digit-balanced test images and 3
seeds; $\pm$ is a 95% confidence interval across seeds.

| $k$ | target $R$ | pred. $R$ | $R_1$ | $R_2$ | $R_3$ (deepest) | target $F$ | $F_3$ | sweep $R^2$ | val MSE | $\rho_{\text{val}}$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 1.01 | 1.12 | 1.006 ± 0.001 | 1.034 ± 0.006 | **1.094 ± 0.010** | 0.397 | 0.356 ± 0.004 | 0.899 | 0.0307 | 1.058 |
| 1   | 1.03 | 1.14 | 1.007 ± 0.001 | 1.036 ± 0.010 | **1.105 ± 0.016** | 0.389 | 0.353 ± 0.005 | 0.900 | 0.0314 | 1.059 |
| 2   | 1.11 | 1.18 | 1.007 ± 0.002 | 1.044 ± 0.007 | **1.137 ± 0.011** | 0.357 | 0.342 ± 0.005 | 0.897 | 0.0338 | 1.056 |
| 5   | 1.61 | 1.42 | 1.011 ± 0.003 | 1.092 ± 0.013 | **1.326 ± 0.015** | 0.209 | 0.283 ± 0.005 | 0.900 | 0.0475 | 1.079 |
| 10  | 2.70 | 1.80 | 1.015 ± 0.003 | 1.130 ± 0.004 | **1.455 ± 0.036** | 0.048 | 0.265 ± 0.016 | 0.862 | 0.0884 | 1.064 |

### Result 1 — All 15 runs are adequately trained, so the comparison is fair

The whole experiment rests on the five models differing *only* in their target, so we check the
pre-registered adequacy conditions before reading anything into activations. All 15 runs (5 values of
$k$ × 3 seeds) pass: validation loss reaches its minimum well before the end (epochs 345–955 of 2000)
and ends 3.9–9.3% above it ($\rho_{\text{val}} \le 1.093$, requirement $\le 1.2$), and the cosine decay
brings training loss to a smooth floor rather than leaving it mid-oscillation. Validation MSE rises
with $k$ (0.031 → 0.088) simply because the sharper targets have larger variance and a harder map.

![train and validation loss curves for the five k settings](plots/training_curves.png)

**Figure 2.** Training adequacy. Left — x: epoch; y: training MSE over the full 1000-image training set
(log scale). Right — x: epoch; y: validation MSE on 2000 held-out images (linear scale); the open
marker on each curve marks its minimum. Series are the five $k$ settings (color + line style, same
encoding as Figure 1); seed 0 shown, the other two seeds are visually identical. The mid-training
oscillation in the left panel is AdamW at a high learning rate; the cosine decay removes it, and every
run ends at its own smooth floor. Sharper targets (larger $k$) sit at higher validation loss throughout.

### Result 2 — The models learn the map, but systematically *soften* the switch

If the models had failed to fit the sharp targets, any negative plateau result would be uninformative.
They fit well: sweep $R^2$ is 0.86–0.90 across all $k$, so on unseen images the predicted curve tracks
the true target closely. But Figure 3 shows something more interesting than a pass/fail: at $k=5$ and
especially $k=10$ the mean prediction is a visibly *gentler* sigmoid than the target it was trained on.
The prediction-curve reference quantifies this — at $k=10$ the target's concentration is 2.70 but the
model's own output only reaches 1.80. The network could in principle place a sharp step in its final
linear layer; it does not.

![target vs mean prediction across brightness, one panel per k](plots/prediction_sweeps.png)

**Figure 3.** Fit quality on 100 held-out test images. x: brightness $b$; y: target / prediction value.
Dotted dark curve = true target $y_k(b)$; solid colored curve = prediction averaged over images and
seeds; the shaded band is $\pm 1$ standard deviation *across images* (nuisance variation from digit
identity and style). The grey vertical band is the central window $[0.64, 0.76]$. Panels are the five
$k$ settings with their sweep $R^2$ in the title. Note the $k=10$ panel: the solid prediction is
noticeably shallower than the dotted target through the transition.

### Result 3 — Movement shifts toward the transition, but never leaves the flanks

This is the direct test of the hypothesis: does the representation stop moving away from $b_0$ and
lurch across it? Figure 4 plots the normalized movement $s_l(b)$ for the target itself and for each
hidden layer. The target panel is the shape a plateau would look like — at $k=10$ it is essentially
zero outside a narrow spike. The hidden layers do not do that. Layer 1 is flat to within a few percent
at every $k$. Layers 2 and 3 develop a bump near $b_0$ that grows with $k$, but even at $k=10$ the
deepest layer is still moving at roughly a third of its peak rate at both extremes of the brightness
range.

A caveat visible in the same figure: even at $k=0.5$, where the target is almost a straight line, the
deeper layers are not perfectly uniform — $s_3$ peaks around $b \approx 0.6$ and falls off toward
$b=1.0$. That baseline curvature is a property of a ReLU network under input rescaling, not an effect
of the target, which is exactly why the $k=0.5$ model is the right control and why we read *changes*
in $R_l$ and $F_l$ with $k$ rather than their absolute values.

![normalized movement vs brightness for the target and the three hidden layers](plots/activation_movement_by_k.png)

**Figure 4.** Where the representation moves. x: brightness $b$; y: normalized local movement
$s(b)$ — the share of the path's total travel occurring at that brightness. Leftmost panel: the target's
own normalized change $|\Delta y_k|$. Remaining panels: hidden layers 1, 2 and 3 (deepest). Series are
the five $k$ settings (color + line style, as in Figure 1); shaded bands are 95% CIs across the 3 seeds.
The horizontal dotted line marks perfectly uniform movement (0.005); the grey vertical band is the
central window $[0.64, 0.76]$. **Each panel has its own y-scale** — note that layer 1 spans only
0.0045–0.0051, i.e. it is essentially uniform, whereas the target panel spans 0–0.015.

### Result 4 — The effect is monotone in $k$, grows with depth, and saturates far below the target

Reducing Figure 4 to one number per curve gives the headline. Concentration $R_l(k)$ rises
monotonically with $k$ in every layer, and the slope grows with depth: layer 1 moves 1.006 → 1.015
(essentially nothing), layer 2 1.034 → 1.130, layer 3 1.094 → 1.455. The $k=10$ and $k=0.5$ confidence
intervals for layer 3 are far apart (1.455 ± 0.036 vs 1.094 ± 0.010), so the trend is real, not seed
noise. Both directional predictions of the hypothesis — more sharpness → more concentration, and deeper
→ stronger — hold.

Variation across *images* is much larger than across seeds, and it grows with $k$: the standard
deviation of $R_3$ over the 100 probe images is 0.036 at $k=0.5$ but 0.188 at $k=10$ (versus 0.010 and
0.036 for the seed-level CIs). Sharper targets therefore make the representation's behaviour not just
more concentrated on average but more image-dependent — individual digits differ in *where* along the
brightness path their representation turns over. The seed-level intervals quoted above are the relevant
uncertainty for the group means, since each mean already averages over the same 100 images.

The magnitude is the problem. At $k=10$ the target scores 2.70 and the model's own output 1.80, but the
deepest representation only reaches 1.455. The flank fraction tells the same story more starkly: the
target's flank movement collapses from 0.397 to 0.048 (an 88% drop), while layer 3's falls only from
0.356 to 0.265 (a 26% drop) — two thirds of the uniform-baseline flank movement survives. A
plateau would require this number near zero.

![concentration ratio and flank movement fraction versus k](plots/concentration_vs_k.png)

**Figure 5.** The two summary scores against target sharpness. Both panels — x: target sharpness $k$
(log scale). **(a)** y: concentration ratio $R = C/0.2$, the share of movement in the central 20% of
brightness relative to its uniform share; the dotted grey horizontal line at $R=1$ is the
uniform-movement baseline, higher = more plateau-like. **(b)** y: flank movement fraction $F$, the share
of movement left in the outer 40% of the range; the dotted grey line at $F=0.4$ is the uniform baseline,
**lower** = more plateau-like. Series in both panels: the target reference (dark dotted, star), the
model's own prediction curve (orange dash-dot, triangle), and hidden layers 1, 2, 3 (blue solid circle /
vermillion dashed square / pink dash-dot triangle). Error bars are 95% CIs across 3 seeds. The gap
between the star and the pink layer-3 curve is the headline: the representation inherits only a fraction
of the sharpness it was trained on.

### Result 5 — Robustness: the checkpoint does not matter; the training-set size raises the effect but not the verdict

Two things could make the ceiling in Result 4 an artifact rather than a fact about the network, and we
check both.

**Checkpoint.** The primary analysis uses the final, slightly overfit checkpoint, so we repeat the whole
measurement at each run's minimum-validation-loss checkpoint. The curves are indistinguishable — at
$k=10$, $R_3 = 1.455$ (final) vs $1.451$ (min-val); at $k=0.5$, $1.094$ vs $1.094$ (Figure 6a).

**Training-set size.** Result 2 showed the models under-sharpen their own output, which could be a
data-limitation rather than a representational one. We therefore retrain the entire grid (5 values of
$k$ × 3 seeds) on **10,000** training images with the identical number of gradient steps, optimizer and
schedule. These models fit far better (sweep $R^2$ 0.978–0.985 vs 0.862–0.900; validation MSE at $k=10$
0.088 → 0.014) and, at $k=10$, their *output* now nearly reproduces the switch (prediction concentration
2.42 vs the target's 2.70, up from 1.80). Representation concentration rises with it: $R_3$ at $k=10$
goes from $1.455 \pm 0.036$ to $1.823 \pm 0.222$, and the flank fraction $F_3$ from 0.265 to 0.204
(Figure 6b). So the effect is real and scales with how well the switch is learned — the primary result
understates it.

The verdict does not change. Even in this better-fitting regime the deepest representation reaches only
1.82 against the target's 2.70, and $F_3 = 0.204$ is still **half** the uniform-movement baseline of
0.4, against the target's 0.048. The layer ordering is also unchanged ($R_1 = 1.018$, $R_2 = 1.179$,
$R_3 = 1.823$ at $k=10$). A model that has essentially learned to output a switch still does not carry a
plateau in its hidden layers.

These 10,000-image runs are reported as a secondary check only: their validation loss ends
$\rho_{\text{val}} \approx 1.005$ above its minimum, i.e. they show essentially **no** overfitting and
therefore fail the pre-registered adequacy condition that the primary models satisfy. This is exactly
the tension that fixed the primary training set at 1000 images: the adequacy gate demands mild
overfitting, which a 10,000-image training set on a one-dimensional target does not produce.

![two-panel robustness check: checkpoint choice and training-set size](plots/checkpoint_robustness.png)

**Figure 6.** Robustness. Both panels — x: target sharpness $k$ (log scale); y: concentration ratio
$R$; error bars are 95% CIs across 3 seeds; the dotted grey horizontal line is the uniform-movement
baseline $R=1$. **(a)** Checkpoint choice. Series: hidden layers 1, 2, 3 (color as in Figure 5), each
shown twice — solid line with filled markers = final checkpoint, dashed line with open markers =
minimum-validation-loss checkpoint. The two lie on top of one another everywhere. **(b)** Training-set
size, deepest layer and model output. Series: target reference (dark dotted, star); deepest hidden layer
$R_3$ (pink, triangle) and the model's output-curve concentration (orange, inverted triangle), each
shown for the primary 1000-image models (solid, filled markers) and the 10,000-image models (dashed,
open markers). More data moves both curves up, but the pink layer-3 curves stay far below the star.

### The whole experiment on one page

Figure 7 puts the four steps of the argument side by side: what the models were asked to predict, what
they actually predicted, where their deepest representation moved, and the resulting scores.

![four-panel summary: targets, predictions, deepest-layer movement, concentration score](plots/main_summary.png)

**Figure 7.** Summary. **(a)** x: brightness $b$, y: target $y_k(b)$ — the five targets, linear to
switch-like. **(b)** x: $b$, y: mean prediction $\hat{y}(b)$; dotted grey curves are the corresponding
targets — the models soften every sharp target. **(c)** x: $b$, y: normalized movement $s_3(b)$ in the
deepest hidden layer, with 95% CI bands across seeds; the dotted horizontal line is uniform movement
(0.005) — a bump appears near $b_0$ at high $k$ but the flanks never go quiet. **(d)** x: $k$ (log
scale), y: concentration ratio $R_l(k)$ for the target (dark dotted, star) and hidden layers 1–3
(color + marker as in Figure 5), error bars 95% CI across seeds — activation concentration rises with
$k$ and with depth, but tracks far below the target. Grey vertical bands in (a)–(c) mark the central
window $[0.64, 0.76]$.

## Conclusion

Making a continuous regression target more switch-like *does* push a ReLU network's internal
representation toward plateau-like structure, and the push behaves exactly as the plateau hypothesis
predicts qualitatively: it is monotone in target sharpness, it is absent in the first hidden layer and
strongest in the deepest, and it is stable across seeds and across checkpoints. That is a real,
measurable effect from changing nothing but one number in the loss.

But it is small, and it stops well short of what "plateau" means. At the sharpest target we tested — a
tanh so steep it is nearly a step — the deepest hidden layer concentrates only 1.46× its uniform share
of movement into the transition window and still spends two thirds of the uniform-baseline amount of
movement out in the flanks. There is no stretch of brightness where the representation is quiet.

Part of that shortfall is a fitting limit rather than a representational one: the 1000-image models
answer a near-binary target with a visibly soft sigmoid (output concentration 1.80 against the target's
2.70), and giving them 10× more data pushes the output to 2.42 and the deepest layer to 1.82. But the
rest of the shortfall survives the fix. Even the well-fit models leave 0.204 of their deepest layer's
movement in the flanks — half of what perfectly uniform movement would leave, and four times what the
target does. Sharpening the *function* a network computes turns out to be much easier than flattening
the *representation* on either side of the transition.

For the broader question this direction serves — *are classification plateaus caused by discrete
targets?* — the answer here is **no, not by target sharpness alone**. A continuous target that is
almost a step function does not reproduce them. Whatever produces plateaus in classifiers must involve
something this setup lacks: genuinely discrete supervision, a many-class output geometry, cross-entropy's
unbounded push on the logit margin, or the fact that a classifier's decision only depends on an argmax.

**Limitations.** (i) One architecture (4-layer, width-200 ReLU MLP), one dataset, one loss (MSE) — as
pre-registered; we deliberately did not compare architectures or losses, so we cannot say the ceiling
is a property of *all* networks. (ii) The input path is a single one-dimensional ray, brightness
rescaling of a fixed image; classification plateaus are usually probed along paths between two
different images, and those paths may behave differently. (iii) The primary models are trained on 1000
images and reach near-zero training loss — the canonical setting for this project's plateau work, and
the only regime that produces the mild overfitting the pre-registered adequacy check demands, but a
memorization regime nonetheless. Result 5 quantifies the cost: the 10,000-image models fit far better
and show a substantially larger effect, so the primary numbers are a lower bound on the effect size,
though not on the verdict. (iv) $k=10$ is sharp but still smooth; we did not test the exact
discontinuous limit, which is outside this direction's continuous-target scope.
