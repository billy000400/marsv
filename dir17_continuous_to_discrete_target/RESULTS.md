# RESULTS — Can a switch-like *continuous* target create activation plateaus?

> CURRENT-BEST ONLY. One row per experiment. No history, no superseded/weaker variants
> (those live in CHANGELOG.md). Full write-up: [REPORT.md](REPORT.md).

## Headline

**No — target sharpness alone is not sufficient.** Morphing a continuous MNIST brightness-regression
target from near-linear ($k=0.5$) to near-binary-switch ($k=10$), with inputs, architecture, loss and
schedule held fixed, does concentrate activation movement near the target's transition — monotonically
in $k$, more strongly in deeper layers, consistently across 3 seeds. But the effect saturates far below
the target's own sharpness and never produces plateau-transition-plateau structure: at $k=10$ the
deepest hidden layer still spends two thirds of the uniform-baseline amount of movement in the outer
40% of the brightness range.

## Metrics

**Setup.** 4-layer ReLU MLP (784→200→200→200→1), MNIST images L2-normalized then scaled to brightness
$b\sim U(0.4,1)$, target $y_k(b)=\tanh(k(b-b_0))/\tanh(0.3k)$ with $b_0=0.7$; MSE, AdamW, 10k steps with
cosine decay; identical for every $k$. Probe: 100 digit-balanced held-out **test** images × 201
brightness values, post-ReLU hidden layers 1–3.

**Scores.** $R_l(k)$ = share of the layer's total activation movement falling in the central 20% of the
brightness range, divided by 0.2 (**1.0 = uniform, higher = more plateau-like**). $F_l(k)$ = share left
in the outer 40% (**0.4 = uniform, lower = more plateau-like**). Sweep $R^2$ = fit of prediction to
target on held-out images. Definitions and equations: [REPORT.md](REPORT.md) § Methods.

### Primary — 1000 training images (passes the pre-registered training-adequacy check)

Final checkpoint; mean over 100 held-out images and 3 seeds; ± = 95% CI across seeds.

| $k$ | target $R$ | prediction $R$ | $R_1$ | $R_2$ | $R_3$ (deepest) | target $F$ | $F_3$ | sweep $R^2$ | val MSE | $\rho_{\text{val}}$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 1.01 | 1.12 | 1.006 ± 0.001 | 1.034 ± 0.006 | **1.094 ± 0.010** | 0.397 | 0.356 ± 0.004 | 0.899 | 0.0307 | 1.058 |
| 1   | 1.03 | 1.14 | 1.007 ± 0.001 | 1.036 ± 0.010 | **1.105 ± 0.016** | 0.389 | 0.353 ± 0.005 | 0.900 | 0.0314 | 1.059 |
| 2   | 1.11 | 1.18 | 1.007 ± 0.002 | 1.044 ± 0.007 | **1.137 ± 0.011** | 0.357 | 0.342 ± 0.005 | 0.897 | 0.0338 | 1.056 |
| 5   | 1.61 | 1.42 | 1.011 ± 0.003 | 1.092 ± 0.013 | **1.326 ± 0.015** | 0.209 | 0.283 ± 0.005 | 0.900 | 0.0475 | 1.079 |
| 10  | 2.70 | 1.80 | 1.015 ± 0.003 | 1.130 ± 0.004 | **1.455 ± 0.036** | 0.048 | 0.265 ± 0.016 | 0.862 | 0.0884 | 1.064 |

All 15 runs adequate: validation minimum reached at epoch 345–955 of 2000, final validation loss
3.9–9.3% above it (requirement ≤ 20%), training loss at a smooth floor. Spread across the 100 probe
*images* is larger than across seeds and grows with $k$: SD of $R_3$ = 0.036 at $k=0.5$, 0.188 at
$k=10$.

### Robustness — same grid, 10× more training data (secondary check)

Fits far better but shows essentially no overfitting ($\rho_{\text{val}}\approx1.005$), so it fails the
adequacy gate and is not the primary result. Effect is **larger**, verdict identical.

| $k$ | prediction $R$ | $R_1$ | $R_2$ | $R_3$ | $F_3$ | sweep $R^2$ | val MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 1.06 | 1.009 | 1.034 | 1.045 ± 0.004 | 0.377 | 0.981 | 0.0059 |
| 1   | 1.07 | 1.009 | 1.036 | 1.055 ± 0.001 | 0.372 | 0.984 | 0.0055 |
| 2   | 1.14 | 1.010 | 1.048 | 1.109 ± 0.002 | 0.353 | 0.985 | 0.0056 |
| 5   | 1.57 | 1.013 | 1.113 | 1.419 ± 0.016 | 0.263 | 0.985 | 0.0068 |
| 10  | 2.42 | 1.018 | 1.179 | **1.823 ± 0.222** | 0.204 | 0.978 | 0.0143 |

Checkpoint robustness: reading the minimum-validation-loss checkpoint instead of the final one changes
$R_3(k=10)$ from 1.455 to 1.451. No conclusion depends on it.

## Figures

The five targets differ only in sharpness — amplitude is normalized away, so $k$ is the only variable:

![five tanh target curves of increasing sharpness](plots/target_functions.png)

**Figure 1.** x: brightness $b$; y: target $y_k(b)$. Series = the five $k$ settings (color + line style).
Dotted vertical line = transition centre $b_0=0.7$; grey band = the central window $[0.64,0.76]$.

A plateau claim from a badly trained model means nothing, so we verify training adequacy first:

![train and validation loss curves for the five k settings](plots/training_curves.png)

**Figure 2.** Left — x: epoch, y: training MSE on the full 1000-image training set (log scale). Right —
x: epoch, y: validation MSE on 2000 held-out images; open marker = each curve's minimum. Series = the
five $k$ settings (seed 0; other seeds visually identical). Every run ends at a smooth floor, slightly
past a validation minimum.

The concentration scores are only meaningful if the models learned the map — and how *sharply* they
learned it is itself part of the story:

![target vs mean prediction across brightness, one panel per k](plots/prediction_sweeps.png)

**Figure 3.** x: brightness $b$; y: target / prediction. Dotted dark = true target; solid colored = mean
prediction over 100 held-out images and 3 seeds; band = ±1 SD across images. Sweep $R^2$ in each title.
At $k=10$ the prediction is visibly softer than the target it was trained on.

The direct test of the hypothesis — where along the brightness path does the representation move?

![normalized movement vs brightness for the target and the three hidden layers](plots/activation_movement_by_k.png)

**Figure 4.** x: brightness $b$; y: normalized local movement $s(b)$ (share of the path's total travel).
Panels: the target's own $|\Delta y_k|$, then hidden layers 1, 2, 3 (deepest). Series = the five $k$
settings; bands = 95% CI across seeds; dotted horizontal line = uniform movement (0.005); grey band =
central window. **Y-scales differ per panel** — layer 1 spans only 0.0045–0.0051 (essentially uniform)
while the target panel spans 0–0.015.

Reduced to one number per curve, both the positive and the negative half of the verdict:

![concentration ratio and flank movement fraction versus k](plots/concentration_vs_k.png)

**Figure 5.** x (both): target sharpness $k$ (log scale). **(a)** y: concentration ratio $R=C/0.2$,
higher = more plateau-like, dotted line = uniform baseline $R=1$. **(b)** y: flank movement fraction
$F$, lower = more plateau-like, dotted line = uniform baseline $F=0.4$. Series: target reference (dark
dotted, star), model output (orange dash-dot, triangle), hidden layers 1/2/3 (blue solid circle /
vermillion dashed square / pink dash-dot triangle). Error bars = 95% CI across 3 seeds.

Neither the checkpoint choice nor the data-limited fit explains the ceiling:

![two-panel robustness check: checkpoint choice and training-set size](plots/checkpoint_robustness.png)

**Figure 6.** x (both): $k$ (log scale); y: concentration ratio $R$; dotted line = uniform baseline.
**(a)** Hidden layers 1/2/3 (color), final checkpoint (solid, filled markers) vs minimum-validation-loss
checkpoint (dashed, open markers) — identical. **(b)** Deepest layer $R_3$ (pink triangle) and model
output (orange inverted triangle) for 1000 training images (solid, filled) vs 10,000 (dashed, open),
with the target reference (dark dotted star). More data lifts both, but layer 3 stays far below the star.

The whole experiment in one figure:

![four-panel summary: targets, predictions, deepest-layer movement, concentration score](plots/main_summary.png)

**Figure 7.** **(a)** x: $b$, y: target $y_k(b)$. **(b)** x: $b$, y: mean prediction (dotted grey =
target). **(c)** x: $b$, y: normalized movement $s_3(b)$ in the deepest layer, 95% CI bands, dotted line
= uniform. **(d)** x: $k$ (log), y: $R_l(k)$ for the target (dark dotted star) and layers 1–3. Grey
vertical bands in (a)–(c) = central window $[0.64,0.76]$.
