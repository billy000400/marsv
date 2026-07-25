# RESULTS — Direction 12.1: stable third-class predictions on MNIST activation paths

> CURRENT-BEST ONLY. One row per experiment. No history, no superseded/weaker variants
> (those live in CHANGELOG.md). Full write-up with all metric definitions: **REPORT.md**.

**Question.** When you slide the first-hidden-layer activations of a trained MNIST classifier from a
real image of digit *a* to a real image of digit *b*, the network often confidently predicts some
third digit *z* in between. Is that prediction backed by the activations — are those interpolated
activations actually close to the activations real *z* images produce?

**Setup.** 4-layer MLP (784→200→200→200→10, ReLU, MSE loss, full 60k MNIST), direction-12 checkpoints
at step 30,000 for seeds 0/1/2. Hook point: post-ReLU **`h1`** (200-d). All 45 unordered digit pairs ×
100 fixed test-image pairs × 50 interpolation points = 4,500 paths per seed. Nothing retrained.

## Metrics

### Stage-1 census — how common is a stable third-class prediction? (seed 0, primary)

| category | count of 45 |
|---|---|
| stable third-class prediction **and** stable sub-plateau | 6 |
| stable third-class prediction only | 13 |
| stable sub-plateau only | **0** |
| neither | 26 |
| **any stable third-class prediction** | **19** |

Prevalence-threshold sensitivity: 23 transitions at 25%, **19 at the frozen 50%**, 7 at 75%.
Leave-one-out over the 100 pairs changes no third-class label (100/100 on all 45 transitions).
All 19 labels are identical on the correctly-classified-endpoint subset (endpoint accuracy 98.7%).
Dominant third digit at seed 0: **7 in 13 cases, 8 in 6 cases** — the third answer comes from a set
of two.

### Stage-4 activation regions — is the prediction backed by the activations? (seed 0)

| quantity | value |
|---|---|
| third-class segment points pooled over the 19 transitions | 14,700 (on 1,376 paths) |
| **fraction inside the real activation region of the predicted digit** | **2.5%** |
| best single transition (2→9, z=7) | 18.5% of points, 14.3% of paths |
| transitions below 5% of points inside | 15 of 19 (9 at exactly 0) |
| median normalized distance to the predicted digit z, on segment | **1.50 – 3.71** (×the held-out 95th pct) |
| median normalized distance to the **nearest of all ten** digits, on segment | **1.23 – 2.09** |
| nearest real region pooled over segment points | digit 5 (6,108), digit 2 (2,073), predicted digit 7 (1,984) |

**Verdict: 19 of 19 "prediction only"; 0 "activation-region match"; 0 "mixed".**
Segment points are not merely in the wrong digit's region — they are outside *every* digit's usual
spread (all ratios > 1).

### Per-transition detail (seed 0)

Prevalence = % of the 100 pairs on which z appears as a contiguous segment; run lengths in
interpolation points out of 50; ratio = normalized distance / that digit's held-out 95th percentile.

| transition | z | prev % | median run | IQR run | shelf (alpha @ d) | points inside real-z | paths majority inside | median ratio to z | median nearest ratio | z at seed 1 / 2 |
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
| 6→9 | 8 | 93 | 16 | 10–20 | 0.37–0.53 @ 0.50 | 0.000 | 0.000 | 2.14 | 1.37 | **8/8** |

"–" = not labelled stable third-class at that seed.

### Cross-seed confirmation (frozen rules, same image pairs, nothing retuned)

| seed | stable third-class | stable sub-plateau | dominant third digits |
|---|---|---|---|
| 0 | 19 of 45 | 6 of 45 | 7 (×13), 8 (×6) |
| 1 | 12 of 45 | 3 of 45 | 1 (×10), 3 (×1), 8 (×1) |
| 2 | 18 of 45 | 3 of 45 | 2 (×9), 8 (×9) |

All three seeds agree on the third-class label for 27 of 45 transitions and the sub-plateau label for
39 of 45. **Only 6→9 is stable at all three seeds AND names the same third digit (8) at all three.**
The phenomenon replicates; the identity of the third digit does not.

### Later-layer follow-up — does the match appear at `h2` or `h3`? (no)

Preregistered conditional follow-up, unlocked by the `h1` null. Same frozen segments, same rules, only
the hook point changes. "Regions containing the point" counts how many of the ten digit regions a
segment point falls inside (1 would be ideal; 10 means the inside test is vacuous).

| hook point | points inside real-z region | ratio to z < 1 | z is the *nearest* region | mean regions containing the point | median ratio to z | C1 | C3 (mean) |
|---|---|---|---|---|---|---|---|
| `h1` | **2.5%** | 2.4% | 14.3% | **0.08** | 1.50 – 3.71 | 0.936 | 0.946 |
| `h2` | **10.6%** | 78.4% | 11.7% | **5.80** | 0.25 – 4.33 | 0.949 | 0.984 |
| `h3` | **0.2%** | 0.2% | 0.6% | **0.00** | 174.8 – 471.0 | 0.950 | 0.966 |

The `h2` rise to 10.6% is **not** a partial match: at `h2` a segment point is inside 5.8 of the 10
regions at once, so "ratio to z < 1" is nearly automatic (78.4%), while the discriminating half of the
criterion does not improve — z is the *nearest* region for only 11.7% of points, at or below the 12.5%
expected from picking one of the eight non-endpoint digits at random. At `h3` the classes are so
tightly clustered that segment points sit 175–471× the held-out 95th percentile from the predicted
digit and inside zero regions. C1 and C3 pass at every layer, so the measure is calibrated throughout.

### Controls on the activation-region measure (all pass)

| control | result | pass? |
|---|---|---|
| C1 — held-out real images nearest their own digit region | 96.0% argmin-correct, 95.0% ratio < 1, 93.6% both | yes |
| C2 — endpoint-plateau points inside the endpoint region (alpha ≤ 0.1 / ≥ 0.9) | median 0.81; 33 of 38 cases > 0.5 (over *all* endpoint-predicted points: median 0.37) | yes, with caveat |
| C3 — within-digit interpolation stays inside its own region | 91.8%–96.5% of points, all ten digits | yes |
| C4 — reversing a path only reverses alpha | max relative deviation 1.7 × 10⁻⁷ | yes |

Caveat on C2: mid-path points predicted as an *endpoint* digit are also often outside that digit's
region, so "far from every real region" is a general property of the middle of a cross-digit path.
What remains specific to the third-class segment is that the distance to the named digit z is flat and
high along the whole path (see the distance-view figure) — the segment is not where the path comes
closest to z.

### Robustness flags

Three sub-plateau labels fail the preregistered bootstrap-agreement bar (≥ 0.90 over 1,000 resamples)
and are reported as borderline: **0→4** (0.684, labelled sub-plateau), **1→9** (0.636, labelled no
sub-plateau; also the only leave-one-out flip, 99/100) and **5→7** (0.895, no sub-plateau). All
third-class labels clear leave-one-out at 100/100.

## Figures

![Ten-by-ten matrix of all 45 digit transitions at seed 0. Rows are endpoint digit a, columns endpoint digit b; diagonal blank. Cell colour = category (grey neither, blue stable third-class only, orange sub-plateau only — never occurs, green both); white number = dominant third digit z.](plots/s1_transition_matrix.png)

![All 45 mean d(alpha) curves at seed 0. In each panel x = alpha (0 = first endpoint image, 1 = second), y = logit-space relative endpoint distance d(alpha) in [0,1]; thick line = pointwise mean over 100 image pairs, shaded band = ±1 standard deviation across pairs; dotted diagonal = reference d = alpha; grey horizontal bar = detected sub-plateau shelf. Line colour encodes category as in the matrix.](plots/s1_mean_curves_grid.png)

![Predicted-class composition for the 19 stable third-class transitions at seed 0. x = alpha from 0 to 1; y = fraction of the 100 image pairs predicting each digit at that alpha, stacked to 1; colours are predicted digits 0-9 per the shared legend.](plots/s1_class_composition.png)

![Per-pair third-class segment width, 19 stable transitions at seed 0. x = transition and its dominant third digit z; y = longest run of consecutive alpha points predicted z (out of 50; 0 = that path never predicts z). Boxes = interquartile range over the 100 pairs, orange line = median, green triangle = mean, circles = outlying pairs; dashed red line = frozen median-run threshold of 3.](plots/s1_segment_widths.png)

![Activation-region membership per transition at seed 0. x = the 19 stable third-class transitions with their z. Bars (left axis, fraction 0-1): blue = fraction of segment points inside the real-z region, orange = fraction of paths with a majority of segment points inside. Markers (right axis, normalized distance ratio): black diamonds = median ratio to z on the segment, red triangles = median ratio to the nearest of all ten digits; dotted line at ratio 1.](plots/s4_region_membership.png)

![Distance from the interpolated h1 activation to each real digit region, 19 stable transitions at seed 0. x = alpha from 0 to 1; y = normalized distance ratio on a log scale, averaged over 100 pairs. Coloured lines = ratio to endpoint digit a, endpoint digit b, and third digit z (labelled in each panel legend). Dashed line = ratio 1 (held-out 95th percentile). Grey band = alpha range where a majority of paths predict z.](plots/s4_distance_view.png)

![Two-dimensional PCA view of the three highest-prevalence transitions (0→1 z=7, 6→9 z=8, 3→4 z=7). Axes = first two principal components of real h1 activations, fitted only on real images of the two endpoint digits and the third digit (arbitrary units). Faint dots = real training images by digit, stars = per-digit means, thin grey lines = twelve interpolation paths, black crosses = interpolation points predicted as z. Visualization only; the conclusion comes from the full 200 coordinates.](plots/s4_pca_view.png)

![Controls on the h1 activation-region measure. Left (C1): histogram over 7,000 held-out real test images of the normalized distance ratio to their own digit region; x = ratio, y = image count, dashed line at 1. Middle (C3): x = digit c, y = fraction of within-digit (c→c) interpolation points with ratio < 1 to digit c. Right (C2): histogram of the fraction of endpoint-predicted points inside that endpoint digit's region, one entry per endpoint per transition (38 values); green = endpoint plateau only, red = all endpoint-predicted points.](plots/s5_controls.png)

![Later-layer follow-up over the 19 stable third-class transitions at seed 0, hook points h1 (blue), h2 (orange), h3 (green). Top: x = transition, y = fraction of third-class segment points inside the real-z activation region. Middle: x = transition, y = median normalized distance ratio on the segment (log scale); solid lines with circles = distance to the predicted digit z, dashed with triangles = distance to the nearest of all ten digits; dotted line at 1. Bottom: x = transition, y = mean number of the ten digit regions containing a segment point (1 ideal, 10 vacuous); dotted line at 1.](plots/s6_later_layers.png)

![Cross-seed comparison over all 45 transitions. Top: stable third-class label, transitions on x, seeds as rows, black = label true. Middle: same for the stable sub-plateau label. Bottom: dominant third digit z where the transition is stable third-class, digit printed and colour-coded; white = not stable at that seed.](plots/s1_seed_agreement.png)

## Headline

**A stable, reproducible class prediction is not evidence that the activations resemble that class.**
19 of 45 MNIST digit transitions make the same third-class prediction on a majority of 100 fixed image
pairs, over a median of 4–16 of 50 interpolation points — yet only **2.5%** of the 14,700 points in
those segments lie inside the real `h1` activation region of the digit they are predicted to be, and
the typical point is outside *every* digit's region. The similarity does not appear at `h2` or `h3`
either. Which digit gets named is a property of the
trained network (7/8 at seed 0, 1 at seed 1, 2/8 at seed 2; only 6→9 → "8" replicates across all
three), not of the digits being interpolated.
