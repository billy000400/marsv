# REPORT — Does a character-level GPT treat meaningful linguistic directions differently from arbitrary ones?

## Summary

Two constructed feature directions steer this model as intended (Figure 1), but the prediction
they test failed: the equal mixture is no later than the singles (Figure 2), the feature plane resembles
a random plane (Figure 3), and random directions moved the output most.

## Motivation

Does a language model have privileged internal directions — a few meaningful axes it computes with — or
is every direction alike? Recent work reports *feature-specific error correction*: models suppress small perturbations along their own
features. The visual prediction: pushing an activation along *one* meaningful feature direction should
change the output sooner than splitting the same push across *two*.

## Setup

**Data & model.** tinyshakespeare (1.1M characters, 65-character vocabulary); a 12-layer, 12-head,
width-240 GPT, 128-character context, trained to 55.0% next-character accuracy. Directions come from the
corpus's first 90%, all measurements from the held-out last 10%. We intervene at the residual stream after block 0,
last character position, chosen in advance.

**Directions.** Each is the unit-normalized mean of 30 matched positive-minus-negative differences.
*Speaker label*: inside an all-capital speaker name starting a line, versus ordinary capitalized text.
*Word continuation*: the word continues with a letter, versus ends next. Pairs share a final character;
a disjoint 30 held-out pairs test each.

**Perturbation.** Move $h_0$ toward unit direction $v$ by size $\alpha$, length held fixed so no change
comes from enlarging it:

```math
h(\alpha) = \lVert h_0 \rVert \cdot \frac{\hat h_0 + \alpha v}{\lVert \hat h_0 + \alpha v \rVert}, \qquad \hat h_0 = \frac{h_0}{\lVert h_0 \rVert}
```

**Metrics.** Does a direction mean what we claim? Score the probability on the characters it predicts,
a set $T$:

```math
m_T(P) = \sum_{c \in T} P(c)
```

$T$ is uppercase-or-colon (speaker label) or any letter (word continuation); higher = the steer worked
(Figure 1). How much did the whole output move? Jensen–Shannon divergence in bits between perturbed and
clean distributions, zero when identical, at most 1 bit (Figures 2 and 3):

```math
\mathrm{JSD}(P, Q) = \tfrac{1}{2}\mathrm{KL}(P \Vert M) + \tfrac{1}{2}\mathrm{KL}(Q \Vert M), \qquad M = \tfrac{1}{2}(P + Q)
```

**Baselines.** Random unit directions drawn isotropically in the 240-dimensional residual space and
perturbed identically, 10 per measurement; a feature direction behaving like these is not special. The
**equal mixture** is the unit vector halfway between the two directions, overlap removed.

## Results

**Both directions steer as intended.** At $\alpha = 1$ speaker label lifts the probability of an
uppercase letter or colon from 0.0002 to 0.31, against 0.004 for random directions; word continuation
lifts letter probability from 0.25 to 0.78, against 0.26. Both are real handles on the output.

![Target-character probability against steering size for both feature directions](plots/fig1_steering.png)

**Figure 1.** x: steering size $\alpha$. Top y: probability mass on the target set — uppercase-or-colon
(left), any letter (right) — mean of 30 held-out contexts. Series: feature direction (solid/circles),
10-random mean (dashed/squares), unperturbed (dotted), single contexts (thin translucent). Bottom: one
context's next-character distribution at three $\alpha$, by bar hatching.

**The singles do not leave the low-change region before the mixture.** Over 30 held-out anchors, the mean
JSD curve crosses 0.05 bits at $\alpha = 0.39$ for the mixture, 0.38 for speaker label, 0.79 for word
continuation: the mixture falls *between* the singles. Per anchor it is near a coin flip: speaker label
exceeds the mixture in 7 of 16 anchors where both cross, word continuation in 12 of 19. **Random
directions cross earliest** (median $\alpha = 0.82$ versus 1.00).

![Output change against perturbation size for both features, their mixture, and random directions](plots/fig2_curves.png)

**Figure 2.** x: perturbation size $\alpha$; y: JSD (bits) from the clean distribution.
Series: speaker label (solid/circles), word continuation (dashed/squares), equal mixture
(dash-dotted/triangles), 10-random mean (dotted gray/diamonds). Left: mean over 30 anchors. Right: three
single anchors, context above each.

**The feature plane looks like a random plane.** With each axis rescaled so its single-direction boundary
sits at $\pm 1$, the low-change region has nearly the same shape and area in both planes (34% versus 32%
of the grid below 0.05 bits). At the 45° point the mean JSD is 0.12 bits against 0.17 — the predicted
direction, but lower in only 5 of 8 anchors (0.008–0.47).

![Mean output change over the feature-feature plane and a random-random plane](plots/fig3_planes.png)

**Figure 3.** Axes: rescaled coefficients; $\pm 1$ (white circles) is where a move along that axis alone
first reaches JSD = 0.05 bits. Color: mean JSD from clean, bits (viridis, dark = unchanged), 8 anchors;
white contour = 0.05 bits. Left: feature plane. Right: random plane.

## What this does and does not show

Both directions change the model as expected (Figure 1), so the test rested on real handles. It shows **no visible support** for the prediction: the mixture leaves the low-change
region at the same size as the singles (Figure 2), and the feature plane is no broader between its axes
than a random plane (Figure 3). We do not use the phrase "consistent with feature-specific error
correction".

This is not evidence *against* error correction. One model, one layer, one readout; the directions are
difference-of-means probes, steering the output without necessarily being the model's own axes. Speaker
label may also encode line position: its positives sit mid-word in an all-caps label, its negatives
usually start a line. Figure 3 drops the 4 of 12 anchors where rescaling is undefined — the weakest. See `RESULTS.md`.
