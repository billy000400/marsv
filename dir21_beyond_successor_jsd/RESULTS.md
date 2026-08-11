# RESULTS — What explains transition-width variation beyond successor JSD?

> CURRENT-BEST ONLY. No history (see CHANGELOG.md). Full definitions and interpretation: REPORT.md.

**Setting.** `pythia-1.4b-deduped` @ `step143000`, residual stream after block 0 at the final token
position, 1,000 token pairs from 123 endpoint tokens × 3 sentence frames × 50 interpolation steps
(pair artifacts inherited from `dir18`; the per-token probes, the forward screen, the anchor-set swap,
the layer sweep, the embedding probe, the vocabulary test, the frame-shape control, the two
embedding interventions, the displacement-norm ladder and the two mode-split experiments are new inference on the same model and
hook point, ~1.6M forward passes). Transition width `w` = fraction of the path over
which the output-distance score `d(t)` climbs from 0.1 to 0.9; smaller = narrower. Analyses run on the
**929 pairs** whose endpoint output movement is at least 0.2 bits in every frame.

## Headline

Beyond corpus successor JSD, transition width is mostly a **per-token additive property**: fitting one
number per token, with no interaction term, raises held-out $R^2$ from **0.149** (corpus JSD alone) to
**0.578**, against a reproducibility ceiling of **0.934**. The per-token term alone (0.365) beats both
corpus JSD (0.149) and the model's own endpoint output difference (0.187).

**And the per-token number can be measured, not just fitted.** Each token's width against **six anchor
tokens that appear in none of the 1,000 pairs** predicts its fitted effect at $\rho = +0.70$
($p = 5\times10^{-19}$); at the pair level two free parameters on $\hat w_u + \hat w_v$ reach held-out
$R^2 = 0.350$, matching the 123-parameter fit.

**Used as a screen it predicts pairs of tokens the analysis never saw.** For 40 tokens absent from the
bank, predicting all 780 of their pairs from anchor widths alone — slope and intercept frozen from the
bank, nothing fitted on the new tokens — gives $R^2 = 0.397$, $\rho = +0.66$, mean absolute error 0.047
on the 718 pairs that pass the gate, beating the model's own endpoint output difference on the same
pairs ($\rho = -0.51$).

**And the per-token number is largely readable from the static embedding.** A ridge probe on the
token's 2048-dimensional embedding row predicts its held-out measured width at $\rho = +0.76$
($R^2 = 0.51$), against $+0.60$ for embedding norm alone and $-0.20$ for shuffled targets; a screen
built from embeddings alone predicts the same 718 unseen pairs at $R^2 = 0.213$, $\rho = +0.53$ with
**no forward pass at any point**. The lookup also holds outside the curated token pool: on 32 tokens
spanning subword fragments, punctuation, numerals, capitalised names and rarer words, predicted and
measured widths correlate at $\rho = +0.60$ ($p = 3.0\times10^{-4}$).

**The ranking is a token property; the level is the context's.** Measured in four differently shaped
contexts (mid-sentence, interrogative, list, code), the token ranking holds at $\rho = +0.84$, $+0.77$,
$+0.74$ and $+0.50$ — the first matching the $+0.82$ that two original frames achieve with each other —
while the median width moves from 0.53 to 0.71.

**The trait is fragile, and what destroys it is behaviour, not geometry.**
Growing an embedding edit along the probe direction until the token's output moves 0.05–0.2 bits does
move width (0.10–0.15 width units), but a random direction matched on output movement moves it just as
much (0.123 vs 0.127, $p = 0.47$), and **all 144 edits widen** where the probe predicts opposite signs:
no direction is a width lever. What the edits do instead is push every token toward a common
$\hat w_u \approx 0.68$. Walking a displacement ladder with the quietest and loudest of 24 directions
rebuilt at each rung separates the two candidate causes: at a **displacement of norm 1.8 the quiet
direction moves the output by 0.049 bits and keeps the token ordering intact ($\rho = +0.94$), while
the loud direction at the same norm moves it by 0.402 bits and destroys the ordering ($\rho = +0.08$)**.
The level of `w` follows the displacement; the ordering — the part a screen uses — follows what the
edit does to the model. Splitting that output change by successor token shows the disruption is
tail-weighted: the token's 32 most likely successors hold 0.71 of its probability mass but absorb only
0.389 of the divergence a loud edit produces, so the behaviour whose loss coincides with the trait's
collapse is not mainly the behaviour corpus successor JSD scores. That tail-weighting cannot be
steered away: directions **constructed** to put 0.86 versus 0.18 of the divergence on the top
successors both land at $S \approx 0.38$ once grown to the 0.4 bits at which width responds, and both
erase the ordering ($\rho = -0.16$ and $-0.28$). The trait belongs to the token's whole output map, not
to an identifiable slice of its next-token distribution.

**In the computation, the trait localises to one component.** Mean-ablating each of 102 early
components one at a time — every attention head and MLP in blocks 0–5 — leaves the token ordering
untouched for 101 of them (median $\rho = +0.99$; every one of the 96 heads $\ge +0.97$). The
exception is the **block-0 MLP**: removing it collapses the spread across tokens from sd 0.084 to
0.018, pushes every token to $\hat w_u \approx 0.82$ and leaves $\rho = -0.10$. It is also the only
early component the model noticeably feels (0.451 bits of output movement, against $\le 0.007$ for
every other component and $\le 0.0004$ for every head), which is both the finding and its caveat.

**The trait is real; the measuring stick is not neutral.** Swapping the six anchors for six function
words or six rare content words still recovers the fitted token effect ($\rho = +0.57$ and $+0.61$),
but the two disjoint sets rank tokens at only $\rho = +0.46$ with each other — so $\hat w_u$ means
"width against this anchor set", and the anchor set must be reported with the method.

## Metrics

Held-out $R^2$ for `w`, 5-fold cross-validation over the 929 gated pairs, identical folds for every
model. The ordering is the direction's main result: every model containing the per-token term
$a_u + a_v$ beats every model without it, and the best pair-level-only model (five covariates plus
corpus JSD) still trails the token-additive model that uses no covariates at all.

| model of `w` | held-out $R^2$ |
|---|---|
| corpus successor JSD `J` | 0.149 |
| corpus JSD + quadratic term | 0.165 |
| model-output JSD $JSD_{\mathrm{out}}$ | 0.187 |
| 5 pair covariates + `J` (entropy, log-frequency, surprisal, $\cos_0$, $d_0$) | 0.399 |
| **token-additive** $\mu + a_u + a_v$ (no pair information) | **0.365** |
| **token-additive + `J`** | **0.578** |
| token-additive + `J` + $JSD_{\mathrm{out}}$ | 0.648 |
| token-additive + `J` + $JSD_{\mathrm{out}}$ + block-0 geometry | 0.723 |
| *reproducibility ceiling (across sentence frames)* | *0.934* |

Replacing the 123 fitted token effects by **measured** anchor widths — one number per token, obtained
from partners the bank never used — costs almost nothing, which is what turns the additive description
into a usable screen. The basin-radius probe, the mechanistic candidate, does much less well.

| model of `w` using measured per-token quantities | free parameters | held-out $R^2$ |
|---|---|---|
| **measured anchor width sum $\hat w_u + \hat w_v$** | **2** | **0.350** |
| *(for comparison: fitted token effects $a_u + a_v$)* | *123* | *0.365* |
| measured anchor width sum + `J` | 3 | 0.452 |
| basin-radius sum + `J` | 3 | 0.299 |
| output-entropy sum + `J` | 3 | 0.237 |

Forward screen — 40 tokens absent from the bank, all 780 pairs among them, slope and intercept frozen
from the bank so **no parameter is fitted on these tokens**. This is prediction, not fit, and it is the
result that makes the per-token screen an auditing tool rather than a description.

| forward screen on unseen tokens | value |
|---|---|
| pairs run / scored after gate and curve-validity | 780 / 718 |
| $R^2$ of the frozen screen on the new pairs | **0.397** |
| Spearman $\rho$ (screen vs observed `w`) | $+0.66$ ($p = 1.5\times10^{-89}$) |
| mean absolute error, on an observed `w` range of 0.34–0.78 | 0.047 |
| median observed `w` by predicted tercile | 0.50 / 0.57 / 0.62 |
| same pairs, baseline: model-output JSD (needs both endpoints of each pair) | $\rho = -0.51$ |

Anchor-set swap — the control on what the per-token measurement means. Two disjoint anchor sets rank
tokens only moderately alike, yet each recovers the same fitted token effect, so a common trait exists
and the anchor set is part of the method rather than a free choice.

| anchor set used to measure $\hat w_u$ | $\rho$ with fitted $a_u$ | held-out $R^2$ for pair `w` |
|---|---|---|
| mixed common words (` and`, ` significant`, ` close`, ` playing`, ` bigger`, ` buried`) | $+0.70$ | 0.350 |
| function words (` he`, ` it`, ` we`, ` but`, ` they`, ` them`) | $+0.57$ | 0.146 |
| rare content words (` surreal`, ` creepy`, ` unbelievable`, ` disgusting`, ` ironic`, ` tempting`) | $+0.61$ | 0.265 |
| *rank agreement between the two disjoint sets* | \- | $\rho = +0.46$ ($p = 1.0\times10^{-7}$) |

Layer sweep — the same per-token measurement with the interpolation site moved down the network. The
ranking of tokens survives; the transitions themselves flatten toward a proportional response, so the
sharpening is done by the blocks below the site.

| interpolation site | $\rho$ with block-0 $\hat w_u$ | $\rho$ with $a_u$ | median $\hat w_u$ | IQR across tokens | held-out $R^2$ for pair `w` |
|---|---|---|---|---|---|
| block 0 | 1.00 | $+0.70$ | 0.553 | 0.102 | 0.350 |
| block 6 | $+0.92$ | $+0.59$ | 0.621 | 0.086 | 0.284 |
| block 12 | $+0.84$ | $+0.52$ | 0.728 | 0.065 | 0.214 |
| block 18 | $+0.72$ | $+0.35$ | 0.800 | 0.020 | 0.146 |

Embedding probe — whether the per-token number has to be measured at all. A ridge probe from the
static embedding row, fitted on 80 tokens and tested on 43 over 50 random splits, recovers most of the
ranking, and beats the obvious deflationary explanation that it is reading token frequency off the
embedding norm.

| predictor of a held-out token's measured width $\hat w_u$ | held-out $\rho$ | held-out $R^2$ |
|---|---|---|
| **ridge probe on the static embedding row $W_E[u]$** | $+0.764 \pm 0.045$ | $0.514 \pm 0.073$ |
| embedding norm $\lVert W_E[u]\rVert$ alone | $+0.597 \pm 0.071$ | $0.190$ |
| same probe, shuffled targets (control) | $-0.201 \pm 0.095$ | $-0.037$ |
| probe target = fitted token effect $a_u$ instead | $+0.505 \pm 0.102$ | $0.270$ |
| anchor width measured at the input embedding instead of block 0 | $+0.79$ vs block 0 | \- |

The lookup replaces the measurement end to end: fitted on the 123 bank tokens and applied to the 40
tokens the analysis never saw, it predicts their 718 pairs with no forward pass anywhere.

| screen for the 718 unseen pairs | $R^2$ | $\rho$ | MAE | terciles of observed `w` |
|---|---|---|---|---|
| measured anchor widths (18 curves per token) | 0.397 | $+0.66$ | 0.047 | 0.50 / 0.57 / 0.62 |
| **static-embedding lookup (no forward pass)** | **0.213** | $+0.53$ | 0.055 | 0.51 / 0.57 / 0.61 |
| model-output JSD (needs both endpoints of each pair) | \- | $-0.51$ | \- | \- |

Vocabulary-wide check — the lookup applied to 32 tokens from four classes the curated pool excludes,
eight per class, spaced over the probe's predicted range. The ranking holds outside the pool, and the
widths found there cover the pool's own range, so the extremes an auditor looks for are present.

| token class (8 tokens each, outside the pool) | $\rho$(predicted, measured) | median measured $\hat w_u$ |
|---|---|---|
| **all 32 together** | $+0.60$ ($p = 3.0\times10^{-4}$) | \- |
| ordinary words the pool excludes | $+0.57$ | 0.632 |
| subword fragments | $+0.31$ | 0.569 |
| punctuation and numerals | $+0.24$ | 0.529 |
| capitalised names | $+0.83$ | 0.527 |
| *reference: the 123 pool tokens* | \- | *0.549* |

MAE 0.046 width units; 576/576 curves valid; measured widths span 0.367–0.686 against the pool's
0.361–0.660. The lookup under-disperses (sd 0.047 predicted vs 0.073 measured) — ridge shrinkage.

Frame-shape control — the per-token measurement repeated in four contexts of different shape. The
ranking is a token property; the level is set by the context, so a threshold calibrated in one context
does not carry to another.

| context the anchor width is measured in | $\rho$ with the original ranking | median $\hat w_u$ | IQR |
|---|---|---|---|
| *reference: two of the three original frames* | *$+0.82$* | *0.549* | *0.102* |
| `She kept walking because everything felt` (mid-sentence) | $+0.844$ | 0.599 | 0.123 |
| `Is it really` (interrogative) | $+0.770$ | 0.623 | 0.107 |
| `The report mentions the following:` (list) | $+0.735$ | 0.530 | 0.118 |
| `def solve(x): … return` (code) | $+0.501$ | 0.705 | 0.049 |

Curve validity 99.6–100%; all $p \le 3.7\times10^{-9}$. For scale, two disjoint anchor sets agree at
$\rho = +0.46$.

Embedding intervention, probe-calibrated steps — the first causal test, and a null with a loophole.
Editing a token's embedding row along the probe's direction by a step the probe says should change
width by a given amount leaves the measured width where it was, and barely disturbs the model at all.

| edit along … | slope of measured vs requested $\Delta\hat w_u$ | mean \|$\Delta\hat w_u$\| | sign agreement | output shift |
|---|---|---|---|---|
| *what a causal direction would give* | *1.0* | *0.0375* | *1.00* | \- |
| **probe direction** | $-0.023$ | 0.0027 | 0.39 | 0.0001 bits |
| random direction, same step norm | $+0.000$ | 0.0008 | 0.50 | 0.0000 bits |

16 tokens × 4 requested changes ($\pm 0.025$, $\pm 0.05$ width units); step norm 0.053 against a median
embedding-row norm of 0.984.

Embedding intervention, behaviour-calibrated steps — the same edit with the step grown until the
model's output moves a fixed number of bits. Width now moves fifty times more, but a random direction
matched on output movement moves it just as much, and every edit widens.

| edit along … | mean \|$\Delta\hat w_u$\| at 0.05 / 0.1 / 0.2 bits | signed $\Delta\hat w_u$, $+$ / $-$ step (0.05 bits) | median step norm | edits that widen |
|---|---|---|---|---|
| **probe direction** | 0.103 / 0.130 / 0.148 | $+0.118$ / $+0.088$ | 1.01 | 72/72 |
| random direction, matched on output movement | 0.109 / 0.125 / 0.135 | $+0.109$ / $+0.109$ | 1.62 | 72/72 |
| *what a causal direction would give* | \- | *opposite signs* | \- | *half* |

12 tokens × 2 directions × 2 signs × 3 budgets = 144 edits; achieved / requested output movement
median 1.00 (IQR 0.91–1.05). Probe vs random on matched edits: 0.127 vs 0.123, Wilcoxon $p = 0.47$;
regression of measured on probe-predicted $\Delta\hat w_u$: slope $-0.002$, $\rho = +0.06$ ($p = 0.61$).

What the edits do instead is destroy the trait. Every token lands near the same width regardless of
where it started and regardless of the direction taken.

| after a 0.2-bit edit | mean $\hat w_u$ | sd across tokens | $\rho$(base $\hat w_u$, $\Delta\hat w_u$) |
|---|---|---|---|
| *before any edit* | *0.543* | *0.083* | \- |
| probe direction | 0.691 | 0.022 | $-0.78$ |
| random direction | 0.678 | 0.015 | $-0.94$ |

Reaching a given output movement takes a smaller step along the probe direction than along a random one
(norm ratio 1.54 / 1.66 / 1.76 at the three budgets), so the probe direction is behaviourally special
but does not carry width. The compression is strong but not total: the post-edit ranking still agrees
with the original at $\rho = +0.73$ / $+0.85$ after a 0.05-bit edit and $+0.57$ / $+0.36$ after a
0.2-bit one (probe / random, 12 tokens).

Displacement-norm ladder — quiet against loud directions at four displacements, with both rebuilt at
each rung by measuring what 24 random directions actually do to the token's output there. This is the
test that separates "the move erased the trait" from "the model's response to the move erased it".

| displacement norm | quiet: bits / mean $\hat w_u$ / $\rho$(before, after) | loud: bits / mean $\hat w_u$ / $\rho$(before, after) | quiet vs loud, paired |
|---|---|---|---|
| *before any edit* | *— / 0.543 / —* | *— / 0.543 / —* | \- |
| 0.15 | 0.0001 / 0.544 / $+1.00$ | 0.0003 / 0.546 / $+1.00$ | $p = 0.09$ |
| 0.40 | 0.0006 / 0.552 / $+0.99$ | 0.0027 / 0.562 / $+0.99$ | $p = 0.02$ |
| 0.90 | 0.0053 / 0.589 / $+0.91$ | 0.0221 / 0.620 / $+0.87$ | $p = 0.0005$ |
| **1.80** | **0.0489 / 0.656 / $+0.94$** | **0.4023 / 0.683 / $+0.08$** | $p = 0.09$ |

Mode split — which successors a disruptive edit moves. The Jensen–Shannon divergence splits exactly by
successor token, so every edit's output change can be scored by the share $S$ of it landing on the
token's 32 most likely successors: the high-mass continuations corpus successor JSD is built from.

| edit at displacement norm 1.8 (12 tokens) | bits | top-mass share $S$ | mean $\hat w_u$ | $\rho$(before, after) |
|---|---|---|---|---|
| *before any edit* | \- | \- | *0.543* | \- |
| loudest of 24 random directions | 0.402 | **0.389** | 0.683 | $+0.08$ |
| most top-heavy, rescaled to matched movement | 0.410 | 0.408 | 0.666 | $-0.08$ ($p = 0.81$) |
| most tail-heavy, rescaled to matched movement | 0.453 | 0.355 | 0.651 | $-0.37$ ($p = 0.24$) |

Those 32 successors hold 0.71 of the probability mass yet absorb only 0.389 of the divergence, and
louder directions are more tail-weighted still ($\rho(B_j, S_j) = -0.36$): the disruption that coincides
with the trait's collapse is not mainly disruption of what corpus statistics score. The steering half is
a null with a stated limit — random directions span only $S = 0.36$–$0.56$, and at matched movement both
extremes flatten the ordering, though the top-heavy edit widens slightly more ($+0.124$ vs $+0.108$,
paired $p = 0.009$) while moving the output less.

The quiet direction lands below the loud one at every rung (12 tokens, Wilcoxon on $\Delta\hat w_u$),
and at the top rung — displacement 1.8, nearly twice a median embedding row — the two differ by 8× in
output movement: the quiet edit keeps the token ordering the screen depends on ($\rho = +0.94$,
$p = 4\times10^{-6}$) while the loud edit at the identical displacement leaves nothing of it
($\rho = +0.08$, $p = 0.80$). The level still rises along every direction (sd across tokens 0.083 →
0.038 quiet, 0.022 loud), so displacement sets the level and behaviour sets whether the trait survives.

Supporting quantities. The first block says how strong the inherited association is on this subset,
the second says the leftover is neither noise nor a normalisation artifact, and the third says the
per-token effect cannot be looked up in a count table.

| quantity | value |
|---|---|
| $\rho(J, w)$, all 1,000 pairs | $-0.486$ ($p = 2.6\times10^{-60}$) |
| $\rho(J, w)$, 929 gated pairs | $-0.409$ ($p = 1.0\times10^{-38}$) |
| $\rho(JSD_{\mathrm{out}}, w)$, gated | $-0.357$ ($p = 3.1\times10^{-29}$) |
| matched narrow-vs-wide contrasts found (consistent in all 3 frames) | 1,529 (21 share a token) |
| largest matched contrast | ` her`/` when` $w{=}0.34$ vs ` kind`/` wrong` $w{=}0.77$ at $J = 0.70$ bits |
| across-frame agreement of additive-model residuals | $\bar r = 0.67$ (0.54 after adding geometry) |
| CV of `w` vs CV of $w \cdot d_0$ | 0.158 vs **0.216** — path-length artifact refuted |
| token effect $a_u$ vs **measured anchor width** $\hat w_u$ | $\rho = +0.70$ ($p = 4.6\times10^{-19}$; $+0.67$ with output entropy partialled out) |
| token effect $a_u$ vs corpus log-frequency | $\rho = -0.33$ ($p = 2.9\times10^{-4}$) |
| token effect $a_u$ vs continuation entropy | $\rho = -0.24$ ($p = 0.008$) |
| token effect $a_u$ vs model surprisal in frame | $\rho = +0.26$ ($p = 0.004$) |
| token effect $a_u$ vs basin radius, anchor directions ($\tau = 0.2$ bits) | $\rho = +0.39$ ($p = 1.1\times10^{-5}$) — sign opposite to the basin prediction |
| token effect $a_u$ vs basin radius, random directions ($\tau = 0.1$ bits) | $\rho = -0.02$ ($p = 0.87$) — no relation |

## Figures

The scatter that motivates the direction, and what the endpoint-movement gate removes:

![Width against corpus successor JSD, and against endpoint output movement](plots/scatter_and_gate.png)

**Figure 1.** Left: width `w` (y) against corpus successor JSD `J` in bits (x), all 1,000 pairs;
circles pass the movement gate, open squares fail it, black diamonds joined by lines are the eight
largest matched contrasts. Right: `w` (y) against endpoint output movement $JSD_{\mathrm{out}}$ in bits
(x), dashed line = the 0.2-bit gate. Gated-out pairs cluster at high `w`: a normalised score cannot
describe a movement that barely happened.

Matched contrasts differ in the shape of the curve, not in how far the output travels:

![Output-distance curves for three matched narrow-vs-wide contrasts](plots/contrast_curves.png)

**Figure 2.** `d(t)` (y) against interpolation position `t` (x) for three matched contrasts, all three
frames per pair. Solid = narrow pair, dashed = wide pair; the two pairs in each panel are matched on
`J` and on $JSD_{\mathrm{out}}$ (panel titles). Dotted lines at $d = 0.1$ and $d = 0.9$ bound `w`.

Which model of `w` predicts held-out pairs best — the direction's core question:

![Held-out R-squared for models of transition width](plots/cv_r2.png)

**Figure 3.** Held-out $R^2$ (x) for seven models of `w` (y). Hatched = pair-level predictors only,
solid = includes the per-token term. Dashed line = the 0.934 reproducibility ceiling.

The gain is a shift of the whole distribution, not a few pairs:

![Predicted versus observed width for two models](plots/prediction.png)

**Figure 4.** Observed `w` (y) against held-out predicted `w` (x), corpus JSD alone (left, squares) and
corpus JSD plus the per-token term (right, circles); dashed line $y = x$. Corpus JSD compresses all
predictions into 0.47–0.67 and never reaches the narrow tail.

What the per-token effect is, and what it is not:

![Fitted per-token width effects, and their relation to corpus frequency](plots/token_effects.png)

**Figure 5.** Left: fitted token effect $a_u$ in width units (y) for the 120 tokens used in at least
four gated pairs, ranked (x), extremes labelled — ` un`, ` in`, ` his`, ` my` pull narrow, ` kind`,
` real`, ` now` push wide. Right: $a_u$ (y) against corpus log-frequency (x); the relation is real but
loose, so frequency is not the underlying variable.

Does the per-token trait survive contact with the model, and is the basin picture behind it?

![Anchor width against fitted token effect, basin radius against fitted token effect, and held-out R-squared](plots/transfer.png)

**Figure 6.** Left: fitted token effect $a_u$ (y) against measured anchor width $\hat w_u$ (x, median
width against six tokens used in no pair) — the transfer test, $\rho = +0.70$. Middle: $a_u$ (y)
against basin radius (x, radians of great-circle travel before the output moves $\tau$ bits); squares
= directions toward anchor tokens ($\tau = 0.2$), triangles = random directions ($\tau = 0.1$). Right:
held-out $R^2$ for pair width (x) for five models (y); hatched = corpus JSD, solid = measured token
widths, dotted = fitted token effects.

Does the screen hold up on tokens the analysis never saw?

![Forward prediction of pair width for unseen tokens, and separation by predicted tercile](plots/forward_screen.png)

**Figure 7.** Left: observed `w` (y) of 718 pairs built from 40 tokens absent from the bank, against the
width predicted from the two tokens' anchor widths alone (x); slope and intercept come from the bank,
so nothing is fitted here. Dashed line $y = x$. Right: observed `w` (y) by tercile of the screen's
prediction (x); boxes are median, quartiles and 1.5 IQR whiskers, hatched distinctly per tercile.

Can the anchor tokens be swapped freely, or is the measuring stick part of the result?

![Anchor width under two disjoint anchor sets, and each set's agreement with the fitted token effect](plots/anchor_swap.png)

**Figure 8.** Left: anchor width against six rare content words (y) versus anchor width against six
function words (x), one marker per endpoint token, extremes of the fitted effect labelled. Right:
Spearman $\rho$ between each anchor set's widths and the fitted token effect $a_u$ (y) for the three
anchor sets (x). The sets agree only in part with each other, yet each recovers the same trait.

Where does the trait come from, and what does depth contribute?

![Anchor width measured at blocks 0, 6, 12 and 18](plots/layer_sweep.png)

**Figure 9.** Left: Spearman $\rho$ across the 123 tokens (y) against the interpolation block (x);
solid/circles = agreement with block-0 anchor widths, dashed/squares = agreement with the fitted token
effect $a_u$. Right, same x-axis: median anchor width (solid, circles), held-out $R^2$ for the block-0
pair widths (dashed, diamonds), and the interquartile range of $\hat w_u$ across tokens (dotted,
triangles); the dash-dotted line is $w = 0.8$, a perfectly proportional response.

Can the screen be a lookup instead of a measurement, and what does that cost?

![Anchor width at the embedding site, a ridge probe from the static embedding, its controls, and the resulting zero-forward-pass screen](plots/embed_probe.png)

**Figure 10.** Far left: $\hat w_u$ measured at the input embedding (y) against $\hat w_u$ measured
after block 0 (x). Centre left: block-0 $\hat w_u$ (y) against the out-of-fold prediction from the
token's static embedding (x). Centre right: mean held-out Spearman $\rho$ on 43 test tokens (y) over 50
random splits for three targets (x); hatched = probe, dotted = shuffled-target control, dotted
horizontal line = embedding-norm baseline; error bars $\pm 1$ sd across splits. Far right: observed `w`
(y) of the 718 unseen pairs against the width predicted from static embeddings alone (x); the narrow
x-range is ridge shrinkage. Dashed lines are $y = x$.

Does the lookup survive contact with token types the pool excludes?

![Predicted versus measured anchor width for 32 tokens outside the curated pool, and the measured width of each token class](plots/vocab_probe.png)

**Figure 11.** Left: measured $\hat w_u$ at block 0 (y) against the width predicted from the static
embedding (x) for 32 tokens outside the pool — circles: ordinary words, squares: subword fragments,
triangles: punctuation and numerals, diamonds: capitalised names; dashed line $y = x$, shaded band =
the range of measured widths over the 123 pool tokens. Right: measured $\hat w_u$ (y) by token class
(x) with the pool as reference; boxes are median, quartiles and 1.5 IQR whiskers, hatched distinctly,
individual tokens overplotted; the dash-dotted line is $w = 0.8$, a proportional response.

Is the per-token measurement about the token, or about the slot it sits in?

![Rank agreement of anchor widths measured in four new contexts with the original ranking, and the widths themselves](plots/frame_control.png)

**Figure 12.** Left: Spearman $\rho$ between each new context's token ranking and the original (y) for
the four contexts (x), bars hatched distinctly; dashed line = mean agreement among the three original
frames ($+0.82$), dotted line = agreement between two disjoint anchor sets ($+0.46$). Right:
$\hat w_u$ measured in each new context (y) against $\hat w_u$ in the original frames (x), one marker
per token per context, markers matching the left panel; dashed line $y = x$. The clouds sit off the
diagonal (level shifts) while keeping their order (rank preserved).

Does editing the embedding along the probe's direction change how sharply the model transitions?

![Measured width change against the width change requested along the probe direction, and per-token response slopes](plots/intervene.png)

**Figure 13.** Left: measured $\Delta\hat w_u$ (y) against the change requested from the probe (x), 16
tokens × 4 step sizes; circles = probe direction, squares = random direction of the same step norm
(jittered). Dashed line = what a causal direction would give ($y = x$). Right: per-token slope of
measured against requested change (y) for the two directions (x), one marker per token, thick bar =
mean, dashed line at 1.0 = the probe's own prediction. Both directions sit at zero.

Once the step is calibrated on the model instead of the probe, does the direction matter — and where do
the edited tokens end up?

![Width change against the calibrated output movement, probe against random at matched movement, signed changes by direction, and where the edited tokens land](plots/intervene2.png)

**Figure 14.** 12 tokens edited along the probe direction and a random direction, both signs, three
output budgets. Top left: mean $|\Delta\hat w_u|$ (y) against the output movement the edit was
calibrated to produce (x, bits, log scale); solid circles = probe, dashed squares = random, faint
markers = individual edits. Top right: $|\Delta\hat w_u|$ along the probe direction (y) against the
same token's random-direction edit at the same budget (x); marker shape = budget, dashed line $y = x$.
Bottom left: mean signed $\Delta\hat w_u$ (y) per direction and sign (x), hatched distinctly — the probe
predicts opposite signs for the two. Bottom right: $\hat w_u$ after a 0.2-bit edit (y) against
$\hat w_u$ before it (x); dotted line = no change, horizontal lines = each direction's mean landing
point.

Does the collapse follow the displacement or the model's response to it?

![Mean anchor width against displacement norm for quiet, loud and random directions, and width after the largest edit against width before it](plots/ladder.png)

**Figure 15.** 12 tokens displaced along the quietest and loudest of 24 directions, rebuilt at each rung
by measuring their actual output movement there, plus one plain random direction. Left: mean $\hat w_u$
after the edit (y) against displacement norm (x, log scale); solid/circles = quiet, dashed/squares =
loud, dotted/triangles = random; error bars 1 s.e. over the 12 tokens; dash-dotted line = the unedited
mean 0.543. Right: $\hat w_u$ after an edit of norm 1.8 (y) against $\hat w_u$ before it (x), quiet
(circles) and loud (squares) with least-squares fits; dotted line = no change. At the same
displacement, the quiet edit preserves the ordering and the loud edit flattens it.

Where the damage lands, and whether steering it changes anything:

![Where random embedding edits move the output, and anchor width after top-heavy and tail-heavy edits matched on total output movement](plots/mode_split.png)

**Figure 16.** Twelve tokens, the same as Figures 14–15. Left: top-mass share $S$ of the output change
(y, fraction of the JSD landing on the token's 32 most likely successors) against the edit's total
output movement (x, bits, log scale); small circles = the 24 random directions per token at
displacement norm 1.8, triangles = most top-heavy and squares = most tail-heavy after rescaling to
0.4 bits; dashed line = the mass those 32 successors hold before the edit (0.71). Right: $\hat w_u$
after the edit (y) against $\hat w_u$ before it (x) for the two selected edits, rank agreement with the
pre-edit ordering in the legend; dotted line = no change.

Whether that split can be steered on purpose — directions built from a generalised eigenproblem rather
than drawn at random:

![Predicted versus achieved top-mass share for constructed top-heavy and tail-heavy edits, and anchor width before versus after each edit](plots/mode_construct.png)

**Figure 17.** Twelve tokens, the same as Figures 14–16. Left: top-mass share $S$ (y, fraction of the
output change landing on the token's 32 most likely successors) per token (x, token strings); open
markers = $S$ predicted for a small step (the generalised eigenvalues), filled markers = $S$ actually
achieved once the same direction is rescaled to 0.4 bits, joined by a dotted line; triangles =
$S$-maximising ("top-heavy"), squares = $S$-minimising ("tail-heavy"); gray band = the range 24 random
directions span; dash-dotted line = the mass those 32 successors hold before any edit (0.71). Right:
$\hat w_u$ after the edit (y) against $\hat w_u$ before it (x); dotted line = no change.

| edit rescaled to 0.4 bits (12 tokens) | predicted $S$ (small step) | achieved $S$ | bits | mean $\hat w_u$ | sd | $\rho$(before, after) |
|---|---|---|---|---|---|---|
| *before any edit* | \- | \- | \- | *0.543* | *0.083* | \- |
| constructed top-heavy | **0.856** | 0.369 | 0.422 | 0.666 | 0.023 | $-0.16$ ($p = 0.62$) |
| constructed tail-heavy | **0.179** | 0.390 | 0.419 | 0.672 | 0.020 | $-0.28$ ($p = 0.38$) |

The construction reaches a 0.68 separation in predicted $S$, three times what 24 random draws supply,
and it survives none of the way to a behaviourally meaningful step: the tail-weighting of a large
embedding edit is set by the step size, not by the direction. So the top-mass hypothesis is untestable
by embedding edits, and both arms agree on the outcome that matters — any disturbance the model
registers erases the token ordering wherever in the distribution it lands.

Component ablation — the first intervention that is not an embedding edit. Each of the 102 attention
heads and MLPs in blocks 0–5 is mean-ablated at the final token position, one at a time, and the
per-token width is re-measured for the same 12 tokens against the same 6 anchors.

| mean-ablated component (12 tokens, 6 anchors, 1 frame) | mean $\hat w_u$ | sd across tokens | $\rho$(before, after) | output movement |
|---|---|---|---|---|
| *nothing ablated* | *0.565* | *0.084* | \- | \- |
| **block-0 MLP** | **0.822** | **0.018** | $-0.10$ | 0.451 bits |
| MLPs of blocks 1–5 (worst of the five) | 0.585 | 0.091 | $+0.90$ | 0.007 bits |
| all 96 attention heads (worst of the 96) | 0.563 | 0.076 | $+0.97$ | 0.0004 bits |
| *median over all 102 components* | \- | *0.084* | *$+0.99$* | \- |

![Spread and ordering of the per-token width after mean-ablating each early component](plots/ablate.png)

**Figure 18.** Each of the 102 attention heads and MLPs in blocks 0–5 mean-ablated one at a time, for
the 12 tokens of Figures 14–17. Left: standard deviation of $\hat w_u$ across the 12 tokens (y) against
the block containing the ablated component (x, heads jittered); open circles = attention heads,
diamonds = MLPs; dash-dotted line = the unablated spread 0.084. Right of it: rank agreement
$\rho$(unablated $\hat w_u$, ablated $\hat w_u$) (y) against the output movement the ablation causes
(x, bits, log scale), same markers; dash-dotted line = perfect agreement. Far right: $\hat w_u$ after
the ablation (y) against $\hat w_u$ before it (x) for the two extreme components; dotted line = no
change. Only the block-0 MLP leaves the cluster on either panel.

The profile is flat everywhere except one point, so the trait is not spread thinly over early
attention: no head carries a detectable share of it, and no MLP above block 0 does either. The block-0
MLP both destroys it and is the only early component whose removal moves the model by more than
0.01 bits — 0.451 bits, essentially the 0.4-bit rung at which the displacement ladder showed that *any*
disturbance flattens the ordering. The two readings cannot be separated by ablation alone: either the
block-0 MLP computes the trait, or it is simply the only single early component large enough to reach
the regime where the trait dies. What the sweep does establish is that nothing else in blocks 0–5 is
load-bearing, which narrows the mechanistic search from 102 components to one.

## Next experiment

**Give the block-0 MLP a dose–response curve, and match it against an equally loud control.** The
ablation sweep found exactly one candidate (Figure 18) and left exactly one confound: the block-0 MLP
is both the component whose removal erases the trait and the only early component whose removal the
model feels at all. Interpolate the ablation — replace the block-0 MLP's final-position output by
$(1-\alpha)$ of itself plus $\alpha$ of its mean, for $\alpha = 0.1 \dots 1$ — and at each $\alpha$
build a control that perturbs the same residual stream by a random vector rescaled to the *same* output
movement in bits. Then plot $\rho$(before, after) against output movement for both. If the MLP arm
loses the ordering at a smaller output movement than the matched control, the block-0 MLP computes the
trait; if the two curves lie on top of each other, the trait dies from disturbance as such and the
sweep's single hit is a size effect, closing the mechanistic line and leaving the static-embedding
lookup (Figures 10–11) as the deliverable. Cost: ~20 conditions at the price of one ablation
component, well under the last experiment's budget.
