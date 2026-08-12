# RESULTS — What explains transition-width variation beyond successor JSD?

> CURRENT-BEST ONLY. No history (see CHANGELOG.md). Full definitions and interpretation: REPORT.md.

**Setting.** `pythia-1.4b-deduped` @ `step143000`, residual stream after block 0 at the final token
position, 1,000 token pairs from 123 endpoint tokens × 3 sentence frames × 50 interpolation steps
(pair artifacts inherited from `dir18`; the per-token probes, the forward screen, the anchor-set swap,
the layer sweep, the embedding probe, the vocabulary test, the frame-shape control, the two
embedding interventions, the displacement-norm ladder, the two mode-split experiments, the component
ablation, the per-token-matched dose–response and the block-0 MLP probe/transplant are new inference on
the same model and hook point, ~1.6M forward passes). The last two sections repeat the cheap end of the
pipeline on `pythia-160m/410m/1b-deduped` at the same checkpoint, on 17 released training
checkpoints of `pythia-410m-deduped` from `step0` to `step143000` — including a transplant of the
block-0 MLP output between two of those checkpoints — and on `gpt2` (124M, a different corpus and
tokenizer). Transition width `w` = fraction of the path over
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
measured widths correlate at $\rho = +0.60$ ($p = 3.0\times10^{-4}$). What that lookup ranks is narrower
than "the width": it predicts how plateau-shaped a token's curves are, and predicts none of the width
ordering that remains once curve shape is removed, so the free tier inherits its accuracy from the two
properties nearly coinciding in this model. That missing component stays unreadable one block deeper —
from the block-0 MLP's output and from the whole post-block-0 residual state alike — so there is no
cheap upgrade to a table that ranks the crossing width specifically.

**And it is not a quirk of one network.** Repeating the measurement in three other Pythia sizes at the
same checkpoint, with the same token ids, anchors and frames, gives the same ranking of tokens:
Pythia-410M, 1B and 1.4B agree at $\rho = +0.88$ to $+0.90$ over the 123 tokens, and at **$+0.98$ to
$+1.00$ once each correlation is divided by what the measurement's own reliability allows**. The level
is the network's — the median width falls 0.749 → 0.658 → 0.620 → 0.549 from 160M to 1.4B — but the
ordering is the token's. The free lookup transfers with no measurable loss: the probe read off
Pythia-1.4B's embedding matrix ranks 410M's measured widths at $\rho = +0.760$ and 1B's at $+0.745$,
against $+0.765$ on the model it was fitted in. The exception is the smallest model: 160M ranks tokens
differently ($+0.21$, and $+0.26$ after the same correction), so the trait is something the family
acquires with scale, somewhere between 160M and 410M.

**It is learned in the first 0.4% of training, and it is not a count table.** Seventeen released
checkpoints of Pythia-410M show no ordering at initialisation (across-token sd 0.003 against 0.060 at
the end; $\rho = +0.015$ with the final ranking) and an ordering that reaches **0.87 of its noise
ceiling by `step512`** and 0.94 by `step2000`, after which it does not change for the remaining 98.6%
of training — while the *level* keeps sharpening (median width 0.833 → 0.595 between `step256` and
`step64000`). Two stages: through `step128` the ranking is purely unigram frequency ($\rho = -0.72$
with $\log_{10}$ count, stronger than the $-0.53$ it ends at, and nothing left after partialling
frequency and successor entropy out), and from `step256` a second component appears that those
statistics do not contain — they explain only **0.375** of the final ranking's rank variance, and
early-to-final agreement survives their removal at $+0.6$ to $+0.8$.

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
every other component and $\le 0.0004$ for every head). A dose–response against a random control
matched **token by token** on output movement breaks that confound in the block-0 MLP's favour, by a
modest margin: at every dose where the ordering is still alive, and for all three control seeds,
blending the MLP toward its mean costs more rank agreement per bit than a random perturbation of the
same residual stream (at 0.014 bits, $\rho = +0.64$ against $+0.91$), and the control needs about
**1.3× more output movement** to do the same damage. Token by token the gap is clearer: the dose moves
a token's width about **twice as far** as that token's own exactly matched control (0.074 against
0.036 width units at 0.0068 bits, Wilcoxon $p = 0.001$), and still does after each arm's mean level
shift is removed ($p = 0.034$). Matching per token is what makes those numbers mean anything — a
control matched only on the 12-token average gives individual tokens 0.08× to 8.5× the movement the
dose gave them, and inflates the apparent margin to 2.8×.

**And that component's output vector is sufficient on its own.** Overwriting one token's block-0 MLP
output $m_u$ with another token's — leaving the rest of the forward pass, and the anchors, untouched —
transports the width almost completely: the recipient's new width tracks the **donor's** width at
$\rho = +0.968$ with slope $+0.913$, while the recipient's own remaining state contributes nothing
($\rho = -0.104$, $p = 0.64$; 66× less variance). A self-transplant reproduces the baseline exactly.
$m_u$ here is computed from the token embedding alone — its cosine across the three sentence frames is
1.0000 — which is why a per-token width exists before any context is read. Reading the number *off*
$m_u$ with a probe, however, is no easier than reading it off the static embedding ($\rho = +0.748$ vs
$+0.764$): the component carries the trait without making it more linearly explicit. Nor is it
compressible — transplanting only the top 64 principal components of $m$, which carry 79% of its
across-token variance, delivers 30% of the transfer while causing 95% of the output movement. What the
write transports is the whole curve: scored on how plateau-shaped the recipient's curves become, the
same 132 transplants give a slope of $+0.970$ alongside $+0.913$ for the width, and each property still
transports with the donor's other property held constant ($+0.796$ and $+0.517$, both above zero in all
12 recipients).

**But the token is a token *of a training corpus*.** The same 123 strings measured in GPT-2 small —
different corpus, different BPE vocabulary — rank at $\rho = -0.22$ with Pythia-1.4B (against $+0.88$
between two Pythia sizes), the free lookup transfers at $-0.20$, and a probe refitted inside GPT-2's own
embedding matrix recovers only 0.30 of the ordering its target allows, against 0.81 inside Pythia — the
free lookup is a Pythia result. Before that, the measurement itself fails there: 88.8% of GPT-2's block-0
curves are non-monotone, so `w` is undefined for them, and its per-token width has a split-half
reliability of 0.32 against Pythia's 0.89 at every site we tried. The screen is per-model, and the
reliability check is the go/no-go test for porting it.

**GPT-2 has its own ordering; Pythia-160M barely has plateaus.** A width can be computed on a curve
with no plateau in it, so a failed transfer needs one more check: **edge drift** `E`, how far the output
moves in the outer tenth of the path at each end (≈ 0 for a plateau, exactly 0.2 for a straight line).
GPT-2's block-0 curves are as plateau-shaped as Pythia-1.4B's ($E = 0.087$ vs 0.081); scoring only its
plateau-shaped curves ($E \le 0.1$, 56% of them) doubles its reliability to **0.66** (ceiling 0.77) and
leaves its disagreement with Pythia at $-0.19$. GPT-2 ranks these tokens reproducibly and differently.
Pythia-160M is the other failure mode: the least plateau-shaped configuration measured ($E = 0.183$,
87% of curves above the 0.1 cut against 22% of Pythia-1.4B's).

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
embedding norm. The shuffled-target control permutes the targets once, which is a noisy estimate of
chance ($\pm 0.2$ for these 123 tokens); where the probe's accuracy is small enough for that to matter
— the GPT-2 probes below — chance is estimated from 50 independent permutations and reported as a
permutation $p$-value, the fraction of permuted draws scoring at least as high as the probe.

| predictor of a held-out token's measured width $\hat w_u$ | held-out $\rho$ | held-out $R^2$ |
|---|---|---|
| **ridge probe on the static embedding row $W_E[u]$** | $+0.764 \pm 0.045$ | $0.514 \pm 0.073$ |
| embedding norm $\lVert W_E[u]\rVert$ alone | $+0.597 \pm 0.071$ | $0.190$ |
| same probe, shuffled targets (control, one permuted draw) | $-0.201 \pm 0.095$ | $-0.037$ |
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
| **measured** $\hat w_u$ vs corpus log-frequency (1.4B / 410M) | $\rho = -0.52$ / $-0.53$ |
| **measured** $\hat w_u$ vs successor entropy (1.4B / 410M) | $\rho = -0.48$ / $-0.46$ |
| rank $R^2$ of the token ranking from both corpus statistics (1.4B / 410M) | 0.378 / 0.375 |
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
disturbance flattens the ordering. The sweep on its own therefore establishes the negative half firmly
(nothing else in blocks 0–5 is load-bearing, narrowing the search from 102 components to one) and
leaves the positive half confounded by size. The dose–response below removes that confound.

### Dose–response: is the block-0 MLP special, or merely loud?

To separate "this component computes the trait" from "this is the only early component big enough to
reach the lethal regime", we soften the ablation into a dose and give every dose a control that is
exactly as loud **on each token separately**. The MLP's final-position output is blended toward its
mean with weight $\alpha$; at each $\alpha$ a random direction is added to the same residual stream
with a scale binary-searched *per endpoint prompt* so that **that token's** output moves the same
number of bits as the dose moved it. Per-token matching is essential because the conclusion is about
the ordering of individual tokens and the dose is uneven across them — at full ablation the per-token
movement spans 0.254–0.710 bits. The comparison is repeated for three random seeds.

![Rank agreement, across-token spread, control matching quality and per-token width change for the block-0 MLP dose and a per-token output-matched random control](plots/dose.png)

**Figure 19.** Dose–response for the block-0 MLP (solid, circles: output blended toward its mean,
$\alpha = 0.1 \dots 1$) against a random direction added to the same residual stream, rescaled so each
token's output moves the same number of bits (dashed, squares; mean of three seeds, error bars 1 sd
across seeds). Panel 1 x: output movement in bits (log scale), the mean over the 12 tokens of the JSD
between perturbed and unperturbed next-token distributions. Panel 1 y: rank agreement $\rho$ between
each token's anchor width before and after the perturbation — 1 = ordering intact, 0 = destroyed;
dotted triangles = the same control matched only on the 12-token *mean* movement. Panel 2, same x: sd
of $\hat w_u$ across the 12 tokens; dash-dotted line = the unperturbed spread 0.084. Panel 3 x: the
dose's output movement; y: each token's ratio of control movement to dose movement (log scale, 1.0 =
exact match), open triangles = mean-matched control, filled squares = per-token-matched. Panel 4 x:
the 12 tokens ordered by unperturbed width; y: $|\Delta\hat w_u|$ at the 0.0068-bit dose.

| output movement (bits) | $\rho$, block-0 MLP dose | $\rho$, per-token-matched control | sd, MLP | sd, control |
|---|---|---|---|---|
| 0.0006 | +0.97 | $+1.00 \pm 0.00$ | 0.076 | 0.081 |
| 0.0027 | +0.92 | $+0.99 \pm 0.01$ | 0.071 | 0.078 |
| 0.0068 | +0.84 | $+0.98 \pm 0.02$ | 0.070 | 0.074 |
| 0.0143 | +0.64 | $+0.91 \pm 0.04$ | 0.069 | 0.068 |
| 0.0292 | +0.62 | $+0.76 \pm 0.12$ | 0.055 | 0.060 |
| 0.1033 | +0.25 | $+0.15 \pm 0.12$ | 0.027 | 0.055 |
| 0.2651 | +0.74 | $+0.24 \pm 0.10$ | 0.021 | 0.035 |
| 0.4506 | −0.10 | $-0.06 \pm 0.04$ | 0.018 | 0.016 |

The curves separate in the band where an ordering still exists. Across the five rungs up to 0.03 bits
the MLP dose sits below its matched control at every one of the 15 rung × seed comparisons; it falls
through $\rho = 0.6$ at 0.031 bits and the control at 0.041, so a random disturbance needs about
**1.3× more output movement** to do the same damage. Above 0.1 bits both arms are at noise
(with $n = 12$ tokens a single $\rho$ carries a standard error near 0.3), and the two cross, so the
top three rungs are reported but not interpreted.

Panel 3 shows why the matching had to be per token, and what it cost. A control matched only on the
12-token average moves individual tokens by 0.08× to 8.5× what the dose moved them; per-token matching
brings every ratio to 1.000. Under the loose control the MLP's margin looked like 2.8× (the control
crossing $\rho = 0.6$ only at 0.086 bits) because at the higher rungs the control was simply
under-dosed — at 0.103 bits it received 0.078 bits and kept $\rho = +0.68$, where an honestly matched
control lands at $+0.15$, below the MLP arm.

| paired per-token test, dose vs that token's own matched control ($n = 12$) | at 0.0068 bits | at 0.0143 bits | at 0.4506 bits |
|---|---|---|---|
| mean $\lvert\Delta\hat w_u\rvert$, MLP dose / control | 0.074 / 0.036 | 0.116 / 0.067 | 0.257 / 0.250 |
| Wilcoxon $p$ | 0.0010 | 0.0005 | 0.27 |
| level-free $\lvert\Delta\hat w_u - \overline{\Delta\hat w}\rvert$, MLP / control | 0.034 / 0.014 | 0.047 / 0.022 | 0.064 / 0.063 |
| Wilcoxon $p$ | 0.034 | 0.016 | 0.79 |

Because a rank correlation over 12 tokens is blunt, the load-bearing evidence is this paired per-token
comparison. The dose moves a token's width roughly twice as far as that token's exactly matched
control at every dose from 0.0006 to 0.265 bits ($p \le 0.005$ throughout), converging only at full
ablation where both have saturated. Removing each arm's mean level shift — leaving only the part that
rearranges tokens — keeps the difference at the two doses where the ordering is still alive
($p = 0.034$, $0.016$) and erases it once the ordering is gone ($p \ge 0.47$ above 0.1 bits).

The spread panel separates two effects the earlier experiments kept confusing. Through 0.014 bits the
across-token *spread* collapses along the same trajectory in both arms (0.070/0.074, 0.069/0.068 at
matched bits) — pushing the residual stream around by any means compresses every token toward
$\hat w_u \approx 0.82$, as the displacement ladder found for embedding edits; beyond that the dose
compresses harder (0.027 vs 0.055 at 0.103 bits), where the ordering has already gone. Level and
ranking are separate channels, and it is the ranking that singles out a component.

This is the direction's positive mechanistic localisation, with an honest margin: one component, one
frame, three control seeds, 12 tokens, and a 1.3× separation in bits (about 2× in per-token width
change) confined to doses below 0.03 bits. It says the block-0 MLP's contribution to the final-position
residual stream is where the per-token width trait is realised — consistent with the layer sweep's
finding that the ordering is already fixed at the input and that the blocks *below* the interpolation
site do the sharpening. The control equalises how far each token's output moves, not the direction it
moves in, so it bounds a size effect rather than every alternative to a carrier.

### Transplant: the block-0 MLP's output vector carries the width

Destroying a component shows it is necessary. To ask whether its output is *sufficient*, we take the
block-0 MLP's final-position output vector $m_u$ — which in this architecture is computed from the
token embedding alone, with no context (its cosine across the three sentence frames is 1.0000) — and
overwrite one token's $m_u$ with another's, leaving every other part of the forward pass in place. The
six anchor prompts are never edited, so a token transplanted with its own $m_u$ must return exactly its
baseline width; it does, for all 12 tokens (max difference $0.0000$). We also probe $m_u$ directly, to
see whether the number is more *readable* there than in the static embedding.

![Ridge probes from three representations, the 12x12 transplant matrix, transplanted width against the donor's own width, and donor-versus-recipient rank agreement](plots/mlp_read.png)

**Figure 20.** Left: held-out Spearman $\rho$ (y) between predicted and measured $\hat w_u$ for ridge
probes from three representations (x), 80 train / 43 test tokens over 50 random splits, error bars
$\pm 1$ sd across splits, hatched distinctly; gray cross-hatched bars = the same probe with shuffled
targets. Centre left: measured width $\hat w$ after transplant (colour, `cividis`) for every recipient
(y) × donor (x) pair, both axes ordered narrow → wide by the token's own unedited width; vertical
banding = the donor sets the value. Centre right: recipient's width after the transplant (y) against
the donor's own width (x); circles = cross transplants, diamonds = self transplants, gray lines join
one recipient's 12 donors, dashed line $y = x$ (complete transfer). Right: Spearman $\rho$ over the 11
partners (y) for each of the 12 tokens sorted by its own value (x) — circles: with the recipient held
fixed, against the donor's width; squares: with the donor held fixed, against the recipient's width.

| transplanting $m_u$ from a donor onto a recipient (12 × 11 pairs, frame 1) | value |
|---|---|
| **width follows the donor: $\rho$ over the 11 donors, per recipient** | **$+0.968$** (min $+0.95$, Wilcoxon $p = 5\times10^{-4}$) |
| **slope of the recipient's new width on the donor's own width** | **$+0.913$** (1.0 = complete transfer) |
| width follows the recipient's remaining state: $\rho$ over the 11 recipients, per donor | $-0.104$ ($p = 0.64$) |
| between-donor variance ÷ between-recipient variance of the transplanted width | $66\times$ |
| self-transplant vs baseline | $\rho = +1.000$, max difference $0.0000$ |
| $m_u$ replaced by the 12-token mean (the ablation, at this dose) | $\hat w = 0.663 \pm 0.017$, from $0.565 \pm 0.084$ |

Swapping one vector transports the whole trait. A wide token given a narrow token's $m_u$ becomes
narrow, at 91% of the full distance, and the width it lands on is set by the donor to the near-exclusion
of anything else in the recipient's forward pass: the recipient's own identity contributes 66× less
variance and is, if anything, mildly *anti*-correlated with the outcome. This is the sufficiency result
the ablation and dose–response could not give, and it also explains why the static-embedding lookup
works at all — $m_u$ is a fixed function of the token's embedding row, so a per-token width is fixed
before any context is read.

The intervention is a large one and should be read as such. The transplant moves the model's output by
a median 0.738 bits, and $m_u$ is 79% of the post-block-0 state's norm and 76% of its across-token
spread, so the hybrid state sits about three-quarters of the way from the recipient toward the donor.
The claim the numbers support is that the width-relevant content of the block-0 state lives in the
MLP's contribution — not that a small edit suffices.

| what the number can be read from (ridge probe, 80 train / 43 test, 50 splits) | held-out $\rho$ | held-out $R^2$ |
|---|---|---|
| static embedding row $W_E[u]$ | $+0.764 \pm 0.045$ | $0.514$ |
| **block-0 MLP output $m_u$** | $+0.748 \pm 0.049$ | $0.511$ |
| post-block-0 residual state $x_u$ | $+0.772 \pm 0.044$ | $0.558$ |
| shuffled targets (control, worst of the three) | $-0.234$ | \- |

The probe half is a null, and an informative one. Reading $\hat w_u$ off $m_u$ is no easier than
reading it off the static embedding ($+0.748$ against $+0.764$) or off the full post-block-0 state
($+0.772$); all three sit at the same accuracy. So the block-0 MLP does not make the width trait more
linearly explicit — it *carries* it rather than *encoding* it in a newly readable direction, which is
consistent with the earlier failure of edits along the probe direction to steer width. For a
practitioner this means the free static-embedding lookup (Figures 10–11) loses nothing to a probe
placed deeper.

### How many directions does the transplant need? All of them

If the width trait were a low-dimensional feature, a few directions of $m_u$ would carry it, and an
auditor could monitor or edit those directions instead of the whole vector. We test this by
transplanting only part of the difference: project $m_d - m_r$ onto the top $k$ principal components of
$m$ across the 123 tokens, transplant that projection alone, and sweep $k$. Two controls: the bottom
$k$ components (the low-variance tail), and a random $k$-dimensional subspace.

![Transfer slope against the number of transplanted directions, against the variance they carry, and the mean transplanted width against output movement](plots/mlp_rank.png)

**Figure 21.** Left: transfer slope on the donor's width (y) against the number of transplanted
directions $k$ (x, log scale); circles = top $k$ principal components of $m$, triangles = bottom $k$,
squares = a random $k$-dimensional subspace; dash-dotted line = the complete transplant's $+0.913$.
Centre: the same slope (y) against the share of the across-token variance of $m$ that the transplanted
subspace carries (x); dashed line = transfer proportional to variance kept. Right: mean $\hat w$ over
the 132 transplants (y, error bars 1 sd across transplants) against the output movement the partial
transplant causes (x, bits, symmetric log scale), each point labelled with its $k$; dash-dotted line =
the unedited mean 0.565.

| directions transplanted | variance carried | transfer slope | $\rho$ | mean $\hat w$ | output movement |
|---|---|---|---|---|---|
| top 8 principal components | 0.24 | $+0.256$ | $+0.40$ | 0.653 | 0.271 bits |
| top 32 | 0.55 | $+0.298$ | $+0.47$ | 0.647 | 0.599 bits |
| top 64 | 0.79 | $+0.274$ | $+0.58$ | 0.613 | 0.713 bits |
| **all 122 (the complete vector)** | **1.00** | **$+0.913$** | **$+0.97$** | **0.573** | 0.750 bits |
| bottom 58 (the low-variance tail) | 0.21 | $-0.022$ | $+0.01$ | 0.601 | 0.016 bits |
| random 64-dimensional subspace | \- | $+0.000$ | $-0.09$ | 0.570 | 0.001 bits |
| *reference: no edit* | \- | \- | \- | *0.565* | \- |

The answer is that the trait needs the whole vector. Keeping the 64 principal components that carry
79% of the across-token variance of $m$ buys only 30% of the transfer ($+0.274$ against $+0.913$) while
already causing 95% of the full transplant's output movement, and the discarded tail on its own carries
nothing ($-0.022$, and it barely moves the model at all: 0.016 bits). Top-64 and bottom-58 together
would give $+0.25$ if their effects added; the complete vector gives $+0.913$.

The third panel makes the dissociation concrete. Every partial transplant behaves like the
disturbances of Figures 14–19 — it inflates the mean width from 0.565 toward 0.65 and compresses the
spread — whereas the complete transplant is not a disturbance in this sense at all: it returns a mean
of 0.573 and a spread of 0.076, essentially the unedited 0.565 and 0.084, having simply exchanged the
tokens' widths. Truncating the vector costs the transfer and keeps the damage.

So the width trait is not a low-dimensional feature of the block-0 MLP's output that could be read off
a handful of directions; it is a property of that vector as a whole, which matches what the mode-split
experiments found from the output side (the trait belongs to the token's whole next-token map, not to
an identifiable slice of it). One caveat is worth stating: a truncated $m$ is a vector no token
actually produces, so the failure of a partial transplant is evidence about the code being distributed
only to the extent that the model's response to off-manifold states is informative.

### Four model sizes: the ordering belongs to the token, the level to the network

Everything above is one network, so the screen's practical value rests on an untested assumption: that
a token's width is a property of its *representation* rather than a calibration of this particular
model. We repeat the cheap end of the pipeline on Pythia **160M, 410M and 1B** at the same checkpoint —
anchor widths for the same 123 tokens against the same 6 anchors in the same 3 frames, the embedding
probe refitted inside each model, and a mean-ablation of every MLP and attention block in blocks 0–5.
The Pythia family shares one tokenizer, so the same token ids carry the same strings everywhere (the
script asserts it).

An agreement between two models is capped by how reliably each one measures width at all. We therefore
split the six anchors into two halves, recompute every token's width from each half, and
Spearman–Brown correct the agreement between the halves: that number is the **noise ceiling** on any
correlation that model can show, and dividing by the geometric mean of two models' ceilings gives the
agreement they would show with a perfect measurement.

| model (same checkpoint, same tokens/anchors/frames) | median $\hat w_u$ | sd across tokens | measurement reliability | $\rho$ with 1.4B (raw / ÷ ceiling) | embedding probe, held-out $\rho$ |
|---|---|---|---|---|---|
| Pythia-160M (12 blocks, $d = 768$) | 0.749 | 0.079 | 0.734 | $+0.207$ / $+0.256$ | $+0.233 \pm 0.104$ |
| **Pythia-410M** (24 blocks, $d = 1024$) | 0.658 | 0.060 | 0.891 | $+0.884$ / $\mathbf{+0.995}$ | $\mathbf{+0.774 \pm 0.055}$ |
| **Pythia-1B** (16 blocks, $d = 2048$) | 0.620 | 0.063 | 0.932 | $+0.898$ / $\mathbf{+0.989}$ | $\mathbf{+0.755 \pm 0.051}$ |
| Pythia-1.4B (24 blocks, $d = 2048$) — this report | 0.549 | 0.066 | 0.885 | \- | $+0.764 \pm 0.045$ |

![Per-token width in each model against Pythia-1.4B, the agreement against model size, and the 1.4B embedding lookup against every model's measured width](plots/cross_model.png)

**Figure 22.** Left: each model's measured anchor width $\hat w_u$ (y) against Pythia-1.4B's (x), one
marker per token; circles = 160M, squares = 410M, triangles = 1B. Centre: Spearman $\rho$ over the 123
tokens (y) against model size (x, log scale, tick labels name the model) — circles/solid = raw
agreement with 1.4B, squares/dashed = the same divided by the noise ceiling, triangles/dotted = that
model's own split-half reliability (its ceiling); dash-dotted line = perfect agreement. Right: every
model's measured width (y) against $\tilde w_u$, the width predicted by the ridge probe read off
**Pythia-1.4B's** embedding matrix (x, out-of-fold predictions); circles = 160M, squares = 410M,
triangles = 1B, diamonds = 1.4B itself.

Three networks of different width and depth — 410M, 1B, 1.4B — rank the 123 tokens **identically to
within the measurement's own noise** ($+0.98$ to $+1.00$ disattenuated; the raw $+0.88$–$+0.90$ is what
is left after each model's ~0.9 reliability). Their absolute widths differ systematically: transitions
sharpen with scale (median $\hat w_u$ 0.658 → 0.620 → 0.549), exactly the relation the frame-shape
control found for context, where the ordering also survived a level shift. Whatever the per-token trait
is, it is fixed by something the three models share — the token, its embedding, and the corpus
statistics behind it — and not by the individual network.

The practical consequence is the third panel. The probe fitted on Pythia-1.4B's embedding matrix, used
with no forward pass in any model, ranks **410M's** measured widths at $\rho = +0.760$ and **1B's** at
$+0.745$, against $+0.765$ on the model it was fitted in. The lookup is therefore a table you build once
and reuse across the family, which is what makes it cheap enough to run over a whole vocabulary; the
caveat is that these models share a tokenizer and a training corpus, so this tests portability across
networks, not across token inventories.

Pythia-160M is the informative exception. Its disagreement is not a noise artefact: its own reliability
is 0.734, so its ceiling against 1.4B is 0.806, and it reaches 0.207 — a quarter of what is available.
Its widths are also higher (median 0.749) and their spread larger, and the 1.4B lookup tells us nothing
about it ($\rho = +0.04$, $p = 0.63$). The trait is something this family acquires between 160M and
410M, not a fixed property of the tokenizer.

The same three models also reproduce the mechanistic localisation. Mean-ablating each MLP and each
attention block in blocks 0–5 at the final position leaves the ordering intact everywhere except one
place, in all three models.

| mean-ablated component (12 tokens, 6 anchors, 1 frame) | 160M | 410M | 1B | 1.4B |
|---|---|---|---|---|
| unablated sd of $\hat w_u$ across tokens | 0.169 | 0.071 | 0.096 | 0.084 |
| **block-0 MLP**: sd after ablation | **0.023** | **0.021** | **0.019** | **0.018** |
| **block-0 MLP**: $\rho$(before, after) | $+0.55$ | $-0.06$ | $-0.14$ | $-0.10$ |
| **block-0 MLP**: output movement | 0.404 bits | 0.438 bits | 0.445 bits | 0.451 bits |
| every other early component: median $\rho$ | $+0.91$ | $+0.97$ | $+0.98$ | $+0.99$ |
| every other early component: worst $\rho$ | $+0.67$ | $+0.93$ | $+0.86$ | $+0.90$ |
| every other early component: loudest | 0.030 bits | 0.017 bits | 0.013 bits | 0.007 bits |

![Held-out accuracy of the embedding probe in each model, and the across-token spread left by ablating each early component](plots/second_repl.png)

**Figure 23.** Left: held-out Spearman $\rho$ between predicted and measured $\hat w_u$ (y) for a ridge
probe fitted inside each model from that model's own embedding matrix, against model size (x, log
scale); circles/solid = probe, error bars $\pm 1$ sd over 50 random 80/43 train–test splits;
squares/dashed = the same probe with shuffled targets. Right: standard deviation of $\hat w_u$ across
the 12 test tokens (y) after mean-ablating one early component (x, the MLP and the whole attention
block of blocks 0–5); circles = 160M, squares = 410M, triangles = 1B; each model's dotted horizontal
line is its unablated spread. Pythia-1.4B's finer sweep over all 102 individual heads and MLPs is
Figure 18.

The block-0 MLP is again the only early component whose removal collapses the across-token spread by a
factor of 3–7 and erases the ordering, and again the only one the model noticeably feels. This is the
sweep's usual confound — 0.4 bits is a regime in which any disturbance flattens the ordering — so we
also rerun the per-token movement-matched dose–response on Pythia-410M with the same protocol, code and
three seeds as Figure 19.

![Rank agreement against output movement for the block-0 MLP dose and its matched control in 410M and 1.4B, and the level-free per-token movement in 410M](plots/second_ctrl.png)

**Figure 24.** Left: rank agreement $\rho$ between each token's anchor width before and after the
perturbation (y) against the output movement the perturbation causes (x, bits, log scale). Circles/solid
= the 410M block-0 MLP dose ($\alpha = 0.1 \dots 1$), squares/dashed = a random direction added to the
same residual stream and rescaled so **each token's** output moves the same number of bits (mean of 3
seeds, error bars 1 sd across seeds); triangles/dotted and diamonds/dash-dotted = the same two arms in
Pythia-1.4B (Figure 19); gray dash-dotted line = ordering intact. Right: mean per-token width change
with each arm's own mean shift removed, $\lvert\Delta\hat w_u - \overline{\Delta\hat w}\rvert$ (y),
at each dose (x, bits); hatched `//` bars = the MLP dose, dotted `..` bars = the matched control;
annotations are Wilcoxon $p$ over the 12 tokens.

**Half of the 1.4B dose–response replicates and half does not, and the half that fails is the one that
carried the localisation claim.** In raw per-token movement the dose again outruns its own matched
control — $\lvert\Delta\hat w_u\rvert$ 0.016 vs 0.008 at 0.0010 bits, 0.049 vs 0.032 at 0.0074 bits,
0.062 vs 0.048 at 0.0117 bits (Wilcoxon $p = 0.002$, $0.005$, $0.012$) — and it compresses the
across-token spread harder at every matched dose (0.038 vs 0.051 at 0.026 bits). But once each arm's
mean level shift is removed, nothing is left: the level-free paired test is null at all nine rungs
($p \ge 0.62$ in the live band, against $p = 0.034$ and $0.016$ at 1.4B), and the ordering itself is not
damaged faster by the dose — across the six rungs below 0.05 bits the MLP arm sits below its matched
control in 9 of 18 rung × seed comparisons, exactly chance, and the $\rho = 0.6$ crossing puts the
control at 0.023 bits against the MLP's 0.035, i.e. the ratio runs backwards (0.66× against 1.3× at
1.4B).

So the second model supports the *site* and not the *specificity*. What replicates across four sizes is
that the block-0 MLP's contribution is where the width-relevant information sits early in the stream,
and that disturbing it changes the level and the spread of $\hat w_u$ more than an equally loud random
disturbance does. What does not replicate is the claim that it rearranges the *ordering* faster per bit
than a generic disturbance of the same stream: on 410M that margin is absent and its sign reverses
within noise. The 1.4B margin was already modest (1.3× in bits, $p = 0.034$ / $0.016$ at two rungs,
$n = 12$); the honest current reading is that it is a small model-specific effect, and the reproducible
statement is the transplant's — the block-0 MLP's *output vector* carries the width, which needs no
matched control because its evidence is the donor's identity, not the size of the damage.

### Seventeen checkpoints: the ordering is learned in the first 512 steps, and it is not a count table

The lookup is worth building only if we know what it reads. Four model sizes showed the ranking is
learned rather than architectural (160M lacks it, 410M has it), which leaves two very different
possibilities: it repackages a statistic of the training data — how often a token occurs, how
predictable its successors are, both computable with no model at all — or it is something the network
builds while learning that token's successors. Training checkpoints separate them, because a property
of the data is available from the first optimizer steps. We repeat the per-token measurement in
**Pythia-410M at 17 released checkpoints** from `step0` (random initialisation) to `step143000`, with
the same 123 tokens, 6 anchors, 3 frames and block-0 site, and score each checkpoint against the final
ranking (raw, and divided by the noise ceiling from its own split-half reliability), against the
token's unigram count $N_u$ and successor entropy $H_u$ from `dir18`'s manifest, and against both
embedding lookups. This sweep's `step143000` reproduces the independent 410M run above at
$\rho = +1.0000$ over the 123 tokens.

| checkpoint | median $\hat w_u$ | sd across tokens | reliability | $\rho$ with final (raw / ÷ ceiling) | $\rho$ with final, corpus stats removed | $\rho$ with $\log_{10} N_u$ | probe refit here | fixed 1.4B lookup |
|---|---|---|---|---|---|---|---|---|
| `step0` (init) | 0.829 | 0.003 | 0.570 | $+0.015$ / $+0.02$ | $+0.02$ | $-0.00$ | $-0.128$ | $-0.059$ |
| `step16` | 0.824 | 0.002 | 0.553 | $+0.033$ / $+0.05$ | $-0.00$ | $-0.03$ | $-0.058$ | $-0.052$ |
| `step32` | 0.822 | 0.010 | 0.241 | $+0.173$ / $+0.37$ | $-0.05$ | $-0.39$ | $-0.016$ | $+0.205$ |
| `step64` | 0.819 | 0.021 | 0.759 | $+0.291$ / $+0.35$ | $-0.08$ | $-0.63$ | $+0.058$ | $+0.395$ |
| `step128` | 0.819 | 0.022 | 0.875 | $+0.443$ / $+0.50$ | $+0.15$ | $\mathbf{-0.72}$ | $+0.027$ | $+0.540$ |
| `step256` | 0.833 | 0.024 | 0.856 | $+0.661$ / $+0.76$ | $+0.45$ | $-0.58$ | $+0.080$ | $+0.706$ |
| **`step512`** | 0.806 | 0.050 | 0.923 | $+0.788$ / $\mathbf{+0.87}$ | $+0.60$ | $-0.61$ | $+0.249$ | $+0.813$ |
| `step2000` | 0.701 | 0.066 | 0.956 | $+0.866$ / $\mathbf{+0.94}$ | $+0.75$ | $-0.64$ | $+0.654$ | $+0.836$ |
| `step8000` | 0.662 | 0.060 | 0.940 | $+0.864$ / $+0.94$ | $+0.75$ | $-0.63$ | $+0.775$ | $+0.823$ |
| `step32000` | 0.620 | 0.061 | 0.931 | $+0.898$ / $+0.99$ | $+0.82$ | $-0.62$ | $+0.790$ | $+0.807$ |
| `step64000` | 0.595 | 0.077 | 0.935 | $+0.892$ / $+0.98$ | $+0.81$ | $-0.55$ | $+0.807$ | $+0.817$ |
| `step143000` (final) | 0.658 | 0.060 | 0.891 | — | — | $-0.53$ | $+0.774$ | $+0.760$ |

(`step2`, `8`, `1000`, `4000` and `16000` were also measured and fall between their neighbours; the
full grid is in `results/checkpoints_summary.json` and in the figures.)

To show when the trait appears and to separate its two channels, Figure 25 plots agreement with the end
of training, and the level and spread of the measurement, against training step.

![Agreement of each checkpoint's per-token width ranking with the final checkpoint, and the level and spread of the measurement, against training step](plots/ckpt_emergence.png)

**Figure 25.** Pythia-410M-deduped at 17 released checkpoints, same 123 tokens, 6 anchors, 3 frames and
block-0 site throughout. x (both panels): training step, log scale; `step0` (random initialisation) is
drawn off-scale to the left of the vertical rule. Left, y: Spearman $\rho$ over the 123 tokens —
circles/solid = raw agreement of that checkpoint's ranking with `step143000`'s, squares/dashed = the
same divided by the noise ceiling $\sqrt{R_M R_{\mathrm{final}}}$, triangles/dotted = that checkpoint's
own split-half reliability $R_M$; dash-dotted line = perfect agreement. The final checkpoint is omitted
from the two agreement series because it would be compared with itself. Right: median $\hat w_u$
(y-left, circles/solid, shaded band $\pm 1$ sd across tokens) and the across-token sd itself (y-right,
squares/dashed).

**The ordering does not exist at initialisation and is essentially complete after 512 of 143,000
steps.** At `step0` there is nothing to rank — the across-token spread is 0.003 against 0.060 at the
end, the measurement's own reliability is 0.570, and agreement with the final ranking is $+0.015$ — and
that is still true at `step16`. Agreement then climbs to $+0.87$ of the noise ceiling by `step512` and
$+0.94$ by `step2000`, and across the remaining 98.6% of training it does not change ($+0.94$ to
$+0.99$). The **level** keeps moving long after: median $\hat w_u$ falls from 0.833 at `step256` to
0.595 at `step64000`, so transitions go on sharpening for two orders of magnitude of training after
which tokens are narrow has been settled (the final checkpoint's 0.658 is the one non-monotone point).
Ordering and level are separate channels in training, exactly as they are across contexts and across
model sizes.

To ask whether that ordering is a repackaged corpus statistic, and to compare the two ways of reading
it off an embedding matrix, Figure 26 plots the corpus correlations and both lookups against training
step.

![Correlation of each checkpoint's widths with two corpus statistics and with the final ranking after partialling them out, and the accuracy of the fixed and refitted embedding lookups](plots/ckpt_source.png)

**Figure 26.** Same sweep and same x-axis as Figure 25. Left, y: Spearman $\rho$ over the 123 tokens —
circles/solid = $-\rho$ between $\hat w_u$ and $\log_{10} N_u$ (unigram count in `dir18`'s corpus
sample), squares/dashed = $-\rho$ with successor entropy $H_u$ (both negated so that "more of the
ordering explained" points up), diamonds/dash-dotted = raw agreement with `step143000`,
triangles/dotted = that agreement with both corpus statistics partialled out. Right, y: Spearman $\rho$
between each checkpoint's measured $\hat w_u$ and two predictions of it — circles/solid = a ridge probe
refitted inside that checkpoint (shaded band $\pm 1$ sd over 50 random 80/43 splits), squares/dashed =
the fixed lookup read off Pythia-1.4B's embedding matrix, triangles/dotted = the refitted probe with
shuffled targets.

**The trait is built in two stages, and only the first is frequency.** Up to `step128` everything the
model knows about width is *rare tokens are narrow*: the correlation with $\log_{10} N_u$ reaches
$-0.72$ there, stronger than the $-0.53$ the finished model shows, while the agreement with the final
ranking net of frequency and successor entropy is zero ($-0.05$, $-0.08$, $+0.15$ at `step32`–`128`).
From `step256` a second component appears — successor-entropy correlation moves $-0.15 \to -0.46$ and
the partial agreement with the final ranking climbs $+0.45 \to +0.60 \to +0.75$ and plateaus near
$+0.80$. In the finished model the two corpus statistics explain 0.375 of the ranking's rank variance
(0.378 in 1.4B). So the free lookup cannot be replaced by a count table, and what it reads is fixed in
the first few hundred optimizer steps rather than refined late.

**A mature model's lookup detects the trait before the young model's own embedding expresses it.** The
fixed 1.4B lookup tracks each checkpoint's measured widths from the step the trait appears ($+0.21$,
$+0.40$, $+0.54$, $+0.71$, $+0.81$ at `step32`–`512`) and holds $+0.77$–$+0.84$ thereafter, ranking
`step2000` ($+0.836$) slightly better than the finished model ($+0.760$). A probe refitted *inside*
each checkpoint is indistinguishable from its shuffled control through `step256` and only reaches its
final $+0.77$–$+0.81$ from `step4000` on. Read with care — each refit trains on 80 tokens (sd
$\pm 0.10$ early) and `step32`'s reliability of 0.241 makes its ceiling-corrected value unstable — but
the asymmetry is useful: a table built on a trained model reads a checkpoint whose own embeddings could
not have produced that table.

### GPT-2: the ordering is a property of the token *in a training corpus*, not of the string

Every generalisation above holds the tokenizer and the training data fixed, so "the ordering belongs to
the token" was still a statement about Pythia and the Pile. GPT-2 small is the cheapest way outside
that family — a different corpus (WebText), a different BPE vocabulary, and a serial rather than
parallel residual block — and it costs nothing in comparability, because **all 123 endpoint token
strings and all 6 anchor strings are single tokens in GPT-2's vocabulary too**. The same strings, the
same anchors, the same three frames, the same block-0 site.

The first thing the test found is not about the ordering at all: **dir18's width `w` is undefined for
most GPT-2 curves.** `w` requires the output-distance score `d(t)` to rise monotonically and to cross
0.1 and 0.9 once each. Every Pythia curve in this report does (validity 1.000 at 410M and at 1.4B,
median backslide 0.000). At GPT-2's block 0, **88.8% of curves fail** — they rise, fall back and rise
again, with a median backslide of 0.107. So we score every curve with an **envelope width** $\hat w_u$,
which replaces `d` by its running maximum and is therefore defined for every curve; on monotone curves
it is exactly `w`. Inside Pythia the substitution changes nothing (rank correlation with `w` = 1.0000
per curve and per token, both models), which is what licenses using it on GPT-2.

To show what breaks, Figure 27 puts three raw curves from each model side by side and then compares the
per-token rankings, with a second Pythia size as the positive control.

![Raw interpolation curves in both models, and scatter plots of per-token width in Pythia-410M and in GPT-2 against Pythia-1.4B](plots/xmodel_agreement.png)

**Figure 27.** Left: raw `d(t)` for three endpoint tokens against the anchor ` close` in the first of the three frames,
Pythia-1.4B (solid) and GPT-2 small (dashed); x = interpolation position `t`, y = relative output
distance `d(t)`; grey lines mark the 0.1 and 0.9 levels whose crossings define the width. Pythia's
curves rise once; GPT-2's fall back below a level they have already crossed. Middle and right: each
model's per-token envelope width $\hat w_u$ (y) against Pythia-1.4B's (x), one point per token, 123
tokens, same axes in both panels; dotted line = equality. Middle is the positive control
(Pythia-410M), right is GPT-2. Titles give raw $\rho$, the noise ceiling $\sqrt{R_A R_B}$ from the two
models' split-half reliabilities, and $\rho$ divided by it.

| | GPT-2 small | Pythia-410M | Pythia-1.4B |
|---|---|---|---|
| strict-validity rate of `w` | **0.112** | 1.000 | 1.000 |
| median curve backslide | 0.107 | 0.000 | 0.000 |
| median $\hat w_u$ (level) | 0.442 | 0.658 | 0.549 |
| sd of $\hat w_u$ across tokens | 0.132 | 0.060 | 0.066 |
| split-half reliability | **0.319** | 0.891 | 0.885 |
| $\rho$ with Pythia-1.4B's ranking (÷ ceiling) | $\mathbf{-0.219}$ ($-0.41$) | $+0.884$ ($+0.99$) | — |
| fixed 1.4B embedding lookup → this model | $-0.200$ | $+0.760$ | $+0.765$ |
| probe refitted inside this model | $+0.295$ | $+0.774$ | $+0.764$ |
| $\rho$ with $\log_{10}$ unigram count $N_u$ | $-0.038$ | $-0.525$ | $-0.517$ |

**Nothing transfers.** GPT-2 ranks the 123 tokens at $\rho = -0.219$ with Pythia-1.4B and $-0.189$ with
410M, against $+0.884$ between the two Pythias; the free lookup read off Pythia-1.4B's embedding matrix
ranks GPT-2's widths at $-0.200$ where it reaches $+0.76$ on both Pythias; a probe refitted inside
GPT-2 reaches $+0.295$ where the same probe reaches $+0.76$–$+0.77$ inside either Pythia (how far above
chance that is needs the 50-draw null below); and even the frequency signal that
survives everything else in Pythia ($-0.52$) is absent ($-0.038$). Removing unigram count and successor
entropy leaves the cross-model figure where it was ($-0.211$), so this is not a corpus-statistic
mismatch.

**And this is not simply GPT-2 being too noisy to say.** Its measurement reliability is 0.319 against
0.885, which caps any correlation it could show at 0.53 — but the observed $|\rho| \le 0.22$ is well
inside that cap, and the sign is not even stable ($-0.22$ at block 0, $+0.01$ to $+0.14$ at every other
site below). The correct reading is *no relationship*, not a reversed one.

The obvious objection is the site: block 0 of GPT-2's 12 blocks is not block 0 of Pythia's 24, and
GPT-2's curves are badly behaved exactly there. Figure 28 answers it by repeating the whole measurement
at six depths.

![GPT-2 depth sweep: curve validity, measurement reliability, agreement with Pythia and width level against the interpolated block](plots/gpt2_sites.png)

**Figure 28.** GPT-2 small, same 123 tokens, 6 anchors and 3 frames, with the interpolation site moved
across blocks 0, 1, 2, 4, 6 and 8 (x, both panels). Left, y: fraction of curves passing dir18's strict
validity (circles/solid) and the split-half reliability of the per-token width (squares/dashed);
horizontal dotted and dash-dotted lines mark Pythia-1.4B's values (1.000 and 0.885). Right, y: Spearman
$\rho$ between GPT-2's ranking at that site and Pythia-1.4B's block-0 ranking (triangles/solid) and
GPT-2's own median envelope width (diamonds/dashed, the level), with Pythia-1.4B's level (0.549) dotted.

**Depth fixes the curves but not the trait.** Moving down GPT-2, the curves become well behaved —
validity 0.112 → 0.801 and median backslide 0.105 → 0.000 between blocks 0 and 8 — and the level rises
0.442 → 0.671, the same sharpening-with-depth direction Pythia shows. But the per-token measurement
never becomes reliable (peak 0.462 at block 6, against Pythia's 0.885) and agreement with Pythia's
ranking never leaves noise (maximum $+0.141$, $p = 0.12$; $p > 0.2$ at every site except block 0's
negative). At GPT-2's most reliable site the ceiling is 0.64 and the observed value is 0.14.

**One thing does replicate.** Mean-ablating each early component of GPT-2 in turn (12 tokens, first frame)
leaves the block-0 MLP as the only one the model feels — 0.228 bits of output movement against
$\le 0.011$ for the other eleven — and the only one that inflates the across-token spread (0.116 →
0.201) and erases what ordering there is ($\rho = +0.06$, against $+0.38$ to $+0.97$ elsewhere). With
12 tokens and a reliability of 0.32 this is suggestive, not established, but the site of the effect is
where Pythia puts it.

**What this costs the report.** The width ordering is a property of a token *as trained in a particular
corpus*, not of the token string. The practical screen is therefore per-model: an auditor must run the
split-half reliability check first, because it is exactly the statistic that separates the two regimes
here (0.89 in a model where the screen works, 0.32 in one where it does not) and it needs no reference
model to compute. The 160M floor found earlier reads the same way — as a fact about that training run,
not about parameter count.

### Two models fail the screen for two different reasons: is there a plateau to measure?

A width can be computed on any curve, including one that rises steadily from the first step — and such
a curve has no plateau, so its "width" is near 1 by construction and means nothing. Two failures
therefore look alike: a model with plateau-shaped transitions in a *different* token order, and a model
with no plateau structure at all. **Edge drift** separates them, using curves already stored:

```math
E \;=\; d(0.1) \;+\; \bigl(1 - d(0.9)\bigr).
```

Because $d(0) = 0$ and $d(1) = 1$, $E$ is the total movement inside the outer tenth of the path at each
end: $\approx 0$ for a plateau, exactly 0.2 for the straight line $d(t) = t$. Lower is more
plateau-shaped. Figure 29 gives the distribution over all 2,214 curves of six configurations, and then
asks whether GPT-2's disagreement with Pythia survives discarding every curve that is not
plateau-shaped ($E \le 0.1$, half the straight-line value).

![Left: cumulative distributions of edge drift for GPT-2 at three blocks and three Pythia sizes. Right: GPT-2's reliability, noise ceiling and agreement with Pythia before and after discarding non-plateau curves](plots/edgedrift.png)

**Figure 29.** Left, x: edge drift $E$ of a single curve (log scale), y: fraction of that
configuration's 2,214 curves (123 tokens × 6 anchors × 3 frames) with drift at most $E$; further left
is more plateau-shaped. Dashed vertical line = the straight-line reference $E = 0.2$; thin solid line =
the $E \le 0.1$ cut used on the right. Series: GPT-2 small at blocks 0 (solid), 4 (dashed), 8 (dotted)
and Pythia-160M (solid), 410M (dashed), 1.4B (dash-dotted) at block 0. Right, y: Spearman $\rho$ —
GPT-2's split-half reliability, the noise ceiling $\sqrt{R_A R_B}$ for agreement with Pythia-1.4B, and
the measured agreement, each computed on all 2,214 curves (hatched `//`) and on the 56% that are
plateau-shaped (dotted fill).

| | GPT-2 block 0 | GPT-2 block 4 | GPT-2 block 8 | Pythia-160M | Pythia-410M | Pythia-1.4B |
|---|---|---|---|---|---|---|
| median edge drift $E$ (straight line = 0.2) | 0.087 | 0.136 | 0.164 | **0.183** | 0.115 | **0.081** |
| 10th–90th percentile of $E$ | 0.028–0.333 | 0.044–0.461 | 0.093–0.418 | 0.091–0.328 | 0.081–0.168 | 0.059–0.116 |
| fraction of curves with $E > 0.1$ | 0.440 | 0.627 | 0.861 | 0.868 | 0.682 | 0.221 |
| median $\hat w_u$ (level) | 0.435 | 0.587 | 0.670 | 0.743 | 0.649 | 0.545 |
| $\rho$ between $E$ and $\hat w_u$ across tokens | $+0.770$ | $+0.731$ | $+0.556$ | $+0.927$ | $+0.963$ | $+0.967$ |
| $\rho$ of this configuration's $E$ ranking with Pythia-1.4B's | $-0.167$ | $+0.122$ | $-0.049$ | $+0.243$ | $+0.887$ | — |

**GPT-2 does have plateaus; the model that does not is Pythia-160M.** GPT-2's block-0 curves are as
plateau-shaped at the median as Pythia-1.4B's (0.087 against 0.081), so its disagreement is not a case
of measuring a width where there is no plateau. The prediction that the Pythia size lacking the trait
would look like GPT-2 fails: 160M is the *least* plateau-shaped of the six configurations (0.183,
essentially a straight ramp). Within Pythia, plateau structure sharpens with scale
(0.183 → 0.115 → 0.081), and the size at which the ordering appears is the size at which the curves
stop looking like ramps — a correspondence between two measurements at one checkpoint each, not a
demonstrated cause. GPT-2's distinguishing feature is the *spread* of its curve shapes (10th–90th
percentile 0.028–0.333), and depth widens it further (0.164 by block 8) even as the same depth repairs
strict validity.

**GPT-2 has a reproducible width ordering of its own, and it is not Pythia's.** Keeping only its 56%
plateau-shaped curves more than doubles its split-half reliability (0.319 → **0.661**) and lifts the
ceiling from 0.53 to 0.77, while the agreement with Pythia-1.4B stays put ($-0.219 \to -0.185$,
$p = 0.04$, 123 tokens). The earlier reading — "too noisy to say" — is now a measured statement: GPT-2's
tokens are ranked consistently by its own plateau-shaped curves, and that ranking is unrelated to
Pythia's at under a quarter of the ceiling. The same filter cannot be run at 160M (only 13.2% of its
curves pass, leaving 83 tokens and reliability $-0.139$), but on all its curves 160M also has a
reliable ordering of its own (0.699) that is nearly unrelated to Pythia-1.4B's ($+0.213$, ceiling
0.787). Both failing models answer consistently; neither answers like Pythia-1.4B.

**A caveat about `w` itself.** Within each Pythia, a token's width and its edge drift rank the tokens
almost identically ($+0.93$, $+0.96$, $+0.97$), and both transfer between 410M and 1.4B to the same
degree ($+0.887$ for $E$, $+0.884$ for $\hat w_u$). The trait can equally be described as *how long the
output stays put near the endpoints*. This is a restatement of one measurement, not a second finding —
but the two descriptions do come apart once a predictor is put in front of them, and the shape one turns
out to be what the embedding lookup holds (Figure 32).

### Does GPT-2's embedding hold GPT-2's own widths?

The cheapest form of this screen is the lookup: inside Pythia a token's width is readable from its
static embedding row at $\rho = +0.76$, so tokens can be ranked with no forward pass. Whether that
shortcut ports decides how much of the method an auditor gets for free, and GPT-2's reliable ordering
above is the first target worth fitting there. Figure 30 (left) shows why the answer needed a better
control first; Figure 30 (right) places each probe against the ceiling its target allows.

![Left: histogram of 50 shuffled-target draws with the probe and the earlier single-draw control marked. Right: held-out accuracy of four probes with their noise ceilings and null bands](plots/gpt2_probe.png)

**Figure 30.** Left, x: mean held-out Spearman $\rho$ over 50 train/test splits, y: how many of 50
independently shuffled targets landed there (bars, hatched). Dashed vertical line = GPT-2's probe on
its all-curve widths ($+0.295$); dash-dotted vertical line = the single shuffled draw used as the
earlier control ($+0.275$). Right, x: mean held-out $\rho$, error bars $\pm 1$ sd across the 50 splits;
y: four probes, top to bottom — GPT-2's embedding against its all-curve widths (circle), against its
plateau-filtered widths (square), two corpus statistics against the plateau-filtered widths (triangle),
and Pythia-1.4B's embedding against its strict widths (diamond, for scale). Gray dotted bands = the
50-draw null (mean $\pm 1$ sd); black tick marked "ceiling" = $\sqrt{R}$ for that row's target
reliability $R$. Pythia's row has no band because only one shuffled draw was run for it.

| | GPT-2, all curves | GPT-2, plateau curves | GPT-2, 2 corpus statistics | Pythia-1.4B (reference) |
|---|---|---|---|---|
| features | 768 embedding dims | 768 embedding dims | $\log_{10} N_u$, $H_u$ | 2048 embedding dims |
| target reliability $R$ | 0.319 | 0.661 | 0.661 | 0.885 |
| ceiling $\sqrt{R}$ | 0.565 | 0.813 | 0.813 | 0.941 |
| held-out $\rho$ | $+0.295 \pm 0.092$ | $+0.244 \pm 0.122$ | $+0.176 \pm 0.129$ | $+0.764 \pm 0.045$ |
| $\rho$ ÷ ceiling | $+0.52$ | $+0.30$ | $+0.22$ | $+0.81$ |
| held-out $R^2$ | $+0.025$ | $-0.021$ | $-0.036$ | $+0.514$ |
| 50-draw null | $-0.002 \pm 0.093$ | $+0.006 \pm 0.108$ | $-0.012 \pm 0.090$ | (one draw: $-0.201$) |
| permutation $p$ | 0.020 | 0.020 | 0.039 | — |

**The earlier "the probe sits on its control" reading was an artifact of a one-draw control.** GPT-2's
control value of $+0.275$ reproduces exactly and is the largest of 50 independent shuffled draws,
whose distribution is centred at $-0.002$ with sd $0.093$ (range $-0.274$ to $+0.157$). One permutation
of 123 targets is worth about $\pm 0.2$ of apparent skill — the size of the effect being tested.
Against the full distribution GPT-2's probe is above chance ($+0.295$, $p = 0.020$). The same caution
applies to this report's other single-draw controls ($+0.032$ at 410M, $-0.201$ at 1.4B), where it
changes nothing because those probes sit at $+0.77$.

**The lookup does not port.** Refitted on the reliable target, GPT-2's probe recovers 0.30 of the
ceiling its target allows, against 0.81 inside Pythia-1.4B, with held-out $R^2 = -0.021$: it orders
tokens slightly better than chance and predicts none of the variance in the width. GPT-2 widths have to
be measured. Two details sharpen this. First, the more reliable target scores $0.051$ *lower* than the
noisy one ($\pm 0.140$ over the 50 shared splits, filtered target ahead in 16 of 50, paired Wilcoxon
$p = 0.023$), so label noise was not the binding constraint — the part of GPT-2's width its embedding
predicts is disproportionately the part the plateau filter discards. Second, a probe using only
$\log_{10} N_u$ and successor entropy $H_u$ reaches $+0.176$ on the same target and splits
($p = 0.039$), and 768 embedding dimensions beat it by just $0.067 \pm 0.164$ (34 of 50 splits, paired
Wilcoxon $p = 0.009$); that margin comes from entropy ($+0.191$, $p = 0.03$), since frequency is absent
in GPT-2 ($-0.018$, $p = 0.84$) though it carries $-0.52$ in both Pythias. The two models' lookups also
disagree: GPT-2's out-of-fold lookup ranks its own filtered widths at $+0.196$ and Pythia-1.4B's at
$-0.174$, and the two lookups rank the 123 tokens at $-0.204$ with each other.

**The porting recipe therefore has three steps, all cheap:** the edge-drift distribution (does the model
have plateaus), split-half reliability on the plateau-shaped curves (are its widths measurable), and —
only if a free lookup is wanted — an embedding probe against a 50-draw permutation null, benchmarked
against a probe on $\log_{10} N_u$ and $H_u$.

### Cross-checkpoint transplant: the vector carries part of the trait, in a code that has changed

Two findings above invite a mechanism they do not establish: one vector, $m_u$ — the block-0 MLP's
output at the final token position — moves a token's width inside the finished model, and the width
ordering appears within a few hundred training steps. Reading those together as "$m_u$ is what training
installs" would be a mechanism inferred from two correlations, so we intervened. Pythia-410M-deduped
gives two networks that differ in how much of the ordering they hold: `step128` agrees with the end of
training at $\rho = +0.443$ (half of the 0.883 its measurement noise allows), and `step143000` is the
target by definition. We wrote `step143000`'s $m_u$ into the `step128` network — no weight changed, the
write applied only to the endpoint state each interpolation path starts from — and re-measured all 123
tokens against the same 6 anchors in the same 3 frames. Because any disturbance flattens the width
ordering (Figure 15), each transplant runs beside a write of **the same vectors with the token
identities shuffled**: the matched control that separates "the ordering was carried in" from "the
network was knocked about". Figure 31 puts all six write conditions on one axis, in both directions.

![Two panels of dot-and-interval plots comparing six write conditions against shuffled controls, in both transplant directions](plots/ckpt_transplant.png)

**Figure 31.** Pythia-410M-deduped, 123 tokens, 6 anchors, 3 frames, block-0 site. x: Spearman $\rho$
over the 123 tokens; y: the six write conditions. Filled circles: agreement with the *donor*
checkpoint's own measured ordering, with 95% bootstrap intervals over tokens. Open squares: agreement
with the *recipient's* own baseline ordering (no interval — the baseline row is a self-comparison and
sits at $+1.0$ by construction). Gray numbers: the median output shift the write causes, in bits of
Jensen-Shannon divergence between the token's next-token distribution before and after. Left:
`step143000`'s $m_u$ written into `step128`. Right: the reverse. "Norm-matched" scales the donor
vectors by one global constant $\kappa$ matching the two checkpoints' median $\lVert m_u \rVert$
(1.94 at `step128`, 11.06 at `step143000`).

**The transfer is real and small.** Writing a network's own $m_u$ back reproduces its baseline exactly
($\rho = +1.000$, 0.000 bits), which is the check that the hook edits what we claim. With the correct
donor's vector, `step128`'s widths agree with `step143000`'s ordering at $+0.329$ (as measured) and
$+0.189$ (norm-matched); the identity-shuffled writes, which move the output by as much or more (0.075
and 0.036 bits against 0.060 and 0.024), give $-0.030$ and $-0.141$. The gaps are $+0.357$
($[+0.151, +0.553]$) and $+0.324$ ($[+0.075, +0.572]$), 2,000-resample paired bootstrap over tokens.
The norm-matched condition is the cleaner one: it erases the recipient's own ordering ($-0.009$ with
`step128`'s untouched widths) while keeping $+0.189$ with `step143000`'s, so that residual can only
have come from the transplanted vectors. Removing the recipient's baseline ranking by partial
correlation leaves $+0.240$, and additionally removing the donor vector's length leaves $+0.272$;
length is not the channel, since $\lVert m_u \rVert$ ranks the final checkpoint's own widths at
$-0.098$.

**But it does not install the ordering.** Untouched, `step128` already ranks at $+0.443$; the best
transplant leaves it at $+0.329$. The write costs more than it delivers, so $m_u$ is *partially*
sufficient across training time — detectable against a matched control, far short of enough — and the
strong sufficiency result above stays a statement about token-to-token substitution inside one network.

**The two checkpoints do not write $m_u$ in the same coordinates**, which bounds what the experiment
could have shown and is measured directly on the same 123 vectors:

| | `step128` vs `step143000` |
|---|---|
| median $\lVert m_u \rVert$ | 1.94 vs 11.06 |
| cosine between the same token's two vectors | $+0.178$ |
| the same, after subtracting the across-token mean | $+0.198$ |
| rank agreement of $\lVert m_u \rVert$ across tokens | $-0.043$ |
| rank agreement of the 7,503 within-checkpoint pairwise cosines | $+0.031$ |
| the same, after centring | $+0.096$ |

Training rewrites this component's output space, so the transplanted vector arrives in a code the
receiving network never learned to read. That makes $+0.19$ to $+0.33$ the surprising half of the
result, and it makes the reverse direction uninformative: writing `step128`'s $m_u$ into the finished
model destroys that model's ordering ($+1.000 \to +0.148$) without replacing it with `step128`'s
($+0.027$; gap over its shuffled control $+0.088$, $[-0.091, +0.263]$). That write moves the output by
0.64 bits, past the 0.4 bits at which any disturbance flattens the ordering (Figure 15), so the
finished model sits in the regime where the measurement stops discriminating. The informative direction
is the one whose disturbance stays under 0.08 bits. One model, one pair of checkpoints: the conclusion
supported is that the trait travels in $m_u$ *together with the weights that consume it*.

### Shape or width: what the free lookup actually ranks

The lookup is what makes this screen cheap, and until now it was described as reading a token's
crossing width. The same curves carry a second property, the edge drift $E$ defined above, and the two
rank the 123 tokens almost together — $\rho = +0.809$ in Pythia-1.4B and $+0.537$ in GPT-2 on the
targets the probes are fitted to. A probe fitted to one therefore scores on the other, so which property
the embedding holds has to be tested directly. Figure 32 fits four probes per model on identical
features and identical 50 train/test splits: shape alone, width alone, and each with the other's ranking
regressed out. The residual targets are the test — if the embedding holds only shape, the width residual
should fall to chance.

The two targets are medians over the token's $c = 1, \dots, 18$ curves (6 anchors $\times$ 3 frames),
the shape target over all of them and the width target over its plateau-shaped ones:

```math
E_u \;=\; \mathrm{median}_{c} \; E_{u,c},
\qquad
w_u \;=\; \mathrm{median}_{c \,:\, E_{u,c} \le 0.1} \; \hat w^{\mathrm{env}}_{u,c}.
```

Writing $r^{w}$ and $r^{E}$ for the across-token ranks of those two, the width residual is the part of a
token's width ranking that its shape ranking does not explain,

```math
\tilde w_u \;=\; r^{w}_u \;-\; \hat a \;-\; \hat b \; r^{E}_u ,
```

fitted by least squares and standardised; the shape residual $\tilde E_u$ swaps the two roles. Each
target is scored against $\sqrt{R}$, the ceiling its own split-half reliability $R$ allows, with $R$ for
a residual target computed by forming the residual inside each half separately and with a 95% interval
from 2,000 bootstrap resamples of the 123 tokens. Chance is 50 target permutations through the identical
protocol, so the smallest permutation $p$ obtainable is 0.020.

![Four probes per model shown as dots with error bars against their noise ceilings and null bands, GPT-2 on the left and Pythia-1.4B on the right](plots/gpt2_shape.png)

**Figure 32.** x: mean held-out Spearman $\rho$ between predicted and measured target over the 50
shared splits, error bars $\pm 1$ sd across splits. y: four targets — the token's median edge drift
$E_u$, its plateau-filtered width $w_u$, the width with shape regressed out, and the shape with width
regressed out. Circles = probes, hatched gray bars = the 50-permutation null (mean $\pm 1$ sd),
triangles = the ceiling $\sqrt{R}$ allowed by that target's split-half reliability $R$ (absent on the
bottom-left row, whose reliability estimate is negative). Left: GPT-2 small, block 0, 768 embedding
dimensions; right: Pythia-1.4B, block 0, 2,048 dimensions. Panel titles give each model's correlation
between the two targets.

| | Pythia-1.4B | | | | GPT-2 | | | |
|---|---|---|---|---|---|---|---|---|
| target | shape | width | width, shape out | shape, width out | shape | width | width, shape out | shape, width out |
| reliability $R$ | 0.859 | 0.734 | 0.397 | 0.546 | 0.099 | 0.661 | 0.543 | $-0.155$ |
| 95% interval for $R$ | [0.80, 0.90] | [0.60, 0.83] | [0.10, 0.59] | [0.31, 0.70] | [$-0.32$, 0.37] | [0.50, 0.78] | [0.34, 0.70] | [$-0.72$, 0.17] |
| held-out $\rho$ | $+0.783$ | $+0.658$ | $+0.072$ | $+0.243$ | $+0.216$ | $+0.244$ | $+0.280$ | $+0.335$ |
| $\rho$ ÷ ceiling | $+0.84$ | $+0.77$ | $+0.11$ | $+0.33$ | — | $+0.30$ | $+0.38$ | — |
| held-out $R^2$ | $+0.543$ | $+0.375$ | $-0.096$ | $+0.043$ | $-0.035$ | $-0.021$ | $-0.016$ | $+0.042$ |
| permutation $p$ | 0.020 | 0.020 | **0.255** | 0.020 | 0.020 | 0.020 | 0.020 | 0.020 |

**Inside Pythia-1.4B the lookup ranks shape, and the width-specific ordering is not in the embedding.**
The shape probe beats the width probe ($+0.783$ vs $+0.658$, ahead in 47 of 50 shared splits, paired
Wilcoxon $p = 3.4\times10^{-14}$). Removing shape from width drops the probe to $+0.072$ — inside the
null band, the only probe here that is not above chance — while removing width from shape leaves
$+0.243$, a third of its ceiling ($+0.171$ apart, 45 of 50 splits, $p = 2.6\times10^{-11}$). This is not
residualising the signal away with the noise: the width residual is reliably measured ($R = 0.397$,
interval [0.098, 0.591]), so tokens really do differ in how narrow their crossing is relative to their
curve shape, and the embedding does not know it. The screen's practical claims are unchanged — the
lookup still ranks measured widths at $+0.76$ and still predicts 718 unseen pairs — but an auditor
should not expect it to separate tokens whose curves have similar shape, and improving the readout of
the embedding cannot recover information the embedding does not hold.

**GPT-2's weak lookup is not merely a shape lookup.** Its two unresidualised probes are
indistinguishable ($-0.028 \pm 0.173$ paired, 26 of 50 splits, $p = 0.40$) and both residuals stay above
chance: width with shape removed reaches $+0.280$ (0.38 of its ceiling), shape with width removed
$+0.335$, the latter ahead by $+0.055$ (31 of 50, $p = 0.011$). So GPT-2's embedding does carry
width-specific information; it is simply weak. Two of its reliabilities are worth flagging: its shape
target reproduces across anchor halves at only $R = 0.099$ with an interval covering zero, so no
fraction of ceiling is quoted for it — in GPT-2, how plateau-shaped a curve is depends more on the
anchor than on the token, where in Pythia-1.4B it is a stable token property ($R = 0.859$) — and its
shape residual returns a negative estimate, which the probe's above-chance accuracy on held-out tokens
shows to be a downward-biased estimate rather than a target made of noise.

$E$ and $w$ come from the same curves, so their noise is shared and each residual probe is conservative;
the width target is also defined using $E$. Neither coupling explains why the shape residual survives in
both models while the width residual survives in only one. Two models, one site each, 123 tokens.

### Does the transplant move shape too, or only width?

The section above is about what can be *read* from a token's embedding. The transplant of $m_u$ is the
strongest *causal* result here, it acts on a vector computed from that same embedding row one block
later, and it was scored on width alone — so it could have been a shape result in disguise. Figure 33
repeats the transplant unchanged (same 12 tokens, 6 anchors, sentence frame and hook) and records $E$ on
every curve alongside $w$, so both scorings come from the same forward passes. Per recipient, across its
11 cross donors, we take the least-squares slope of the value it lands on against the donor's own value
(1.0 = the donor's value arrives whole) and the rank correlation of the same pair with the donor's
*other* property regressed out, which is needed because baseline width and baseline shape rank these 12
tokens at $\rho = +0.937$ and a marginal statistic on one can be inherited from the other.

![Post-transplant width against the donor's own width, post-transplant edge drift against the donor's own edge drift, and the two scorings paired over the 12 recipients](plots/transplant_shape.png)

**Figure 33.** Pythia-1.4B, block 0, 132 cross transplants of the original 12 endpoint tokens. (a) x:
the donor's own unedited median width $w$; y: the width the recipient lands on after the donor's $m_u$
is written into it. (b) the same transplants with both axes replaced by edge drift $E$. Thin lines join
one recipient's 11 donors; dashed line $y = x$ = the donor's value arriving whole. (c) the two scorings
paired over the same 12 recipients — transported slope (left pair) and partial rank correlation with the
donor's other property held constant (right pair); circles = scored on width, squares = scored on edge
drift, gray lines join one recipient's two values, dashed line at 1.0 = complete transport, $p$-values
from Wilcoxon signed-rank tests on the 12 paired differences.

| transplanting $m_u$ (12 recipients × 11 donors, frame 1) | width $w$ | edge drift $E$ |
|---|---|---|
| donor dependence $\rho$ | $+0.968$ ($p = 5\times10^{-4}$) | $+0.940$ ($p = 5\times10^{-4}$) |
| **transported slope** (1.0 = donor's value arrives whole) | **$+0.913$** | **$+0.970$** |
| partial $\rho$, donor's other property held constant | $+0.796$ (min $+0.49$) | $+0.517$ (min $+0.27$) |
| recipient dependence $\rho$ (control) | $-0.104$ ($p = 0.64$) | $-0.025$ ($p = 0.75$) |
| baseline reliability $R$, 3 anchors vs 3 | 0.671 [0.196, 0.871] | 0.552 [$-0.036$, 0.865] |

**The write hands over the whole curve.** Shape transports at least as completely as width — slope
$+0.970$ against $+0.913$, with shape ahead in 11 of the 12 recipients ($p = 0.0015$) — and both are
close enough to 1.0 that the donor's value effectively arrives intact. The control agrees in both
scorings: with the donor fixed, the recipient's untouched state predicts neither landing value. Two
checks tie this to the original experiment: the baseline widths reproduce it to the last stored digit
(max difference 0.0000) and the width control reproduces its $\rho = -0.104$ exactly. So the readout
result and the causal result are not in conflict, and an intervention on $m_u$ changes where the model's
behaviour is sensitive rather than one summary statistic of it.

**And the width-specific component does travel with $m_u$**, even though no probe on the embedding could
recover it ($+0.072$, $p = 0.255$ above): with the donor's shape held constant, the donor's width still
predicts the recipient's landing width at $+0.796$, above zero in all 12 recipients. A failed probe
bounds what a linear readout of that form extracts, not what the vector contains.

The gap between the two partial correlations ($+0.796$ against $+0.517$) is **not** evidence that width
transports more specifically than shape: a partial correlation is attenuated by noise in the quantity
held constant, and the shape baseline is noisier here, with a reliability interval that covers zero. No
disattenuation is defensible at that width, so the supported claim is that each property transports
specifically, not the ordering between them. The partials also rest on the small part of the donor
ranking the two properties do not share, over 11 donors per recipient. 12 tokens, one frame, one model,
and both properties come from the same curves.

### Can the width-specific component be read from the vector that carries it?

The two sections above leave one question open. The part of the width ordering that curve shape does not
explain is unreadable from a token's embedding row ($+0.072$, $p = 0.255$), yet writing $m_u$ into
another token transports that same part at $+0.796$. A failed probe bounds the probe, so both hold — but
it is worth knowing whether the component becomes readable one block later, in the vector that carries
it. If it did, the free lookup could be upgraded to a one-forward-pass lookup that ranks the crossing
width specifically, which is the screen an auditor wants. Figure 34 refits the same four probes, with
the same targets, ceilings and 50 splits, from two new feature sets at the same final token position:
the block-0 MLP output $m_u$ (the vector the transplant overwrites) and the full post-block-0 residual
state $x_u$ (the embedding plus everything block 0 wrote). Each is averaged over the three sentence
frames to give one vector per token; for $m_u$ that costs nothing, since its cosine across the three
frames is 1.0000. The embedding probes are refit alongside as a reference and must reproduce the earlier
numbers exactly.

![Three panels of four probe accuracies each, for the block-0 MLP output, the post-block-0 residual state, and the static embedding row](plots/early_shape.png)

**Figure 34.** Pythia-1.4B, 123 endpoint tokens, 50 shared train/test splits. Each panel is the feature
set the ridge probe reads: (left) block-0 MLP output $m_u$, (middle) post-block-0 residual state $x_u$,
(right) static embedding row $W_E[u]$. y: the four targets — median edge drift $E_u$, plateau-filtered
width $w_u$, and each with the other's across-token rank regressed out. x: mean held-out Spearman
$\rho$ between predicted and measured target, error bars $\pm 1$ sd across splits. Hatched gray bars =
the 50-permutation null (mean $\pm 1$ sd), carets = the ceiling $\sqrt{R}$ that target's split-half
reliability allows (identical in all three panels, since ceilings belong to the targets), printed $p$ =
a permutation $p$-value above 0.05.

| held-out $\rho$ (123 tokens, 50 shared splits) | $m_u$ | $x_u$ | $W_E[u]$ | ceiling |
|---|---|---|---|---|
| shape $E_u$ | $+0.789$ | $+0.808$ | $+0.783$ | 0.927 |
| width $w_u$ | $+0.634$ | $+0.666$ | $+0.658$ | 0.857 |
| **width, shape removed** | **$+0.084$** ($p = 0.31$) | **$+0.115$** ($p = 0.22$) | **$+0.072$** ($p = 0.26$) | 0.630 |
| shape, width removed | $+0.271$ | $+0.281$ | $+0.243$ | 0.739 |

**The width-specific component is not linearly readable at any of the three sites, including the one
that transports it.** Every probe except the three in the bold row is above chance at the smallest
attainable permutation $p$ of 0.020; the bold row sits inside its own null everywhere, reaching at most
0.18 of a ceiling of 0.630 that the target's reliability does allow ($R = 0.397$, [0.098, 0.591]). No
probe changes by more than $+0.042$ between the three sites, and the refit embedding probes reproduce
the earlier numbers to three decimals, which is what makes the three panels comparable. The detectable
differences are small and one-directional: read split by split, $x_u$ beats the embedding row on shape
($+0.025$, 80% of splits), on the width residual ($+0.042$, 80%) and on the shape residual ($+0.038$,
82%), while $m_u$ alone is $0.024$ behind it on the width target (26% of splits). Block 0 does add a
little linearly readable information, and it shows up in the residual stream rather than in the MLP's
contribution by itself — but $+0.042$ on a probe worth $+0.072$ leaves it inside the null.

Transport and linear decodability come apart here, which is the useful part. The same vector that hands
a donor's crossing width to a recipient at $+0.796$ with shape held constant does not make that
component readable to a ridge probe with 2,048 dimensions and 80 training tokens. For the deliverable
this closes a door: there is no cheap upgrade from the shape-ranking lookup to a width-ranking one by
moving the probe one block deeper.

"Not readable" means "not by this probe" — a different probe family or a larger training set could
still recover the component. The two properties are also entangled in the data ($\rho = +0.809$ across
the 123 tokens), so the width residual is built from the smaller part of a strong correlation and is
the least reliable of the four targets. And these are the three earliest representations only; nothing
here tests the deeper sites where the layer sweep shows the crossing actually sharpening.

## Next experiment

**Read the same four targets from the middle of the network.** The three earliest representations are
now tested and the width-specific component is at chance in all of them. The layer sweep says the
sharpening happens further down: measured at blocks 6, 12 and 18 the token ordering survives
($\rho = +0.72$ between block 0 and block 18) while the median width climbs 0.553 → 0.800 and the
spread across tokens collapses fivefold. That is the natural place to look for a representation in
which the crossing width is explicit. The test refits the same four probes, unchanged in protocol,
targets and splits, from the residual state at blocks 6, 12 and 18 at the same final position, on the
same 123 tokens. If the width residual rises above chance at some depth, the component becomes explicit
downstream of where it is decided and a mid-network probe is the screen that ranks it; if it stays at
chance everywhere while the transplant keeps transporting it, the negative strengthens from "no early
site makes it explicit" to "no residual-stream site at any depth does, under a linear readout". One
forward pass per token per frame with hidden states retained — 123 tokens $\times$ 3 frames, no
interpolation curves.
