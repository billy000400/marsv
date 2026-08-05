# RESULTS — Do tokens with more different next-token distributions have narrower transitions?

> CURRENT-BEST ONLY. One result per experiment, no history (that is in CHANGELOG.md).
> Full method definitions, pair construction and robustness checks are in REPORT.md.

**The question.** For each token we count which token comes **immediately after** it in 2.05 billion
tokens of the stream Pythia was actually trained on, and compare two tokens by the Jensen-Shannon
divergence (JSD) between those two next-token distributions — 0 bits = identical continuation habits,
1 bit = no overlap. We then slide the model's internal state (the residual stream after block 0, at
the final token position) from one token's state to the other's and watch the output. The
**output-distance score** `d(t)` says how far the current logits have travelled from the first token's
logits toward the second's, 0 at the start and 1 at the end. The outcome is **`w`, the fraction of the
path over which `d(t)` climbs from 0.1 to 0.9** — smaller means a narrower transition; a model whose
output moved in proportion to the path position would give `w = 0.8`. So:

> Do token pairs with more different immediate-next-token distributions tend to have narrower
> transitions in the trained model's output-distance score `d(t)`?

**The answer.** Yes. On `pythia-1.4b-deduped` at its final checkpoint, over **1,000 token pairs**,
Spearman $\rho = -0.486$ between corpus next-token JSD and $w$, with a 95% interval of
$[-0.603, -0.353]$ from an uncertainty calculation that accounts for tokens reused across pairs. A
**controlled 60-pair analysis** in which no token is reused and pairs are matched on corpus frequency
and model surprisal gives $\rho = -0.525$ ($p = 1.7\times10^{-5}$). Both are near zero on the same
pairs in the untrained network. The claim goes no further than this:

> Across a large 1,000-pair analysis and a controlled 60-pair analysis, tokens with more different
> immediate-next-token distributions tend to have narrower transitions in the trained model's
> output-distance score. This is an observational endpoint-level relationship; it does not show that
> each plateau corresponds to one continuation distribution or that corpus JSD causes the transition.

## 1. Main result — 1,000 token pairs at the final checkpoint

1,000 pairs built from 123 eligible tokens (200 per JSD group, at most 20 pairs per token, median 17),
selected without looking at any interpolation curve, run in three fixed sentence frames at 50
interpolation positions each. **Tokens are reused across these pairs**, so they are not 1,000
independent observations; every interval below comes from a token-level procedure and no ordinary
$p$-value is reported.

![Left: transition width against corpus next-token JSD for 1,000 pairs with ten binned medians. Right: three intervals for the same correlation.](plots/large_bank.png)

**Figure 1.** More divergent next-token distributions go with narrower transitions, smoothly across
the whole JSD range. *Left:* x = corpus next-token JSD $J(u,v)$ in bits (measurement sample); y = $w$
(smaller = narrower). Each of the 1,000 small markers is one pair, its $w$ the median over the three
sentence frames; marker shape and hue give which of the five JSD groups it was drawn from (group 1 =
most similar continuations), a selection label only. The dashed `x`-marked line is the median $w$ in
ten non-overlapping equal-count JSD bins (100 pairs each, bars = interquartile range) — a summary of
the same pairs, not extra data. Bin medians fall 0.649 → 0.499 with one 0.002 reversal.
*Right:* x = Spearman $\rho$ between $J$ and $w$ with 95% interval bars; rows are the controlled
60-pair analysis (round marker), the 1,000-pair analysis accounting for token reuse (square marker),
and the same pairs with reuse ignored (triangular marker, invalid, shown to size the error).

Between the lowest and highest JSD bin the typical transition narrows by about 0.15 of the
interpolation path — roughly a fifth of the 0.8 a purely proportional response would occupy — from a
predictor that needs no forward pass through the model.

| 1,000 pairs, trained `pythia-1.4b-deduped` step143000 | Value |
|---|---|
| Pairs / tokens / uses per token (min, median, max) | 1,000 / 123 / (1, 17, 20) |
| Spearman $\rho$ between $J$ and $w$ | **−0.486** |
| 95% interval accounting for tokens reused across pairs (4,000 resamples) | [−0.603, −0.353] |
| Token-relabelling permutation $p$ (4,000 relabellings) | **< 0.00025** (0 of 4,000 reached that magnitude) |
| Interval ignoring token reuse — invalid here, shown for contrast | [−0.533, −0.437] |
| Spearman $\rho$ using the pair-selection sample's JSD instead | −0.485 |
| Spearman $\rho$ between $J$ and model-output JSD | +0.729 |
| Median $w$ (interquartile range) / valid-curve rate over 3,000 curves | 0.555 (0.129) / 1.000 |
| **Same 1,000 pairs at step 0:** $\rho$, interval, permutation $p$ | **−0.008** [−0.126, +0.109], $p = 0.86$ |
| Same 1,000 pairs at step 0: $\rho$ with model-output JSD / median $w$ | +0.001 / 0.831 |

Accounting for token reuse widens the interval by a factor of 2.6 (bootstrap standard deviation 0.064
against 0.025), and the association still survives: the permutation test, which keeps the reuse
structure and destroys only the pairing between JSD and width, never reached $|\rho| = 0.486$ in 4,000
relabellings (97.5th percentile 0.116). The untrained network gives a tightly bounded null, so the
relationship is acquired during training — with the caveat that untrained widths span an
interquartile range of only 0.005, leaving little variation for any predictor to explain.

## 2. Controlled analysis — 60 pairs, no token reused

60 pairs in which **no token appears in more than one pair**, chosen to span the full JSD range while
sitting near the middle of the eligible-token distribution on corpus frequency and model surprisal.
Bootstrap intervals over pairs are valid here (10,000 resamples). Construction and the full listing
are in REPORT.md Appendix A.

![Three scatter panels of transition width against corpus JSD: trained 1.4B, untrained step 0, and 410M.](plots/jsd_vs_width.png)

**Figure 2.** The relationship holds in the controlled set and needs training. In all three panels
x = corpus next-token JSD $J(u,v)$ in bits, y = $w$; each dot is one of the same 60 pairs, its $w$ the
median over three sentence frames. Marker shape and hue give the pair's JSD group; the dashed
`x`-marked line is the median $w$ in five non-overlapping equal-count JSD bins. **The panels have very
different y-ranges.** *Left,* trained 1.4B: 0.40–0.80, $\rho = -0.525$. *Middle,* untrained step 0:
the whole panel spans 0.820–0.840, $\rho = -0.056$. *Right,* 410M trained: 0.47–0.82,
$\rho = -0.512$.

Matching pairs one by one on frequency and surprisal removes noise the larger set carries, which is
why the controlled estimate is slightly stronger. The two analyses have opposite weaknesses — reused
tokens and unmatched properties on one side, only 60 observations on the other — and they agree.

| Quantity, 60 controlled pairs | Trained 1.4B (step143000) | Untrained 1.4B (step0) | 410M (step143000) |
|---|---|---|---|
| Spearman $\rho$ between $J$ and $w$ | **−0.525** [−0.701, −0.304], $p=1.7\times10^{-5}$ | −0.056 [−0.314, +0.211], $p=0.67$ | −0.512 [−0.711, −0.272], $p=2.9\times10^{-5}$ |
| Same, using the pair-selection sample's JSD | −0.526 | −0.053 | −0.511 |
| $\rho$ after accounting for the 5 measured pair properties | −0.384 | −0.142 | −0.396 |
| Spearman $\rho$ between $J$ and model-output JSD | **+0.751** [+0.615, +0.843], $p=4.9\times10^{-12}$ | +0.145 [−0.122, +0.394], $p=0.27$ | +0.749 [+0.611, +0.838] |
| Median $w$ (interquartile range) | 0.541 (0.169) | 0.831 (0.006) | 0.640 (0.133) |
| Median $w$ by JSD group 1→5 | 0.619, 0.608, 0.462, 0.502, 0.479 | 0.831, 0.832, 0.833, 0.830, 0.828 | 0.723, 0.683, 0.610, 0.582, 0.578 |
| Median edge drift $E$ (0 = flat ends; 0.184 = no transition) | **0.076** | 0.213 | 0.109 |
| Valid-curve rate under the three criteria, in every group | 1.000 | 1.000 | 1.000 |

The 410M model gives essentially the same correlation on the identical pairs, so the relationship is
not specific to one size in this family. The group medians fall in a stepwise but noisy way at 1.4B
(group 3 dips below groups 4 and 5) at roughly 12 pairs per group. No single sentence frame carries
the result: inside `The thing was`, `They said it was` and `I thought it was` separately, $\rho$ is
−0.486, −0.411 and −0.504.

## 3. Checks that make those numbers mean something

**The corpus statistic is reliable** — otherwise the correlations above would be correlations with
sampling noise. Both gates were fixed before any interpolation ran.

![Left: pair-selection-sample JSD against measurement-sample JSD for 10,000 pairs. Right: between-token and same-token divergence histograms.](plots/jsd_reliability.png)

**Figure 3.** The predictor is stable across two disjoint corpus samples and sits far above its noise
floor. *Left:* 10,000 token pairs; x = JSD on the pair-selection sample, y = JSD on the measurement
sample (the axis labels read "selection-split" and "held-out"); dashed line is $y = x$; Spearman
0.9998 against a 0.90 gate. *Right:* x = JSD in bits, y = count. `//`-hatched = between-token JSD,
median 0.673; `\\`-hatched = same-token half-sample divergence (the sampling-noise floor), median
0.049. Ratio 0.072, against a gate of 0.25.

**The curves are well behaved, so `w` is well defined.**

![Small multiples: all 180 output-distance curves at two checkpoints, one panel per JSD group, trained on top and untrained below.](plots/all_curves.png)

**Figure 4.** Every curve rises once and cleanly; trained curves bend into an S, untrained ones do
not. x = interpolation position $t$ (0 = token $u$'s state, 1 = token $v$'s); y = output-distance
score $d(t)$. Columns are the five JSD groups (panel titles read Q1–Q5, Q1 = most similar
continuations); top row = trained 1.4B, bottom row = untrained step 0. Thin lines are the three
sentence frames of each pair, one line style per frame; the thick dark line is the group's pointwise
median; dotted horizontals mark $d = 0.1$ and $0.9$. Across the 6,000 curves of the two 1,000-pair
runs and the 540 curves of the three 60-pair runs, **zero** failed the span, single-crossing or
monotonicity criteria, and the largest backslide was 0.0000.

**The corpus statistic predicts a difference the model actually makes** in these specific frames —
without this, a result about $w$ would be hard to interpret.

![Scatter of model-output JSD against corpus next-token JSD for the 60 controlled pairs.](plots/output_jsd_validation.png)

**Figure 5.** Corpus JSD strongly predicts the model's own output difference. x = corpus next-token
JSD in bits; y = model-output JSD in the sentence frame, in bits (median over the three frames of the
JSD between the two tokens' output distributions, over the 50,060 corpus-observed target IDs). Marker
shape and hue give the JSD group. $\rho = +0.751$ [+0.615, +0.843], $p = 4.9\times10^{-12}$; on the
same pairs at step 0, $+0.145$ ($p = 0.27$). This is the strongest relationship in the report: corpus
counts predict what the model encodes about two tokens better than they predict the shape of the path
between them.

**How much of the width association survives accounting for everything else measured.**

![Forest plot of the 60-pair association before and after accounting for other pair properties.](plots/adjustment.png)

**Figure 6.** The overall association is strong; the fully adjusted one is not significant. x =
Spearman $\rho$ between corpus next-token JSD and $w$ with 95% bootstrap interval bars; rows top to
bottom: unadjusted, after accounting for the five measured pair properties, after accounting for the
model-output JSD, after accounting for both. Filled marker = $p < 0.05$, open marker = $p > 0.05$.

| Association with corpus next-token JSD, 60 controlled pairs | $\rho$ | 95% interval | $p$ |
|---|---|---|---|
| Unadjusted (overall association) | **−0.525** | [−0.701, −0.304] | $1.7\times10^{-5}$ |
| After accounting for the 5 measured pair properties | −0.384 | [−0.623, −0.110] | 0.0024 |
| After accounting for the model-output JSD | −0.277 | [−0.509, −0.002] | 0.032 |
| After accounting for both | −0.204 | [−0.471, +0.080] | **0.119 (n.s.)** |

The five measured pair properties are mean token log-frequency, mean continuation entropy, mean token
surprisal in the frames, and the block-0 cosine similarity and Euclidean distance between the two
states. Read plainly: the overall association is strong and survives adjustment for frequency,
entropy, surprisal and block-0 geometry; accounting for the model's own output difference as well
leaves $-0.204$ with $p = 0.119$, not significant at n = 60. Corpus JSD is a good *predictor* of
transition width, and this design gives no significant evidence that it explains width beyond the
output separation the model has already learned.

## 4. What the score does not capture

**Flatness and width are nearly the same measurement here.**

![Left: histogram of edge drift at three model settings against the no-transition reference. Right: edge drift against transition width.](plots/edge_drift.png)

**Figure 7.** The trained curves do have flat ends, but flatness adds almost nothing beyond width.
*Left:* x = edge drift $E$ (mean movement of $d$ away from its endpoint values inside the outer 20% of
the path; 0 = perfectly flat ends), y = number of pairs. `//`-hatched = trained 1.4B (median 0.076),
`\\`-hatched = untrained step 0 (0.213), `..`-hatched = 410M (0.109); dashed vertical = the
no-transition reference $E = 0.184$ for a straight line. *Right:* x = $w$, y = $E$; round markers =
trained 1.4B, square markers = step 0. Spearman between them is $+0.971$, so this experiment cannot
tell "flatter ends" apart from "narrower transitions"; the claim made is the second, weaker one.

**The trend is distributional, not pair-by-pair.**

![Raw output-distance curves for the three lowest- and three highest-JSD pairs, all sentence frames drawn separately.](plots/reference_curves.png)

**Figure 8.** Individual pairs deviate from the trend. x = interpolation position $t$; y = $d(t)$.
Solid lines with round/square/triangle markers = the three **lowest**-JSD pairs (` of`/` in`,
` on`/` with`, ` never`/` always`; 0.14–0.27 bits); dashed lines = the three **highest** (` out`/
` your`, ` un`/` better`, ` extremely`/` happening`; 0.85–0.94 bits); all three sentence frames drawn
separately, no averaging. The two function-word pairs at the bottom of the JSD range are the widest
here, but ` never`/` always` — also low-JSD — is among the narrowest. Even the narrowest curve is far
from a step.

**`d(t)` is uninformative when the two endpoint outputs are already almost identical.** Two named
example pairs make this concrete: interpolating between *"My house is big"* and *"My house is large"*,
and between *"My house is big"* and *"My house is in"*, in the frame `My house is` plus the three
project frames, at all three model settings. Alongside `d(t)` we record the **absolute output
movement** $M(t)$, the JSD in bits between the output distribution at position $t$ and the one at the
start.

![Four panels: output-distance curves for the two named pairs in the trained model, their absolute output movement in bits, the same curves untrained, and both pairs placed against the 60 controlled pairs.](plots/house_reference.png)

**Figure 9.** ` big`/` in` gives a narrow transition; ` big`/` large` gives the straight line of a
pair whose outputs never separate. *(a)* x = $t$, y = $d(t)$, trained 1.4B, frame `My house is`; solid
with round markers = ` big`/` large` ($w = 0.773$), dashed with square markers = ` big`/` in`
($w = 0.357$); gray dotted diagonal = the no-transition reference $d(t) = t$; faint horizontals mark
$d = 0.1$ and $0.9$. *(b)* Same pairs and styles; y = absolute output movement $M(t)$ in bits.
*(c)* The same prompts untrained, axes as in (a): both lie on the diagonal. *(d)* Where the two pairs
fall against the controlled set: x = corpus next-token JSD in bits, y = $w$; small gray dots = the 60
controlled pairs, dash-dotted `x`-marked line = their median $w$ in five equal-count JSD bins, large
open circle and open square = the two named pairs at their `My house is` width, vertical bars = that
pair's width across the other three sentence frames.

The numbers below separate a real narrow transition from an artefact of the score: the two pairs
differ by a factor of two in width, but by a factor of 27 in how far the output moves at all.

| Quantity, trained 1.4B, frame `My house is` | ` big`/` large` | ` big`/` in` |
|---|---|---|
| Corpus next-token JSD $J(u,v)$ [bits] | 0.412 | 0.701 |
| Token occurrences in the measurement sample | 122,257 / 175,159 | 122,257 / 9,821,847 |
| Transition width $w$ (no-transition reference $\approx 0.8$) | 0.773 | **0.357** |
| Edge drift $E$ (no-transition reference 0.184) | 0.162 | **0.043** |
| Absolute output movement $M(1)$ / $M(0.5)$ [bits] | 0.035 / 0.008 | 0.935 / 0.505 |
| $w$ across the three project sentence frames | 0.767–0.793 | 0.348–0.500 |
| Untrained step 0: $w$ / $E$ | 0.834 / 0.216 | 0.829 / 0.211 |
| 410M trained: $w$ / $E$ | 0.794 / 0.198 | 0.494 / 0.075 |

The trained model's outputs after *"My house is big"* and *"My house is large"* differ by only 0.035
bits, 0.008 of which has accumulated by the midpoint. $d(t)$ divides that near-zero movement by itself
and records the leftover as a straight line, so any $w$ computed for such a pair describes noise
rather than a transition; checking $M(1)$ first costs two forward passes and is the cheapest guard.
` big`/` in`, whose outputs are 0.935 bits apart, has a genuine transition and it is narrow
($w = 0.357$, narrower than all 60 controlled pairs, whose minimum is 0.401). Both behaviours are
learned, and the 410M model reproduces them. As an illustration the pair points the right way — the
higher-JSD pair is the narrower one — but the gap is far wider than the trend predicts (controlled
pairs near 0.41 bits have median $w = 0.639$, near 0.70 bits 0.502), and ` in` occurs about 80 times
more often than ` big`, so this pair would fail the controlled set's factor-of-two frequency rule.

## 5. Limitations

**Observational.** We did not intervene on corpus JSD. The one adjustment available points against an
independent contribution: $-0.204$, $p = 0.119$, after accounting for the model's own output
difference and the five measured pair properties.

**Only the two endpoints were measured.** We measure the corpus distribution of the single token
following $u$ and following $v$, and the model's outputs at the two ends and along the path. No
continuation distribution was measured at any intermediate point, so nothing here says a flat stretch
corresponds to one continuation distribution, or that continuation distributions jump anywhere.

**`w` and edge drift are almost the same measurement** ($\rho = +0.971$), so the association cannot be
attributed to flatness specifically. A flat $d(t)$ means this one relative distance score changes
slowly; it does not establish that the logits or the output distribution are stationary.

**The untrained baseline is a restricted-range control**: step-0 widths span an interquartile range of
0.006 just under 0.8.

**Scope.** One model family, one hook point (post-block-0, final position), three sentence frames, 50
interpolation positions, a context-averaged single-token corpus statistic from 0.68% of the released
training stream, and a token pool limited to high-frequency word-start tokens in the model's top-256
continuations of all three frames. The 410M run shares the corpus estimates and the pair set, so it
checks scale rather than replicating independently. Transitions are moderate: even the narrowest
controlled pair has $w \approx 0.40$.

## Robustness (details and two further figures in REPORT.md Appendix B)

Swapping which corpus sample supplies the predictor changes nothing ($\rho(J_{\mathrm{sel}}, J) =
0.99972$ on the controlled set; $-0.526$ against $-0.525$ at trained 1.4B, $-0.485$ against $-0.486$
on the 1,000 pairs). A looser post-hoc pool of 75 pairs (top-512 filter) gives $-0.419$
[−0.587, −0.225] trained, $-0.155$ at step 0, $-0.320$ at 410M; dropping the one pair whose token
` un` is a word-start fragment gives $-0.502$ ($p = 5.2\times10^{-5}$, n = 59). Every self-test of the
interpolation experiment passes: patching at $t = 0, 1$ reproduces the unpatched logits to within
$6.3\times10^{-5}$ relative error, swapping which token is $u$ and which is $v$ changes $w$ by at most
$1.1\times10^{-5}$ against a grid spacing of 0.0204, and the shared prefix's block-0 residuals within
a pair differ by exactly 0.0. A scan of where the patch is applied shows the median width rising
0.599 → 0.804 as the patch moves from block 0 to block 23, converging on the no-transition value.

**Auditability.** Every raw 50-point $d(t)$ curve is committed — `results/curves_*.npy` plus a
plain-text `results/curves_*.csv.gz` export — so every width, flatness and validity number above can
be recomputed without a GPU.
