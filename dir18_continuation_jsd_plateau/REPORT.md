# Does training-corpus continuation divergence predict how sharply a model separates two words?

> Final, presentable, current-best only (no history — see CHANGELOG.md).

## Summary

Large language models appear to carve their internal activation space into discrete regions: if you
take the hidden state the model computes for one input and slide it continuously toward the hidden
state for a different input, the model's *output* often does not move smoothly. It stays put, then
flips. These flat stretches are called **activation plateaus**. They matter for safety because a
model that computes over a small number of discrete internal states is a model whose behaviour might
be enumerable and auditable — and because a sharp flip is a place where a small perturbation
produces a large behavioural change.

This report asks a simple observational question: **can you predict, from the training corpus alone,
which pairs of inputs the model will separate sharply?** We estimate, from 2.05 billion tokens of
Pythia's actual released training stream, how differently two words are continued in text
(Jensen-Shannon divergence between their next-token distributions), and we test whether
higher-divergence word pairs produce sharper output transitions inside the trained model.

**They do.** On `pythia-1.4b-deduped` at its final checkpoint, corpus continuation divergence
predicts narrower transitions with Spearman $\rho = -0.525$ (95% CI $[-0.701, -0.304]$,
$p = 1.7\times10^{-5}$, n = 60 endpoint-disjoint pairs; negative means *higher divergence → narrower
transition*). The same bank at **step 0**, before any training, shows no relationship
($\rho = -0.056$, CI $[-0.314, +0.211]$) — though it also has almost no variation in width to predict
(interquartile range 0.006), so that control is partly a floor effect.
`pythia-410m-deduped` replicates the effect ($\rho = -0.512$, $p = 2.9\times10^{-5}$). Corpus
divergence predicts even more strongly a distinction the model demonstrably learned — its own output
divergence in the same context ($\rho = +0.751$, $p = 4.9\times10^{-12}$).

This matches the prespecified verdict branch *"corpus JSD predicts model-output JSD and smaller $w$;
step 0 does not"*. Stated precisely, and this precision matters: **corpus divergence predicts (i) how
far apart the trained model puts the two words' output distributions and (ii) the overall width of
the transition between them.** Three qualifications. First, the width metric $w$ measures the whole
10%→90% transition; the trained curves *are* plateau-shaped in level terms (our flatness metric gives
0.076 against 0.184 for a straight line), but flatness and width correlate at $+0.971$ across pairs,
so this experiment cannot separate "predicts flatter plateaus" from "predicts narrower transitions"
and we claim only the latter. Second, **the headline is a total association.** It weakens to $-0.384$
after adjusting for endpoint frequency, continuation entropy, surprisal and the block-0 geometry of
the two endpoint states; to $-0.277$ after adjusting instead for the model's own output divergence
(the obvious mediator); and to $-0.204$ ($p = 0.119$, **not significant**) after adjusting for both.
Corpus divergence is therefore a good *predictor* of transition width, but we have no significant
evidence that it explains width *independently* of the output separation the model learned. Third,
four intermediate checkpoints **contradict the expected pattern**: the plan predicted the negative
relationship would *strengthen* during training, but it is already at full strength by step 1000
($\rho = -0.582$) and afterwards moves within overlapping confidence intervals — while the transitions
themselves sharpen through step 64000 (median $w$ 0.831 → 0.512) and then show a modest late reversal
to 0.541 at the final checkpoint. This is an observational predictor test: it does not show that
divergence *causes* plateaus.

---

## Methods

### Data & Model

**Model.** `EleutherAI/pythia-1.4b-deduped` (1.4B parameters, 24 transformer blocks, residual width
2048), at revision `step143000` (the final checkpoint) and revision `step0` (the untrained
initialisation). Formation subset: the same model at revisions `step1000`, `step8000`, `step32000`
and `step64000`. Scale check: `EleutherAI/pythia-410m-deduped` at `step143000`. Native Hugging Face
GPT-NeoX modules, `eval()` mode, `torch.inference_mode()`, float32. Every checkpoint is run on the
**same frozen 60-pair bank** with the same corpus estimate.

**Hook point.** The residual stream at the **final token position, immediately after transformer
block 0**. This is the single site we interpolate and patch; blocks 1–23 then run normally and we
read the final-position logits after the final LayerNorm and unembedding. One control (Figure 11)
repeats the assay patching after blocks 0, 6, 12, 18 and 23 instead.

**Corpus.** `EleutherAI/pile-deduped-pythia-preshuffled` — the exact tokenised, pre-shuffled stream
Pythia was trained on. We did **not** reconstruct the full 602 GB. The dataset is one concatenated
`uint16` array of 146,432,000 sequences of exactly 2049 tokens; we verified this against the official
Megatron index header (magic `MMIDIDX`, version 1, dtype code 8 = `uint16`, length 146,432,000, every
listed sequence size 2049) and confirmed the arithmetic is byte-exact: the index file is 1,757,184,042
bytes as predicted by $34 + 12L + 8D$, and the data shards total 600,078,336,000 bytes, which equals
$146{,}432{,}000 \times 2049 \times 2$. The byte offset of training row $i$ is therefore exactly
$4098i$, so a row-aligned sample is a plain HTTP byte range.

We took **two distant, row-aligned samples of 500,000 rows each** — split **A** starting at global row
1,000,000 and split **B** starting at global row 73,300,000, roughly halfway through the run. Each
split is 1,024,500,000 tokens (2.05B total, ~4.1 GB). We count only the 2,048 adjacent transitions
*inside* each row and **never join two rows**.

**Sample sizes.** 10,000 word pairs for the reliability bank; 60 endpoint-disjoint pairs in the frozen
assay bank (14/13/11/10/12 across the five divergence quintiles); 3 carrier contexts per pair; 50
interpolation points per curve, so 180 raw curves per checkpoint; 50,060 valid target token IDs.

### Metrics

Everything below is motivated by one chain of questions: *(i) is our corpus estimate stable enough to
be a predictor at all? (ii) how sharp is the model's transition, and is it a genuine plateau? (iii)
does corpus divergence predict that sharpness? (iv) is the prediction about something the model
actually learned, or about architecture and token geometry?*

**(i) Corpus continuation divergence** — the predictor. For an endpoint word $a$ we estimate its
context-averaged next-token distribution directly by counting, over the training stream, what token
follows each occurrence of $a$:

```math
p_a(y) \;=\; \frac{N(a, y)}{\sum_{y'} N(a, y')},
```

where $N(a,y)$ counts adjacent $(a,y)$ pairs inside training rows. The divergence between two
endpoints is the symmetric, unsmoothed, base-2 Jensen-Shannon divergence (JSD), restricted to target
IDs that actually occur in training:

```math
JSD(p_a, p_b) \;=\; \tfrac12 D_{KL}\!\left(p_a \,\Vert\, m\right)
                  + \tfrac12 D_{KL}\!\left(p_b \,\Vert\, m\right),
\qquad m = \tfrac12 (p_a + p_b).
```

It is measured in **bits** and ranges from 0 (the two words are continued identically) to 1 (their
continuations never overlap). We use plain JSD rather than, say, cosine similarity of embeddings
because it is a property of the *data*, computable without ever consulting the model — which is the
whole point of asking whether the corpus predicts the model. $JSD_A$ (from split A) is used **only**
to select and bin pairs; $JSD_B$ (from split B, disjoint training rows) is the predictor in every
reported analysis, so the reported correlations are not inflated by selection on the same noise.
Consumed by Figures 1, 4, 5, 7, 8, 9 and 10.

**Reliability and the sampling-noise floor** — a count-based divergence is only meaningful if it is
stable across samples. Two checks, both prespecified as gates before any plateau curve was viewed.
The first is the rank agreement of the two independent estimates, the Spearman correlation of $JSD_A$
with $JSD_B$ on 10,000 pairs (gate: at least 0.90). The second asks how much divergence we would
measure between two estimates of the *same* word — pure sampling noise. Splitting split A into two
disjoint halves $A_1$ and $A_2$, the noise ratio is the median same-word divergence over the median
between-word divergence:

```math
\text{noise ratio} \;=\; \frac{\mathrm{median}_a \; JSD\!\left(p^{A_1}_a,\; p^{A_2}_a\right)}
                              {\mathrm{median}_{(a,b)} \; JSD\!\left(p^{B}_a,\; p^{B}_b\right)}
```

Gate: below 0.25, i.e. the between-word signal must be at least four times the same-word noise.
Consumed by Figure 1.

**(ii) Relative logit distance and transition width** — the outcome. We build the two endpoint prompts
(a carrier context plus endpoint token $a$ or $b$), take their final-position post-block-0 residual
states $x_A$ and $x_B$, and interpolate between them with **norm-rescaled spherical linear
interpolation (SLERP)** at 50 evenly spaced positions $t$ in $[0,1]$. Writing $\hat u$ for a unit
vector and $\Omega$ for the angle between $\hat u_A$ and $\hat u_B$:

```math
x(t) \;=\; \big[(1-t)\lVert x_A\rVert + t\lVert x_B\rVert\big]\cdot
           \frac{\sin\!\big((1-t)\Omega\big)\,\hat u_A + \sin\!\big(t\Omega\big)\,\hat u_B}{\sin \Omega}
```

We use SLERP rather than straight-line interpolation because residual states have a large, roughly
constant norm; a straight line dips through a low-norm region the model never sees. When $\sin\Omega$
falls below $10^{-6}$ the formula is numerically unstable and we fall back to renormalised linear
interpolation (this never triggered in the reported runs). We patch $x(t)$ into the final position
only, run the remaining blocks, and read the final-position logit vector $z(t)$ restricted to valid
target IDs. The outcome curve measures how far the *output* has travelled from endpoint A to
endpoint B:

```math
d(t) \;=\; \frac{\lVert z(t) - z_A \rVert_2}{\lVert z(t) - z_A \rVert_2 + \lVert z(t) - z_B \rVert_2}
```

so $d(0) = 0$ and $d(1) = 1$. A **plateau** is a stretch where $d$ barely moves while $t$ does,
followed by a rapid rise. We summarise each curve by its **transition width**:

```math
w \;=\; t(d = 0.9) \;-\; t(d = 0.1),
```

linearly interpolated on the 50-point grid. **Smaller $w$ means the output flips over a shorter
stretch of the path.** A perfectly linear output response gives $w \approx 0.8$; a step function gives
$w$ near 0. Each pair's outcome is the **median $w$ across its valid carrier contexts**. Consumed by
Figures 2, 3, 4, 5, 6, 8, 9, 10 and 11.

**Curve validity** — $w$ is only meaningful for a curve that rises once, cleanly, through both levels.
A curve that wanders back down, or crosses a level several times, has no well-defined width, and the
plan requires such curves to be *shown* rather than forced into the correlation. We therefore apply
three explicit criteria to every one of the 180 raw curves per checkpoint. **Span:** $d(0) \le 0.1$
and $d(1) \ge 0.9$, so both levels are actually attained. **Single crossing:** the curve crosses
$d = 0.1$ exactly once and $d = 0.9$ exactly once, counting crossings in either direction.
**Monotonicity:** the largest *backslide* — the furthest the curve ever falls below its own running
maximum —

```math
B \;=\; \max_{t}\Big(\max_{s \le t} d(s) \;-\; d(t)\Big)
```

must be at most 0.02. A curve failing any criterion gets $w = $ NaN and is dropped from the
correlations; a pair with fewer than two valid contexts is itself dropped. Invalid rates are reported
per divergence bin, and all raw curves are committed (`results/curves_*.npy`, and
`results/curves_*.csv.gz` as a plain-text export) so the criteria can be re-applied independently.
Consumed by Figure 2 and the validity table.

**Curve flatness (edge drift)** — width alone cannot tell a *plateau* (flat, flat, jump) from a
*steeper straight line*, and the word "plateau" is the whole reason this assay is interesting. So we
also measure how far the curve moves away from its endpoint values inside the outer 20% of the path:

```math
E \;=\; \frac{1}{|T_0|}\sum_{t \in T_0}\big(d(t) - d(0)\big)
    \;+\; \frac{1}{|T_1|}\sum_{t \in T_1}\big(d(1) - d(t)\big),
\qquad T_0 = \{t \le 0.2\},\; T_1 = \{t \ge 0.8\}.
```

$E = 0$ means perfectly flat ends — a real plateau. The **no-plateau reference** is the straight line
$d(t) = t$, which gives $E = 0.184$ on our grid; anything near or above that has no plateau at all.
Lower is flatter. Consumed by Figure 6.

**Learned sharpening** — the trained width $w$ mixes two things: how sharp a pair *starts out* under
random initialisation and how much training narrowed it. Comparing trained and untrained models at the
group level (Figure 4) does not remove the first. Since the same frozen bank is run at both
checkpoints, we can subtract each pair's own untrained baseline and ask what training *did* to that
specific pair:

```math
\Delta w \;=\; w_{\text{trained}} \;-\; w_{\text{step }0}.
```

$\Delta w$ is negative when training narrowed the transition, and more negative means more sharpening.
Correlating $JSD_B$ with $\Delta w$ is the within-pair version of the primary test. Consumed by
Figure 8.

**Mediation by the model's own output divergence** — the two results above suggest an obvious causal
story: the corpus makes the model separate the two words' outputs ($JSD_{\mathrm{out}}$, below), and
that separation is what produces a narrow transition. If so, $JSD_{\mathrm{out}}$ is a *mediator*, and
adjusting for it should remove most of the association. We therefore report the same partial Spearman
as in the sensitivity model, adjusting for (a) $JSD_{\mathrm{out}}$ alone and (b)
$JSD_{\mathrm{out}}$ together with the five covariates. This is a rank-based *adjustment*, not a
formal causal mediation estimate: with an observational design and one hook point we cannot separate a
mediator from a confounder, so a shrinking coefficient is consistent with the mediation story but does
not establish it. $p$-values for these adjusted correlations come from the rank correlation of the
residuals, not corrected for the covariate degrees of freedom, so they are mildly optimistic. Consumed
by Figure 8.

**Late-reversal test** — median width falls from step 0 to step 64000 and then rises slightly at the
final checkpoint. A change in a median over 60 pairs can easily be noise, so we test it at the pair
level with a two-sided **paired Wilcoxon signed-rank test** on $w_{143000} - w_{64000}$ (the same 60
pairs at both checkpoints), and report how many pairs ended blunter. Consumed by Figure 9.

**(iii) Association** — reported as the Spearman rank correlation $\rho$ between $JSD_B$ and $w$, with
a 95% confidence interval from 10,000 bootstrap resamples. The bank is **endpoint-disjoint** (no token
appears in two pairs), so resampling pairs resamples endpoints as intact clusters; there is no hidden
reuse inflating the effective sample size. Consumed by Figures 4, 5, 8, 9 and 10.

**(iv) Model output divergence** — the validity check on the predictor. Corpus divergence is a global,
context-free statistic, while the assay runs in one specific carrier context. If corpus divergence did
not even predict how differently the model itself continues the two endpoints *in that context*, a
width null would be uninterpretable. So we compute the base-2 JSD between the two endpoint next-token
distributions the model actually outputs:

```math
JSD_{\mathrm{out}} \;=\; JSD\!\big(\mathrm{softmax}(z_A),\; \mathrm{softmax}(z_B)\big)
```

and correlate it with $JSD_B$. Higher means the model draws a bigger distinction between the two
endpoints. It is also the mediator in the adjustment ladder above. Consumed by Figures 7, 8 and 9.

**Sensitivity (partial Spearman)** — divergent words might simply be rarer, more surprising, or
geometrically further apart at block 0, and any of those could drive sharpness. We rank-transform
$JSD_B$, $w$ and five covariates — mean endpoint log-frequency in the corpus, mean continuation
entropy in bits, mean endpoint surprisal under the model in the carrier context, and the block-0
cosine similarity and Euclidean distance between $x_A$ and $x_B$ — regress the first two on the
covariates, and correlate the residuals. Reported alongside the unadjusted result, never in place
of it.

### Baselines and controls

**Step 0 (untrained) checkpoint** — the primary baseline. The identical frozen bank and identical
assay, run on `pythia-1.4b-deduped` revision `step0`. Any width relationship surviving here is
produced by architecture, tokenisation, and random initialisation rather than by learning. Its
interpretation has a limit worth stating up front: the untrained network's widths are almost constant
(IQR 0.006), so a null correlation there is partly a *floor effect* — there is very little variation
for any predictor to explain.

**Same-token split-half divergence** — the noise baseline for the predictor, defined by the noise-ratio
equation above. It answers "how large a JSD would we see for two *identical* words, purely from finite
counts?"

**Linear-response reference** — if the model's output moved proportionally with the interpolation
position, $w$ would be $0.9 - 0.1 = 0.8$ and edge drift would be $E = 0.184$. Values near those mean
"no plateau"; the step-0 medians (0.831 and 0.213) sit at or slightly beyond this reference, and so
does the deepest block-scan point.

**Post-hoc top-512 bank** — a secondary bank of 75 pairs built by relaxing the endpoint filter (see
below) from the prespecified top-256 to top-512. It is **not** a prespecified fallback; it exists only
to check that the conclusion does not depend on where the filter is drawn, and it is reported as a
clearly labelled post-hoc analysis in Figure 10.

**Fragment-dropped bank** — the prespecified top-256 bank minus the single pair whose endpoint `un` is
a word-start fragment rather than a complete word (n = 59). A second robustness check on the same
figure, for readers who expected the filter to admit only complete words.

**Block scan (blocks 0, 6, 12, 18, 23)** — patching later leaves fewer blocks to compute a sharp
response. If sharpness is produced by downstream computation rather than by readout geometry, $w$
should grow as the patch moves later. Run on 10 frozen pairs (the 5 lowest and 5 highest $JSD_B$).

**Assay self-tests** — patching at $t=0$ and $t=1$ must reproduce the unpatched endpoint logits (worst
case across all runs: $4.6\times10^{-5}$ relative error); swapping which endpoint is A and which is B
must leave $w$ unchanged (worst case over 20 pairs: $1.1\times10^{-5}$, against a grid spacing of
0.0204); and within a pair the two prompts must share every prefix token and every prefix block-0
residual (measured difference: exactly 0.0).

### Pair bank construction (frozen before any curve of this bank was seen)

Endpoints are lowercase alphabetic **word-start tokens** of at least two characters (GPT-NeoX BPE
marks a word start with `Ġ`) that are among
the trained model's **top-256** eligible word continuations of **all three** carrier contexts
(`The thing was`, `They said it was`, `I thought it was`), so every prompt is in-distribution. This is
the filter the plan prespecified. Further rules, all fixed in advance: each endpoint occurs at least
20,000 times in **each** corpus split (123 of the top-256 tokens qualify); the two endpoint
frequencies within a pair differ by at most a factor of two (1,763 candidate pairs survive); **no
endpoint token is reused anywhere in the bank**, so pairs are statistically independent; and pairs are
taken in each $JSD_A$ quintile, round-robin across quintiles, choosing at each step the pair closest to
the bank-wide median in corpus log-frequency and model surprisal.

A word-start token need not be a **complete word**: the filter tests the `Ġ` marker, lowercase
alphabetic characters and length, so a multi-letter word-start prefix passes it. Exactly one such
token survives into the bank — `un` (in the pair `un`/`better`) — out of 120 endpoints; every other
endpoint is a complete word. We do **not** treat that as a defect to be patched after the fact,
because the bank was frozen before any curve of it was seen; instead we report the bank as
"word-start tokens" and add a prespecified-style sensitivity check that drops that one pair (Figure
10). Its per-pair count and divergence are estimated exactly as for every other endpoint.

The endpoint-disjointness rule caps the bank at $\lfloor 123/2 \rfloor = 61$ pairs; we obtain **60**,
distributed 14/13/11/10/12 across quintiles Q1→Q5. The covariate balance holds (Kruskal-Wallis
$p = 0.52$ for log-frequency, $p = 0.21$ for surprisal — a large $p$ means the bins are statistically
indistinguishable on that covariate). $JSD_B$ across the bank ranges from 0.14 (`of`/`in`) to 0.94
(`extremely`/`happening`). The bank is stored with all token IDs, counts, and both JSD values in
`results/pair_manifest_top256.json`, and a 15-pair calibration subset (three per quintile) passed the
prespecified dynamic-range gate before the full analysis (IQR of $w$ = 0.109, gate $\ge 0.05$; all
curves valid, gate $\ge 0.80$).

---

## Results

### The corpus predictor is reliable

Before looking at a single plateau curve we checked whether a count-based divergence estimated from
2.05B tokens is stable. Figure 1 shows both prespecified gates, and both pass by a wide margin.

![Left: JSD from split A against JSD from split B for 10,000 word pairs. Right: overlaid histograms of between-word and same-word divergence.](plots/jsd_reliability.png)

**Figure 1.** The corpus predictor is highly reliable. *Left:* each point is one of 10,000 word pairs;
x is $JSD_A$ (bits, estimated from split A), y is $JSD_B$ (bits, split B, completely disjoint training
rows). The dashed line is $y = x$. Rank agreement is Spearman 0.9998, far above the 0.90 gate.
*Right:* x is JSD in bits, y is the number of word pairs (or words) per histogram bin. The `//`-hatched
distribution is between-word $JSD_B$ (median 0.673); the `\\`-hatched distribution is the same-word
split-half divergence — the sampling-noise floor — with median 0.049. The noise ratio is 0.072, well
under the 0.25 gate, so roughly 93% of the measured between-word divergence is real signal rather than
counting noise.

### Every raw curve, and the validity audit

Transition width $w$ is our summary, but the original post defined no such summary, so the raw curves
are the primary evidence — and the only way to check the validity criteria is to look at all of them.
Figure 2 shows all 180 curves per checkpoint, grouped by divergence bin.

![Small multiples: all 180 raw d(t) curves per checkpoint, one panel per divergence quintile, trained on top and untrained below.](plots/all_curves.png)

**Figure 2.** Every raw curve in the frozen bank passes the strict validity criteria. x is the
interpolation position $t$ along the block-0 residual SLERP path (0 = endpoint A's state, 1 = endpoint
B's state); y is the relative logit distance $d(t)$. Columns are the five $JSD_A$ quintiles
(Q1 = most similar continuations); the top row is the trained 1.4B model and the bottom row the
untrained step-0 model. Thin lines are the three carrier contexts of every pair in that bin, drawn
separately with one line style per context; the thick dark line with markers is the bin's pointwise
median; dotted horizontals mark $d = 0.1$ and $d = 0.9$. Across all six checkpoints and 1,080 curves,
**zero** failed the span, single-crossing or monotonicity criteria, and the largest backslide anywhere
was $0.0000$ — the curves are not merely "monotone enough", they are strictly monotone. The untrained
network (bottom) is a straight line in every bin; the trained one (top) bends into an S, more so in
the higher-divergence bins.

Individual pairs are noisy, so the effect should be read as distributional, not pair-by-pair. Figure 3
makes that concrete with the extremes of the divergence range.

![Raw curves for the three lowest- and three highest-divergence pairs, all carrier contexts drawn separately.](plots/reference_curves.png)

**Figure 3.** The trend does not hold pair by pair. x is $t$, y is $d(t)$, as in Figure 2. Solid lines
with round/square/triangle markers are the three **lowest**-$JSD_B$ pairs (`of`/`in`, `on`/`with`,
`never`/`always`; $JSD_B$ = 0.14–0.27); dashed lines are the three **highest** (`out`/`your`,
`un`/`better`, `extremely`/`happening`; $JSD_B$ = 0.85–0.94). All three carrier contexts of each pair
are drawn separately, with no averaging. The two function-word pairs at the bottom of the divergence
range are indeed the widest curves here, but `never`/`always` — also low divergence — is among the
sharpest. Note too that even the sharpest curve is far from a step function: these are moderate
plateaus, not hard switches.

### Primary result: corpus divergence predicts narrower transitions, but only after training

Figure 4 is the main test. It plots the predictor against the outcome for the trained 1.4B model, the
untrained step-0 baseline, and the 410M scale check.

![Three scatter panels of transition width against corpus divergence: trained 1.4B, untrained step 0, and 410M.](plots/jsd_vs_width.png)

**Figure 4.** Corpus continuation divergence predicts transition width in trained models and not in an
untrained one. In every panel x is $JSD_B$ (bits) and y is the transition width $w$ (**smaller =
sharper**); each point is one of the 60 endpoint-disjoint pairs, with marker shape and hue giving its
$JSD_A$ quintile (Q1 = most similar continuations, Q5 = most divergent), and the dashed line with `x`
markers traces the running median of $w$ in five equal-count $JSD_B$ bins. **The three panels have very
different y-ranges** — the trained 1.4B spans 0.40–0.80, while the untrained step-0 panel spans only
0.820–0.840. *Left (trained 1.4B):* $\rho = -0.525$, CI $[-0.701, -0.304]$. *Middle (step 0):*
$\rho = -0.056$, CI $[-0.314, +0.211]$ — consistent with zero, but note the whole panel is squeezed
into a 2%-wide band around the linear-response value 0.83, so the untrained network has very little
variation for *any* predictor to explain. *Right (410M):* $\rho = -0.512$, CI $[-0.711, -0.272]$.

The bin view shows how much of the trend survives aggregation, and how far it is from a clean monotone
staircase.

![Box plots of transition width by divergence quintile for the three checkpoints.](plots/width_by_jsd_bin.png)

**Figure 5.** Lower width in the higher-divergence bins — monotonically at 410M, noisily at 1.4B, not
at all at step 0. x is the $JSD_A$ quintile of the frozen bank (Q1 = most similar continuations); y is
the transition width $w$. Three box-and-scatter groups sit side by side at each quintile, distinguished
by hatch and marker: `//` with round markers = trained 1.4B, `\\` with square markers = step-0 1.4B,
`..` with triangular markers = 410M. Boxes show the interquartile range with the median as a horizontal
bar; individual pairs are overplotted. 410M medians run 0.723 → 0.683 → 0.610 → 0.582 → 0.578 across
Q1→Q5, a clean monotone fall; trained 1.4B runs 0.619 → 0.608 → 0.462 → 0.502 → 0.479, where Q3 dips
below Q4 and Q5, so at n ≈ 12 pairs per bin the bin-level trend is real but noisy; step 0 runs
0.831 → 0.832 → 0.833 → 0.830 → 0.828, a total spread of 0.005. **All 60 pairs were valid in every bin
at every checkpoint**, so no bin's result comes from selective exclusion.

### Are these really plateaus?

A smaller $w$ could mean a genuine plateau (flat, flat, jump) or merely a steeper straight line. Since
"plateau" is the concept under test, we measure endpoint flatness separately with the edge-drift
metric $E$ and compare it to the no-plateau reference $E = 0.184$.

![Left: histogram of edge drift for the three checkpoints against the no-plateau reference. Right: edge drift against transition width.](plots/edge_drift.png)

**Figure 6.** The trained curves are genuinely plateau-shaped, but flatness and width are redundant.
*Left:* x is edge drift $E$ (mean movement of $d$ away from its endpoint value inside the outer 20% of
the path; 0 = perfectly flat ends), y is the number of pairs. `//`-hatched = trained 1.4B (median
0.076), `\\`-hatched = untrained step 0 (0.213), `..`-hatched = 410M (0.109); the dashed vertical is
the no-plateau reference $E = 0.184$ for a straight line. Every trained pair sits well below the
reference — the ends really are flat — while the untrained ones sit slightly *above* it. *Right:* x is
$w$, y is $E$; round markers = trained 1.4B, square markers = step 0; the dashed horizontal is again
the reference. Spearman$(w, E) = +0.971$: at the pair level the two metrics carry the same information,
which is why we report the association in terms of width and do not claim a separate result about
flatness.

### The predictor tracks something the model actually learned

A global next-token distribution could have been too coarse to matter in one specific sentence.
Figure 7 shows it is not: corpus divergence strongly predicts how differently the trained model itself
continues the two endpoints in the carrier context. This is the strongest association in the report.

![Scatter of model output divergence against corpus divergence for 60 pairs.](plots/output_jsd_validation.png)

**Figure 7.** Corpus divergence predicts a distinction the trained model demonstrably encodes. x is
$JSD_B$ (bits, corpus); y is $JSD_{\mathrm{out}}$ (bits), the JSD between the two endpoint next-token
distributions the 1.4B model outputs in the carrier context; marker shapes and hues give the $JSD_A$
quintile as in Figure 4. $\rho = +0.751$, CI $[+0.615, +0.843]$, $p = 4.9\times10^{-12}$. At step 0 the
same correlation is $+0.145$ (CI $[-0.122, +0.394]$, $p = 0.27$), as expected for an untrained readout.
This rules out the prespecified "global next-token distribution is too coarse" verdict, under which a
width null would have been uninterpretable.

### How much of the width association is independent of that learned separation? Not a significant amount

Figure 7 gives the natural causal story a name: corpus divergence → learned output separation →
narrow transition. Two analyses test it, both in Figure 8. First, because the same frozen bank runs at
step 0, we can ask what training did to *each pair*, using the learned-sharpening outcome
$\Delta w = w_{\text{trained}} - w_{\text{step }0}$ instead of the trained width. Second, we adjust the
association for the candidate mediator $JSD_{\mathrm{out}}$, alone and together with the five
covariates.

![Left: scatter of the training-induced width change against corpus divergence. Right: forest plot of the association before and after adjustment.](plots/mediation.png)

**Figure 8.** Corpus divergence predicts how much training sharpened each pair, but almost nothing
survives adjustment for the model's own output separation. *Left:* x is $JSD_B$ (bits); y is the
learned sharpening $\Delta w = w_{\text{trained}} - w_{\text{step }0}$, where more negative means
training narrowed that pair's transition more; marker shape and hue give the $JSD_A$ quintile as in
Figure 4; the dotted horizontal is "training changed nothing". Every pair lies below it — training
narrowed all 60 — and the higher-divergence pairs narrow most: $\rho = -0.517$, CI $[-0.694, -0.294]$,
$p = 2.3\times10^{-5}$, median $\Delta w = -0.287$. *Right:* a forest plot of four analyses (listed top
to bottom on the y-axis); x is the Spearman $\rho$ with $JSD_B$ and the bars are 95% bootstrap CIs;
a filled marker means $p < 0.05$, an open marker $p > 0.05$; the dotted vertical is zero. The total
association is $-0.525$; the learned-sharpening version is essentially the same, $-0.517$, so the
result is not an artefact of pairs that started out sharp. Adjusting for the mediator
$JSD_{\mathrm{out}}$ halves it to $-0.277$ ($p = 0.032$), and adjusting for the mediator plus the five
covariates leaves $-0.204$ with $p = 0.119$ — **not significant**. Using $\Delta w$ in the adjusted
rows gives the same picture ($-0.263$, $p = 0.042$; $-0.198$, $p = 0.129$).

The honest reading: **the headline $-0.525$ is a total association**, and it is a real, strongly
significant one. But once you know how far apart the model puts the two words' output distributions,
corpus divergence tells you little more about transition width that is statistically distinguishable
from noise at n = 60. That is exactly what the mediation story predicts — and also what a plain
common-cause story predicts, which is why we cannot decide between them here.

### When during training does the relationship appear? By step 1000, and then it stops changing

Figure 4 shows the relationship needs training, but not *how much*. The plan predicted it would grow
stronger as training proceeded. Running the same frozen bank at four intermediate checkpoints shows
otherwise, and Figure 9 separates two things that move differently: how well corpus divergence
*predicts* width, and how sharp the transitions actually *are*.

![Left: Spearman correlations against training step. Middle: median transition width against training step. Right: per-pair width at step 64000 against the final checkpoint.](plots/formation.png)

**Figure 9.** The corpus predictor is at full strength by step 1000; transitions sharpen through step
64000 and then partly reverse. The left and middle panels share an x-axis: training step on a log
scale, with step 0 drawn at the left edge (it cannot sit on a log axis) and ticks at 0, 1k, 8k, 32k,
64k, 143k. *Left:* y is the Spearman $\rho$ of corpus $JSD_B$ with each outcome. The solid line with
round markers is $\rho$ with transition width $w$ (shaded `//`-hatched band = 95% bootstrap CI); the
dashed line with square markers is $\rho$ with the model's own output divergence $JSD_{\mathrm{out}}$;
the dotted horizontal is zero. Both jump from ≈ 0 at step 0 to full magnitude by step 1000 ($-0.582$
and $+0.791$); afterwards the width correlation moves between $-0.408$ and $-0.628$ with heavily
overlapping CIs — no reliable trend either way — and the output correlation is flat at about $+0.75$.
*Middle:* y is median $w$ across the 60 pairs, with a `//`-hatched band spanning median ± IQR/2; the
dashed horizontal marks the linear-response value $w = 0.8$. Median $w$ falls 0.831 → 0.753 → 0.601 →
0.555 → 0.512 and then **rises** to 0.541. *Right:* the per-pair check on that rebound. x is $w$ at
step 64000, y is $w$ at step 143000, one point per pair; triangles are the 38 pairs that end blunter,
circles the 22 that end sharper; the dashed line is $y = x$ (no change). The reversal is systematic,
not a median artefact: two-sided paired Wilcoxon $p = 0.0052$, median per-pair $\Delta w = +0.012$.

Read together: **transitions get sharper over the first tens of thousands of steps, sharpest at step
64000, and then blunt slightly by the end of training — while the corpus statistic's ability to say
*which* pairs are sharp is established almost immediately and never improves.** The plan's expected
pattern — a relationship that strengthens with training — is not what happens. The late reversal is
small (about 6% of the 64k median) but consistent across pairs; we have no mechanism for it, and with
one trajectory, one bank, one model, and no resolution below step 1000 or between 64k and 143k, both
the reversal and the flat correlation are suggestive observations rather than established
training-dynamics results.

### The result does not depend on how the bank was filtered

The bank above uses the prespecified top-256 endpoint filter, which caps it at 60 pairs. Two
alternative banks test whether that choice matters: a larger 75-pair bank built by relaxing the filter
to top-512 (post-hoc, so it cannot carry the headline), and the top-256 bank minus the one pair whose
endpoint `un` is a word-start fragment rather than a complete word.

![Spearman rho with 95% CIs for the top-256, top-512 and fragment-dropped banks at three checkpoints.](plots/bank_comparison.png)

**Figure 10.** All three versions of the bank give the same conclusion. x is the checkpoint; y is
Spearman $\rho(JSD_B, w)$ with 95% bootstrap CI bars; round markers = prespecified top-256 bank
(n = 60), square markers = post-hoc top-512 bank (n = 75), triangular markers = top-256 without the
`un`/`better` fragment pair (n = 59); the dotted horizontal is zero. Trained 1.4B: $-0.525$ /
$-0.419$ / $-0.502$ (the last with $p = 5.2\times10^{-5}$). Step 0: $-0.056$ / $-0.155$ / $-0.019$, all
consistent with zero. 410M: $-0.512$ / $-0.320$ / $-0.491$. The CIs overlap heavily everywhere, so the
differences between the banks are not themselves findings — the point is that neither the filter
threshold nor the single word-fragment endpoint drives the result.

### The sharpness is produced downstream of the patch

Finally, a control on the assay itself: if the sharp transition were an artefact of readout geometry
rather than of computation, moving the patch later would not matter.

![Transition width against patched block index for low- and high-divergence pairs.](plots/block_scan.png)

**Figure 11.** Sharpness requires the blocks that follow the patch. x is the patched block index $L$
(the residual stream is interpolated after this block; 23 is the last of the 24 blocks, so almost no
computation remains); y is the transition width $w$. The solid line with round markers is the median
over the 5 **lowest**-$JSD_B$ pairs, the dashed line with square markers the median over the 5
**highest**; faint lines are individual pairs. Median $w$ rises monotonically 0.599 → 0.661 → 0.741 →
0.805 → 0.804 as the patch moves from block 0 to block 23, converging on the linear-response value of
about 0.8.

### Current-best numbers

| Result | Trained 1.4B (step143000) | Untrained 1.4B (step0) | 410M (step143000) |
|---|---|---|---|
| Spearman $\rho$ of $JSD_B$ with $w$ | **−0.525** [−0.701, −0.304], $p=1.7\times10^{-5}$ | −0.056 [−0.314, +0.211], $p=0.67$ | −0.512 [−0.711, −0.272], $p=2.9\times10^{-5}$ |
| Partial $\rho$ (5 covariates adjusted) | −0.384 | −0.142 | −0.396 |
| Spearman $\rho$ of $JSD_B$ with $JSD_{\mathrm{out}}$ | **+0.751** [+0.615, +0.843] | +0.145 [−0.122, +0.394] | +0.749 [+0.611, +0.838] |
| Median $w$ (IQR) | 0.541 (0.169) | 0.831 (0.006) | 0.640 (0.133) |
| Median edge drift $E$ (no-plateau reference 0.184) | 0.076 | 0.213 | 0.109 |
| Valid-curve rate (strict criteria) | 1.000 | 1.000 | 1.000 |
| Max endpoint-patch relative error | $4.6\times10^{-5}$ | $3.3\times10^{-6}$ | $6.3\times10^{-5}$ |

Learned sharpening and the adjustment ladder, trained 1.4B against its own step-0 baseline (n = 60):

| Association with corpus $JSD_B$ | $\rho$ | 95% CI | $p$ |
|---|---|---|---|
| Trained width $w$ — **headline, total association** | **−0.525** | [−0.701, −0.304] | $1.7\times10^{-5}$ |
| Learned sharpening $\Delta w = w_{\text{trained}} - w_{\text{step }0}$ | **−0.517** | [−0.694, −0.294] | $2.3\times10^{-5}$ |
| $w$, adjusted for the mediator $JSD_{\mathrm{out}}$ | −0.277 | [−0.509, −0.002] | 0.032 |
| $w$, adjusted for $JSD_{\mathrm{out}}$ + the 5 covariates | −0.204 | [−0.471, +0.080] | **0.119 (n.s.)** |
| $w$, adjusted for the 5 covariates only | −0.384 | [−0.623, −0.110] | 0.0024 |
| $\Delta w$, adjusted for $JSD_{\mathrm{out}}$ / for $JSD_{\mathrm{out}}$ + 5 covariates | −0.263 / −0.198 | — | 0.042 / 0.129 |

Median $\Delta w = -0.287$, and all 60 pairs have $\Delta w < 0$: training narrowed every pair's
transition, by about 0.29 of the interpolation path at the median.

Formation subset, same frozen bank on `pythia-1.4b-deduped` at six checkpoints:

| Training step | 0 | 1000 | 8000 | 32000 | 64000 | 143000 |
|---|---|---|---|---|---|---|
| Spearman $\rho$ of $JSD_B$ with $w$ | −0.056 | **−0.582** | −0.456 | −0.408 | −0.628 | −0.525 |
| 95% CI | [−0.31, +0.21] | [−0.77, −0.36] | [−0.66, −0.21] | [−0.62, −0.16] | [−0.77, −0.44] | [−0.70, −0.31] |
| Spearman $\rho$ of $JSD_B$ with $JSD_{\mathrm{out}}$ | +0.145 | +0.791 | +0.721 | +0.766 | +0.750 | +0.751 |
| Median $w$ (IQR) | 0.831 (0.006) | 0.753 (0.107) | 0.601 (0.150) | 0.555 (0.131) | **0.512** (0.150) | 0.541 (0.169) |
| Median edge drift $E$ | 0.213 | 0.153 | 0.088 | 0.077 | 0.069 | 0.076 |
| Valid-curve rate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Supporting checks (trained 1.4B): reliability Spearman 0.9998; noise ratio 0.072; per-context $\rho$ =
−0.486, −0.411 and −0.504 for `The thing was`, `They said it was` and `I thought it was` respectively,
so no single context drives the result; reversal changes $w$ by at most $1.1\times10^{-5}$; prefix
block-0 residuals within a pair differ by exactly 0.0; zero curves out of 1,080 failed any validity
criterion, with a largest backslide of $0.0000$; dropping the one word-fragment pair (`un`/`better`)
leaves $\rho = -0.502$ ($p = 5.2\times10^{-5}$, n = 59); the 64k → 143k late reversal has 38 of 60
pairs blunter, paired Wilcoxon $p = 0.0052$, median $\Delta w = +0.012$.

---

## Conclusion

Corpus continuation divergence is a usable, purely data-side predictor of how sharply a trained Pythia
model separates two words. The evidence supports two claims, in this order of strength. **First, it
predicts learned output separation:** the JSD between the two endpoints' next-token distributions in
the carrier context tracks corpus divergence at $\rho = +0.751$ ($p = 4.9\times10^{-12}$). **Second,
it predicts the overall transition width** of the interpolation curve at $\rho = -0.525$ at 1.4B and
$-0.512$ at 410M, consistently across all three carrier contexts, and not in the untrained network.
The trained curves are plateau-shaped in absolute terms (edge drift 0.076 against 0.184 for a straight
line), so "plateau" is a fair description of what we are measuring — but width and flatness are
$+0.971$ correlated across pairs, so this design cannot attribute the association to plateau flatness
specifically rather than to overall transition width. The honest headline is the width claim, and it
is a **total** association: corpus divergence also predicts how much training narrowed each pair's
transition ($\rho = -0.517$ on $\Delta w$), but once the model's own output divergence is adjusted for,
what remains ($-0.277$, and $-0.204$ with the five covariates as well, $p = 0.119$) is no longer
statistically significant at n = 60.

**The training trajectory did not go as predicted.** The plan expected the negative relationship to
strengthen during training. Instead it is essentially fully formed by step 1000 — the earliest
checkpoint we ran — and afterwards fluctuates within overlapping confidence intervals. Meanwhile the
transitions themselves sharpen through step 64000 (median $w$ 0.831 → 0.512) and then partly reverse,
ending at 0.541 with 38 of 60 pairs blunter than they were at 64k (paired Wilcoxon $p = 0.0052$).
Sharpening continues long after the corpus statistic has stopped explaining more of it, and then
undoes a little of itself. A natural next test is whether a *context-conditioned* divergence estimate
does better at the late checkpoints, where the global one has plateaued.

**What this does not show.** (1) *Not causation.* This is an observational predictor test on 60 pairs;
we did not intervene on divergence. (2) *Not an independent effect.* Adjusting for endpoint frequency,
continuation entropy, surprisal, and block-0 cosine/distance cuts the association from $-0.525$ to
$-0.384$; adjusting for the model's own output divergence cuts it to $-0.277$; adjusting for both
leaves $-0.204$, $p = 0.119$ — not significant. So we can say corpus divergence *predicts* width, but
not that it explains width beyond the learned separation and endpoint geometry. Because that geometry
plausibly lies *on the causal path* from training targets to transition shape, the adjusted numbers
are lower bounds rather than "true" effects — but the honest summary is that this design cannot
demonstrate an independent contribution. (3) *Not a clean untrained control.* The step-0 network's widths have an IQR of 0.006,
so its null correlation is partly a floor effect, not purely an absence of association. (4) *Not sharp
switching.* Even the sharpest pairs have $w \approx 0.40$; these are moderate plateaus, not step
functions. (5) *Not necessity.* Nothing here says plateaus are required for low training loss.

**Scope of the data and compute.** 2.05B tokens (2 × 500,000 rows) byte-range-sampled from a 300B-token
released stream — about 0.68% of it — so $p_a(y)$ is a well-estimated but *global, context-free*
statistic; a context-conditioned estimate was explicitly out of scope. Three carrier contexts, one hook
point (post-block-0, final position), 60 pairs, 50 interpolation points, six `pythia-1.4b-deduped`
checkpoints plus one 410M checkpoint, on one shared RTX PRO 4500 GPU. The 410M result is a scale
check, not an independent replication — it uses the same frozen bank and the same corpus estimate, as
does every formation checkpoint.

**Reproduction.** `experiments/download_splits.py` (byte-range corpus sample), `count_jsd.py` (bigram
counts and reliability gates), `select_endpoints.py` and `build_pairs.py --pool strict` (frozen bank),
`assay.py` with `run_assay.py` (plateau assay), `curve_metrics.py` with `rescore.py` (validity criteria
and raw-curve export), `block_scan.py`, `checks.py`, `formation.py` (intermediate checkpoints),
`revisions.py` (learned sharpening, adjustment ladder, late-reversal test, fragment sensitivity), and
`analyze.py` (figures and statistics). Manifests, per-pair summaries and all summary statistics are in
`results/`. **The raw curves are committed**, as `results/curves_*.npy` and as a plain-text
`results/curves_*.csv.gz` export — one row per pair × context × grid point — so every width, flatness
and validity number in this report can be recomputed from disk without a GPU. (The repo-wide
`.gitignore` excludes `*.npy` and `*.gz`; this direction ships its own `.gitignore` that un-ignores
`results/curves_*`, about 1.6 MB in total.)
