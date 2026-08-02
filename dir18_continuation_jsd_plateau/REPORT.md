# Does training-corpus continuation divergence predict activation-plateau sharpness?

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
higher-divergence word pairs produce sharper plateau transitions inside the trained model.

**They do.** On `pythia-1.4b-deduped` at its final checkpoint, corpus continuation divergence
predicts sharper transitions with Spearman $\rho = -0.419$ (95% CI $[-0.585, -0.222]$,
$p = 1.8\times10^{-4}$, n = 75 endpoint-disjoint pairs; negative means *higher divergence → narrower
transition → sharper plateau*). The same bank at **step 0**, before any training, shows no
relationship ($\rho = -0.155$, CI $[-0.368, +0.068]$) and essentially no plateau structure at all
(median width 0.831 with an interquartile range of 0.004 — the transition is nearly the trivial
linear one). `pythia-410m-deduped` replicates the effect ($\rho = -0.320$, $p = 5.1\times10^{-3}$).
Corpus divergence also strongly predicts a distinction the model demonstrably learned — its own
output divergence in the same context ($\rho = +0.729$, $p = 1.2\times10^{-13}$).

This matches the prespecified verdict **"predictive divergence is associated with learned plateau
sharpening."** Two qualifications matter. First, after adjusting for endpoint frequency, continuation
entropy, surprisal, and the block-0 geometry of the two endpoint states, the association weakens from
$-0.419$ to $-0.267$; we therefore report the *total* association and explicitly do **not** claim that
corpus divergence explains sharpness beyond learned endpoint geometry. Second, four intermediate
checkpoints **contradict the expected pattern**: the plan predicted the negative relationship would
*strengthen* during training, but it is already at its strongest by step 1000 ($\rho = -0.660$) and
then *weakens* to $-0.419$ by step 143000 — even while the plateaus themselves keep sharpening
throughout (median $w$ falls monotonically from 0.831 to 0.562). This is an observational predictor
test: it does not show that divergence *causes* plateaus.

---

## Methods

### Data & Model

**Model.** `EleutherAI/pythia-1.4b-deduped` (1.4B parameters, 24 transformer blocks, residual width
2048), at revision `step143000` (the final checkpoint) and revision `step0` (the untrained
initialisation). Formation subset: the same model at revisions `step1000`, `step8000`, `step32000`
and `step64000`. Scale check: `EleutherAI/pythia-410m-deduped` at `step143000`. Native Hugging Face
GPT-NeoX modules, `eval()` mode, `torch.inference_mode()`, float32. Every checkpoint is run on the
**same frozen 75-pair bank** with the same corpus estimate.

**Hook point.** The residual stream at the **final token position, immediately after transformer
block 0**. This is the single site we interpolate and patch; blocks 1–23 then run normally and we
read the final-position logits after the final LayerNorm and unembedding. A control (Figure 6)
repeats the assay patching after blocks 0, 6, 12, 18 and 23 instead (Figure 6 is a training-time
result, not a hook-point change).

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

**Sample sizes.** 10,000 word pairs for the reliability bank; 75 endpoint-disjoint pairs (15 per
divergence quintile) in the frozen assay bank; 3 carrier contexts per pair; 50 interpolation points
per curve; 50,060 valid target token IDs.

### Metrics

Everything below is motivated by one chain of questions: *(i) is our corpus estimate stable enough to
be a predictor at all? (ii) how sharp is the model's transition? (iii) does corpus divergence predict
that sharpness? (iv) is the prediction about something the model actually learned, or about
architecture and token geometry?*

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
Consumed by Figures 1, 3, 4 and 5.

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

linearly interpolated on the 50-point grid. **Smaller $w$ means a sharper transition, i.e. a stronger
plateau.** A perfectly linear output response gives $w \approx 0.8$; a step function gives $w$ near 0.
A curve that never rises monotonically through both levels is recorded as **invalid** and excluded
from the correlation rather than forced into it; invalid rates are reported per bin. Because $w$ is a
summary the original post did not define, the raw curves are shown as primary evidence (Figure 2).
Each pair's outcome is the **median $w$ across the three carrier contexts**. Consumed by Figures 2, 3,
4 and 6.

**(iii) Association** — reported as the Spearman rank correlation $\rho$ between $JSD_B$ and $w$, with
a 95% confidence interval from 10,000 bootstrap resamples. The bank is **endpoint-disjoint** (no token
appears in two pairs), so resampling pairs resamples endpoints as intact clusters; there is no hidden
reuse inflating the effective sample size. Consumed by Figures 3 and 4.

**(iv) Model output divergence** — the validity check on the predictor. Corpus divergence is a global,
context-free statistic, while the assay runs in one specific carrier context. If corpus divergence did
not even predict how differently the model itself continues the two endpoints *in that context*, a
plateau null would be uninterpretable. So we compute the base-2 JSD between the two endpoint
next-token distributions the model actually outputs:

```math
JSD_{\mathrm{out}} \;=\; JSD\!\big(\mathrm{softmax}(z_A),\; \mathrm{softmax}(z_B)\big)
```

and correlate it with $JSD_B$. Higher means the model draws a bigger distinction between the two
endpoints. Consumed by Figure 5.

**Sensitivity (partial Spearman)** — divergent words might simply be rarer, more surprising, or
geometrically further apart at block 0, and any of those could drive sharpness. We rank-transform
$JSD_B$, $w$ and five covariates — mean endpoint log-frequency in the corpus, mean continuation
entropy in bits, mean endpoint surprisal under the model in the carrier context, and the block-0
cosine similarity and Euclidean distance between $x_A$ and $x_B$ — regress the first two on the
covariates, and correlate the residuals. Reported alongside the unadjusted result, never in place
of it.

### Baselines and controls

**Step 0 (untrained) checkpoint** — the primary baseline. The identical frozen bank and identical
assay, run on `pythia-1.4b-deduped` revision `step0`. Any plateau structure or divergence-width
relationship surviving here is produced by architecture, tokenisation, and random initialisation
rather than by learning. This is the control that decides between "learned separation" and "geometry
confound".

**Same-token split-half divergence** — the noise baseline for the predictor, defined by the noise-ratio
equation above. It answers "how large a JSD would we see for two *identical* words, purely from finite
counts?"

**Linear-response reference** — if the model's output moved proportionally with the interpolation
position, $w$ would be $0.9 - 0.1 = 0.8$. Values near 0.8 mean "no plateau"; the step-0 median of
0.831 sits at this reference, and so does the deepest block-scan point.

**Block scan (blocks 0, 6, 12, 18, 23)** — patching later leaves fewer blocks to compute a sharp
response. If sharpness is produced by downstream computation rather than by readout geometry, $w$
should grow as the patch moves later. Run on 10 frozen pairs (the 5 lowest and 5 highest $JSD_B$).

**Assay self-tests** — patching at $t=0$ and $t=1$ must reproduce the unpatched endpoint logits (worst
case across all runs: $4.7\times10^{-5}$ relative error); swapping which endpoint is A and which is B
must leave $w$ unchanged (worst case over 20 pairs: $1.1\times10^{-5}$, against a grid spacing of
0.0204); and within a pair the two prompts must share every prefix token and every prefix block-0
residual (measured difference: exactly 0.0).

### Pair bank construction (frozen before any curve was seen)

Endpoints are lowercase alphabetic word-start tokens that decode as one complete word and are among
the trained model's top-K eligible word continuations of **all three** carrier contexts
(`The thing was`, `They said it was`, `I thought it was`), so every prompt is in-distribution. The
plan prespecified K = 256, which yields only 134 tokens — at most 67 endpoint-disjoint pairs, short of
the 75-pair target. Following the prespecified fallback order we relaxed to **K = 512** (258 tokens,
still the top 2.8% of 18,714 eligible word tokens) rather than adopt a dependent all-pairs design; 12
of the 75 final pairs have both endpoints inside the stricter top-256.

Further rules, all applied before any plateau curve was viewed: each endpoint occurs at least 20,000
times in **each** split (393 of the 527-token counting pool qualify; 222 after the top-512
restriction); the two endpoint frequencies within a pair differ by at most a factor of two; **no
endpoint token is reused anywhere in the bank**; and 15 pairs are taken in each $JSD_A$ quintile with
corpus log-frequency and model surprisal balanced across the five bins. The balance holds
(Kruskal-Wallis $p = 0.92$ for log-frequency, $p = 0.81$ for surprisal — a large $p$ means the bins are
statistically indistinguishable on that covariate). The bank was never revised afterwards; it is
stored with all token IDs, counts, and both JSD values in `results/pair_manifest.json`.

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

### Raw curves: high-divergence pairs flip more abruptly

Transition width $w$ is our summary, but the original post defined no such summary, so the raw curves
are the primary evidence. Figure 2 shows the three lowest- and three highest-divergence pairs in the
bank.

![Six d(t) curves against interpolation position t, three low-divergence and three high-divergence pairs.](plots/reference_curves.png)

**Figure 2.** Higher corpus divergence produces a visibly more abrupt output transition. x is the
interpolation position $t$ along the block-0 residual SLERP path (0 = endpoint A's state, 1 = endpoint
B's state); y is the relative logit distance $d(t)$, the fraction of the way the output has travelled
from A to B. Solid lines with round/square markers are the three **lowest**-$JSD_B$ pairs
(`making/getting`, `later/done`, `nothing/someone`; $JSD_B$ = 0.38–0.41); dashed lines with
diamond/triangle markers are the three **highest** (`un/before`, `gonna/happening`, `ra/okay`;
$JSD_B$ = 0.93–0.97). Dotted horizontals mark $d = 0.1$ and $d = 0.9$, whose horizontal separation is
$w$. The high-divergence curves stay flatter for longer and then rise more steeply. Note that even the
sharpest curve here is far from a step function — these are moderate plateaus, not hard switches.

### Primary result: corpus divergence predicts sharper transitions, but only after training

Figure 3 is the main test. It plots the predictor against the outcome for the trained 1.4B model, the
untrained step-0 baseline, and the 410M scale check.

![Three scatter panels of transition width against corpus divergence: trained 1.4B, untrained step 0, and 410M.](plots/jsd_vs_width.png)

**Figure 3.** Corpus continuation divergence predicts plateau sharpness in trained models and not in an
untrained one. In every panel x is $JSD_B$ (bits) and y is the transition width $w$ (**smaller =
sharper**); each point is one of the 75 endpoint-disjoint pairs, with marker shape and hue giving its
$JSD_A$ quintile (Q1 = most similar continuations, Q5 = most divergent), and the dashed line with `x`
markers traces the running median of $w$ in five equal-count $JSD_B$ bins. **The three panels have very
different y-ranges** — the trained 1.4B spans 0.41–0.74, while the untrained step-0 panel spans only
0.820–0.837. *Left (trained 1.4B):* $\rho = -0.419$, CI $[-0.585, -0.222]$. *Middle (step 0):*
$\rho = -0.155$, CI $[-0.368, +0.068]$ — consistent with zero, and the whole panel is squeezed into a
1.7%-wide band around the linear-response value 0.83, meaning the untrained network has essentially no
plateau structure to predict. *Right (410M):* $\rho = -0.320$, CI $[-0.526, -0.087]$.

The bin view makes the untrained baseline's flatness unmistakable and shows the trend is monotone
rather than driven by a few outliers.

![Box plots of transition width by divergence quintile for the three checkpoints.](plots/width_by_jsd_bin.png)

**Figure 4.** The sharpening trend is monotone across all five bins in both trained models and
completely absent at step 0. x is the $JSD_A$ quintile of the frozen bank (Q1 = most similar
continuations); y is the transition width $w$. Three box-and-scatter groups sit side by side at each
quintile, distinguished by hatch and marker: `//` with round markers = trained 1.4B, unhatched with
square markers = step-0 1.4B, `..` with triangular markers = 410M. Boxes show the interquartile range
with the median as a horizontal bar; individual pairs are overplotted. Trained 1.4B medians run
0.611 → 0.568 → 0.532 → 0.516 → 0.516 across Q1→Q5; 410M runs 0.699 → 0.679 → 0.648 → 0.647 → 0.629;
step 0 runs 0.832 → 0.831 → 0.831 → 0.831 → 0.829, a total spread of 0.003. **All 75 curves were valid
at every checkpoint** — the invalid-curve rate is 0.000 in every bin, so no bin's result comes from
selective exclusion.

### The predictor tracks something the model actually learned

A global next-token distribution could have been too coarse to matter in one specific sentence.
Figure 5 shows it is not: corpus divergence strongly predicts how differently the trained model itself
continues the two endpoints in the carrier context.

![Scatter of model output divergence against corpus divergence for 75 pairs.](plots/output_jsd_validation.png)

**Figure 5.** Corpus divergence predicts a distinction the trained model demonstrably encodes. x is
$JSD_B$ (bits, corpus); y is $JSD_{\mathrm{out}}$ (bits), the JSD between the two endpoint next-token
distributions the 1.4B model outputs in the carrier context; marker shapes and hues give the $JSD_A$
quintile as in Figure 3. $\rho = +0.729$, CI $[+0.599, +0.818]$, $p = 1.2\times10^{-13}$. At step 0 the
same correlation is $-0.144$ (CI $[-0.363, +0.085]$), as expected for an untrained readout. This rules
out the prespecified "global next-token distribution is too coarse" verdict, under which a plateau null
would have been uninterpretable.

### When during training does the relationship appear? Early — and then it fades

Figure 3 shows the relationship needs training, but not *how much*. The plan predicted it would grow
stronger as training proceeded. Running the same frozen bank at four intermediate checkpoints shows
the opposite, and Figure 6 separates two things that turn out to move differently: how well corpus
divergence *predicts* sharpness, and how sharp the plateaus actually *are*.

![Left: Spearman correlations against training step. Right: median transition width against training step.](plots/formation.png)

**Figure 6.** The corpus predictor is strongest early in training, while plateaus keep sharpening to
the end. Both panels share the x-axis: training step on a log scale, with step 0 drawn at the left
edge (it cannot sit on a log axis) and ticks at 0, 1k, 8k, 32k, 64k, 143k. *Left:* y is the Spearman
$\rho$ of corpus $JSD_B$ with each outcome. The solid line with round markers is $\rho$ with
transition width $w$ (shaded band = 95% bootstrap CI); the dashed line with square markers is $\rho$
with the model's own output divergence $JSD_{\mathrm{out}}$; the dotted horizontal is zero. Both jump
from ≈0 at step 0 to their full magnitude by step 1000 ($-0.660$ and $+0.779$). The sharpness
correlation then *weakens* to $-0.419$ by step 143000, while the output correlation stays flat
(+0.78 → +0.73). *Right:* y is median $w$ across the 75 pairs, with a `//`-hatched band spanning
median ± IQR/2; the dashed horizontal marks the linear-response value $w = 0.8$. Median $w$ falls
monotonically 0.831 → 0.758 → 0.624 → 0.582 → 0.541 → 0.562, crossing below the linear-response
reference between step 0 and step 1000.

Read together: **plateaus keep getting sharper throughout training, but a global corpus statistic
explains a shrinking share of *which* pairs are sharp.** A natural reading is that early training is
dominated by corpus-level continuation statistics, and later training adds context-sensitive
structure that a context-free $P(y \mid a)$ cannot capture — the predictor does not break (its link
to the model's own outputs is flat from step 1000 on), it just stops being the whole story for
sharpness. This is one trajectory on one bank, so it is a suggestive observation rather than an
established training-dynamics result.

### The sharpness is produced downstream of the patch

Finally, a control on the assay itself: if the sharp transition were an artefact of readout geometry
rather than of computation, moving the patch later would not matter.

![Transition width against patched block index for low- and high-divergence pairs.](plots/block_scan.png)

**Figure 7.** Sharpness requires the blocks that follow the patch. x is the patched block index $L$
(the residual stream is interpolated after this block; 23 is the last of the 24 blocks, so almost no
computation remains); y is the transition width $w$. The solid line with round markers is the median
over the 5 **lowest**-$JSD_B$ pairs, the dashed line with square markers the median over the 5
**highest**; faint lines are individual pairs. Median $w$ rises monotonically 0.549 → 0.646 → 0.726 →
0.796 → 0.805 as the patch moves from block 0 to block 23, converging on the linear-response value of
about 0.8. The high-divergence group stays below the low-divergence group at every depth.

### Current-best numbers

| Result | Trained 1.4B (step143000) | Untrained 1.4B (step0) | 410M (step143000) |
|---|---|---|---|
| Spearman $\rho$ of $JSD_B$ with $w$ | **−0.419** [−0.585, −0.222], $p=1.8\times10^{-4}$ | −0.155 [−0.368, +0.068], $p=0.18$ | −0.320 [−0.526, −0.087], $p=5.1\times10^{-3}$ |
| Partial $\rho$ (5 covariates adjusted) | −0.267 | −0.146 | −0.251 |
| Spearman $\rho$ of $JSD_B$ with $JSD_{\mathrm{out}}$ | **+0.729** [+0.599, +0.818] | −0.144 [−0.363, +0.085] | +0.717 [+0.584, +0.808] |
| Median $w$ (IQR) | 0.562 (0.111) | 0.831 (0.004) | 0.655 (0.075) |
| Valid-curve rate | 1.000 | 1.000 | 1.000 |
| Max endpoint-patch relative error | $4.7\times10^{-5}$ | $3.8\times10^{-6}$ | $7.8\times10^{-5}$ |

Formation subset, same frozen bank on `pythia-1.4b-deduped` at six checkpoints:

| Training step | 0 | 1000 | 8000 | 32000 | 64000 | 143000 |
|---|---|---|---|---|---|---|
| Spearman $\rho$ of $JSD_B$ with $w$ | −0.155 | **−0.660** | −0.605 | −0.524 | −0.539 | −0.419 |
| 95% CI | [−0.363, +0.068] | [−0.779, −0.496] | [−0.734, −0.433] | [−0.674, −0.323] | [−0.678, −0.355] | [−0.586, −0.219] |
| Spearman $\rho$ of $JSD_B$ with $JSD_{\mathrm{out}}$ | −0.144 | +0.779 | +0.693 | +0.726 | +0.714 | +0.729 |
| Median $w$ (IQR) | 0.831 (0.004) | 0.758 (0.087) | 0.624 (0.088) | 0.582 (0.098) | **0.541** (0.114) | 0.562 (0.111) |
| Valid-curve rate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Supporting checks (trained 1.4B): reliability Spearman 0.9998; noise ratio 0.072; per-context $\rho$ =
−0.326, −0.398 and −0.437 for `The thing was`, `They said it was` and `I thought it was` respectively,
so no single context drives the result; reversal changes $w$ by at most $1.1\times10^{-5}$; prefix
block-0 residuals within a pair differ by exactly 0.0; the 12 pairs whose endpoints both sit in the
stricter top-256 give the same point estimate, $\rho = -0.406$, but are far too few to be conclusive on
their own ($p = 0.19$).

---

## Conclusion

Corpus continuation divergence is a usable, purely data-side predictor of activation-plateau sharpness
in trained Pythia models. The effect is moderate ($\rho \approx -0.42$ at 1.4B and $-0.32$ at 410M),
monotone across all five divergence bins, present in all three carrier contexts, and absent in the
untrained network — which also has almost no plateau structure to begin with. Together with the strong
corpus-to-output-divergence link ($\rho = +0.729$), this supports the prespecified verdict that
**predictive divergence is associated with learned plateau sharpening**, not with architecture or
tokenisation geometry.

**The training trajectory did not go as predicted.** The plan expected the negative relationship to
strengthen during training. Instead it is essentially fully formed by step 1000 — the earliest
checkpoint we ran, where it is at its *strongest* ($\rho = -0.660$) — and then decays to $-0.419$ by
the final checkpoint, while median $w$ keeps falling from 0.831 to 0.562. So plateau sharpening
continues throughout training, but the share of it explained by a context-free corpus statistic
shrinks. We report this as an unexpected observation on one bank and one model, not as an established
result about training dynamics; a natural next test would be whether a *context-conditioned*
divergence estimate holds its predictive power at the late checkpoints where the global one fades.

**What this does not show.** (1) *Not causation.* This is an observational predictor test on 75 pairs;
we did not intervene on divergence. (2) *Not "beyond geometry".* Adjusting for endpoint frequency,
continuation entropy, surprisal, and block-0 cosine/distance cuts the association from −0.419 to
−0.267. The relationship survives adjustment but is substantially attenuated, so part of what corpus
divergence predicts is already carried by the learned endpoint geometry — and since that geometry
plausibly lies *on the causal path* from training targets to plateau shape, the adjusted number is a
lower bound rather than the "true" effect. We report the total association as the headline and the
adjusted one beside it. (3) *Not sharp switching.* Even the sharpest pairs have $w \approx 0.41$; these
are moderate plateaus, not step functions. (4) *Not necessity.* Nothing here says plateaus are required
for low training loss.

**Scope of the data and compute.** 2.05B tokens (2 × 500,000 rows) byte-range-sampled from a 300B-token
released stream — about 0.68% of it — so $p_a(y)$ is a well-estimated but *global, context-free*
statistic; a context-conditioned estimate was explicitly out of scope. Three carrier contexts, one hook
point (post-block-0, final position), 75 pairs, 50 interpolation points, six `pythia-1.4b-deduped`
checkpoints plus one 410M checkpoint, on one shared RTX PRO 4500 GPU. The 410M result is a scale
check, not an independent replication — it uses the same frozen bank and the same corpus estimate, as
does every formation checkpoint.

**Reproduction.** `experiments/download_splits.py` (byte-range corpus sample), `count_jsd.py` (bigram
counts and reliability gates), `select_endpoints.py` and `build_pairs.py` (frozen bank), `assay.py`
with `run_assay.py` (plateau assay), `block_scan.py`, `checks.py`, `formation.py` with
`plot_formation.py` (intermediate checkpoints), and `analyze.py` (figures and statistics). Manifests
and all summary statistics are in `results/`.
