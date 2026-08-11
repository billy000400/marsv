# What explains transition-width variation beyond successor JSD?

> Final, presentable, current-best only (no history — see CHANGELOG.md).

## Summary

A language model does not always respond smoothly to its own internal state. If you take the hidden
state the model computes for one input, slide it gradually toward the hidden state for a different
input, and read the model's output at every step, the output sometimes barely moves for most of the
path and then swings quickly. **Transition width** `w` is the fraction of the path over which that
swing happens. Narrow transitions matter for safety auditing: they are places where a small change to
an internal state produces a large behavioural change, so an auditor would like to know where they are
without running an interpolation experiment for every pair of inputs — of which there are quadratically
many.

Earlier work in this project (direction `dir18`) showed that one cheap corpus statistic —
how different two tokens' immediate-next-token distributions are, measured by Jensen-Shannon
divergence (JSD) — predicts `w` on average, with a rank correlation of $-0.486$ over 1,000 token
pairs. It also leaves a lot unexplained: pairs with nearly identical corpus JSD can have widths of
0.34 and 0.77. This report asks what the leftover variation is.

**The leftover is mostly a property of the individual tokens, not of the pairing.** Fitting
$w(u,v) \approx \mu + a_u + a_v$ — one number per token, no interaction — takes held-out $R^2$ from
0.149 (corpus JSD alone) to 0.578 (corpus JSD plus the per-token term), against a reproducibility
ceiling of 0.934 estimated from agreement across sentence frames. The per-token term alone (0.365)
beats corpus JSD (0.149) and beats the model's own endpoint output difference (0.187). So a token
carries its own width contribution into whatever pair it appears in: ` un`, ` in`, ` his`, ` my` pull
transitions narrow; ` kind`, ` real`, ` now`, ` never` push them wide.

**The per-token quantity can be measured directly, and it transfers.** The additive fit is a
description with 123 free parameters, so we tested it against the model. We measured each token's
width against **six anchor tokens that appear in none of the 1,000 pairs**, giving one number per
token from outside the pair bank. That measured number predicts the fitted effect at Spearman
$\rho = +0.70$ ($p = 5\times10^{-19}$), and at the pair level **two free parameters** on the sum of two
measured token widths reach held-out $R^2 = 0.350$ — matching the 123-parameter fit (0.365) and more
than doubling corpus JSD (0.149).

**As a screen, it works on tokens the fit never saw.** We took 40 tokens appearing nowhere in the
1,000-pair bank, measured only their anchor widths, and predicted the width of all 780 pairs among them
using a slope and intercept taken from the bank — no parameter fitted on the new tokens. On the 718
pairs that pass the gate, the prediction lands at $R^2 = 0.397$, Spearman $\rho = +0.66$, mean absolute
error 0.047 width units, and it beats the model's own endpoint output difference on the same pairs
($\rho = -0.51$) while needing 40 per-token measurements instead of one per pair. Sorting the new pairs
by the screen's prediction separates median widths of 0.50 / 0.57 / 0.62 across terciles — before any
of those pairs is run.

**The number the screen needs is largely readable from the token's static embedding.** A ridge probe
on the 2048-dimensional embedding row predicts a held-out token's measured width at $\rho = +0.76$
($R^2 = 0.51$ over 50 random 80/43 splits), well above the embedding-norm baseline ($\rho = +0.60$) and
a shuffled-target control ($\rho = -0.20$). Used forward, with the probe fitted only on the original
123 tokens, it predicts the widths of the 718 unseen pairs at $R^2 = 0.213$ and $\rho = +0.53$ without
running the model at all. That is weaker than measuring the widths (0.397), because ridge shrinkage
compresses the predicted range, but it makes the screen a table over the vocabulary rather than a
per-token experiment, and it still ranks unseen pairs about as well as the model's own endpoint output
difference, which needs both endpoints of every pair to be run.

**The ranking is a token property; the level belongs to the context.** Re-measuring the same 123
tokens in four differently shaped contexts — a mid-sentence continuation, an interrogative, a
colon-list and a code prefix — keeps the token ranking at $\rho = +0.844$, $+0.770$, $+0.735$ and
$+0.501$ against the original. The first of those matches the $+0.82$ that two of the original frames
achieve with each other, so in a nearby context the measurement transfers essentially without loss, and
even the code prefix stays above the agreement between two disjoint anchor sets. The absolute widths do
move: the median runs from 0.530 in the list context to 0.705 in code. A vocabulary table therefore
ranks tokens usefully in an unfamiliar context, while a width threshold calibrated in one context
should not be carried into another.

Three results bound the claim. First, the measurement is not anchor-free: two disjoint anchor sets —
six function words and six rare content words — rank the 123 tokens at only $\rho = +0.46$, though each
recovers the fitted token effect ($\rho = +0.57$ and $+0.61$, against $+0.70$ for the mixed set used
throughout). A common per-token component is there whichever anchors are used, with an anchor-specific
component on top, so the anchor set belongs to the method rather than being a free choice. Second, the
per-token contribution is **not** a simple corpus statistic: it correlates only weakly with corpus
frequency ($\rho = -0.33$), continuation entropy ($-0.24$), and how surprising the token is to the model
in context ($+0.26$), so it has to be read from the model rather than looked up in a count table.
Third, genuine pair-specific structure survives the additive fit: residuals from the additive model
still agree at $r = 0.67$ across independent sentence frames, and block-0 endpoint geometry recovers
part of it (held-out $R^2$ 0.723). Additivity is the dominant term, not the whole story.

Moving the interpolation site through the network shows where the effect comes from. The ranking of
tokens is stable — anchor widths measured at block 18 still agree with the block-0 ranking at
$\rho = +0.72$ — but the transitions themselves flatten out: the median anchor width rises from 0.553
at block 0 to 0.800 at block 18, exactly the value of a perfectly proportional response, and the spread
across tokens shrinks fivefold. Which token is narrow is settled early; how sharp any transition gets
depends on how much of the network still lies below the site. Pushing the site the other way, below
block 0 to the input embedding, leaves the ranking essentially intact ($\rho = +0.79$), which is what
makes the embedding lookup possible.

What produces the trait is still open, and the attempt to find a handle on it produced the one causal
result in this report. A per-token basin of output insensitivity, the mechanism we set out to test,
does not explain the trait. Nor does the probe's own embedding direction: editing a token's embedding
along it, with the step grown until the model's next-token distribution moves 0.05–0.2 bits, does move
width — by 0.10 to 0.15 width units — but a random direction matched on that same output movement moves
it just as much, and all 144 edits widen even though the probe predicts opposite signs for opposite
steps. Instead of sliding tokens along a width axis, every edit lands them near a common width of 0.68,
with the spread across tokens collapsing from 0.083 to 0.02 and the narrowest tokens moving furthest.
A displacement ladder then says what is doing the destroying, and the answer is behaviour rather than
geometry: at a fixed displacement of norm 1.8, an edit chosen to move the token's output as little as
possible (0.049 bits) leaves the ordering of the twelve tokens intact ($\rho = +0.94$), while an edit
of the same size chosen to move it as much as possible (0.402 bits) leaves nothing of it
($\rho = +0.08$). Narrow transitions are a fragile property, but the property is behavioural — which is
what makes the vocabulary-wide lookup a reading of what a token does to the model rather than of where
its embedding row happens to sit.

Moving the intervention out of embedding space narrows the mechanism to one component. Mean-ablating
each of the 102 attention heads and MLPs in blocks 0–5 in turn leaves the token ordering untouched in
101 cases (median $\rho = +0.99$); only the block-0 MLP collapses the spread across tokens (sd
$0.084 \to 0.018$) and erases the ordering ($\rho = -0.10$). That component is also the only one whose
removal the model feels (0.451 bits of output movement, against $\le 0.007$ for every other), so the
localisation arrived with a confound attached. A dose–response settles it: softening the ablation and
matching every dose against a random perturbation of the same residual stream *at the same output
movement in bits*, the MLP arm loses the ordering at roughly 3.5× less output movement than the control
(at 0.014 bits, $\rho = +0.64$ against $+0.91$), while the collapse of the across-token spread is
identical in the two arms. Disturbance of any kind flattens the level; the block-0 MLP specifically
carries the ordering.

We also ruled out the most deflationary explanation. Because `w` is a *fraction* of the path, a
transition of fixed absolute size would look narrower on a longer path. If that were the mechanism,
converting `w` into residual-stream distance units would make it more homogeneous. It does the
opposite: the coefficient of variation rises from 0.158 to 0.216.

---

## Methods

### Data & model

**Model and hook point.** `EleutherAI/pythia-1.4b-deduped` (1.4B parameters, 24 blocks, residual width
2048) at revision `step143000`, the final checkpoint. All quantities are read at the **final token
position of the residual stream immediately after transformer block 0**; blocks 1–23 then run
normally and the final-position logits are read after the final LayerNorm and unembedding.

**Pairs and frames.** The 1,000 token pairs built in `dir18` from 123 eligible endpoint tokens, each
run in three fixed sentence frames — `The thing was`, `They said it was`, `I thought it was` — with the
pair's token as the final token, at 50 interpolation positions per curve. A pair's `w` is the median
over its three frames. Everything about those 1,000 pairs is re-analysed from the stored artifacts.
The probes and the forward screen described below add new inference on the same model and hook point:
the basin sweep (123 tokens × 3 frames × 12 directions × 26 steps), the anchor widths (123 tokens ×
3 frames × 6 anchors × 50 steps), and the forward screen (40 unseen tokens, their anchor widths plus
all 780 pairs among them × 3 frames × 50 steps) and the anchor-set swap (123 tokens × 3 frames ×
12 further anchors × 50 steps) and the layer sweep (the anchor-width run repeated at blocks 6, 12 and
18, and once more at the input embedding) and the vocabulary-wide test (32 tokens from outside the pool
× 3 frames × 6 anchors × 50 steps), the frame-shape control (123 tokens × 4 new contexts × 6 anchors ×
50 steps) and the two embedding interventions (16 tokens × 9 edits and 12 tokens × 13 edits, each edit
re-measured over 3 frames × 6 anchors × 50 steps) and the displacement-norm ladder (12 tokens × 4 rungs
× 24 probed directions plus 3 re-measurements each) and the two mode-split experiments (12 tokens ×
24 probed directions plus 2 calibrated re-measurements each, twice) — about 1.6 million forward passes
in total, roughly four and a half hours on one GPU shared with three other jobs.

**Corpus.** `EleutherAI/pile-deduped-pythia-preshuffled`, the tokenised stream Pythia was trained on.
Corpus quantities come from a 500,000-row sample (1.02 billion tokens) starting at global row
73,300,000, counting only adjacent token transitions inside a row.

### Inherited quantities

Three quantities come from `dir18` unchanged and are defined here so this report stands alone.

**Corpus successor JSD** `J(u,v)` — the predictor whose leftovers we are studying. For token $u$,
$P_u(y)$ is the empirical distribution of the token that immediately follows $u$ in the corpus sample.
`J` is the base-2 Jensen-Shannon divergence between the two tokens' successor distributions, in bits,
running from 0 (identical continuation habits) to 1 (disjoint ones):

```math
J(u,v) \;=\; \tfrac{1}{2} D_{\mathrm{KL}}\!\left(P_u \,\Vert\, m\right)
           + \tfrac{1}{2} D_{\mathrm{KL}}\!\left(P_v \,\Vert\, m\right),
\qquad m = \tfrac{1}{2}\left(P_u + P_v\right).
```

**The interpolation and the output-distance score** `d(t)` — how the outcome is measured. With $x_u$
and $x_v$ the two endpoint residual states, $\hat e$ the corresponding unit vectors and $\Omega$ the
angle between them, the path is norm-rescaled spherical interpolation, which keeps the interpolated
state at realistic residual norms:

```math
x(t) \;=\; \big[(1-t)\lVert x_u\rVert + t\lVert x_v\rVert\big]\cdot
           \frac{\sin\!\big((1-t)\Omega\big)\,\hat e_u + \sin\!\big(t\Omega\big)\,\hat e_v}{\sin \Omega}.
```

Writing $z(t)$ for the final-position logits produced by patching in $x(t)$, the output-distance score
says how far along the segment from $z_u$ to $z_v$ the output currently sits, so $d(0)=0$ and
$d(1)=1$ by construction:

```math
d(t) \;=\; \frac{\lVert z(t) - z_u \rVert_2}{\lVert z(t) - z_u \rVert_2 + \lVert z(t) - z_v \rVert_2}.
```

**Transition width** `w` — the outcome variable, the fraction of the path needed for the output to
cross from one endpoint's neighbourhood to the other's, linearly interpolated on the 50-point grid:

```math
w \;=\; t(d = 0.9) \;-\; t(d = 0.1).
```

**Smaller `w` = narrower transition.** A model whose output moved in exact proportion to $t$ gives
$w = 0.8$; a step function gives $w$ near 0. Observed values run from 0.32 to 0.80.

### Is `w` informative for this pair? The endpoint-movement gate

`d(t)` is normalised to run from 0 to 1 however little the output actually changes, so for a pair
whose two outputs are nearly identical `w` measures the shape of a movement that barely happened.
To separate those cases we use the **endpoint output movement**: the base-2 JSD between the two
tokens' output distributions in a given frame $c$, in bits, and we take the **minimum over the three
frames** so that a pair is kept only if the model separates the two tokens in every frame it is
measured in:

```math
JSD_{\mathrm{out}}^{\min}(u,v) \;=\; \min_{c}\;
JSD\Big(\mathrm{softmax}\big(z^{(c)}_u\big),\; \mathrm{softmax}\big(z^{(c)}_v\big)\Big).
```

The gate is $JSD_{\mathrm{out}}^{\min} \ge 0.2$ bits, which keeps 929 of the 1,000 pairs. Figure 1
shows what the gate removes; every later analysis runs on the 929 gated pairs.

### How much of `w` is explainable at all? The reproducibility ceiling

No model can predict the part of `w` that is frame-to-frame noise, so we need a ceiling to compare
against. Each pair is measured in three sentence frames; treating the frames as parallel measurements,
$\bar r$ is the mean correlation between frames across pairs, and the Spearman-Brown formula gives the
reliability of a median over three frames:

```math
R^2_{\mathrm{ceiling}} \;=\; \frac{3\,\bar r}{1 + 2\,\bar r}.
```

This is the fraction of the variance of `w` that reproduces when the sentence frame changes, and it is
the upper bound for every held-out $R^2$ reported below.

### Matched narrow-vs-wide contrasts

To show that the leftover variation is real rather than a smear of noise around a trend, we look for
*matched contrasts*: two pairs whose corpus JSD and whose endpoint output movement are both nearly the
same, but whose widths are far apart. Matching on $JSD_{\mathrm{out}}$ as well as `J` matters, because
otherwise a difference in how far apart the two endpoints are could masquerade as a difference in the
shape of the movement. A contrast is a pair of gated pairs $p, q$ with

```math
|J_p - J_q| \le 0.02 \text{ bits},\qquad
|JSD_{\mathrm{out},p} - JSD_{\mathrm{out},q}| \le 0.05 \text{ bits},\qquad
w_q - w_p \ge 0.15,
```

plus a reproducibility requirement that the ordering hold in every frame:
$\max_c w_p^{(c)} < \min_c w_q^{(c)}$. Figures 1 and 2 show the contrasts this yields.

### The decomposition: is the leftover per-token or per-pair?

The central question is whether width is a property each token carries with it, or something that
arises from the chemistry of a specific pairing. The two hypotheses make different predictions, so we
fit the **token-additive model**, with one free parameter $a_u$ per endpoint token and no interaction
term:

```math
w(u,v) \;=\; \mu + a_u + a_v + \varepsilon.
```

If width is carried by individual tokens, this fits well; if it comes from the pairing, it cannot. The
model has 123 free parameters for 929 observations, so an in-sample fit would flatter it. We therefore
score every model by **held-out $R^2$** under 5-fold cross-validation over pairs (identical folds for
every model, ridge term $10^{-3}$ for numerical stability), where $\hat w_{-k(i)}$ is the prediction
for pair $i$ from a fit that excluded the fold containing $i$:

```math
R^2_{\mathrm{CV}} \;=\; 1 \;-\;
\frac{\sum_i \big(w_i - \hat w_{-k(i)}\big)^2}{\sum_i \big(w_i - \bar w\big)^2}.
```

$R^2_{\mathrm{CV}} = 0$ means no better than predicting the mean width; 1 means perfect prediction;
negative means worse than the mean. Repeating the whole comparison under five different random fold
assignments moves each reported value by at most 0.01 (token-additive + `J`: 0.575–0.586; corpus JSD:
0.149–0.153), so none of the gaps below depends on a lucky split. Figures 3 and 4 report it for these
models:

- **corpus JSD** `J` (and a quadratic version, to check the shortfall is not just curvature);
- **model-output JSD** $JSD_{\mathrm{out}}$, the model's own endpoint separation;
- **five pair covariates + `J`**: summed continuation entropy, mean corpus log-frequency, summed model
  surprisal of the two tokens, and the two block-0 geometry terms below;
- **token-additive**, alone and combined with `J`, with $JSD_{\mathrm{out}}$, and with block-0 geometry.

**Block-0 geometry** is the pair's endpoint arrangement at the interpolation site: the cosine
similarity $\cos_0$ between $x_u$ and $x_v$ and their Euclidean distance $d_0 = \lVert x_u - x_v\rVert$,
median over frames. These are pair-specific by construction, so they are the natural candidate for
whatever additivity misses.

### Two probes that measure a single token, with no partner involved

A fitted $a_u$ is a description, not a measurement: it exists only inside the pair bank and could be
absorbing anything that varies token by token. To turn it into something an auditor could compute, we
ran two probes on the model itself. Both use six **anchor tokens** — ` and`, ` significant`, ` close`,
` playing`, ` bigger`, ` buried` — drawn from the same eligible pool but appearing in **none** of the
1,000 pairs, so nothing measured here shares a partner with the bank. Both probes run on
`pythia-1.4b-deduped` @ `step143000` at the same block-0 hook point, in the same three frames.

**Anchor width** $\hat w_u$ — the transfer test. We run the interpolation protocol exactly as above
between token $u$ and each anchor, apply the same validity criteria, and take the median width over
the six anchors and three frames (18 curves per token; all 18 valid for every token):

```math
\hat w_u \;=\; \mathrm{median}_{\,c,\;\alpha \in \text{anchors}} \; w\big(u, \alpha; c\big).
```

If width is a per-token property, $\hat w_u$ — measured against partners the bank never used — should
predict $a_u$, and the sum $\hat w_u + \hat w_v$ should predict the width of a bank pair with only a
slope and an intercept to fit. Figure 6 reports both.

**Basin radius** $r_u(\tau)$ — a mechanistic candidate for *why* a token has the width it does. If each
token sits in a region within which the model's output hardly changes, the size of that region should
set how long the output stays put. Starting from the token's post-block-0 state $x_u$, we travel along
a great circle at fixed norm in a direction $\hat\delta$ orthogonal to $x_u$,

```math
x(\theta) \;=\; \lVert x_u \rVert \,\big(\cos\theta \cdot \hat e_u + \sin\theta \cdot \hat\delta\big),
```

and record how far the output distribution has moved from the token's own output, in bits:

```math
M_u(\theta) \;=\; JSD\Big(\mathrm{softmax}\big(z(\theta)\big),\; \mathrm{softmax}\big(z_u\big)\Big).
```

The radius is the first angle at which that movement crosses a threshold,
$r_u(\tau) = \min\lbrace \theta : M_u(\theta) \ge \tau \rbrace$, linearly interpolated on a 26-point
grid over $\theta \in [0, 1]$ radians. We use two families of directions and report both, because they
ask different questions: **random** directions (6 fixed Gaussian directions, shared across tokens) ask
whether the token is generically insensitive, and **anchor** directions (toward each anchor's state)
ask whether it is insensitive along the kind of direction the pair experiment actually travels.
Thresholds $\tau \in \lbrace 0.05, 0.1, 0.2 \rbrace$ bits; we report $\tau = 0.2$ for anchor directions
and $\tau = 0.1$ for random ones, the largest threshold each family reaches inside the sweep. Because
a peaked output distribution registers more bits of JSD per unit of logit movement than a flat one, we
also record each token's output entropy and report the radius correlation with entropy partialled out.

### The forward screen: predicting pairs of tokens the fit never saw

Every quantity above is computed on the same 123 tokens, so a good fit could still be a property of
that particular token set. The screen is therefore tested forward. We take **40 tokens that appear in
none of the 1,000 pairs and are not anchors**, measure only their anchor widths $\hat w_u$, and predict
the width of **all 780 pairs** among them with

```math
\hat w(u,v) \;=\; \beta_0 + \beta_1\big(\hat w_u + \hat w_v\big),
```

where $\beta_0$ and $\beta_1$ are estimated **on the 929 bank pairs only** and then frozen. No
parameter, and no token, is shared with the pairs being predicted. The new pairs are then run through
the full interpolation protocol and scored with the same validity criteria and the same 0.2-bit
movement gate (718 of 780 pairs survive), using

```math
R^2_{\mathrm{fwd}} \;=\; 1 - \frac{\sum_i \big(w_i - \hat w_i\big)^2}{\sum_i \big(w_i - \bar w\big)^2},
```

which is a genuine prediction error: the mean $\bar w$ is the new set's own mean, so a model that
merely knew the average width of the bank would score below zero. Corpus JSD cannot enter this
comparison — the corpus count arrays are no longer on disk — so the baseline here is the model's own
endpoint output difference $JSD_{\mathrm{out}}$, which requires running both endpoints of each pair.
Figure 7 reports the result.

### The anchor-set swap: is $\hat w_u$ a token trait or a relation to the anchors?

The six anchors above are ordinary content words, so a high $\hat w_u$ might mean "this token behaves
this way generally" or only "this token is far from a typical content word". To separate those we
recompute the anchor width for all 123 endpoint tokens against two further anchor sets, disjoint from
each other and from the original: six **function words** (` he`, ` it`, ` we`, ` but`, ` they`,
` them`) and six **rare content words** (` surreal`, ` creepy`, ` unbelievable`, ` disgusting`,
` ironic`, ` tempting`, selected as the highest-id alphabetic tokens in the pool, since GPT-NeoX BPE
merges are frequency-ordered so a high id means a rare token). If $\hat w_u$ is a trait, the anchor set
is an arbitrary measuring stick and the two rankings agree; if it is a relation, they diverge. Figure 8
reports the rank agreement between the sets and each set's agreement with the fitted effect $a_u$.

### The layer sweep: where is the trait established?

Every measurement above is taken at one site, after block 0, which leaves open whether the per-token
trait is already present in the state entering the network or is built by the blocks that follow. We
therefore repeat the anchor-width measurement with the interpolation site moved to **blocks 6, 12 and
18**, using the same six anchors and the same protocol — only the hook point changes, so blocks
$L+1 \ldots 23$ are what remains to produce the output. Two things are compared across sites: the rank
agreement of $\hat w_u^{(L)}$ with the block-0 values (does the network reorder the tokens?), and how
much of the block-0 *pair* widths each site's measurement predicts. Figure 9 reports both.

### The embedding probe: can the per-token number be looked up instead of measured?

The screen still costs 18 interpolation curves per token, which is cheap but not free, and the layer
sweep says the token ranking is already present at the earliest site tested. That raises a practical
question: is the trait visible in the token's **static embedding** — the row $W_E[u]$ of the embedding
matrix, the 2048-dimensional vector the model looks up before any computation happens? If so, an
auditor needs no forward pass at all. Two measurements answer it.

**The embedding site.** We repeat the anchor-width measurement with the interpolation site moved
*below* block 0, to the input embedding, using the same six anchors and the same protocol, and compare
the resulting $\hat w_u^{\mathrm{emb}}$ with the block-0 values. This asks how much of the trait exists
before block 0 runs.

**The probe.** We fit a ridge regression from the static embedding row to the block-0 anchor width,

```math
\hat w_u \;\approx\; \beta_0 + \boldsymbol{\beta}^{\top} \tilde W_E[u],
\qquad
\boldsymbol{\beta} = \arg\min_{\boldsymbol{b}} \sum_{u \in \mathrm{train}}
\big(\hat w_u - \beta_0 - \boldsymbol{b}^{\top}\tilde W_E[u]\big)^2 + \lambda \lVert \boldsymbol{b}\rVert^2 ,
```

where $\tilde W_E[u]$ is the embedding row standardised using the training tokens' means and standard
deviations. The probe is fitted on 80 of the 123 endpoint tokens and scored on the other 43, repeated
over 50 random splits; $\lambda$ is chosen inside each training set by 5-fold cross-validation over
$10^{-1} \ldots 10^{5}$. We report the mean held-out Spearman $\rho$ and $R^2$ across splits, and two
controls. The first is a **shuffled-target control**: the identical procedure with the targets randomly
permuted, which measures how much apparent skill 2048 features can manufacture from 80 training points.
The second is an **embedding-norm baseline**: the same probe using only $\lVert W_E[u]\rVert$ as its
single feature, since embedding norm tracks token frequency in Pythia and would be the boring
explanation for any probe success.

**The lookup screen.** Finally we run the forward screen again with the measurement replaced by the
lookup. The probe is fitted on the 123 bank tokens, applied to the embedding rows of the 40 tokens
outside the bank, and the pair-level slope and intercept are re-estimated on the bank pairs using
out-of-fold probe predictions and then frozen. The same 718 gated pairs are scored, so the two screens
— measured and looked-up — are directly comparable. Nothing in this path runs a forward pass on the new
tokens. Figure 10 reports all of it.

### The vocabulary-wide test: does the lookup hold outside the curated pool?

Every token used anywhere above comes from `dir18`'s eligibility pool of common, single-token,
alphabetic words, so a probe that works there might have learned a property of that word class rather
than of tokens in general. We therefore apply the probe — fitted on the 123 bank tokens — to all 50,304
embedding rows and select 32 tokens from **four classes the pool excludes or under-samples**: ordinary
words outside the pool, subword fragments (no leading space), punctuation and numerals, and capitalised
names. Within each class the eight tokens are spaced evenly over that class's predicted-width
quantiles, so each class covers the probe's range instead of clustering at one end. Their anchor widths
are then measured at block 0 with the same six anchors, the same three frames and the same protocol,
and compared with the predictions. Figure 11 reports the comparison and where each class sits.

### The frame-shape control: a token property or a token-in-this-slot property?

All three frames used so far are short declarative prefixes with the token in final position, so
$\hat w_u$ might describe the token in that particular slot rather than the token. We therefore repeat
the anchor-width measurement for the same 123 tokens and the same six anchors in four contexts of
deliberately different shape — a mid-sentence continuation (`She kept walking because everything
felt`), an interrogative (`Is it really`), a colon-list (`The report mentions the following:`) and a
code prefix (`def solve(x):` followed by a newline and `    return`) — and correlate each context's
token ranking with the original one.

That correlation needs a reference, because two measurements of the *same* quantity in different frames
do not agree perfectly either. We therefore also compute the agreement among the three original frames,
each summarised the same way (median over the six anchors within one frame), and read the new contexts
against it. Figure 12 shows both, together with the widths themselves so that changes in level are
visible separately from changes in ranking.

### The embedding intervention: is the probe's direction a lever?

A probe that predicts a quantity has found a direction along which that quantity varies; it has not
shown that the direction *sets* it. The test is to edit the model. Writing $\boldsymbol{\beta}$ for the
probe's weights on standardised features and $s_j$ for the training standard deviation of coordinate
$j$, the gradient of the probe's prediction with respect to the raw embedding row is

```math
g_j \;=\; \frac{\beta_j}{s_j},
\qquad\text{so a step}\quad
\delta \;=\; \frac{\Delta}{\lVert g\rVert^2}\, g
\quad\text{changes the probe's own prediction by exactly } \Delta .
```

For 16 tokens spread evenly over the measured-width range we add $\delta$ to the token's embedding row
for $\Delta \in \lbrace -0.05, -0.025, +0.025, +0.05\rbrace$ width units, re-measure that token's
anchor width with the standard protocol, and restore the row. If the direction is causal, the measured
$\Delta\hat w_u$ tracks the requested $\Delta$ with slope 1. Two controls come with it: a **random
direction** rescaled to the same step norm, which says whether any perturbation of that size moves
width; and the **output shift**, the JSD in bits between the token's next-token distribution before and
after the edit, which says whether the edit changed the model's behaviour at all. Figure 13 reports the
result.

### The behaviour-calibrated intervention: was the null just too small a step?

The intervention above sizes its step by the *probe's* opinion of it, and the output shift it reports —
a ten-thousandth of a bit — says the model never noticed. An edit the model does not notice cannot be
expected to change anything the model does, so the null is uninformative until the edit is made large
enough to matter. The fix is to let the model set the step size. For a unit direction $\hat d$ we grow
the step $c\,\hat d$ until the token's next-token distribution has moved a chosen number of bits,

```math
c^{\star}(B) \;=\; \min\lbrace c > 0 : \tfrac{1}{3}\textstyle\sum_{f=1}^{3}
\mathrm{JSD}\big(p_f^{\text{base}},\, p_f^{c\hat d}\big) = B \rbrace ,
\qquad B \in \lbrace 0.05,\,0.1,\,0.2 \rbrace \text{ bits},
```

where $p_f$ is the model's next-token distribution with the token in frame $f$ and JSD is measured in
bits as everywhere else in this report. We find $c^{\star}$ by scanning a geometric ladder of step
norms and interpolating in log–log, then re-measure the token's anchor width at that step and record
the output shift actually achieved.

Three things make this a sharper test than the first one. The step is taken in **both signs**: the
probe says $+\hat d$ widens and $-\hat d$ narrows, so a lever must produce opposite-signed width
changes, and a perturbation effect must not. The control is a **random unit direction calibrated to the
same output shift**, not merely to the same step norm — matching on behaviour rather than on geometry
is what makes "the probe direction does more than an arbitrary one" a meaningful comparison. And the
three budgets $B$ trace out a dose–response curve, so a small effect at 0.05 bits growing at 0.2 bits
is distinguishable from no effect. We run 12 tokens spread over the measured-width range, giving
12 tokens × 2 directions × 2 signs × 3 budgets = 144 re-measurements. Figure 14 reports the result.

### The displacement-norm ladder: is the collapse about the move or about the model's response?

The calibrated intervention grows the step norm and the output movement together, so it cannot say
which of the two the collapse follows. Is width tied to *where* the embedding sits, so that any large
displacement erases it, or to the *behaviour* the embedding induces, so that only displacements the
model responds to erase it? The two readings have opposite consequences for the screen: if the trait is
positional, the vocabulary-wide lookup is reading a geometric accident of where training happened to
place each row; if it is behavioural, the lookup is reading something about what the token makes the
model do.

To separate them we compare, **at the same displacement norm**, a direction the model barely responds
to against one it responds to strongly. For each token $u$ and each rung $c$ of the ladder
$c \in \lbrace 0.15,\, 0.4,\, 0.9,\, 1.8 \rbrace$ — against a median embedding-row norm of 0.98 — we
draw $N = 24$ random unit directions $u_j$ and measure what each one actually does to the token's
next-token distribution **at that norm**, averaged over the three frames:

```math
B_j(c) \;=\; \tfrac{1}{3}\sum_{f=1}^{3} \mathrm{JSD}\big(p_f^{\text{base}},\;
p_f^{\,W_E[u] + c\, u_j}\big) \quad \text{bits}.
```

The **quiet** direction at rung $c$ is $\arg\min_j B_j(c)$ and the **loud** one is $\arg\max_j B_j(c)$;
a fixed $u_1$ serves as a plain random control. Selecting by measured response at the rung itself is
what makes the contrast real: the earlier version of this test built the two directions from the
model's *linear* response at a step of 0.05 and extrapolated, and at a displacement of 1.8 the
extrapolation had failed badly enough that the "quiet" direction was no quieter than a random one. We
then re-measure anchor width at each of the three directions and every rung, giving 12 tokens × 4 rungs
× 3 directions = 144 re-measurements.

Two quantities separate the readings. The **level**, mean $\hat w_u$ after the edit, says how far the
tokens have been pushed toward a generic width. The **ordering**, $\rho$(before, after) across the 12
tokens, says whether the trait itself — the thing a screen reads — is still there. Positional
destruction predicts that quiet and loud behave alike at equal norm; behavioural destruction predicts
the quiet direction preserves the ordering where the loud one does not. Figure 15 reports the result.

### The mode split: which successors does a disruptive edit move?

The ladder says the trait dies when the model's output moves, but the output is a distribution over
50,304 successors and "it moved" does not say which part of it moved. The distinction matters for the
whole direction: corpus successor JSD, the statistic this project started from, is dominated by a
token's few high-mass continuations, so if the width collapse tracks disturbance of *those*, the
per-token trait is a rediscovery of the same object and a cheaper corpus-side estimator should exist.
If instead the collapse tracks the tail, the embedding probe is reading something corpus counts cannot
see, and the two quantities are genuinely separate.

We therefore split every edit's output change by successor token. The Jensen–Shannon divergence is a
sum of non-negative per-successor terms, so it partitions exactly. With $p_f$ the token's next-token
distribution in frame $f$ before the edit, $q_f$ the same after it, $m_f = (p_f + q_f)/2$, and
$T_f$ the set of the $K = 32$ successors with the largest $p_f$, the per-successor term and the
**top-mass share** of the divergence are:

```math
J_f(v) \;=\; \tfrac{1}{2}\Big( p_f(v)\log_2\tfrac{p_f(v)}{m_f(v)} \;+\; q_f(v)\log_2\tfrac{q_f(v)}{m_f(v)} \Big),
\qquad
S \;=\; \frac{1}{3}\sum_{f=1}^{3} \frac{\sum_{v \in T_f} J_f(v)}{\sum_{v} J_f(v)}.
```

$S = 1$ means the edit rearranged only the token's top successors; $S = 0$ means it left them alone and
churned the tail. The reference point is the base mass $\frac{1}{3}\sum_f \sum_{v \in T_f} p_f(v)$: an
edit that disturbs the distribution in proportion to mass has $S$ near that value, so $S$ below it means
the damage is tail-weighted.

The experiment reuses the ladder's top rung. For each of the same 12 tokens we draw $N = 24$ random
unit directions, apply each at norm 1.8, and record both the total movement $B_j$ in bits and the share
$S_j$. That gives the descriptive half: where the loud direction — the one that erased the ordering —
actually puts its damage. For the causal half we take the most top-heavy direction $\arg\max_j S_j$ and
the most tail-heavy $\arg\min_j S_j$, **rescale each by a log-log calibration scan until both move the
output by the same 0.4 bits**, and re-measure $\hat w_u$. Matching on total movement is what makes the
comparison about *which* successors moved rather than *how much* moved. Top-heavy edits doing more
damage to the ordering would tie the trait to high-mass continuations; equal damage says the collapse
is indifferent to where in the distribution the disturbance lands. Figure 16 reports the result.

### Can the split be steered on purpose?

Selecting the extreme of 24 random draws is a weak instrument: the draws barely differ in $S$, so the
matched-movement comparison above has little contrast to work with. The last experiment therefore
**constructs** the two directions. For a small displacement the per-successor divergence is quadratic
in the logit response, $J_v \propto p(v)\tilde\delta(v)^2$ with $\tilde\delta$ the logit change
centred under $p$, so inside the span of $m$ probe directions the share $S$ is a Rayleigh quotient in
the mixing coefficients $c \in \mathbb{R}^m$:

```math
S(c) \;=\; \frac{c^{\top} A c}{c^{\top} B c},
\qquad
A_{jl} = \sum_{f}\sum_{v \in T_f} p_f(v)\,\tilde\delta_{j,f}(v)\,\tilde\delta_{l,f}(v),
\qquad
B_{jl} = \sum_{f}\sum_{v} p_f(v)\,\tilde\delta_{j,f}(v)\,\tilde\delta_{l,f}(v).
```

$A$ restricts the sum to the token's top-$K$ successors and $B$ runs over the whole vocabulary, so the
generalised eigenvectors of $(A, B)$ give the $S$-maximising and $S$-minimising combinations in closed
form. We build them from $m = 24$ random probe directions applied at displacement norm 0.6 — small
enough for the quadratic approximation, so each eigenvalue is a **predicted** $S$ for a small step in
that combination. Both constructed directions are then rescaled to the same 0.4 bits by the same
log-log calibration, and $\hat w_u$ is re-measured. Reporting the predicted $S$ next to the $S$ the
rescaled edit actually achieves is what makes this a test of the method as well as of the hypothesis.
Figure 17 reports the result.

### Component ablation: which early computation carries the trait?

Every intervention so far edits the token's embedding, and all of them end the same way — the trait
dies from disturbance rather than moving where the probe points. The remaining place to look is the
computation between the embedding and the interpolation site. The layer sweep says which tokens are
narrow is already fixed at the input while the sharpening is produced by the blocks below the site, so
if the trait is carried by a small number of components they are early ones.

We therefore **mean-ablate** one component at a time in blocks 0–5: each of the 16 attention heads per
block (by replacing that head's 128-dimensional slice of the input to the block's output projection)
and each block's MLP (by replacing its output), always at the final token position only, and always by
the mean that component produces at that position over the 18 endpoint prompts (12 tokens + 6 anchors)
run with nothing ablated. Endpoints, interpolation bank and $\hat w_u$ are then recomputed from
scratch with the ablation in place, so a component below the interpolation site is scored through its
effect on the endpoint states as well as on the readout. Two numbers score each of the 102 components:

```math
\mathrm{sd}_c = \sqrt{\frac{1}{n-1}\sum_{u}\bigl(\hat w^{(c)}_u - \overline{\hat w^{(c)}}\bigr)^2},
\qquad
\rho_c = \rho\bigl(\hat w^{(0)}_u,\; \hat w^{(c)}_u\bigr),
\qquad
B_c = \frac{1}{n}\sum_u \mathrm{JSD}\bigl(p^{(0)}_u \,\Vert\, p^{(c)}_u\bigr)
```

Here $\hat w^{(c)}_u$ is token $u$'s anchor width with component $c$ ablated, $\hat w^{(0)}_u$ the
unablated one, and $p^{(c)}_u$ the model's unpatched next-token distribution after token $u$. A
component that carries the trait shows a small $\mathrm{sd}_c$ (the tokens stop differing) and a low
$\rho_c$ (the ordering is gone); $B_c$, in bits, says how much of the model's ordinary behaviour the
ablation destroyed, which is what tells a genuine carrier apart from a component that is merely large.
Run for the 12 tokens of the intervention experiments against the 6 anchors in the first frame.
Figure 18 reports the result.

### Separating a carrier from a loud component: the matched-bits dose–response

A single ablation cannot tell a component that *computes* a quantity from one that is merely the
largest thing in the neighbourhood, because a big enough disturbance destroys the trait wherever it is
applied. The fix is to make loudness the x-axis instead of a nuisance. We soften the ablation into a
dose $\alpha \in [0,1]$ applied to the block-0 MLP's final-position output $m_u$, and pair every dose
with a control that perturbs the same residual stream by a fixed random unit direction $r$ at a scale
$c$ chosen so both arms move the model's output by the same number of bits:

```math
m_u^{\mathrm{mlp}}(\alpha) = (1-\alpha)\, m_u + \alpha\, \bar m,
\qquad
m_u^{\mathrm{ctrl}}(c) = m_u + c\, r,
\qquad
c(\alpha): \; B\bigl(m^{\mathrm{ctrl}}(c)\bigr) = B\bigl(m^{\mathrm{mlp}}(\alpha)\bigr)
```

Here $\bar m$ is the same mean replacement vector the ablation used, $r$ is drawn once from a standard
normal and normalised, and $B(\cdot)$ is the mean output movement in bits defined above, $B_c$. The
scale $c(\alpha)$ is found by bisection on $B$, which needs no interpolation curves and is therefore
cheap. Both arms are then scored with the same two numbers as the ablation — the rank agreement
$\rho$ and the across-token spread $\mathrm{sd}$ of $\hat w_u$ — and plotted against $B$, so the two
curves are read at matched loudness. Separated curves place the trait in the component; coincident
curves say only that disturbance kills it. Figure 19 reports the result.

### Is anything left after the additive model, or is it noise?

A held-out $R^2$ below the ceiling could mean either that real structure is missing or that the
ceiling is optimistic. To tell those apart we fit the additive model **separately within each sentence
frame**, take the residuals $r^{(c)}$, and correlate them **across frames**. Structure that reproduces
in an independent frame is real pair-specific signal; noise correlates at zero. We report the mean
across the three frame pairs, and convert it to a variance share with the same Spearman-Brown formula.

### The deflationary alternative: is `w` just path length?

`w` is a fraction of a path, so a transition of fixed absolute size sitting on a longer path would
automatically look narrower. That would make width variation an artifact of how far apart the two
endpoint states happen to be. The check converts `w` into residual-stream distance units,

```math
w_{\mathrm{abs}} \;=\; w \cdot d_0 ,
```

and compares the spread of the two, using the coefficient of variation
$\mathrm{CV}(x) = \mathrm{sd}(x)/\mathrm{mean}(x)$. If path length were the mechanism, $w_{\mathrm{abs}}$
would be markedly *less* variable than `w`.

---

## Results

### Observed patterns

This section reports direct evidence only; interpretation is in **Candidate hypotheses** below.

**1. `w` is highly reproducible, and the movement gate removes interpretation risk, not noise.**
Across all 1,000 pairs the mean between-frame correlation of `w` is $\bar r = 0.825$, giving a
reproducibility ceiling of $R^2_{\mathrm{ceiling}} = 0.934$. The 71 pairs failing the 0.2-bit movement
gate are not noisier (their median across-frame spread in `w` is 0.02–0.04, no larger than the rest),
but they are systematically wide (median `w` 0.687 versus 0.545 for gated pairs) and sit at low corpus
JSD (median 0.401 versus 0.665 bits). Keeping them would let "the model barely distinguishes these two
tokens" be read as "this pair has a wide transition". On the 929 gated pairs the headline correlation
falls from $\rho = -0.486$ ($p = 2.6\times10^{-60}$) to $\rho = -0.409$ ($p = 1.0\times10^{-38}$), so
part of the original association came from the uninformative tail. The model's own endpoint separation
gives $\rho(JSD_{\mathrm{out}}, w) = -0.357$ ($p = 3.1\times10^{-29}$).

To show both the scatter that motivates this direction and what the gate does, Figure 1 plots width
against the corpus statistic and against endpoint movement.

![Width against corpus successor JSD, and against endpoint output movement](plots/scatter_and_gate.png)

**Figure 1.** Left: transition width `w` (y) against corpus successor JSD `J` in bits (x) for all 1,000
pairs. Circles pass the endpoint-movement gate, open squares fail it. Black diamonds joined by lines
are the eight largest matched contrasts — pairs of pairs at nearly the same `J` and nearly the same
endpoint movement whose widths differ by up to 0.44. The dotted line at $w = 0.8$ is the width of a
model whose output moves in proportion to `t`. Right: the same `w` (y) against endpoint output
movement $JSD_{\mathrm{out}}$ in bits (x); the dashed line is the 0.2-bit gate. Pairs below the gate
cluster at high `w` because a normalised score cannot describe a movement that barely happened.

**2. Matched narrow-vs-wide contrasts are abundant, not anecdotal.** Under the criteria above there
are **1,529** matched contrasts among the 929 gated pairs; only 21 of them share a token between the
two pairs, so the effect is not one token dragging its partners around. The largest is
` her` / ` when` ($w = 0.34$) against ` kind` / ` wrong` ($w = 0.77$) at $J \approx 0.70$ bits and
endpoint movements of 0.86 and 0.90 bits — a 0.44 difference in width at matched endpoints.

The strongest contrasts are listed below; each row is a narrow pair and a wide pair matched on corpus
JSD to within 0.02 bits and on endpoint movement to within 0.05 bits, with the width ordering holding
in all three sentence frames. The narrow side is consistently built from function words (determiners,
possessives, prepositions, `when`), the wide side from evaluative adjectives and adverbs
(`kind`, `wrong`, `never`, `nothing`, `most`, `now`) — the pattern that motivates the per-token
analysis that follows.

| narrow pair | `w` | wide pair | `w` | $\Delta w$ | `J` (bits) | $JSD_{\mathrm{out}}$ narrow / wide |
|---|---|---|---|---|---|---|
| ` her` / ` when` | 0.34 | ` kind` / ` wrong` | 0.77 | 0.44 | 0.70 | 0.86 / 0.90 |
| ` our` / ` very` | 0.32 | ` never` / ` nothing` | 0.69 | 0.37 | 0.73 | 0.84 / 0.82 |
| ` from` / ` one` | 0.43 | ` kind` / ` wrong` | 0.77 | 0.35 | 0.70 | 0.90 / 0.90 |
| ` our` / ` very` | 0.32 | ` most` / ` now` | 0.67 | 0.35 | 0.72 | 0.84 / 0.83 |
| ` one` / ` when` | 0.38 | ` hard` / ` kind` | 0.72 | 0.34 | 0.74 | 0.91 / 0.96 |
| ` completely` / ` interesting` | 0.44 | ` kind` / ` wrong` | 0.77 | 0.34 | 0.70 | 0.90 / 0.90 |
| ` because` / ` being` | 0.45 | ` kind` / ` wrong` | 0.77 | 0.33 | 0.69 | 0.90 / 0.90 |

**3. The contrast is a difference in curve shape, not in how far the output travels.** Figure 2 shows
the full `d(t)` curves behind three of these contrasts, one curve per sentence frame, and answers
whether the width difference is a summary-statistic artifact.

![Output-distance curves for three matched narrow-vs-wide contrasts](plots/contrast_curves.png)

**Figure 2.** Output-distance score `d(t)` (y) against interpolation position `t` (x) for three matched
contrasts, all three sentence frames drawn per pair. Solid = the narrow pair, dashed = the wide pair;
the pairs in each panel are matched on corpus JSD `J` and on endpoint output movement
$JSD_{\mathrm{out}}$ (values in each panel title). Dotted horizontal lines mark $d = 0.1$ and $d = 0.9$,
whose separation along `t` defines `w`. The narrow pairs are flat near both endpoints and swing in the
middle; the wide pairs rise almost in proportion to `t`. The three frames of one pair lie nearly on top
of each other, which is the reproducibility the ceiling quantifies.

**4. The leftover variation is mostly a per-token additive effect.** Figure 3 answers the direction's
core question by asking which model of `w` predicts held-out pairs best.

![Held-out R-squared for models of transition width](plots/cv_r2.png)

**Figure 3.** Held-out $R^2$ for `w` under 5-fold cross-validation over the 929 gated pairs (x), for
seven models (y). Hatched bars use pair-level predictors only; solid bars include the per-token term
$a_u + a_v$. The dashed vertical line is the reproducibility ceiling of 0.934 estimated from
across-frame agreement — no model can exceed it. Corpus JSD reaches 0.149; adding one number per token
takes it to 0.578, and the per-token term on its own (0.365) outperforms every pair-level predictor
tested.

Reading the bars: corpus JSD 0.149, a quadratic in corpus JSD 0.165 (so the shortfall is not simply
curvature), model-output JSD 0.187, five pair covariates plus corpus JSD 0.399, token-additive alone
0.365, token-additive + corpus JSD **0.578**, plus model-output JSD 0.648, plus block-0 geometry 0.723.
Against a ceiling of 0.934, the token-additive term plus corpus JSD captures 62% of the explainable
variance where corpus JSD alone captures 16%.

The gain is a shift of the whole distribution, not a few lucky pairs, which Figure 4 makes visible.

![Predicted versus observed width for two models](plots/prediction.png)

**Figure 4.** Observed `w` (y) against held-out predicted `w` (x) for every gated pair, under corpus
JSD alone (left, squares) and corpus JSD plus the per-token term (right, circles). Dashed line is
$y = x$. Corpus JSD compresses every prediction into 0.47–0.67 and cannot reach the narrow tail at all;
adding the per-token term spreads predictions across the observed range and removes most of the
vertical scatter.

**5. The per-token contribution is real but is not a simple corpus statistic.** Figure 5 shows the
fitted token effects and tests the obvious explanation, that frequent tokens behave differently.

![Fitted per-token width effects, and their relation to corpus frequency](plots/token_effects.png)

**Figure 5.** Left: the fitted token effect $a_u$ in width units (y) for the 120 endpoint tokens
appearing in at least four gated pairs, ranked (x); the extremes are labelled. The range is 0.24 width
units, comparable to the full observed spread of `w`. Right: $a_u$ (y) against the token's corpus
log-frequency (x), one marker per token. The relation is real but loose (Spearman
$\rho = -0.33$, $p = 2.9\times10^{-4}$), so frequency is not the underlying variable.

Narrow-pulling tokens are ` un`, ` in`, ` his`, ` my`, ` when`, ` our`; wide-pushing tokens are
` kind`, ` real`, ` now`, ` never`, ` perfect`. All three corpus-side candidates explain only a slice:
corpus log-frequency $\rho = -0.33$ ($p = 2.9\times10^{-4}$), corpus continuation entropy
$\rho = -0.24$ ($p = 0.008$), and the model's surprisal at seeing the token in the frame
$\rho = +0.26$ ($p = 0.004$). Together with the five-covariate model in Figure 3 reaching only 0.399,
these say the per-token effect has to be measured from the model rather than looked up in a count
table.

**6. The per-token effect is real, measurable, and transfers to partners outside the bank.** Figure 6
tests whether the additive description survives contact with the model: it compares each token's width
measured against six anchor tokens used in no pair with the effect fitted inside the bank, and asks
what the two probes buy at the pair level.

![Anchor width against fitted token effect, basin radius against fitted token effect, and held-out R-squared](plots/transfer.png)

**Figure 6.** Left: fitted token effect $a_u$ (y, width units, from the 1,000-pair bank) against anchor
width $\hat w_u$ (x, median width against six tokens that appear in no pair), one marker per token,
extremes labelled. Middle: the same $a_u$ (y) against basin radius (x, radians of great-circle travel
before the output moves $\tau$ bits) — squares are directions toward anchor tokens ($\tau = 0.2$),
triangles are random directions ($\tau = 0.1$). Right: held-out $R^2$ for pair width `w` (x) for five
models (y); hatched = corpus JSD, solid = the two *measured* token widths, dotted = the 123 *fitted*
token effects.

Anchor width predicts the fitted effect at $\rho = +0.70$ ($p = 5\times10^{-19}$), and still
$\rho = +0.67$ with output entropy partialled out. At the pair level, replacing 123 fitted numbers by
two measured ones costs almost nothing: held-out $R^2 = 0.350$ against 0.365 fitted, rising to 0.452
with corpus JSD added (against 0.578 fitted). Anchor width and the fitted effect are the same metric
computed over disjoint partner sets, so this is a transfer test, not a re-description: the six anchors
are shared by all 123 tokens, so the spread in $\hat w_u$ can only come from the token.

**7. The screen predicts pairs of tokens it has never seen.** Figure 7 asks the question an auditor
would ask: measure 40 unfamiliar tokens once, and can you say which of their pairings will have sharp
transitions before running any of them?

![Forward prediction of pair width for unseen tokens, and separation by predicted tercile](plots/forward_screen.png)

**Figure 7.** Left: observed width `w` (y) of 718 pairs built from 40 tokens absent from the 1,000-pair
bank, against the width predicted from the two tokens' anchor widths alone (x); the slope and intercept
come from the bank, so nothing is fitted here. Dashed line is $y = x$. Right: observed `w` (y) grouped
by tercile of the screen's prediction (x), boxes showing median, quartiles and 1.5 IQR whiskers, with
distinct hatching per tercile.

The forward prediction reaches $R^2 = 0.397$, $\rho = +0.66$ ($p = 1.5\times10^{-89}$), and a mean
absolute error of 0.047 width units on an observed range of 0.34–0.78. On the same pairs the model's
own endpoint output difference gives $\rho = -0.51$, so the per-token screen is the stronger predictor
while also being the cheaper one: 40 token measurements cover all 780 pairings, and the cost of the
endpoint statistic grows with the number of pairs. The 40 new tokens are a broader mix than the bank's
(` re`, ` do`, ` time`, ` life`, ` maybe`, ` delicious`, ` extraordinary`, ` awkward` …), and the
screen's slope transfers to them without adjustment.

**8. The trait is real, but the measuring stick is not neutral: $\hat w_u$ depends on the anchor set.**
Figure 8 asks whether the anchor tokens can be swapped freely.

![Anchor width under two disjoint anchor sets, and each set's agreement with the fitted token effect](plots/anchor_swap.png)

**Figure 8.** Left: anchor width measured against six rare content words (y) against anchor width
measured against six function words (x), one marker per endpoint token, extremes of the fitted effect
labelled. Right: Spearman $\rho$ between each anchor set's widths and the fitted token effect $a_u$
(y), for the original mixed anchors, the function-word anchors and the rare-content anchors (x).

The two disjoint sets rank the 123 tokens at $\rho = +0.46$ ($p = 1.0\times10^{-7}$) — agreement well
above chance, well below identity. Yet each set independently recovers the fitted effect: $\rho = +0.57$
for function-word anchors and $+0.61$ for rare-content anchors, against $+0.70$ for the original mixed
set. As pair-level predictors they separate more sharply: held-out $R^2 = 0.146$ (function),
0.265 (rare content), 0.350 (mixed), and 0.318 when the function and rare-content widths are used
together. So there is a common per-token component that every anchor set finds, and on top of it a
component specific to which anchors were used. The practical consequence is that **the anchor set is
part of the method and must be reported with it**, and that a mixed set of common words is the better
measuring stick — which is also the set the forward screen used.

**9. The token ranking is set early; the size of the effect needs the depth below it.** Figure 9 moves
the interpolation site down the network and asks what survives.

![Anchor width measured at blocks 0, 6, 12 and 18](plots/layer_sweep.png)

**Figure 9.** Left: Spearman $\rho$ across the 123 tokens (y) against the block at which the state is
interpolated (x) — solid, circles: agreement with the block-0 anchor widths; dashed, squares: agreement
with the fitted token effect $a_u$. Right, same x-axis: median anchor width (solid, circles), held-out
$R^2$ for the block-0 pair widths from that site's measurement (dashed, diamonds), and the interquartile
range of $\hat w_u$ across tokens (dotted, triangles). The dash-dotted line marks $w = 0.8$, the width
of a perfectly proportional response.

Which token is relatively narrow barely changes with depth: $\rho$ with block 0 is $+0.92$ at block 6,
$+0.84$ at block 12 and $+0.72$ at block 18. What collapses is the *effect itself*. The median anchor
width climbs 0.553 → 0.621 → 0.728 → 0.800, arriving exactly at the proportional-response value, and the
spread across tokens shrinks by a factor of five (interquartile range 0.102 → 0.020). Interpolate five
blocks from the output and there is almost no transition left to measure, in any token. Sharp,
token-specific transitions are therefore produced by the stack *below* the interpolation site, while the
ordering of tokens is fixed early — consistent with the plateau literature's attribution of sharpening
to downstream MLPs. The falling $\rho$ with $a_u$ (0.70 → 0.35) and the falling held-out $R^2$
(0.350 → 0.146) partly reflect that compression: with an interquartile range of 0.02 there is little
signal left to correlate.

**10. Most of the per-token trait is already in the static embedding, so the screen can be a lookup.**
Figure 10 asks whether the number the screen needs can be read off the embedding matrix instead of
measured, and what that costs.

![Anchor width at the embedding site, a ridge probe from the static embedding, its controls, and the resulting zero-forward-pass screen](plots/embed_probe.png)

**Figure 10.** Far left: anchor width $\hat w_u$ measured with the interpolation site at the input
embedding (y) against the same quantity measured after block 0 (x), one marker per endpoint token,
dashed line $y = x$. Centre left: block-0 $\hat w_u$ (y) against the value predicted from the token's
static embedding row alone (x), out-of-fold so no token predicts itself. Centre right: mean held-out
Spearman $\rho$ on the 43 test tokens (y) over 50 random 80/43 splits, for three targets (x); hatched
bars are the probe, dotted bars the shuffled-target control, and the dotted horizontal line is the
embedding-norm baseline for the block-0 target. Error bars are $\pm 1$ standard deviation across
splits. Far right: observed width `w` (y) of the same 718 unseen pairs as Figure 7 against the width
predicted from static embeddings alone (x) — no forward pass anywhere in the prediction — with the
measured screen's $R^2$ quoted for comparison; dashed line $y = x$.

Three things follow. First, the trait exists before the network runs: anchor widths measured at the
input embedding agree with the block-0 values at $\rho = +0.79$ ($p = 2.0\times10^{-27}$) and recover
the fitted token effect at $\rho = +0.60$. Second, a ridge probe on the 2048-dimensional static
embedding predicts a held-out token's block-0 anchor width at $\rho = +0.764 \pm 0.045$ and
$R^2 = 0.514 \pm 0.073$, positive in 50 of 50 splits, against $\rho = -0.20$ for the same procedure with
shuffled targets and $\rho = +0.597 \pm 0.071$ ($R^2 = 0.190$) for embedding norm alone. Norm — which
tracks token frequency — carries a good part of it, and the rest of the embedding carries more. The
probe reaches $\rho = +0.505$ against the *fitted* effect $a_u$, about what the measured anchor width
achieves through a different route.

Third, the lookup survives being used forward. Fitting the probe on the 123 bank tokens and applying it
to the 40 tokens the analysis never saw reproduces their measured anchor widths at $\rho = +0.66$
($p = 3.4\times10^{-6}$), and predicting all 718 of their gated pairs from static embeddings alone gives
$R^2 = 0.213$, $\rho = +0.526$, mean absolute error 0.055, with tercile medians 0.51 / 0.57 / 0.61.
That is a real loss against the measured screen (0.397, $\rho = +0.66$, MAE 0.047) and it is the
expected one: a ridge probe shrinks its predictions, so the looked-up widths span a narrower range than
the measured ones (visible as the compressed x-axis in Figure 10, far right). What it buys is that the
prediction costs **no forward passes at all** — the screen becomes a table over the vocabulary — while
still ranking unseen pairs about as well as the model's own endpoint output difference ($\rho = -0.51$),
which requires running both endpoints of every pair.

**11. The lookup keeps working on token types the analysis never touched.** Every token used above is
a common single-token alphabetic word from `dir18`'s eligibility pool, so the probe could have learned
a property of that word class. Figure 11 tests it outside the pool.

![Predicted versus measured anchor width for 32 tokens outside the curated pool, and the measured width of each token class](plots/vocab_probe.png)

**Figure 11.** Left: measured anchor width $\hat w_u$ at block 0 (y) against the width predicted from
the static embedding (x) for 32 tokens drawn from four classes outside the pool — circles: ordinary
words the pool excludes, squares: subword fragments, triangles: punctuation and numerals, diamonds:
capitalised names. Eight tokens per class, spaced evenly over that class's predicted range. Dashed
line $y = x$; the shaded band is the range of measured widths over the 123 pool tokens. Right: measured
$\hat w_u$ (y) by token class (x), with the 123 pool tokens as the reference group; boxes are median,
quartiles and 1.5 IQR whiskers, hatched distinctly, individual tokens overplotted. The dash-dotted
line marks $w = 0.8$, a perfectly proportional response.

The ranking transfers: over the 32 tokens the predicted and measured widths correlate at
$\rho = +0.60$ ($p = 3.0\times10^{-4}$), with mean absolute error 0.046 and all 576 curves valid. Within
each class of eight the estimates are noisy, as expected at that sample size ($\rho$ from $+0.24$ for
punctuation to $+0.83$ for capitalised names), but no class inverts the relationship. Two honest
caveats show in the figure. The lookup under-disperses — measured widths spread over a standard
deviation of 0.073 against the prediction's 0.047, the same ridge shrinkage seen in pattern 10 — and
the classes sit at different levels: rarer ordinary words are wider (median 0.632) than fragments
(0.569), punctuation and numerals (0.529) or capitalised names (0.527), against 0.549 for the pool.
The practically useful part is that measured widths outside the pool span 0.367–0.686, essentially the
pool's own range, so the extremes an auditor would want to find are present in this wider vocabulary and
the lookup ranks them.

**12. The token ranking survives a change of context; the level of `w` does not.** The three frames
used everywhere above share one shape — a short declarative prefix with the token in final position —
so $\hat w_u$ could have been a property of that slot. Figure 12 re-measures it in four structurally
different contexts.

![Rank agreement of anchor widths measured in four new contexts with the original ranking, and the widths themselves](plots/frame_control.png)

**Figure 12.** Left: Spearman $\rho$ between the token ranking measured in each new context and the
original ranking (y) for the four contexts (x), each bar hatched distinctly. The dashed line is the
mean agreement among the three *original* frames (+0.82) — the ceiling this comparison should be read
against, since it is what two measurements of the same shape achieve. The dotted line is the agreement
between two disjoint anchor sets (+0.46) from pattern 8, for scale. Right: $\hat w_u$ measured in each
new context (y) against $\hat w_u$ in the original three frames (x), one marker per token per context,
markers and colours matching the contexts on the left; dashed line $y = x$.

The ranking transfers. A mid-sentence continuation (`She kept walking because everything felt`) agrees
with the original at $\rho = +0.844$ — indistinguishable from the +0.82 that two original frames achieve
with each other — and an interrogative (`Is it really`) and a colon-list (`The report mentions the
following:`) reach $+0.770$ and $+0.735$. The furthest context, a code prefix
(`def solve(x):\n    return`), still gives $+0.501$, above the agreement between two disjoint anchor
sets. Curve validity stays at 99.6–100%.

What does move is the level. Median $\hat w_u$ runs 0.530 in the list context, 0.549 in the original
frames, 0.599 mid-sentence, 0.623 in the question and 0.705 in code, and the code context also
compresses the spread across tokens (interquartile range 0.049 against 0.107–0.123 elsewhere). Reading
the two panels together: which tokens have narrow transitions is a property the token carries between
contexts, while how narrow transitions are in absolute terms is set by the context — the same split
seen across depth in pattern 9 and across anchor sets in pattern 8. For the screen this means a
vocabulary-wide table ranks tokens usefully in unfamiliar contexts, but a width threshold calibrated in
one context should not be reused in another.

**13. The probe's embedding direction is a correlate, not a lever — editing the embedding along it
does not move width.** Everything above is correlational. Figure 13 reports the first intervention.

![Measured width change against the width change requested along the probe direction, and per-token response slopes](plots/intervene.png)

**Figure 13.** Left: measured change in a token's anchor width $\Delta\hat w_u$ (y) against the change
the probe was asked for (x), for 16 tokens at four step sizes; circles are steps along the probe
direction, squares are random directions of the same step norm, jittered slightly for visibility. The
dashed line is what a causal direction would give ($y = x$); the horizontal line is no change. Right:
the per-token slope of measured against requested change (y) for the two directions (x), one marker per
token, thick bar = mean; the dashed line at 1.0 is what the probe predicts.

Adding a step to a token's embedding row along the probe's direction, sized so the probe's own
prediction moves by 0.05 width units, moves the measured width by 0.0027 on average — 5% of what was
asked — with no consistent sign (slope $-0.023$, sign agreement 0.39, per-token slopes scattered from
$-0.13$ to $+0.15$). A random direction of the same norm moves it by 0.0008. The steps are not
negligible in size: their norm is 0.053 against a median embedding-row norm of 0.984, about 5%. They
are, however, nearly invisible to the model — the token's next-token distribution shifts by 0.0001 bits
— which is the honest caveat on this null: at a perturbation this functionally small the model's
behaviour hardly changes at all, so what the experiment establishes is that the probe's direction is
not an *efficient* lever, not that no lever exists. What it rules out is the strong reading of pattern
10: the probe finds a direction along which embeddings of narrow and wide tokens differ, and that
direction does not itself set width.

**14. Once the edit is large enough for the model to notice, width moves a great deal — but every
direction moves it the same way, toward a common value.** Pattern 13 leaves one loophole: the edits
were behaviourally invisible. Calibrating the step on the model instead of on the probe closes it, and
turns the null into a positive statement about where the trait lives. Figure 14 reports it.

![Width change against the calibrated output movement, probe against random at matched movement, signed changes by direction, and where edited tokens end up](plots/intervene2.png)

**Figure 14.** Twelve tokens, each edited along the probe direction and along a random direction, in
both signs, at three calibrated output budgets. Top left: mean $|\Delta\hat w_u|$ (y) against the
output movement the edit was calibrated to produce (x, bits, log scale); solid circles = probe
direction, dashed squares = random direction, faint markers = individual edits. Top right:
$|\Delta\hat w_u|$ along the probe direction (y) against $|\Delta\hat w_u|$ along the random direction
for the same token and budget (x); marker shape gives the budget, dashed line $y = x$. Bottom left:
mean *signed* $\Delta\hat w_u$ (y) for each direction and sign (x), hatched distinctly — the probe
predicts that $+$ and $-$ have opposite signs. Bottom right: $\hat w_u$ after a 0.2-bit edit (y)
against $\hat w_u$ before it (x); dotted line = no change, horizontal lines = the mean landing point
for each direction.

The step sizes now reach the model: a 0.05-bit budget needs a step of norm 0.87 along the probe
direction, about 90% of a median embedding row, and the calibration lands on target (median achieved /
requested output movement 1.00, interquartile range 0.91–1.05). At that size width moves by 0.10 on
average and by 0.15 at the 0.2-bit budget — fifty times pattern 13's 0.003, so the earlier null was
indeed a step-size null.

The specificity test fails all the same, and it fails in every way it can. A random direction matched
to the same output movement moves width by 0.123 against the probe direction's 0.127 (Wilcoxon
$p = 0.47$ over 72 matched edits; the probe is larger in 53% of them). The probe says $+\hat d$ widens
and $-\hat d$ narrows; instead **all 144 edits widen**, mean $+0.118$ for $+\hat d$ and $+0.088$ for
$-\hat d$ at the smallest budget. Regressing measured against predicted width change gives a slope of
$-0.002$ ($\rho = +0.06$, $p = 0.61$).

What the edits do instead is visible in the bottom-right panel, and it is the informative part. The
landing point is nearly the same wherever the token started: after a 0.2-bit edit the twelve tokens sit
at mean $\hat w_u$ 0.691 (probe) and 0.678 (random) with a standard deviation across tokens of 0.022
and 0.015, against 0.083 before the edit — a fourfold collapse of the spread that carries the whole
result of this direction. Tokens that started narrow move furthest ($\rho$(base $\hat w_u$,
$\Delta\hat w_u$) $= -0.85$ to $-0.94$). So a behaviourally sized displacement of the embedding row, in
any direction, does not slide a token along a width axis: it **compresses** the token's width trait
toward a generic value near 0.68 — close to the 0.8 of a perfectly proportional response and well above
the pool median of 0.549. The compression is strong but not total, and the residue decays with edit
size: the ranking of the edited tokens still agrees with the original at $\rho = +0.73$ and $+0.85$
after a 0.05-bit edit and at $+0.57$ and $+0.36$ after a 0.2-bit one (12 tokens, so these are
indicative). Narrow transitions are a fragile property of the particular embedding training produced,
not a robust consequence of the region it sits in.

One asymmetry is worth recording because it says the probe direction is not arbitrary, only irrelevant
to width. Reaching a given output movement takes a *smaller* step along the probe direction than along
a random one — norm ratio 1.54, 1.66 and 1.76 at the three budgets — so the probe found a direction the
model is unusually sensitive to. It is a behaviourally special direction that does not carry width.

**15. What destroys the trait is behaviour, not displacement: at the same step norm, a quiet direction
keeps the token ordering and a loud one erases it.** Pattern 14 leaves the collapse's cause ambiguous,
because its steps grew in norm and in output movement together. The ladder separates them by putting
directions of very different loudness at identical displacement.

![Mean anchor width against displacement norm for quiet, loud and random directions, and width after the largest edit against width before it](plots/ladder.png)

**Figure 15.** Twelve tokens edited along the quietest and loudest of 24 random directions, rebuilt at
each rung from their measured output movement there, plus one plain random direction. Left: mean
$\hat w_u$ after the edit (y) against the displacement norm of the edit (x, log scale, four rungs
0.15–1.8 against a median embedding-row norm of 0.98); solid/circles = quiet, dashed/squares = loud,
dotted/triangles = random, error bars 1 s.e. over the 12 tokens; the dash-dotted line is the unedited
mean $\hat w_u = 0.543$. Right: $\hat w_u$ after an edit of norm 1.8 (y) against $\hat w_u$ before it
(x) for the quiet (circles, solid fit) and loud (squares, dashed fit) directions, with the median
output movement each produced in the legend; dotted line = no change.

| displacement norm | quiet: bits / mean $\hat w_u$ / $\rho$(before, after) | loud: bits / mean $\hat w_u$ / $\rho$(before, after) | paired $p$ |
|---|---|---|---|
| *before any edit* | *— / 0.543 / —* | *— / 0.543 / —* | \- |
| 0.15 | 0.0001 / 0.544 / $+1.00$ | 0.0003 / 0.546 / $+1.00$ | 0.09 |
| 0.40 | 0.0006 / 0.552 / $+0.99$ | 0.0027 / 0.562 / $+0.99$ | 0.02 |
| 0.90 | 0.0053 / 0.589 / $+0.91$ | 0.0221 / 0.620 / $+0.87$ | 0.0005 |
| **1.80** | **0.0489 / 0.656 / $+0.94$** | **0.4023 / 0.683 / $+0.08$** | 0.09 |

The construction now works: selecting by measured response at each rung separates the two directions by
a factor of 8 in output movement at norm 1.8 (0.049 against 0.402 bits), where the previous
linear-extrapolation version separated them not at all. That is what makes the comparison in the last
row possible, and it is the row that carries the result. **At one and the same displacement — 1.8,
nearly twice a median embedding row — the quiet edit leaves the token ordering essentially untouched
($\rho = +0.94$, $p = 4\times10^{-6}$) and the loud edit leaves nothing of it ($\rho = +0.08$,
$p = 0.80$).** Geometry cannot distinguish those two edits; only what they do to the model can. The
paired comparison points the same way at every rung — the quiet direction widens less than the loud one
in 12 of 12 tokens at norm 0.9 (Wilcoxon $p = 5\times10^{-4}$) and at norm 0.4 ($p = 0.02$) — and the
two rungs where the paired test only reaches $p = 0.09$ are the ones where nothing happens at all
(norm 0.15, both directions under 0.0003 bits) and where the loud direction has saturated (norm 1.8).

The *level* of `w` is a different matter, and it does follow the displacement. Every direction, quiet
included, raises the mean: 0.543 → 0.656 for the quietest direction at norm 1.8, with the spread across
tokens falling from 0.083 to 0.038 (loud: 0.022). So a large enough edit compresses everything toward
a generic width whatever it does to the model, but only a behaviourally loud edit scrambles which
token is which. Read together with pattern 14, where a random direction matched on *output movement*
moved width as much as the probe direction did, the picture is consistent: width responds to how much
the edit disturbs the token's behaviour and not to which direction delivers the disturbance. For the
screen this is the reassuring answer — the vocabulary-wide lookup (pattern 10) is reading a property
tied to what the token makes the model do, not an accident of where its row happens to sit.

**16. The damage an edit does is tail-weighted, and the ordering dies wherever in the distribution it
lands.** Pattern 15 says the trait dies when the model's output moves, and the natural follow-up is
whether "the output" here means the token's few high-mass continuations — the object corpus successor
JSD is built from — or the rest of the distribution. Figure 16 answers both halves of that: where a
large embedding edit actually puts its damage, and whether steering the damage toward the top
successors changes what the edit does to width.

![Where random embedding edits move the output, and anchor width after top-heavy and tail-heavy edits matched on total output movement](plots/mode_split.png)

**Figure 16.** Twelve tokens, the same ones as Figures 14–15. Left: the top-mass share $S$ of the
output change (y, fraction of the JSD landing on the token's 32 most likely successors) against the
total output movement of the edit (x, bits, log scale); small circles are the 24 random directions per
token applied at displacement norm 1.8, triangles the most top-heavy and squares the most tail-heavy
direction after rescaling to 0.4 bits; the dashed line is the mean probability mass those 32 successors
hold before any edit (0.71). Right: anchor width $\hat w_u$ after the edit (y) against $\hat w_u$
before it (x) for the top-heavy (triangles, solid fit) and tail-heavy (squares, dashed fit) edits, with
each one's rank agreement with the pre-edit ordering in the legend; dotted line = no change.

| edit at displacement norm 1.8 (12 tokens) | output movement (bits) | top-mass share $S$ | mean $\hat w_u$ | sd across tokens | $\rho$(before, after) |
|---|---|---|---|---|---|
| *before any edit* | \- | \- | *0.543* | *0.083* | \- |
| loudest of 24 random directions | 0.402 | **0.389** | 0.683 | 0.022 | $+0.08$ |
| most top-heavy, rescaled | 0.410 | 0.408 | 0.666 | 0.027 | $-0.08$ ($p = 0.81$) |
| most tail-heavy, rescaled | 0.453 | 0.355 | 0.651 | 0.025 | $-0.37$ ($p = 0.24$) |

The descriptive half is clear and points away from high-mass continuations. The 32 most likely
successors carry 0.71 of the token's probability mass, but only **0.389** of the divergence a loud edit
produces lands on them: relative to mass, the disturbance is concentrated in the tail. That tilt grows
with the size of the edit — across the 24 random directions per token, louder directions are *more*
tail-weighted ($\rho(B_j, S_j) = -0.36$, median over tokens). So the behaviour whose disruption
coincides with the trait's collapse is not mainly the behaviour corpus successor JSD scores.

The causal half is a null with a real limit. Selecting the extreme directions from a random draw buys
only a narrow contrast — $S$ spans 0.358 to 0.564 across 24 directions for the median token, never
reaching the mass-proportional 0.71 — and at matched output movement the two extremes do the same
thing: both widen every token toward a common value (0.666 and 0.651 against a pre-edit 0.543) and both
leave nothing of the pre-edit ordering ($\rho = -0.08$ and $-0.37$, neither distinguishable from zero
at $n = 12$). One difference survives the pairing: the top-heavy edit widens slightly more than the
tail-heavy one (mean $\Delta\hat w_u$ $+0.124$ vs $+0.108$, Wilcoxon $p = 0.009$) even though it moved
the output *less* (0.410 vs 0.453 bits), a weak hint that top-mass disturbance is the more efficient
way to inflate the level. It does not extend to the ordering, which is the quantity a screen consumes.

Read with pattern 15, the picture is that any disturbance the model genuinely feels destroys the token
ordering, and it does not matter much which successors it rearranges. For the direction's central
question that is a mildly favourable answer: the per-token trait does not reduce to the token's
high-mass continuations, which is consistent with the embedding lookup carrying information that corpus
successor JSD does not (patterns 3 and 10). The honest caveat is that random directions are a blunt
instrument for this question — a direction built from the unembedding rows of the token's top
successors would give a much larger contrast in $S$, and that is the test that could turn this
suggestion into a result.

**17. The top-mass split is steerable only while the edit is small; at a behaviourally meaningful step
the damage is tail-weighted whatever direction delivers it.** Pattern 16 could not decide whether
steering the damage toward a token's high-mass successors matters, because random draws span too little
of $S$. Constructing the directions from the generalised eigenproblem removes that excuse, and Figure 17
shows what happens when it does.

![Predicted versus achieved top-mass share for constructed top-heavy and tail-heavy edits, and anchor width before versus after each edit](plots/mode_construct.png)

**Figure 17.** Twelve tokens, the same ones as Figures 14–16. Left: top-mass share $S$ (y, fraction of
the output change landing on the token's 32 most likely successors) for each token (x, token strings);
open markers = the $S$ the construction predicts for a small step (the generalised eigenvalues),
filled markers = the $S$ the same direction actually delivers once rescaled to 0.4 bits, joined by a
dotted line; triangles = the $S$-maximising ("top-heavy") combination, squares = the $S$-minimising
("tail-heavy") one; the gray band is the range 24 random directions span (pattern 16) and the
dash-dotted line is the probability mass those 32 successors hold before any edit (0.71). Right:
anchor width $\hat w_u$ after the edit (y) against $\hat w_u$ before it (x), triangles/solid fit =
top-heavy, squares/dashed fit = tail-heavy; dotted line = no change.

| edit rescaled to 0.4 bits (12 tokens) | predicted $S$ (small step) | achieved $S$ | output movement (bits) | mean $\hat w_u$ | sd across tokens | $\rho$(before, after) |
|---|---|---|---|---|---|---|
| *before any edit* | \- | \- | \- | *0.543* | *0.083* | \- |
| constructed top-heavy | **0.856** | 0.369 | 0.422 | 0.666 | 0.023 | $-0.16$ ($p = 0.62$) |
| constructed tail-heavy | **0.179** | 0.390 | 0.419 | 0.672 | 0.020 | $-0.28$ ($p = 0.38$) |

The construction works where it is supposed to. In the small-step regime the two combinations are
predicted to put $0.856$ and $0.179$ of the divergence on the top-32 successors — a separation of
0.68, three times the 0.21 a random draw of 24 supplies, and the top-heavy end reaches past the
mass-proportional 0.71. The instrument the previous experiment lacked exists.

It buys nothing. Rescaled to the 0.4 bits at which width actually responds, the two edits land at
$S = 0.369$ and $S = 0.390$ — indistinguishable, both far below the base mass 0.71, and both inside
the band a random direction already occupies. The paired difference is not significant and runs
*backwards* (the top-heavy construction ends marginally more tail-weighted, $p = 0.09$). Everything
downstream follows: both edits move the output by the same amount (0.422 vs 0.419 bits), both widen
every token to a common $\hat w_u \approx 0.67$, and neither leaves any of the pre-edit ordering
($\rho = -0.16$ and $-0.28$, $n = 12$).

Two things are settled by that. First, about the model: **the tail-weighting of a large embedding edit
is not a property of the direction, it is a property of the step size.** A direction chosen to hit the
top successors and nothing else does so only while the perturbation is small; grown until the model's
output moves 0.4 bits, it churns the tail like any other. The linear picture of an embedding edit
expires well before the displacement at which width responds — the same expiry that made the earlier
fixed-displacement test misleading (pattern 15). Second, about this direction's question: since $S$
cannot be held apart at a step the model feels, the tail-versus-top-mass hypothesis is not just
unsupported but **untestable by embedding edits**, and the two arms agree on the outcome that matters
— any disturbance large enough to register erases the token ordering regardless of where it lands. The
per-token trait is a property of the token's whole output map rather than of any identifiable slice of
its next-token distribution, which is the reading under which the vocabulary-wide static-embedding
lookup (pattern 10) is the right level of description: there is no smaller behavioural object to point
at.

**18. The basin picture is only weakly supported, and not in the direction the simple version
predicts.** Radius along random directions is unrelated to the token effect ($\rho = -0.02$,
$p = 0.87$): generic insensitivity of the residual stream explains nothing. Radius along anchor
directions does correlate ($\rho = +0.39$, $p = 1.1\times10^{-5}$; $+0.33$ with output entropy
partialled out) but with the *opposite* sign to the naive prediction — tokens that hold their output
distribution longer contribute *wider*, not narrower, transitions. At the pair level the radius sum
plus corpus JSD reaches only 0.299 and adds 0.012 on top of anchor width. Two of the probe's own
by-products behave the same way: output entropy $\rho = -0.30$, endpoint logit norm $\rho = -0.23$. So
"how far the state can move before the output moves in absolute terms" is not the quantity behind
width; what transfers is the *shape* measure itself.

**19. Pair-specific structure survives the additive model.** Residuals of the additive-plus-`J` model,
fitted separately in each sentence frame, correlate across frames at $\bar r = 0.67$ (variance share
0.86 of what is left) — a large amount of reproducible pair-specific structure. Adding model-output
JSD and block-0 geometry to the fit lowers that residual agreement to $\bar r = 0.54$, so the endpoint
arrangement at the interpolation site accounts for part of it and something still unmeasured accounts
for the rest.

**20. Width is not a fixed absolute transition divided by path length.** Converting `w` into
residual-stream distance units makes the distribution *more* dispersed, not less: coefficient of
variation 0.158 for `w` against 0.216 for $w_{\mathrm{abs}} = w \cdot d_0$ (median $d_0 = 24.0$). The
sign is wrong too — longer endpoint separations go with slightly *wider* transitions
($\rho(d_0, w) = +0.17$, $p = 4.2\times10^{-7}$), where the artifact story predicts narrower. Endpoint
angle carries a little independent signal ($\rho(\cos_0, w) = -0.25$, $p = 2.4\times10^{-14}$): more
nearly parallel endpoint states go with narrower transitions.

**21. Of 102 early components, only the block-0 MLP carries the trait — and it is also the only one the
model feels.** Mean-ablating each attention head and MLP in blocks 0–5 one at a time leaves the token
ordering intact in 101 cases (median $\rho = +0.99$ across components; every one of the 96 heads
$\ge +0.97$, every MLP above block 0 $\ge +0.90$). Removing the block-0 MLP instead collapses the
across-token spread from sd $0.084$ to $0.018$, lifts every token to $\hat w_u \approx 0.82$ and leaves
$\rho = -0.10$. Figure 18 shows both panels of that contrast.

| mean-ablated component (12 tokens, 6 anchors, 1 frame) | mean $\hat w_u$ | sd across tokens | $\rho$(before, after) | output movement (bits) |
|---|---|---|---|---|
| *nothing ablated* | *0.565* | *0.084* | \- | \- |
| **block-0 MLP** | **0.822** | **0.018** | $-0.10$ | **0.451** |
| MLPs of blocks 1–5, worst of the five | 0.585 | 0.091 | $+0.90$ | 0.007 |
| attention heads, worst of the 96 | 0.563 | 0.076 | $+0.97$ | 0.0004 |
| *median over all 102 components* | \- | *0.084* | *$+0.99$* | \- |

![Spread and ordering of the per-token width after mean-ablating each early component](plots/ablate.png)

**Figure 18.** Each of the 102 attention heads and MLPs in blocks 0–5 mean-ablated one at a time, for
the same 12 tokens as Figures 14–17. Left: standard deviation of $\hat w_u$ across the 12 tokens (y)
against the block containing the ablated component (x, heads jittered horizontally); open circles =
attention heads, diamonds = MLPs; dash-dotted line = the unablated spread 0.084. Centre: rank agreement
$\rho$(unablated $\hat w_u$, ablated $\hat w_u$) (y) against the output movement the ablation causes
(x, bits, log scale), same markers; dash-dotted line = perfect agreement. Right: $\hat w_u$ after the
ablation (y) against $\hat w_u$ before it (x) for the two extreme components; dotted line = no change.
Only the block-0 MLP leaves the cluster on either panel.

The negative half is the informative half for the mechanism: the trait is not spread thinly across
early attention, because no head carries a detectable share of it, and it is not re-derived layer by
layer, because no MLP above block 0 matters either. The positive half comes with a confound that
ablation alone cannot remove. The block-0 MLP moves the output by 0.451 bits, sixty times more than
any other component here and almost exactly the 0.4-bit rung at which pattern 15's displacement ladder
showed that *any* disturbance flattens the ordering. So either the block-0 MLP computes the per-token
width trait, or it is merely the only single early component large enough to reach the regime where the
trait dies. Pattern 22 separates those two readings.

**Pattern 22 — at matched output movement, the block-0 MLP destroys the ordering 3.5× more cheaply
than a random perturbation, while both flatten the level identically.** Softening the ablation into a
dose and matching each dose to an equally loud random perturbation of the same residual stream gives
the two curves of Figure 19. Over the band 0.007–0.103 bits, where the ordering is still partly alive,
the MLP arm sits below its matched control at every rung; it falls through $\rho = 0.6$ at about 0.03
bits and the control only at about 0.10.

![Rank agreement and across-token spread against output movement for the block-0 MLP dose and an output-matched random control](plots/dose.png)

**Figure 19.** Dose–response for the block-0 MLP (solid, circles: final-position output blended toward
its mean, $\alpha = 0.1 \dots 1$) against a random direction added to the same residual stream and
rescaled to move the output by the same number of bits (dashed, squares). x (both panels): output
movement $B$ in bits, log scale — the mean JSD between the perturbed and unperturbed next-token
distributions of the 12 tokens. Left y: rank agreement $\rho$ between each token's anchor width
$\hat w_u$ before and after the perturbation (1 = ordering intact, 0 = destroyed). Right y: sd of
$\hat w_u$ across the 12 tokens, with the unperturbed spread 0.084 marked (dotted). The arms separate
on the left panel and lie on top of each other on the right.

| output movement $B$ (bits) | $\rho$, block-0 MLP dose | $\rho$, matched random control | sd, MLP | sd, control |
|---|---|---|---|---|
| 0.001 | +0.97 | +0.97 | 0.076 | 0.084 |
| 0.003 | +0.92 | +0.97 | 0.071 | 0.081 |
| 0.007 | +0.84 | +0.99 | 0.070 | 0.074 |
| 0.014 | +0.64 | +0.91 | 0.069 | 0.067 |
| 0.029 | +0.62 | +0.79 | 0.055 | 0.053 |
| 0.103 | +0.25 | +0.61 | 0.027 | 0.026 |
| 0.265 | +0.74 | −0.32 | 0.021 | 0.020 |
| 0.451 | −0.10 | −0.76 | 0.018 | 0.013 |

Two things follow, and they matter for different reasons. First, the confound is broken in the block-0
MLP's favour: an ordinary disturbance of the same loudness leaves the ordering largely intact where the
MLP dose has already half-destroyed it, so the ablation's single hit is not a size effect. This is the
direction's first positive mechanistic localisation, and it says the trait is realised in one
component's contribution to the final-position residual stream — which fits the layer sweep, where the
ordering is already fixed at the input and the blocks *below* the interpolation site do the sharpening.
Second, the right-hand panel finally separates two effects the intervention experiments had been
conflating. The across-token spread collapses along an identical trajectory in both arms
(0.069/0.067, 0.055/0.053, 0.027/0.026 at matched bits): pushing the residual stream around by any
means compresses every token toward $\hat w_u \approx 0.82$, exactly as pattern 15's displacement
ladder found for embedding edits. The ordering does not behave that way. Level and ranking are separate
channels, and only the ranking singles out a component.

The caveats are the same scale as the experiment: one frame, 12 tokens, one random-control seed, and
above 0.25 bits both arms are at noise ($\rho$ from $+0.74$ to $-0.76$; with $n = 12$ a single $\rho$
has a standard error near 0.3), so the top two rungs are reported but carry no ranking information.

### Candidate hypotheses

This section is interpretation, ranked by how well each fits the evidence above.

**H1 — Transition width is a per-token trait that each token carries into any pairing.** This is the
hypothesis the evidence supports best. It predicts additivity (pattern 4), the abundance of matched
contrasts at fixed corpus JSD (pattern 2), and — the part it was tested on and passed — that a token's
width measured against strangers predicts its behaviour inside the bank (patterns 6 and 7). *What it
does not explain:* the reproducible pair-specific remainder (pattern 19), which is roughly a third of the
explainable variance, and the anchor-set dependence of $\hat w_u$ (pattern 8): the trait is real but
each measurement of it also carries a component specific to the anchors used. *Alternative reading:*
$\hat w_u$ is largely "how far this token is from a typical token" — a similarity statistic in
disguise. The swap partly adjudicates this: a pure similarity statistic would not survive replacing the
anchors with a disjoint set from a different word class, and here both replacements still recover the
fitted effect at $\rho \approx 0.6$. *Cheapest remaining experiment:* measure $\hat w_u$ with a
single anchor repeated across many frames rather than many anchors in three frames. If the trait is a
token property, the frame-averaged single-anchor measurement should track $a_u$ about as well as the
six-anchor one; if it is a relation, it should not.

**H2 — The trait is how sharply the model reads the token in context, not the token string.** Corpus
frequency ($\rho = -0.33$), continuation entropy ($-0.24$), in-frame surprisal ($+0.26$), output entropy
($-0.30$) and endpoint logit norm ($-0.23$) all point the same way: tokens the model reads confidently
and predicts sharply from pull transitions narrow. Every one of those correlations is weak, so this is
a component, not the mechanism. *Alternative reading:* all five are proxies for a function-word /
content-word split, and the operative variable is which of the model's output modes the token belongs
to. *Cheapest discriminating experiment:* hold the token fixed and change the frame so that its
surprisal moves by several bits (a frame where ` kind` is expected against one where it is not), then
re-measure $\hat w_u$. A within-token manipulation separates the trait from the string.

**H3 — The pair-specific remainder is endpoint geometry at the interpolation site.** Adding $\cos_0$
and $d_0$ raises held-out $R^2$ from 0.648 to 0.723 and cuts the reproducible residual agreement from
0.67 to 0.54, so the arrangement of the two endpoint states carries pair-specific information beyond
both corpus JSD and the model's output separation. It is not a path-length normalisation artifact
(pattern 19). *Alternative reading:* $\cos_0$ and $d_0$ may be re-expressing corpus JSD in geometric
clothing, though they add signal on top of `J`, which argues against a pure proxy. *Cheapest
discriminating experiment:* hold the token pair fixed and change only the path — interpolate through a
third state, or use linear interpolation — and see whether `w` tracks the geometry of the new path.

The basin picture we set out to test — a per-token region of output insensitivity whose size sets the
width — is **not** supported in its simple form (pattern 18), and we have dropped it.

### Recommended next experiment

The screen generalises across tokens (pattern 7), across the vocabulary (pattern 11), across contexts
(pattern 12) and down to a free lookup (pattern 10). Three mechanistic stories have now failed — the
basin of output insensitivity (pattern 18), the probe direction as a lever (pattern 13), and any single
embedding direction as a lever (pattern 14). What replaced them is the compression result, and pattern
15 has now pinned down what drives it: at a fixed displacement, an edit the model barely feels keeps
the token ordering ($\rho = +0.94$ at 0.049 bits) and an edit of the same size that it feels strongly
does not ($\rho = +0.08$ at 0.402 bits). The trait is behavioural, so the search should move from
directions in embedding space to **which part of the token's output behaviour carries it**.

Patterns 16 and 17 closed that line of attack. The damage a large edit does is tail-weighted — 0.389
of the divergence on successors holding 0.71 of the mass — and it stays tail-weighted even when the
edit is built to do the opposite: constructed directions predicted to split $S$ 0.86 against 0.18 both
land at $S \approx 0.38$ once they are grown to the 0.4 bits at which width responds, and both erase
the token ordering. Embedding edits cannot hold the top-mass share apart at a behaviourally meaningful
size, so no further variant of this experiment will separate the two arms.

Pattern 21 moved the intervention out of embedding space and into the computation, and returned a
single candidate: of 102 attention heads and MLPs in blocks 0–5, only the block-0 MLP destroys the
per-token ordering. It also returned a single confound, because that component is the only one whose
removal the model registers at all (0.451 bits against $\le 0.007$ for every other), and 0.4 bits is
precisely where pattern 15 showed any disturbance flattens the ordering.

Pattern 22 removed that confound: at matched output movement the MLP dose destroys the ordering about
3.5× more cheaply than a random perturbation of the same residual stream, while the two arms flatten
the across-token spread identically. The mechanism now has a location, and every intervention so far
has been destructive.

The next experiment should therefore **read the block-0 MLP's contribution rather than break it**, and
test whether that vector is *sufficient* as well as necessary. Two halves, sharing one forward pass over
the 123 endpoint tokens. First, fit a ridge probe from the block-0 MLP's final-position output $m_u$ to
the measured anchor width $\hat w_u$, under the same held-out protocol and shuffled-target control as
the embedding probe. Second, transplant $m_u$ from a narrow token onto a wide token's forward pass,
leaving everything else untouched, and re-measure the recipient's width. If the probe beats the
embedding probe's $\rho = +0.76$ and the transplant moves the recipient toward the donor, the trait is
carried by a readable vector and the mechanism is settled at the level of a feature rather than a
component. If the probe adds nothing over the embedding and the transplant does nothing, the block-0
MLP is a necessary stage rather than the place the number is stored, and the free static-embedding
lookup (patterns 10 and 11) remains the practical deliverable. Cost: 123 forwards plus roughly 12
transplant measurements, below pattern 22's budget.

---

## Conclusion

Corpus successor JSD predicts transition width on average but explains only 0.149 of the held-out
variance, against 0.934 that is reproducible. Most of the gap is a **per-token additive effect**: one
number per token raises held-out $R^2$ to 0.578, and that per-token term outperforms every pair-level
predictor we tested, including the model's own endpoint output difference. Width is mostly something
each token brings with it, and it is measurable — a token's width against six anchor tokens it was
never paired with predicts its fitted effect at $\rho = 0.70$, and two such measured numbers do the
work of 123 fitted ones. Used as a screen on 40 tokens the analysis had never seen, it predicts the
width of their 718 pairs at $R^2 = 0.397$ with nothing fitted on them. That makes the per-token screen
the practical unit for finding sharp transitions: score tokens once, and the pairs follow.

The sharpening itself happens below the interpolation site: repeating the per-token measurement at
blocks 6, 12 and 18 keeps the token ranking ($\rho = +0.72$ with block 0 even at block 18) while the
median width climbs to 0.800 and the spread across tokens collapses fivefold. Which tokens are narrow
is decided early; how narrow anything gets is decided by the remaining depth.

How early is now quantified: the ranking is already present in the static embedding. A ridge probe on
the embedding row predicts a held-out token's measured width at $\rho = +0.76$, beating an
embedding-norm baseline of $+0.60$, and a screen built from embeddings alone — no forward pass at any
point — predicts the widths of 718 unseen pairs at $R^2 = 0.213$ and $\rho = +0.53$. Measuring the
widths remains the more accurate route (0.397), so the practical reading is a two-tier screen: a free
vocabulary-wide table for triage, and 18 interpolation curves per token when a sharper number is
needed. The table is not confined to the tokens the analysis was built on: on 32 tokens drawn from
outside the curated pool — subword fragments, punctuation, numerals, capitalised names and rarer words
— predicted and measured widths still correlate at $\rho = +0.60$, and the widths measured there cover
the pool's own range. Nor is it confined to the context it was built in: measured in a mid-sentence
continuation, an interrogative, a colon-list and a code prefix, the token ranking holds at $\rho$ from
$+0.84$ down to $+0.50$, against $+0.82$ for two frames of the original shape. The level of `w` is a
different matter — its median moves from 0.53 to 0.71 across those contexts — so the ordering travels
and the calibration does not.

Three things qualify this. The measurement is anchor-dependent: two disjoint anchor sets rank the
tokens at $\rho = 0.46$ while each still recovers the fitted effect at $\rho \approx 0.6$, so the trait
is real but the anchor set is part of the method. A pair-specific remainder survives the additive model
and is partly endpoint geometry. And the mechanism we set out to test — a basin of output insensitivity around each
token whose size sets the width — does not hold up: radius along random directions is unrelated to the
token effect, and along anchor directions it points the wrong way. What each token carries is the shape
of its transition, and we cannot yet say what in the network produces it. Editing a token's embedding
along the probe's own direction, by a step the probe says should change width by 0.05, moves the
measured width by 0.003 with no consistent sign — but that edit shifts the model's output by only
0.0001 bits, so it says nothing on its own. Re-running the edit with the step grown until the model's
output moves 0.05, 0.1 or 0.2 bits changes the picture and gives the direction its clearest statement
about the trait's nature: width then moves by 0.10–0.15, fifty times more, yet a random direction
matched on output movement moves it just as much (0.123 against 0.127, $p = 0.47$), and **all 144 edits
widen** although the probe predicts opposite signs for opposite steps. What the edits do is compress the
trait: whatever a token's width was, after a 0.2-bit edit it sits near 0.68, with the spread across
tokens falling from 0.083 to 0.02, and the tokens that started narrowest moving furthest. A ladder of
displacements, with the quietest and loudest of 24 directions rebuilt at each rung from their measured
effect on the token's output, then separates the two things that grew together in that edit. The level
follows the displacement — every direction raises the mean width — but the *ordering* follows the
behaviour: at a displacement of norm 1.8 the quiet direction moves the output by 0.049 bits and keeps
the ordering at $\rho = +0.94$, while the loud direction at the identical norm moves it by 0.402 bits
and drops to $\rho = +0.08$. Narrow transitions are fragile, but they are a behavioural property, not a
positional one: an edit the model does not feel does not take them away. The deflationary explanation,
that narrow transitions are just fixed-size transitions on long paths, is refuted.

**Limitations.** One model, one hook point (after block 0) and one checkpoint. The pair bank, the
fitted token effects and both screens rest on three sentence frames of a single shape, so
frame-specific effects there are common to both sides of the transfer test; the frame-shape control
addresses this for the per-token measurement (the ranking survives four other context shapes) but not
for the pair-level results, which were never re-run in a new context.
Anchor widths are measured against six anchors at a time and the value depends on which six, so
$\hat w_u$ should be read as "width against this anchor set", not as an absolute constant. The forward
screen covers 40 new tokens in the same three frames and the same model; it says nothing yet about
other models or contexts, and the layer sweep covers five sites in this one model. The pair bank, the
anchors and the probe's training tokens all come from `dir18`'s pool of common single-token alphabetic
words; the vocabulary-wide test extends the check to fragments, punctuation, numerals and capitalised
names, but with only eight tokens per class, so the per-class correlations there are indicative rather
than established. At block 18 the
interquartile range of $\hat w_u$ is 0.02, so the correlations reported there are attenuated by the
small dynamic range and should not be read as evidence that the trait has disappeared. Both
interventions are small: 16 and 12 tokens with one random direction each, so the compression result
(patterns 14 and 15) rests on 12 tokens and its correlations — including the ordering after the edit —
carry wide intervals. The ladder's quiet direction is the quietest of 24 random draws, not the quietest
direction that exists, so it bounds how much of the trait a behaviour-preserving edit can keep from
below; and at norm 1.8 it still moves the output by 0.049 bits, so "quiet" there means 8× quieter than
the loud direction, not silent. `w` describes movement along the $z_u \to z_v$ direction only, so a pair whose logits
move sideways would be scored as flat. The 0.2-bit movement gate is a judgement call: it keeps 929 of
1,000 pairs, and the headline correlation is reported both with and without it.
