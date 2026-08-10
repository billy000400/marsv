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

Moving the interpolation site down the network shows where the effect comes from. The ranking of
tokens is stable — anchor widths measured at block 18 still agree with the block-0 ranking at
$\rho = +0.72$ — but the transitions themselves flatten out: the median anchor width rises from 0.553
at block 0 to 0.800 at block 18, exactly the value of a perfectly proportional response, and the spread
across tokens shrinks fivefold. Which token is narrow is settled early; how sharp any transition gets
depends on how much of the network still lies below the site.

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
18) — about 900,000 forward passes in total, roughly two hours on one GPU shared with three other jobs.

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

**10. The basin picture is only weakly supported, and not in the direction the simple version
predicts.** Radius along random directions is unrelated to the token effect ($\rho = -0.02$,
$p = 0.87$): generic insensitivity of the residual stream explains nothing. Radius along anchor
directions does correlate ($\rho = +0.39$, $p = 1.1\times10^{-5}$; $+0.33$ with output entropy
partialled out) but with the *opposite* sign to the naive prediction — tokens that hold their output
distribution longer contribute *wider*, not narrower, transitions. At the pair level the radius sum
plus corpus JSD reaches only 0.299 and adds 0.012 on top of anchor width. Two of the probe's own
by-products behave the same way: output entropy $\rho = -0.30$, endpoint logit norm $\rho = -0.23$. So
"how far the state can move before the output moves in absolute terms" is not the quantity behind
width; what transfers is the *shape* measure itself.

**11. Pair-specific structure survives the additive model.** Residuals of the additive-plus-`J` model,
fitted separately in each sentence frame, correlate across frames at $\bar r = 0.67$ (variance share
0.86 of what is left) — a large amount of reproducible pair-specific structure. Adding model-output
JSD and block-0 geometry to the fit lowers that residual agreement to $\bar r = 0.54$, so the endpoint
arrangement at the interpolation site accounts for part of it and something still unmeasured accounts
for the rest.

**12. Width is not a fixed absolute transition divided by path length.** Converting `w` into
residual-stream distance units makes the distribution *more* dispersed, not less: coefficient of
variation 0.158 for `w` against 0.216 for $w_{\mathrm{abs}} = w \cdot d_0$ (median $d_0 = 24.0$). The
sign is wrong too — longer endpoint separations go with slightly *wider* transitions
($\rho(d_0, w) = +0.17$, $p = 4.2\times10^{-7}$), where the artifact story predicts narrower. Endpoint
angle carries a little independent signal ($\rho(\cos_0, w) = -0.25$, $p = 2.4\times10^{-14}$): more
nearly parallel endpoint states go with narrower transitions.

### Candidate hypotheses

This section is interpretation, ranked by how well each fits the evidence above.

**H1 — Transition width is a per-token trait that each token carries into any pairing.** This is the
hypothesis the evidence supports best. It predicts additivity (pattern 4), the abundance of matched
contrasts at fixed corpus JSD (pattern 2), and — the part it was tested on and passed — that a token's
width measured against strangers predicts its behaviour inside the bank (patterns 6 and 7). *What it
does not explain:* the reproducible pair-specific remainder (pattern 11), which is roughly a third of the
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
(pattern 12). *Alternative reading:* $\cos_0$ and $d_0$ may be re-expressing corpus JSD in geometric
clothing, though they add signal on top of `J`, which argues against a pure proxy. *Cheapest
discriminating experiment:* hold the token pair fixed and change only the path — interpolate through a
third state, or use linear interpolation — and see whether `w` tracks the geometry of the new path.

The basin picture we set out to test — a per-token region of output insensitivity whose size sets the
width — is **not** supported in its simple form (pattern 10), and we have dropped it.

### Recommended next experiment

The layer sweep (pattern 9) says the token ranking is present at block 0 and the sharpening is done
below the site, which makes the next question **whether the trait is in the embedding**. Measure anchor
widths with the interpolation site at the *input* embedding, before block 0, and — the decisive half —
predict $\hat w_u$ for a token from its static embedding alone, using a simple probe fitted on 80 of
the 123 tokens and tested on the rest. If a linear probe on the embedding recovers the ranking, the
screen becomes free at inference time: no interpolation, no forward passes, just a lookup, which is the
form an auditor actually wants. If it does not, the trait lives in how the frame and the token combine,
and the search moves to the attention pattern at block 0. The experiment costs one anchor-width run
plus a trivial probe fit, under half an hour.

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

Three things qualify this. The measurement is anchor-dependent: two disjoint anchor sets rank the
tokens at $\rho = 0.46$ while each still recovers the fitted effect at $\rho \approx 0.6$, so the trait
is real but the anchor set is part of the method. A pair-specific remainder survives the additive model
and is partly endpoint geometry. And the mechanism we set out to test — a basin of output insensitivity around each
token whose size sets the width — does not hold up: radius along random directions is unrelated to the
token effect, and along anchor directions it points the wrong way. What each token carries is the shape
of its transition, and we cannot yet say what in the network produces it. The deflationary explanation,
that narrow transitions are just fixed-size transitions on long paths, is refuted.

**Limitations.** One model, one hook point (after block 0), one checkpoint, and three sentence frames,
shared by every measurement — so frame-specific effects are common to both sides of the transfer test.
Anchor widths are measured against six anchors at a time and the value depends on which six, so
$\hat w_u$ should be read as "width against this anchor set", not as an absolute constant. The forward
screen covers 40 new tokens in the same three frames and the same model; it says nothing yet about
other models or contexts, and the layer sweep covers four sites in this one model. At block 18 the
interquartile range of $\hat w_u$ is 0.02, so the correlations reported there are attenuated by the
small dynamic range and should not be read as evidence that the trait has disappeared. `w` describes movement along the $z_u \to z_v$ direction only, so a pair whose logits
move sideways would be scored as flat. The 0.2-bit movement gate is a judgement call: it keeps 929 of
1,000 pairs, and the headline correlation is reported both with and without it.
