# Do language models pass through a stable *third* prediction when you interpolate between two activations?

> Final, presentable, current-best only (no history — see CHANGELOG.md).

## Summary

Take two random 32-token passages of held-out English, read GPT-2 Large's residual stream at an early
block, walk along the line between the two activations, and put each intermediate point back into the
model. **About one in six such paths (16.9%, 95% CI [16.1%, 17.8%] of 7,611 eligible paths) shows a
persistent third next-token prediction** `C` — a token that is neither endpoint's prediction, stays
top-1 for at least 3 of the 50 interpolation steps, and beats both endpoint tokens at every one of
them. The rate reproduces at 17.7% on a completely disjoint bank of 300 pairs with the rule frozen and
untouched, is unchanged under plain linear interpolation (16.1%), and is exactly 0% on self-pairs, so
it is neither a detector artefact nor a property of the spherical interpolation.

But *what* those third regions are matters more than *how often* they occur, and here the answer is
sobering. The typical third region is **weak**: its distribution is flatter than at the endpoints
(6.97 vs 5.70 bits of entropy), its token is usually a frequency default such as `' the'` or `'.'`
(32% of C tokens are among the ten commonest endpoint predictions), it occupies only 3–5 of 50 grid
points, it sits **further** from real activations than the endpoint-region points do (median cosine
distance to the nearest of 2,000 natural activations 0.160 vs 0.086 for a natural context), and its
natural neighbours rarely predict `C` (4.5% vs 14.1% for a natural query). It also appears at 11.1%
between contexts that *agree* on the next token, so much of it is not a contest between A and B at
all. A minority is different: 3.7% of eligible paths are exactly `A, C, B` with no other transient
token, 3.6% of candidates hold a dominance margin above 0.2, and the top-ranked candidates generate
fluent, reproducible continuations from inside the third region.

Asking the same question **geometrically** — is the third region a *plateau*, a flat shelf of the
model's output as you walk along the path? — sharpens that split. Measuring how far the output moves
inside the third region relative to the no-plateau diagonal (flatness ρ, defined in Methods), the
median candidate scores ρ = 2.05: the output is racing through the boundary while the third token
happens to be on top. Only **8.2% of candidates (1.39% of eligible paths, CI [1.15%, 1.68%], about 1
path in 72) are flat enough (ρ < 0.5) to be true sub-plateaus** — but those are textbook staircases,
they sit squarely between the two endpoint outputs, and the candidate score frozen before any curve
was drawn ranks them at the top (median ρ falls from 2.65 to 0.93 across score deciles).

**Verdict: a robust but mostly fragile third output region.** Language models do show the MNIST-style
`A → C → B` behaviour far too often for it to be a curiosity, but at random, in-distribution pairs it
is usually a low-confidence, off-manifold, generic-token band rather than a crisp third state. Anyone
who steers or edits activations along such lines should expect to pass through these bands, and should
not read a third top-1 token as evidence of a meaningful intermediate concept without checking its
margin, entropy and neighbourhood.

## Methods

### Why this experiment exists

When you take two inputs a network handles confidently and walk along a straight-ish line between
their internal activations, the output usually flips once: class A, then class B. In earlier MNIST
work in this project the path sometimes did something else — it left A, settled into a *third* class
C for a stretch, and only then moved to B (`A → C → B`). If large language models do the same thing,
then the space between two ordinary activations contains extra, stable model states that no input in
the pair points at. That matters for safety-relevant interpretability: activation steering,
representation editing and model-diffing all move activations along such lines, and a hidden stable
region is a behaviour you did not ask for and cannot see from the endpoints.

Previous LLM experiments in this project used a handful of hand-picked prompt pairs, so they could
easily have missed (or manufactured) the phenomenon. This direction runs an **unbiased random
screen**: pair up random held-out natural-text contexts, interpolate, and count how often the
downstream model shows a persistent third top-1 token.

### Data & Model

**Model.** GPT-2 Large (774M parameters, 36 transformer blocks, model width 1280), HuggingFace
`gpt2-large`, revision `main`, float32, single GPU. No fine-tuning, no training of any kind.

**Data.** The raw validation split of **WikiText-103** (`Salesforce/wikitext`,
`wikitext-103-raw-v1`, split `validation`). The split's non-empty lines are concatenated in dataset
order, tokenized with the GPT-2 tokenizer, and cut into **non-overlapping windows of exactly 32
tokens**. A window is one *context*. 5,980 windows were built.

**Hook point.** For each context we read the residual stream *after* transformer block `L` at the
**final token position** (`resid_post`, the block's output hidden state), for the preregistered early
blocks **L ∈ {0, 2, 4, 6}**. Blocks **12, 18, 24 and 30** are also screened, but only as a
clearly separated exploratory depth sweep (Section 9) that feeds no headline number.

**Banks.** One shuffle (seed 0) splits the 5,980 windows into three **disjoint** pools, frozen
before any interpolation curve was computed:

| bank | contents | use |
|---|---|---|
| primary | 1,000 random pairs | the prevalence estimate |
| validation | 300 random pairs | confirming the frozen rule without retuning |
| reference | 2,000 single contexts | the natural-activation neighbour bank (S5) |

Within a pool, consecutive shuffled windows are paired without replacement. The **only** rejection
rule is that the two contexts must not have the same unpatched top-1 next-token prediction (30 pairs
rejected in the primary pool, 8 in the validation pool). Nothing is filtered on semantics,
confidence, activation distance, or on anything visible in the interpolation.

### What we compute along a path

**Interpolation (`slerp_rescale`).** Given the two contexts' final-position activations $h_A$ and
$h_B$ at block $L$, we interpolate their *directions* on the sphere and their *lengths* linearly —
the same rule used in this project's other LLM directions, because plain linear interpolation
shrinks the vector norm in the middle and can make midpoints atypically small:

```math
h(\alpha)=\Big[(1-\alpha)\lVert h_A\rVert+\alpha\lVert h_B\rVert\Big]\cdot
\frac{\sin\big((1-\alpha)\theta\big)\,\hat h_A+\sin(\alpha\theta)\,\hat h_B}{\sin\theta},
\qquad \theta=\arccos\big(\hat h_A\cdot\hat h_B\big)
```

with $\hat h=h/\lVert h\rVert$ and 50 evenly spaced $\alpha\in[0,1]$ including both endpoints. Plain
linear interpolation $h(\alpha)=(1-\alpha)h_A+\alpha h_B$ is run as a separate geometry control.

**Patching.** $h(\alpha)$ replaces the final-position residual stream at block $L$ of a *conditioning
context*, and the model then runs every remaining block and the LM head normally. Because the two
contexts are different token sequences, the downstream computation must attend to *some* context's
earlier positions. We therefore run **every pair twice**: once conditioned on context A's tokens and
once on context B's. A *path* is one (pair, block, conditioning context) triple, so the primary bank
gives 1,000 × 4 × 2 = 8,000 paths.

**Path orientation.** Along a path, $A$ is defined as the model's top-1 next token at $\alpha=0$ and
$B$ as its top-1 at $\alpha=1$, *in that conditioning context*. A path is **eligible** only if
$A\neq B$. Under its own context, the $\alpha=0$ (or $\alpha=1$) end is an exact no-op — this is the
endpoint-fidelity check below.

At every $\alpha$ we compute the complete 50,257-dimensional next-token distribution $p_\alpha$ and
retain: the top-1 token and its probability, the probability trajectory of *every* token that is
top-1 somewhere on the path, the predictive entropy, and the divergence between neighbouring grid
points. All 50,257 tokens are therefore scored at every point, but storing the raw distributions for
all 12,400 screened paths would need ≈125 GB, so **only these summaries are retained**; they are
sufficient to recompute every classification and every number in this report.

**Predictive entropy** says how sharp the prediction is at a given point on the path; a "third
region" made only of a flat, unsure distribution would be far less interesting than a confident one:

```math
H(\alpha)=-\sum_{v} p_\alpha(v)\,\log_2 p_\alpha(v)\quad\text{[bits]}
```

**Adjacent-alpha Jensen–Shannon divergence (JSD)** measures how much the *whole* distribution changes
between two neighbouring grid points. We need it because a top-1 label flip can be a numerical
near-tie between two tokens that barely moved; a real transition shows up as an actual change of the
distribution. JSD is a symmetric, bounded (0–1 bit) version of KL divergence:

```math
\mathrm{JSD}(p\Vert q)=\tfrac12 D_{\mathrm{KL}}\!\left(p\,\middle\Vert\,\tfrac{p+q}{2}\right)
+\tfrac12 D_{\mathrm{KL}}\!\left(q\,\middle\Vert\,\tfrac{p+q}{2}\right)
```

Here $p=p_{\alpha_k}$, $q=p_{\alpha_{k+1}}$. Higher means the prediction genuinely moved.

### The frozen `A | C | B` rule

All five conditions and every threshold below were fixed before any interpolation curve or token
string was looked at. Let $k=0\ldots49$ index the alpha grid and let the top-1 sequence be
run-length encoded into maximal runs. A path is an **`A|C|B` candidate** when:

1. the first run is $A$ and the last run is $B$ (endpoints reproduce, by construction);
2. some intermediate maximal run has top-1 token $C\notin\lbrace A,B\rbrace$, appearing after the $A$ run and
   before the $B$ run;
3. **persistence** — that $C$ run is at least `MIN_RUN = 3` consecutive grid points long;
4. **dominance margin** — $C$ beats *both* endpoint tokens everywhere inside its run, i.e. the
   minimum margin is strictly positive;
5. **two distinct transitions** — the JSD at both boundaries of the $C$ run exceeds
   `JSD_FLOOR = 0.005` bits, so entry and exit are real distribution changes rather than one A→B
   crossing with an isolated tie.

The **minimum dominance margin** is the metric behind condition 4. It answers "is C actually winning,
or is it a coin-flip at the top of a flat distribution?", and it is reported for every candidate:

```math
m=\min_{k\in[k_{\text{in}},\,k_{\text{out}}]}\Big[p_{\alpha_k}(C)-\max\big(p_{\alpha_k}(A),\,p_{\alpha_k}(B)\big)\Big]
```

The **C-segment width** is the fraction of the alpha grid the third region occupies, and the
**transition separation** is the alpha gap between the entry and exit crossings:

```math
w=\frac{k_{\text{out}}-k_{\text{in}}+1}{50},\qquad
s=\frac{\alpha_{k_{\text{out}}}+\alpha_{k_{\text{out}}+1}}{2}-\frac{\alpha_{k_{\text{in}}-1}+\alpha_{k_{\text{in}}}}{2}
```

The **candidate score** ranks candidates. It is deliberately the simplest combination of "how wide"
and "how dominant" — an area of dominance in (probability × alpha) units — and was frozen before any
token string was decoded:

```math
\mathrm{score}=m\cdot s
```

A candidate is **clean** when the compressed top-1 path is exactly `A, C, B` (three runs), and
**complex with a persistent C region** otherwise; the two are counted separately and never merged.

**Prevalence** is reported per eligible path and per pair (a pair counts once if any of its 8 paths
qualifies), with 95% **Wilson** binomial confidence intervals, which stay inside $[0,1]$ and remain
usable for small counts:

```math
\mathrm{CI}=\frac{\hat p+\frac{z^2}{2n}\pm z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}},\qquad z=1.96
```

### Baselines and controls

Each control answers "could this rate have come from something other than a genuine third region?"

**Endpoint fidelity** — with the patch applied at $\alpha=0$ under its own context, the patched
logits must equal the unpatched logits. Reported as $\max_v|\Delta \text{logit}_v|$; it is a
correctness check on the patching code, not a scientific result.

**Endpoint transfer** — the *foreign* end of a path (e.g. $h_B$ inserted into context A) need not
reproduce context B's own prediction. The fraction of paths where it does is reported, and the
prevalence is recomputed on that transfer-consistent subset.

**Same-prediction pairs** — pairs of *different* contexts drawn from the reference pool that share
the same unpatched top-1 token. These are outside the primary denominator. Under the frozen rule they
show how often a third region appears when the two endpoints do not disagree at all.

**Self-pairs** — a context paired with itself: every $h(\alpha)$ is identical, so no path may be
eligible and none may be a candidate. A non-zero rate here would mean the detector is broken.

**Linear interpolation** — the same pairs and rule with $h(\alpha)=(1-\alpha)h_A+\alpha h_B$, showing
whether the phenomenon is an artefact of the spherical geometry.

**Threshold sensitivity** — the whole screen is re-scored at persistence 2, 3 and 5 grid points and
at minimum-margin floors 0, 0.02 and 0.05. The headline rate always uses the frozen default
(persistence 3, margin > 0).

**Validation bank** — the 300 disjoint pairs are screened with the identical, unmodified rule.

**Determinism and batching** — 20 primary paths are re-run from scratch in a different batch layout
and compared with the stored summaries, and 32 contexts are run one at a time and as a single batch of
32. Every context is exactly 32 tokens, so no padding or attention masking is ever involved.

### Do C regions behave like real model states? (S4, S5)

Passing the rule only means a third token is top-1 for a stretch. Two follow-ups ask whether that
region behaves like a *state* of the model.

**Continuations (S4).** For a frozen inspection set — the 3 top-scoring candidates plus 3 drawn at
random (seed 7) from the rest — we patch the centre of the A run, of the C run and of the B run and
decode 20 tokens greedily (primary, deterministic view) and at temperature 0.8 under 3 fixed seeds
(secondary view). Unpatched continuations of both endpoint contexts are the control, and the same C
activation is also run under the *other* endpoint's context. Reproducibility inside the C region is
measured as the **common greedy prefix length**: the number of leading tokens that are identical
across continuations decoded at the first, middle and last alpha of the C run (0–20; higher means the
C region behaves the same way throughout, not just at one grid point).

**Nearest natural activations (S5).** If a C region were an ordinary place in activation space, its
points should sit near real activations that also predict C. For every candidate path we take the
C-run centre activation and search the 2,000 reference-bank activations at the same block (exact
search, cosine similarity). Two numbers are reported per query:

```math
d_{\mathrm{NN}}=1-\max_{j}\ \frac{h^\top r_j}{\lVert h\rVert\,\lVert r_j\rVert}
```

the cosine distance to the nearest natural activation (lower = the query sits where natural
activations sit), and

```math
\mathrm{agree}@10=\frac{1}{10}\sum_{j\in\mathcal{N}_{10}(h)}\mathbf{1}\big[\text{top-1}(r_j)=t\big]
```

the fraction of the 10 nearest natural activations whose own unpatched top-1 next token equals the
query's top-1 token $t$ ($t=C$ for a C-region query). Controls: the A-region and B-region centre
points of the same paths (distance-matched in the sense that they lie on the same path), and 400
natural reference activations used as queries with themselves excluded — the level a genuinely
natural point achieves.

### Is the third region a *plateau*? Matthew-style output geometry

The frozen rule above is a statement about **labels**: which token happens to be top-1. Matthew
Shinkle & StefanHex's *Activation Plateaus: Where and How They Emerge* asks a **geometric** question
instead — as you walk from one activation to the other, how does the model's *output vector* move?
Their signature curve is the output's normalized distance from endpoint A, and a **plateau** is a flat
stretch of it. Under that lens, the MNIST-style `A → C → B` picture predicts a *staircase*: flat near
A, sharp boundary, **flat shelf at an intermediate height** (the sub-plateau), sharp boundary, flat
near B. We therefore re-ran every candidate path and a matched control set, keeping the full
final-position logit vector at all 50 alphas.

**Relative output distance $d(t)$** — *does the output at this point look like A, like B, or does it
sit somewhere in between?* Raw logit distances are not comparable across pairs (endpoint separations
vary a lot), so we use Matthew's normalized form, with $x(\alpha)$ the final-position logit vector at
interpolation coefficient $\alpha$ (written $t$ on the figures) and $x_A=x(0)$, $x_B=x(1)$:

```math
d(\alpha)=\frac{\lVert x(\alpha)-x_A\rVert_2}{\lVert x(\alpha)-x_A\rVert_2+\lVert x(\alpha)-x_B\rVert_2}
```

Read it as: $d\approx 0$ means "the output still looks like A", $d\approx 1$ "like B", and
$d\approx 0.5$ "halfway between the two". By construction $d(0)=0$ and $d(1)=1$. A model with no
plateau at all traces the diagonal $d=\alpha$; a plateau–boundary–plateau response hugs 0, jumps, and
hugs 1. This is the y-axis of the two example figures below.

**Flatness $\rho$ of the C window** — *is the third-token run a shelf, or is it simply the middle of
one boundary?* A third token can be top-1 for several grid points while the output is still racing
from A to B; that is not a plateau. So we compare how far $d$ travels inside the C run with how wide
that run is, using the diagonal as the unit of comparison:

```math
\rho=\frac{\max_{k\in[k_{\text{in}},k_{\text{out}}]}d(\alpha_k)-\min_{k\in[k_{\text{in}},k_{\text{out}}]}d(\alpha_k)}{\alpha_{k_{\text{out}}}-\alpha_{k_{\text{in}}}}
```

$\rho=1$ means the output moves through the C run exactly as fast as the no-plateau diagonal;
$\rho<1$ means it is **flatter than the diagonal** — a genuine sub-plateau; $\rho\ll 1$ is a flat
shelf; $\rho>1$ means the C run sits *inside* the transition, where the output is moving faster than
average. Lower is more plateau-like. We also report the mean of $d$ over the C run, $\bar d_C$, which
says *how high* the shelf sits between the endpoints.

**Matched non-candidate control** — $\rho$ depends on the width and position of the window, so a bare
number means nothing. For each of the 1,290 candidates we drew (seed 13, without replacement) one
eligible **non**-candidate path — one with no persistent third token at all — and scored $\rho$ on the
*same* alpha window $[k_{\text{in}},k_{\text{out}}]$. This is the "an arbitrary stretch of an ordinary
A→B path" baseline that the C runs must beat to count as plateaus.

**Transition width $w_{10\to 90}$** — Matthew's sharpness summary for a whole path, the fraction of the
path over which $d$ climbs from 0.1 to 0.9, read off the raw curve by linear interpolation:

```math
w_{10\rightarrow 90}=\alpha(d=0.9)-\alpha(d=0.1)
```

Smaller is sharper. One clean boundary gives a small $w$; a staircase with two boundaries and a shelf
between them necessarily gives a large one, so we use it to check that candidate paths really are
two-step objects rather than single sigmoids.

Because this analysis was added after the primary screen (in response to operator feedback), no
threshold in it feeds the headline prevalence. The $\rho<0.5$ line used to pick the illustration
gallery is **descriptive and post hoc**, and is labelled as such wherever it appears.

## Results

### 1. How often does a third token appear? (primary screen)

Of 8,000 paths, 7,611 (95.1%) are eligible — the top-1 prediction at $\alpha=0$ differs from the one
at $\alpha=1$. **1,290 of them (16.9%, CI [16.1%, 17.8%]) satisfy the frozen `A|C|B` rule.** Counted
per pair (a pair qualifies if any of its 8 paths does), 610 of 991 eligible pairs (61.6%,
CI [58.5%, 64.5%]) show the behaviour somewhere — but that number rewards testing 8 paths per pair, so
the path-level rate is the honest headline.

Only **283 candidates (21.9%) are clean** — the compressed top-1 sequence is exactly `A, C, B`. The
other 1,007 have additional transient top-1 tokens somewhere on the path and are reported separately
as *complex with a persistent C region*; they are never relabelled as clean `A|C|B`.

Patching is exact where it must be: at the end of the path whose activation belongs to the
conditioning context, the patched logits match the unpatched logits to
$\max_v|\Delta\text{logit}_v| = 1.5\times10^{-5}$ and the top-1 token is reproduced on 100% of paths.

To show where the effect comes from and that it is not an artefact of the machinery, we plot the rate
per block, per conditioning context, and for every control:

![A|C|B rate per eligible path by interpolation block (left; blocks 0/2/4/6 of GPT-2 Large) and by conditioning context and control condition (right). Bars are rates per eligible path; error bars are 95% Wilson intervals. "self pairs" is exactly zero because a constant path has no eligible endpoints.](plots/candidate_prevalence_by_layer.png)

Two things stand out. First, the rate **rises monotonically with depth of the interpolated block** —
8.2% at block 0, 15.4% at block 2, 16.4% at block 4, 27.7% at block 6 — so the later the injection,
the more room the residual stream has to hold a third state that the remaining blocks then read out.
Second, conditioning on context A (17.2%) and on context B (16.7%) give the same answer, so the effect
is not an artefact of which sequence supplies the attention keys and values.

**Controls.** Self-pairs produce **0 eligible paths and 0 candidates**, as they must (every
interpolant is the same vector), confirming the detector cannot fire on a constant path. Linear
interpolation gives 16.1% (CI [14.5%, 17.8%]), statistically indistinguishable from `slerp_rescale`'s
16.9%, so the phenomenon is not created by the spherical geometry. Same-prediction pairs — different
contexts that agree on the next token — still yield **11.1%** (CI [9.5%, 12.9%]): two thirds of the
effect survives when there is no A-versus-B contest at all, which already argues that much of what the
rule catches is "interpolation wanders through a generic token", not "the model holds a third
opinion". Finally, the *foreign* end of a path reproduces its home context's own prediction on only
17.6% of paths (early-block activations are largely overwritten by the conditioning context); on that
transfer-consistent subset the rate is 14.0% (CI [12.3%, 15.9%]), close to the headline.

**Determinism and batching.** Re-running 20 paths in a different batch layout reproduces the top-1
token at all 20 × 50 grid points exactly, with a maximum top-1-probability deviation of
$2.7\times10^{-6}$ (float32 kernels are not bit-identical across batch shapes). Running 32 contexts
singly versus as one batch changes no top-1 prediction and moves logits by at most
$4.2\times10^{-5}$. No padding is involved anywhere, since every context is exactly 32 tokens.

### 2. Does it replicate on untouched data?

The 300-pair validation bank, disjoint from the primary bank at the context level, was screened with
the identical rule and no retuning: **401 of 2,261 eligible paths, 17.7% (CI [16.2%, 19.4%])**,
overlapping the primary interval. The rule transfers.

### 3. What do these third regions look like?

A single hand-picked example would be misleading, so we show the three top-scoring candidates
alongside three drawn at random from the qualifying set:

![Next-token probability of the A, C and B tokens versus the interpolation coefficient alpha, for the three top-scoring candidates (upper row) and three randomly drawn candidates (lower row). Solid/circles = A, dashed/squares = C, dash-dot/triangles = B. The grey band marks the detected C run; the dotted grey curve on the right axis is the Jensen-Shannon divergence between neighbouring alphas, in bits.](plots/top_candidate_probability_paths.png)

The contrast is the result. The top-ranked paths look exactly like the MNIST phenomenon: `C` climbs to
0.5–0.9 probability, holds for a fifth of the path, and the JSD trace shows two clearly separated
spikes — entry and exit are genuinely distinct events, not one near-tie. The randomly drawn
candidates are the opposite: a 3-point blip at 0.05–0.1 probability sitting exactly where A and B
cross.

Aggregated over all 1,290 candidates, the random-draw picture dominates:

![Distributions over the 1,290 candidate paths: C-segment width as a fraction of the alpha grid (left), minimum dominance margin of C over both endpoint tokens (middle), and entry alpha versus exit alpha of the C run (right; dashed line is the diagonal).](plots/segment_width_margin_distribution.png)

Most C segments occupy 3–5 of 50 grid points (width 0.06–0.10) and win by a margin under 0.05; 39.9%
of candidates exceed a margin of 0.05 and only 3.6% exceed 0.2. Entry and exit alphas cluster in the
middle of the path (0.3–0.8) and always sit above the diagonal by construction.

### 4. Worked examples: which two texts, which sequence, and is it really a *sub-plateau*?

The previous section is about probabilities of three tokens. This one answers three concrete
questions: *between which two pieces of text are we interpolating, what is the full top-1 sequence
along the path, and does the third region show up as a plateau in Matthew's sense?*

**A worked example, end to end.** This is the highest-scoring candidate of the 1,290, at block 6,
conditioned on context B's tokens:

| | |
|---|---|
| **context A** (32 tokens, WikiText-103 validation) | `" , emerging at night to feed . The diet of H. gammarus mostly consists of other benthic invertebrates . These include crabs ,"` |
| **context B** (the conditioning context) | `" in early 1942 to repair a damaged light cruiser and ordered to return home in May . She was sunk en route by the American submarine USS Salmon , although most of"` |
| **interpolate from → to** | context A's block-6 `resid_post` at its last token → context B's, 50 steps |
| **endpoint predictions** | A = `' which'` at α = 0, B = `' her'` at α = 1 |
| **full top-1 sequence** | `' which'` (α 0–0.10) → `' a'` (0.12) → `' including'` (0.14–0.18) → **`' if'` (0.20–0.41)** → `' her'` (0.43–1.00) |
| **what the C region writes** | *"if not all of her crew survived. The USS Bismarck was sunk by a…"* (identical for all 20 tokens across the whole C run) |
| **plateau geometry** | shelf at $\bar d_C = 0.44$, flatness ρ = 0.16 — six times flatter than the diagonal |

The third prediction `' if'` is not a generic filler here: *"…although most of **if** not all of her
crew survived"* is the completion a human would write, and the model holds it for 11 of 50 steps
between two sharp boundaries.

To show whether that is typical, we plot $d(t)$ for the **six pre-frozen inspection paths** — the 3
top-scoring candidates and 3 drawn at random — in exactly the plateau-post format (output distance
against interpolation coefficient, with the no-plateau diagonal for reference):

![Matthew-style plateau curves for the six pre-frozen inspection paths. x-axis: interpolation coefficient t from 0 (context A's activation) to 1 (context B's). y-axis: relative output distance d(t) on the final logits, 0 = output looks like endpoint A, 1 = like endpoint B; the dashed grey line is the no-plateau reference d = t. The hatched grey band is the detected third-token (C) run and the thin vertical lines mark every top-1 token change. Top row: the 3 highest-scoring candidates; bottom row: 3 candidates drawn at random. Panel titles give the interpolation block, the C run's alpha range and its flatness ρ.](plots/matthew_dt_frozen.png)

The two rows tell opposite stories. The top-scoring paths are **staircases**: the top-1 example holds
a shelf at $d\approx0.44$ across its whole C run (ρ = 0.16) and the third-ranked one holds
$d\approx0.35$ (ρ = 0.30) — plateau, boundary, sub-plateau, boundary, plateau. The randomly drawn
candidates are not: their C runs have ρ = 0.71, 3.07 and 3.71, i.e. the third token is top-1 while
the output is sweeping through the boundary at up to three times the diagonal rate.

That anecdote is confirmed on all 1,290 candidates. To separate "the C run is a shelf" from "the C
run is the middle of the boundary", we compare its flatness with the same alpha window measured on a
randomly matched path that has *no* third region:

![Left: distribution of the flatness ρ of the C window (range of d divided by width in t) for the 1,290 A|C|B candidates (solid) and for the same alpha windows measured on 1,290 matched non-candidate paths (dashed); values above 6 are clipped into the last bin. The dashed vertical line is ρ = 1 (as steep as the no-plateau diagonal), the dotted vertical line the post-hoc ρ = 0.5 sub-plateau cut. Middle: median ρ (circles) with inter-quartile range (hatched band) against the decile of the frozen candidate score, 10 = highest-scoring. Right: histogram of the mean output distance d across the C run, over the 1,290 candidates; the vertical rules mark the two endpoints d = 0 and d = 1.](plots/subplateau_dwell.png)

Three readings, and the first is a negative result:

1. **The median candidate is not a plateau.** Median ρ is **2.05** (IQR 1.15–3.38): inside the typical
   C run the output moves *twice as fast* as the no-plateau diagonal. The matched control windows are
   flatter (median 1.09, IQR 0.47–2.99), simply because a random window of an ordinary A→B path is
   usually far from its single boundary. Only **20.2%** of candidates have ρ < 1 and **8.2%** have
   ρ < 0.5, against 47.3% and 26.4% of the matched controls. A persistent third top-1 token is
   therefore, in the majority of cases, a label event *inside* the transition — not a shelf.
2. **The pre-frozen score finds the real ones.** Median ρ falls monotonically from 2.65 in the lowest
   score decile to 0.93 in the highest (Spearman ρ<sub>s</sub> = −0.34, p ≈ 2 × 10⁻³⁶). The score was
   frozen before any curve was drawn and only combines margin and transition separation, yet it ranks
   candidates by a plateau property it never saw.
3. **When there is a shelf, it sits in the middle.** $\bar d_C$ is centred on 0.52 and 97.3% of
   candidates fall between 0.2 and 0.8 — the third region is genuinely *between* the two endpoint
   outputs, not a wobble next to one of them. Candidate paths also have a much wider overall
   transition than non-candidates ($w_{10\to 90}$ median 0.46 vs 0.30), as a two-boundary staircase
   must.

**So where does the sub-plateau show up?** In the flat tail: **106 of 1,290 candidates (8.2%) have
ρ < 0.5**, which is **1.39% of all eligible paths (95% CI [1.15%, 1.68%])**, or about 1 path in 72.
These are longer than average C runs (8.1 vs 5.2 grid points) and concentrate at the deepest block we
interpolated (55 of 106 at block 6; median ρ per block 2.52 / 2.58 / 2.38 / 1.54 for blocks
0 / 2 / 4 / 6). Their curves are textbook:

![The six flattest sub-plateaus among the 1,290 candidates (post-hoc selection: ρ < 0.5 and C run at least 5 grid points). Axes as in the previous figure: x = interpolation coefficient t, y = relative output distance d(t) on the final logits, dashed grey = the no-plateau diagonal d = t, hatched band = the C run, thin vertical lines = top-1 token changes. Panel titles give the block, the C run's alpha range and ρ.](plots/matthew_dt_gallery.png)

Each of these is flat, jump, flat, jump, flat — the sub-plateau shape the MNIST work predicted. Two of
them, with their source texts and full top-1 sequences:

| | flattest candidate (ρ = 0.04, block 2) | third-flattest (ρ = 0.08, block 2) |
|---|---|---|
| context A | `" Art exhibitions were originally held in Lamar Hotel in downtown Meridian , but after a name change to Meridian Art Association in 1949 , exhibitions were held at various locations around the"` | same context A |
| context B | `" the dance appears in The Pirate by Sir Walter Scott . The writer and journalist John Sands lived on Papa Stour and Foula for a while during"` | same context B |
| top-1 sequence | `' year'` (α 0–0.47) → `','` (0.49) → `' was'` (0.51) → **`'.'` (0.53–0.61)** → `' the'` (0.63–1.00) | `' city'` (0–0.51) → **`','` (0.53–0.65)** → `' the'` (0.67–1.00), a *clean* `A, C, B` |
| shelf height | $\bar d_C = 0.48$ | $\bar d_C = 0.51$ |

Note the honest caveat this exposes: the *flattest* sub-plateaus are usually headed by punctuation or
generic tokens (`'.'`, `','`, `' the'`) rather than by a semantically interesting third word like the
`' if'` above, and only 16.0% of them are clean `A, C, B` (against 21.9% of all candidates). Flatness
in output space and interestingness of the token are different things, and this screen finds far more
of the former than of the latter.

### 5. Is the third region a confident state, or just a flat spot?

This is the question that separates "extra model state" from "the model briefly has no idea". We
compare the top-1 probability and the predictive entropy at the centre of the C run against the mean
of the two path endpoints:

![Top-1 probability (left) and predictive entropy in bits (right) at the centre of the C region (solid) versus the mean of the two path endpoints (dashed), over the 1,290 candidate paths.](plots/c_region_confidence.png)

The third region is **less** confident than the endpoints: top-1 probability 0.227 ± 0.165 versus
0.323 ± 0.182, and entropy 6.97 ± 1.99 versus 5.70 ± 1.82 bits. Only 26.8% of candidates are sharper
in the C region than at their endpoints. So the typical third region is a flatter part of the path
where some third token happens to hold the top of a broad distribution — not a sharpened alternative
prediction.

Its identity points the same way. If a C region encoded a distinct intermediate concept we would
expect its tokens to differ from the model's generic defaults:

![Left: the 15 most common intermediate (C) tokens across the 1,290 candidate paths. Right: the 15 most common endpoint (A) tokens across the 7,611 eligible paths. Bars count paths; y-axis labels are the decoded tokens.](plots/intermediate_token_census.png)

They do not. The C census is headed by `' the'` (96 paths), `'.'` (79), `'-'` (46), `' of'` (44) — the
same generic pool that heads the endpoint census, and 32.3% of all C tokens are among the ten
commonest endpoint predictions. The intermediate token is usually the model's frequency default, which
is what one expects from a distribution that has flattened rather than moved to a new mode.

### 6. How sensitive is the headline rate to the frozen thresholds?

Both thresholds (persistence ≥ 3 points, margin > 0) were fixed before any curve was seen, but a
reader should know how much rides on them:

![A|C|B rate per eligible path versus the persistence threshold (2, 3 or 5 consecutive alpha points), for minimum-dominance-margin floors of 0, 0.02 and 0.05 (three line styles). The dotted vertical line marks the frozen default.](plots/threshold_sensitivity.png)

The rate falls smoothly from 29.5% (persistence 2, any positive margin) through the frozen 16.9% to
2.6% (persistence 5, margin > 0.05). No setting makes the phenomenon vanish, and no setting makes it
the norm; the ordering of the conclusions above is unchanged throughout.

### 7. Do C-region activations sit where natural activations sit?

If a third region were an ordinary place in activation space, its points should have natural
neighbours, and those neighbours should predict `C` too:

![Left: distribution of cosine distance to the nearest of 2,000 held-out natural activations, for A-region, C-region and B-region interpolation points and for natural contexts used as queries (control). Right: fraction of the 10 nearest natural neighbours whose own unpatched top-1 next token equals the query's top-1 token, with 95% bootstrap intervals.](plots/natural_neighbor_comparison.png)

Neither holds. C-region points are the **furthest** from the natural bank (median cosine distance
0.160, CI [0.154, 0.166]) — further than A-region (0.140) and B-region (0.153) points, and much
further than a natural context is from its nearest natural neighbour (0.086, CI [0.061, 0.131]). And
only **4.5%** (CI [3.8%, 5.3%]) of a C point's ten nearest natural neighbours predict `C` themselves,
against 8.1% for the A- and B-region points and 14.1% for a natural query. With a 2,000-context bank
these distances are inflated for every query type, so the comparison — not the absolute number — is
the result: **the third prediction is not supported by nearby natural activations.**

### 8. Does the third region generate coherent text?

For the six frozen inspected candidates (3 top-scoring, 3 random) we decode 20 tokens from the A-, C-
and B-region centres. Every C-region continuation is fluent, context-appropriate English — for the
top-ranked candidate (`A=' which'`, `C=' if'`, `B=' her'`, block 6), the C region produces
*"if not all of her crew survived. The USS Bismarck was sunk by a…"* against B's *"her crew survived.
The USS Bismarck was sunk by a German U-boat"*, and three temperature-0.8 samples stay on the same
theme. So the third region is a usable model state, not noise.

Reproducibility across the region is mixed, which is why we measure it rather than assert it:

![Number of leading greedy-decoded tokens (out of 20) that are identical across continuations generated at the first, middle and last alpha of the C run, for each of the six inspected candidates. Bar labels give the C token and the interpolation block; the dotted line at 1 is the trivial floor, since the first token is C by construction.](plots/continuation_stability.png)

Two of six candidates keep all 20 tokens identical across the whole C run, one keeps 8, and three
agree only on the first token. And in **6 of 6** cases, inserting the same C-region activation into
the *other* endpoint's context reproduces that context's own unpatched continuation almost verbatim —
consistent with the 17.6% endpoint-transfer rate. The third state therefore belongs to the pair
(activation, context), not to the activation alone.

### 9. Where in the network does this live? (exploratory depth sweep)

Everything above interpolates at blocks 0–6, which were preregistered *before* any result was seen.
Inside that window both the rate (8.2% → 27.7%) and the flatness (median ρ 2.52 → 1.54) improve with
depth, which leaves an obvious question the frozen protocol cannot answer: does it keep getting
better deeper in the network? We therefore re-ran the **same 1,000 primary pairs with the same frozen
detector** at blocks **12, 18, 24 and 30** of the 36 (block set chosen before running). This sweep is
**exploratory and reported separately**: it reuses the primary pairs rather than a fresh bank, and it
contributes nothing to the headline prevalence.

![Left: percentage of eligible paths that contain a persistent third top-1 token (solid, circles) and that contain a true sub-plateau with flatness ρ < 0.5 (dashed, squares), against the interpolation block L of GPT-2 Large; error bars are 95% Wilson intervals and the hatched region marks the preregistered blocks 0–6. Right: median flatness ρ of the C window against L, with the preregistered blocks (solid, circles) and the exploratory blocks (dashed, squares); the dashed horizontal line is ρ = 1 (as steep as the no-plateau diagonal) and the dotted line the ρ = 0.5 sub-plateau cut.](plots/depth_sweep.png)

| interpolation block | 0 | 2 | 4 | 6 | 12 | 18 | 24 | 30 |
|---|---|---|---|---|---|---|---|---|
| third-token rate (% of eligible paths) | 8.2 | 15.4 | 16.4 | **27.7** | 22.8 | 13.6 | 5.9 | 1.7 |
| true sub-plateau rate, ρ < 0.5 (%) | 0.95 | 1.16 | 0.58 | **2.87** | 0.10 | 0.00 | 0.00 | 0.00 |
| median flatness ρ | 2.52 | 2.58 | 2.38 | 1.54 | 2.07 | 2.03 | 1.47 | 1.24 |
| clean `A,C,B` share of candidates (%) | — | — | — | — | 22 | 28 | 31 | 45 |

The trend does **not** continue — it turns over. The label-level rate peaks between blocks 6 and 12
and then collapses to **1.7%** by block 30, and the sub-plateau rate peaks at block 6 (2.87%) and is
**exactly zero** at blocks 18, 24 and 30 (0 of 269, 118 and 33 candidates). Median ρ drifts towards
1 with depth, which sounds like flattening but is the opposite: it means the whole output curve is
converging on the no-plateau diagonal, so a C run neither dwells nor sits inside a sharp boundary —
it is simply a point the straight line passes through. The mechanism is intuitive: patching at block
30 leaves only six blocks to fold the interpolant into anything, so the output tracks the interpolated
activation almost linearly, while patching early lets the network re-read the vector and snap it to a
discrete state.

So the preregistered window happened to bracket the interesting region. **The sub-plateau is an
early-to-mid-network phenomenon, maximal around block 6 of 36, and it is gone by block 18.** One
caveat worth stating: at deep blocks the third-token *label* rate falls partly because more paths
become uninteresting overall, and the surviving candidates are cleaner (45% clean at block 30 versus
22% at block 12) but shorter and diagonal — a cleaner label sequence is not a stronger plateau.

## Conclusion

Interpolating between two random held-out natural activations in GPT-2 Large produces a persistent
third top-1 prediction on **16.9%** of eligible paths (CI [16.1%, 17.8%]), replicated at **17.7%** on
an untouched bank. The MNIST-style `A → C → B` behaviour is therefore not an artefact of hand-picked
prompts; it is a common feature of early-layer activation interpolation, and it is more common the
deeper the block you interpolate at (8.2% at block 0 → 27.7% at block 6).

The strong claim, however, does not survive. Most of these third regions are **fragile**: narrow (3–5
of 50 steps), low-margin, *higher*-entropy than the endpoints, headed by generic tokens like `' the'`
and `'.'`, further off the natural activation manifold than the endpoint-region points, and unsupported
by natural neighbours that share the prediction. They also appear at 11.1% between contexts that agree
on the next token, so a large share is not a competition between A and B. A minority — 3.7% of
eligible paths are clean `A, C, B`; 3.6% of candidates exceed a 0.2 margin; two of six inspected
candidates hold a 20-token continuation across the whole region — behaves like the crisp third state
seen in MNIST.

Looking at the same paths through the plateau lens says the same thing more precisely. A *sub*-plateau
means the model's output holds still at an intermediate height, and by that definition only
**1.39% of eligible paths** qualify (ρ < 0.5, CI [1.15%, 1.68%]); the median candidate's C run has
ρ = 2.05, i.e. it lies inside the boundary rather than on a shelf. The real sub-plateaus exist, look
exactly as the MNIST work predicted, sit at $\bar d_C \approx 0.5$ between the endpoints, favour the
deepest block preregistered (55 of 106 at block 6), and are found by the pre-frozen score — but they
are one path in seventy-two, not one in six.

An exploratory sweep to blocks 12/18/24/30 shows the depth trend **turns over** rather than
continuing: the third-token rate peaks between blocks 6 and 12 and falls to 1.7% by block 30, and the
sub-plateau rate peaks at block 6 (2.87% of eligible paths) and is exactly zero from block 18 on. The
phenomenon belongs to the early-to-middle residual stream, where the remaining blocks still have the
capacity to snap an interpolated vector onto a discrete state.

**Limitations.** (i) One model (GPT-2 Large), one corpus (WikiText-103 validation), one context length
(32 tokens). The four preregistered blocks are all early; the exploratory sweep to blocks 12-30
(Section 9) extends the depth picture but reuses the same pairs, and nothing here says how any of it
changes in a larger model. (ii) Because the two contexts are different token sequences, every path must be conditioned on
one of them; the foreign end reproduces its own prediction only 17.6% of the time, so most paths run
between "context A's prediction" and "what B's activation does inside context A". Restricting to
transfer-consistent paths leaves the rate at 14.0%, so this does not drive the result, but it does
mean the endpoints are not always two natural predictions. (iii) The reference bank is 2,000 contexts,
which inflates all nearest-neighbour distances; only the between-condition comparison is meaningful.
(iv) The continuation analysis covers six candidates — enough to show that coherent C-region text
exists and that reproducibility varies, not enough to estimate what fraction is reproducible.
(v) The plateau-geometry analysis (Section 4) was added after the screen, in response to operator
feedback; the ρ < 0.5 cut is descriptive and post hoc, chosen to illustrate the flat tail, and it
feeds no prevalence estimate. The continuous statistics — median ρ, the score-decile trend, the
matched-control comparison — carry the argument and do not depend on that cut.

**What this means for interpretability and safety.** Activation steering, representation editing and
model-diffing all move activations along lines like these. This screen says you will cross a
third-prediction band roughly one time in six, that the band is usually a low-confidence generic-token
zone off the natural manifold, and that a third top-1 token on its own is therefore **not** evidence
of a meaningful intermediate concept. The margin, the entropy, and the neighbourhood of the point are
what separate the fragile majority from the small, genuine minority.
