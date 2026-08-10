# A plateau in a single-token activation interpolation signals *dissimilar* continuations, not shared ones

> Final, presentable, current-best only (history is in CHANGELOG.md).

## Summary

A common move in interpretability is to take the internal activation vector a language model computes
for prompt A, the one it computes for prompt B, and walk continuously from one to the other, watching
how the model's output changes. When the output stays put for a while and then switches abruptly — a
**plateau** followed by a jump — it is tempting to read that as a discrete internal feature flipping
state, and in particular as a sign that the two prompts share a continuation while differing in some
hidden property. This report tests that reading directly: if two prompts differ in one final token
but predict *nearly the same next token*, does interpolating that token's activation give a sharper
plateau?

It gives a **wider** one. Working in two models (GPT-2 medium, 355M, and Pythia-410m-deduped) with a
bank of 200 prompt pairs per model mined from WikiText-103 — each pair a shared prefix plus one
differing final token, spanning the full range of endpoint next-token divergence — the rank
correlation between how differently the two prompts predict the next token and how sharp the
transition is comes out **negative and strongly significant in GPT-2 medium**: $\rho = -0.55$
(95% cluster-bootstrap CI $[-0.66, -0.41]$, $p = 6 \times 10^{-17}$, $n = 200$). More divergent
endpoints give sharper plateaus. Restricting to pairs away from the divergence ceiling, where the
divergence measure can still order pairs, the effect is stronger and appears in both models:
$\rho = -0.61$ (GPT-2 medium, $n = 142$) and $\rho = -0.45$ (Pythia-410m, $n = 127$), both
$p < 10^{-7}$. Three separate sharpness statistics agree on the sign, and the effect survives
controlling for how far apart the two patched activations are geometrically.

Two supporting facts complete the picture. First, plateaus are the *default*: 82% of the mined GPT-2
medium pairs and 48% of the Pythia pairs cross our sharpness threshold, and a deliberately dissimilar
hand-picked control pair plateaus as hard as four hand-picked "same continuation, different feature"
pairs. Second, the sharpening is supplied by depth — one block after the patch every condition still
responds almost exactly linearly, and the width falls monotonically over the following 23 blocks.

For practice, this inverts a diagnostic. An observed plateau is not evidence that two prompts share a
continuation; the sharpest plateaus come from the pairs whose predictions agree least. Any claim
resting on plateau shape needs an interpolation between prompts of known dissimilarity to calibrate
against, and the *difference* between the two, not the absolute sharpness, is the only part that
carries information.

## Methods

### Data & Model

**Models.** Two final pretrained checkpoints, both frozen and in evaluation mode, float32, no
sampling: `gpt2-medium` (355M parameters, 24 blocks, $d_{model}=1024$) and
`EleutherAI/pythia-410m-deduped` at `revision="step143000"` (24 layers, $d_{model}=1024$). Two model
families are used so that any effect specific to one tokenizer or one architecture shows up as a
disagreement between columns.

**Hand-picked pairs (Experiment 1).** Five hand-written prompt pairs, each a shared prefix plus one
differing final token. Four are test pairs, chosen so the two versions of the sentence plausibly
continue the same way while differing in some internal property (identity vs. pronoun, word-form vs.
numeral, lower vs. upper case, chemical symbol vs. atomic number):

1. `Mary and John went to the store. John gave a book to` + ` Mary` / ` her`
2. `Two plus two is` + ` four` / ` 4`
3. `The answer is` + ` four` / ` Four`
4. `Which chemical element does this clue identify?` + ` Au` / ` 79`

The fifth is the **control**: `The house was` + ` big` / ` in`, an adjective against a preposition,
whose continuations are genuinely different. It plays the role of a negative example — under the
hypothesis it should be the one pair that does *not* plateau.

**Mined pair bank (Experiment 2).** Five hand-written pairs cannot measure an association: $n = 10$
model-pair cells only has the power to detect a correlation above about $\rho = 0.75$. We therefore
mine a bank of pairs automatically. We take the first 40 paragraphs of at least 400 characters from
the WikiText-103 validation split (natural English, not written by us), truncate each to a prefix of
$L$ tokens with $L$ drawn uniformly from $[10, 40]$, and run the model on the prefix. Final token A is
always the model's **top-1** next token; final token B is the token at rank $r$, with five values of
$r$ drawn log-uniformly from $[1, 5000]$ per prefix. A rank-1 partner produces a near-tie between two
plausible continuations, a rank-5000 partner an implausible one, so the endpoint divergence defined
below spans its whole range by construction. That gives **200 pairs per model** from 40 prefixes
(the same prefixes and ranks in both models; the tokens themselves are each model's own). Building the
inputs as `prefix_ids + [token_id]` makes the "identical prefix, one differing single final token"
condition exact by construction.

**Validity check (Experiment 1).** For the hand-written pairs we require, per model, that the two
prompts tokenize to an identical prefix and exactly one differing single final token; a pair failing
this in a model would be dropped for that model. All 5 pairs passed in both models (prefix lengths
3–13 tokens), so all 10 model-pair cells are reported and no multi-token interpolation was performed.

**Hook point and sample sizes.** We read and patch `resid_post` after block 0 — the residual stream
immediately after the first transformer block — at the **final token position only**. Because the
prefix is identical and attention is causal, every earlier position is bit-identical between the two
prompts, so one forward pass per interpolation point fully determines the run. For the hand-picked
pairs, downstream `resid_post` is also recorded at the final token of every later block (blocks 1–23)
plus the final logits; for the mined bank only the final logits are recorded. Every sweep uses 101
evenly spaced interpolation values on $[0,1]$, under `torch.no_grad()` with fixed seeds. Total:
$(5 + 200)$ pairs $\times$ 2 models $\times$ 101 points.

### Metrics

The whole study depends on making "the model's output moved from A to B" a number, so we start there
and build up to the sharpness statistics and the association tests that consume them.

**Interpolation.** We need a path between the two activation vectors that does not shrink toward zero
in the middle, as a straight line between two high-dimensional vectors does. We therefore interpolate
the direction along the sphere (SLERP) and the length linearly. With $h_A, h_B$ the patched-layer
activations, $\hat h = h / \lVert h \rVert$ and $\Omega = \arccos(\hat h_A \cdot \hat h_B)$:

```math
h_\alpha \;=\; \Big[(1-\alpha)\lVert h_A\rVert + \alpha\lVert h_B\rVert\Big]\cdot
\frac{\sin\!\big((1-\alpha)\Omega\big)\,\hat h_A + \sin\!\big(\alpha\Omega\big)\,\hat h_B}{\sin\Omega}
```

$h_\alpha$ replaces the block-0 output at the final token and is run forward through the rest of the
model. At $\alpha=0$ and $\alpha=1$ this is the identity, which gives a free correctness check on the
harness (reported in Results). The angle $\Omega$ is also recorded per pair and reused later as a
control variable: it measures how far apart the two endpoints are geometrically, independently of what
they predict.

**Endpoint divergence (JSD)** — how differently the two complete prompts predict the next token, and
so the independent variable of the whole study. It is measured at inference from the full-vocabulary
softmax distributions $P_A, P_B$ at the final position, never from corpus counts. Jensen–Shannon
divergence is symmetric and stays finite when one distribution puts near-zero mass where the other
does not, where a plain KL divergence diverges. Units are nats; 0 means identical predictions:

```math
\mathrm{JSD}(P_A, P_B) \;=\; \tfrac{1}{2} D_{KL}\!\big(P_A \,\Vert\, M\big) + \tfrac{1}{2} D_{KL}\!\big(P_B \,\Vert\, M\big),
\qquad M = \tfrac{1}{2}\big(P_A + P_B\big)
```

JSD is bounded above by $\ln 2 \approx 0.693$ nats, attained when the two distributions have disjoint
support. That ceiling matters for the analysis: pairs that both predict completely different tokens
all pile up at the same $x$ value and can no longer be ordered, so we report the association both on
the full bank and on the **unsaturated** subset with $\mathrm{JSD} < 0.65$.

**Relative distance $d(\alpha)$** — where the model's output sits on the way from A to B. A raw
distance is not comparable across pairs whose endpoints are far apart to different degrees, so we
normalize by the total: $d=0$ means the output is exactly A's, $d=1$ exactly B's. For a vector
$x_\alpha$ read at any hook point (a downstream `resid_post`, or the final logit vector):

```math
d(\alpha) \;=\; \frac{\lVert x_\alpha - x_A\rVert_2}{\lVert x_\alpha - x_A\rVert_2 + \lVert x_\alpha - x_B\rVert_2}
```

This is the quantity plotted in Figure 1. A model that responds proportionally to the input edit gives
$d(\alpha)=\alpha$, the diagonal; a plateau is any large flat stretch followed by a fast rise.

**Transition width $w_{10-90}$** (primary sharpness statistic, fixed by the plan) — how much of the
sweep the output spends actually moving. It is the $\alpha$-distance between the first upward
crossings of $d=0.1$ and $d=0.9$, so a small value means the output ignored most of the interpolation
and then switched:

```math
w_{10\text{-}90} \;=\; \alpha(d=0.9) - \alpha(d=0.1)
```

Crossings are linearly interpolated on the 101-point grid. Following the plan, $w_{10-90} < 0.5$ is
called a clear plateau, against $0.8$ for the linear response. It drives the verdict column of Table 1
and the top row of Figure 3, and is the statistic tracked across depth in Figure 4.

**Total-variation width $w_{TV}$** (threshold-free sharpness statistic) — the same idea without the
fixed crossing levels. Non-monotonic curves are common (Table 1; only 7.5% of the mined GPT-2 medium
curves are monotonic): they dip and re-cross, which pushes the $d=0.1$ crossing far to the left and
makes a visibly sharp curve score as wide. Let $C(\alpha)$ be the fraction of the curve's total
variation accumulated by position $\alpha$; then $w_{TV}$ is the $\alpha$-span carrying the middle
half of all the movement:

```math
C(\alpha) = \frac{\int_0^{\alpha} \lvert d'(u)\rvert\,du}{\int_0^{1} \lvert d'(u)\rvert\,du},
\qquad w_{TV} \;=\; C^{-1}(0.75) - C^{-1}(0.25)
```

It is $0.5$ for a linear response and tends to $0$ for a step, so we call $w_{TV} < 0.25$ sharp.
Because it survives non-monotonicity it is the statistic used for the prevalence count in Figure 2 and
for the headline correlation.

**Plateau fraction PF** (second robustness statistic) — how much of the sweep sits pinned at an
endpoint, computed directly from the grid without any crossing logic, so it is immune both to
non-monotonicity and to the choice of which crossing counts. Over the $N=101$ grid points:

```math
\mathrm{PF} \;=\; \frac{1}{N}\,\#\big\lbrace \alpha_i : d(\alpha_i) < 0.1 \ \ \text{or} \ \ d(\alpha_i) > 0.9 \big\rbrace
```

Higher is more plateau-like; the linear response gives $0.2$. Its role in Results is to check that all
three sharpness statistics agree on the *sign* of the JSD relationship — with the sign convention
flipped, since large PF and small $w$ both mean "sharp".

**Association test.** The hypothesis predicts that low endpoint JSD goes with low width, so we score
it with the Spearman rank correlation $\rho$ between JSD and each sharpness statistic. Rank
correlation is used because the prediction is about ordering and the relationship need not be linear.
Pairs that share a prefix are not independent, so all confidence intervals come from a **cluster
bootstrap**: resample the 40 prefixes with replacement, take all pairs of the drawn prefixes, and
report the 2.5th and 97.5th percentiles of $\rho$ over 2000 resamples. We also report the ordinary
least-squares slope of each width on JSD (units: width per nat) with the same bootstrap, which gives
the effect an interpretable size.

**Partial correlation (confound control).** A pair whose prompts predict different next tokens might
simply have more distant activations at the patch site, and distance alone could drive sharpness. To
separate the two we compute the partial Spearman correlation of JSD with $w_{TV}$ controlling for the
block-0 angle $\Omega$: rank-transform all three variables, linearly regress the JSD and $w_{TV}$ ranks
on the $\Omega$ ranks, and correlate the residuals.

### Baselines

**Linear response** — the null shape, the behavior of a model whose output moves in proportion to the
activation edit:

```math
d(\alpha) = \alpha \quad\Longrightarrow\quad w_{10\text{-}90} = 0.8,\quad w_{TV} = 0.5,\quad \mathrm{PF} = 0.2
```

It appears as the gray dashed diagonal in Figure 1 and the gray dashed line in Figures 2, 3 and 4. Any
value below these is a plateau of some strength.

**Dissimilar-continuation control** — the `The house was` + ` big` / ` in` pair, run through identical
machinery. It fixes the value the plateau statistics take when two prompts emphatically do not share a
continuation, and is drawn with a thick frame in Figure 1 and a thick marker edge in Figures 2 and 3.

**Mined bank as a null distribution** — the 200 corpus-derived pairs per model give the distribution of
plateau sharpness for prompt pairs picked with no regard to continuation similarity at all. A
hand-picked pair is only interesting insofar as it sits away from this distribution; Figure 2 places
the five of them inside it.

**Smallest detectable correlation** — the reference against which a null claim is judged. For a
two-sided test at $\alpha = 0.05$ using the Fisher $z$ transform, the smallest $|\rho|$ that would
reach significance at sample size $n$ is:

```math
\rho_{\min}(n) \;=\; \tanh\!\Big(\frac{1.96}{\sqrt{n-3}}\Big)
```

This is $0.75$ at $n=10$ (the hand-picked cells, which is why they can only be suggestive) and $0.14$ at
$n=200$ (the mined bank).

**Harness identity check** — patching $h_0$ and $h_1$ must reproduce the unpatched runs, giving
$d(0)=0$ and $d(1)=1$ exactly. Deviation from this measures implementation error, reported below.

## Results

**The harness is correct.** All 5 hand-written pairs tokenized validly in both models, and across all
410 model-pair sweeps the patched runs at the endpoints reproduced the clean forward passes to
$|d(0)| \le 4 \times 10^{-4}$ and $|d(1) - 1| \le 4 \times 10^{-4}$. The numbers below are therefore
about the model, not about patching artifacts.

### Plateaus appear everywhere, including where they should not

**Table 1 — endpoint divergence and plateau strength for the five hand-picked pairs.** Reading the
table: JSD is the independent variable (small = the two prompts predict nearly the same next token);
$w_{10-90}$ and $w_{TV}$ are smaller when the transition is sharper; PF is larger when more of the
sweep is pinned at an endpoint. Bold marks the sharpest cell per model. The hypothesis predicts the
small-JSD rows should carry the small widths.

| Model | Prompt pair (final tokens) | endpoint JSD (nats) | $w_{10-90}$ | $w_{TV}$ | PF | monotonic |
|---|---|---|---|---|---|---|
| gpt2-medium | gave a book to ` Mary` / ` her` | 0.068 | 0.586 | 0.114 | 0.51 | no |
| gpt2-medium | Two plus two is ` four` / ` 4` | 0.138 | 0.454 | 0.232 | 0.55 | yes |
| gpt2-medium | The answer is ` four` / ` Four` | 0.377 | **0.120** | **0.058** | 0.88 | no |
| gpt2-medium | clue identify? ` Au` / ` 79` | 0.342 | 0.358 | 0.117 | 0.64 | no |
| gpt2-medium | *control:* The house was ` big` / ` in` | 0.659 | 0.516 | 0.272 | 0.50 | no |
| pythia-410m | gave a book to ` Mary` / ` her` | 0.033 | 0.582 | 0.268 | 0.43 | yes |
| pythia-410m | Two plus two is ` four` / ` 4` | 0.056 | 0.758 | 0.451 | 0.25 | yes |
| pythia-410m | The answer is ` four` / ` Four` | 0.271 | **0.340** | **0.135** | 0.66 | yes |
| pythia-410m | clue identify? ` Au` / ` 79` | 0.385 | 0.598 | 0.254 | 0.41 | yes |
| pythia-410m | *control:* The house was ` big` / ` in` | 0.665 | 0.425 | 0.137 | 0.57 | yes |

Nine of the ten cells land at $w_{TV} \le 0.27$ against the linear-response value of $0.5$, so this
style of single-token interpolation does induce plateaus, at strengths from mild to near-step
(gpt2-medium `four`/`Four` moves through 80% of the gap in $0.12$ of the sweep). The exception,
pythia-410m on `four`/`4` at $w_{TV} = 0.451$, is one of the two *most* similar pairs in the study.
The control — an adjective against a preposition, at the largest divergence measured — plateaus at
$w_{TV} = 0.272$ / $0.137$, sharper than both low-JSD pairs in gpt2-medium and sharper than three of
four test pairs in pythia-410m. A diagnostic that fires just as hard on its own negative control is
not yet measuring what it was supposed to measure.

The verdict rests first on the shape of the raw sweeps, so we show all ten before any summary
statistic. If plateaus tracked continuation similarity, the top two rows (lowest JSD) would be the
flattest and the bottom row (the control) the most linear.

![Relative distance versus interpolation position for five prompt pairs in two models](plots/final_logit_curves.png)

**Figure 1.** Final-logit response to interpolating one token's block-0 activation. x: interpolation
position $\alpha$ from prompt A (0) to prompt B (1); y: relative distance $d$ (0 = at A's logits,
1 = at B's logits). Solid curve with circles = measured $d(\alpha)$; gray dashed = the linear reference
$d = \alpha$. Rows are prompt pairs, columns are models; the bottom row (thick frame) is the control
pair, whose continuations differ most. Every panel except pythia-410m `four`/`4` bends well away from
the diagonal into a flat-then-jump shape, and the control bends as much as the test pairs. The wiggles
in the gpt2-medium column are the non-monotonicity flagged in Table 1 and are why $w_{TV}$ exists.

### In a corpus-mined bank, the plateau is the default response

Five pairs cannot say whether a plateau is rare enough to be informative. To find out we ask how often
an arbitrary prompt pair plateaus, using the 200 WikiText-derived pairs per model, and where the
hand-picked pairs fall inside that distribution.

![Distribution of transition sharpness over 200 mined prompt pairs per model, with the five hand-picked pairs marked](plots/bank_prevalence.png)

**Figure 2.** Plateaus are the norm for arbitrary prompt pairs. x: $w_{TV}$ at the final logits
(smaller = sharper); y: number of mined pairs in each bin (gray hatched histogram, $n=200$ per model).
Gray dashed = the linear-response value ($0.5$), dotted = our sharpness threshold ($0.25$). The five
markers in the strip above each histogram are the hand-picked pairs of Table 1 at their $w_{TV}$
values (marker shape and color per the legend; the control has a thick black edge) — they are placed
on a separate strip because their y position carries no meaning. In gpt2-medium the mined mass piles
up near zero; in pythia-410m it centers on the threshold.

In gpt2-medium, **82% of arbitrary mined pairs are sharp** by our threshold (median $w_{TV} = 0.080$,
median $w_{10-90} = 0.241$); in pythia-410m, 48% are (median $w_{TV} = 0.266$, median
$w_{10-90} = 0.593$). Every one of the hand-picked pairs sits inside the bulk of its model's
distribution — none is an outlier in the sharp direction. Observing a plateau for a hand-chosen pair
in gpt2-medium therefore carries almost no information: four in five random pairs do the same thing.

### Sharper plateaus go with *less* similar continuations

With $n=200$ per model the association test finally has power: it can detect $|\rho| \ge 0.14$, where
the five hand-picked pairs could only have detected $|\rho| \ge 0.75$. Figure 3 plots every mined pair.

![Endpoint divergence against two sharpness statistics for 200 mined pairs per model, with fits](plots/bank_regression.png)

**Figure 3.** Endpoint divergence predicts sharpness — with the sign opposite to the hypothesis.
x (all panels): endpoint JSD in nats, larger = the two prompts predict more different next tokens;
the dash-dot vertical line is the $\ln 2$ ceiling that JSD attains for disjoint predictions.
y: $w_{10-90}$ (top row) and $w_{TV}$ (bottom row) at the final logits, smaller = sharper. Columns are
models. Light circles are the 200 mined pairs, the solid line is the OLS fit, the dashed line with
squares joins quintile means of JSD with $\pm 1$ standard error, and the stars are the five
hand-picked pairs (thick black edge = control). Gray dashed = linear response, dotted = plateau
threshold. Both models trend downward; pythia-410m's quintile means turn back up in the last two
quintiles, which sit at the JSD ceiling.

The correlations are collected below. All are Spearman $\rho$ between endpoint JSD and the named
statistic, with 95% cluster-bootstrap intervals over the 40 prefixes. Remember the sign convention:
negative $\rho$ for a width and positive $\rho$ for PF both mean "more divergent endpoints, sharper
plateau" — the *opposite* of the hypothesis, which predicts positive $\rho$ for widths.

**Table 2 — association between endpoint divergence and plateau sharpness, mined bank.**

| Model | statistic | $\rho$ | 95% CI | $p$ | OLS slope (per nat) |
|---|---|---|---|---|---|
| gpt2-medium | $w_{10-90}$ | $-0.47$ | $[-0.59, -0.33]$ | $1.4\times10^{-12}$ | $-0.64\ [-0.78, -0.49]$ |
| gpt2-medium | $w_{TV}$ | $-0.55$ | $[-0.66, -0.41]$ | $6.2\times10^{-17}$ | $-0.42\ [-0.54, -0.31]$ |
| gpt2-medium | PF | $+0.44$ | $[+0.31, +0.56]$ | $5.7\times10^{-11}$ | — |
| pythia-410m | $w_{10-90}$ | $-0.12$ | $[-0.30, +0.05]$ | $0.090$ | $-0.24\ [-0.35, -0.11]$ |
| pythia-410m | $w_{TV}$ | $-0.11$ | $[-0.30, +0.07]$ | $0.123$ | $-0.20\ [-0.31, -0.07]$ |
| pythia-410m | PF | $+0.12$ | $[-0.06, +0.31]$ | $0.090$ | — |

In gpt2-medium the effect is large and unambiguous, and its size is easy to read off the slope: moving
from two prompts that predict the same token to two that predict disjoint tokens ($0 \to 0.69$ nats)
shortens the transition by $0.29$ of the sweep on $w_{TV}$ — more than half of the $0.5$ that a linear
response would spend. The quintile means in Figure 3 fall monotonically ($w_{TV}$: $0.35 \to 0.13 \to
0.09 \to 0.07 \to 0.08$), so this is not one extreme group doing the work. All three sharpness
statistics agree on the direction, which is what separates a real effect from the noise-driven sign
disagreement the five hand-picked pairs produced.

Pythia-410m looks weak in Table 2, and the reason is visible in Figure 3: 37% of its mined pairs sit
at or above JSD $0.65$, essentially at the $\ln 2$ ceiling, where the divergence measure can no longer
order them and they contribute a vertical stripe of noise. Removing the saturated pairs sharpens both
models and brings them into agreement:

**Table 3 — the same test restricted to pairs below the JSD ceiling ($\mathrm{JSD} < 0.65$).**

| Model | $n$ | $\rho$ ($w_{TV}$) | $p$ | $\rho$ ($w_{10-90}$) | $p$ |
|---|---|---|---|---|---|
| gpt2-medium | 142 | $-0.61$ | $1.5\times10^{-15}$ | $-0.54$ | $4.9\times10^{-12}$ |
| pythia-410m | 127 | $-0.45$ | $9.0\times10^{-8}$ | $-0.47$ | $2.3\times10^{-8}$ |

Both models now show a moderate-to-strong negative association at $p < 10^{-7}$. The finding is
therefore not a GPT-2 idiosyncrasy: in the regime where the independent variable is measurable at all,
prompts that predict more different continuations produce sharper plateaus in both model families.

**The effect is not just endpoint geometry.** Pairs that disagree about the next token might simply
have activations further apart at the patch site, and larger separation might mechanically produce a
sharper crossover. Controlling for the block-0 angle $\Omega$ between the two patched vectors leaves
the association essentially intact in gpt2-medium — partial $\rho = -0.55$ against a raw $-0.55$,
because $\Omega$ barely tracks JSD there ($\rho = 0.03$) — and reduces it in pythia-410m, from $-0.11$
to $-0.16$ with $\rho(\Omega, \mathrm{JSD}) = 0.31$. In both models $\Omega$ correlates only weakly
with sharpness ($\rho = 0.16$). Whatever produces the effect, it is carried by what the prompts
predict, not by how far apart their activations start.

### Depth, not prompt content, manufactures the plateau shape

If sharpness is not evidence about shared continuations, it needs an explanation of its own. To locate
where it arises we recompute $w_{10-90}$ at every block's residual stream between the patch site and
the output, for the five hand-picked pairs.

![Transition width versus recording block for five prompt pairs in two models](plots/layerwise_widths.png)

**Figure 4.** The plateau is built up gradually across depth, not created at the patch site. x: the
block whose `resid_post` is read out at the final token (the patch is applied after block 0; the last
x value is the final logits); y: $w_{10-90}$ at that read-out point. One line per prompt pair, with
color, line style and marker all varying together (see legend). Gray dashed = linear response (0.8),
dotted = plateau threshold (0.5). Every pair starts near 0.8 immediately after the patch and narrows
steadily with depth in both models; the control (triangles, dash-dot) is among the fastest to sharpen.

One block after the patch, all ten cells sit at $w_{10-90} \approx 0.79$–$0.81$: the residual stream
still responds almost exactly proportionally to the edit. The width then falls monotonically across the
following 23 blocks, reaching $0.12$–$0.76$ at the logits. A deep stack of nonlinear layers repeatedly
compresses an interpolated direction toward whichever endpoint dominates, and it does so for the
control just as readily as for the test pairs. Read together with Table 3, the most plausible account
of the sign we measure is that this compression is a competition: when the two endpoints predict
disjoint token sets, the two candidate outputs are well separated and the winner changes abruptly
somewhere in the middle of the sweep, whereas two prompts predicting nearly the same token differ only
in small logit components that the stack carries across smoothly. We measure the sign and its size
here; testing that mechanism directly (for instance by patching at a middle block) is left open.

## Conclusion

Interpolating a single token's early-layer activation between two prompts reliably produces a
plateau-then-jump logit response — in 82% (gpt2-medium) and 48% (pythia-410m) of 200 corpus-mined
prompt pairs, and in 9 of 10 hand-picked model-pair cells. The plateau does not indicate that the two
prompts predict the same continuation. Across the mined bank the relationship runs the other way and
is statistically strong: $\rho = -0.55$ in gpt2-medium over the full bank, and $-0.61$ / $-0.45$ in the
two models once pairs at the JSD ceiling are excluded, all with $p < 10^{-7}$, agreeing across three
sharpness statistics and surviving a control for activation geometry. The hand-picked negative control,
chosen so its two continuations clearly differ, plateaus as sharply as the four test pairs.

For anyone using interpolation as an interpretability probe, the actionable point is that plateau shape
needs calibration before it can be evidence for anything. Report an interpolation between prompts of
known dissimilarity alongside the pair of interest and treat the difference as the signal; reading a
sharp plateau as "these two prompts share a continuation" gets the sign backwards, since in our bank
the sharp end of the distribution is where the *most* divergent pairs live. Figure 4 explains why the
bar is that high: the sharpening is supplied by the 23 blocks downstream of the patch, which are the
same 23 blocks in every condition.

**Limitations.** The pairs are constructed by swapping the final token for a lower-ranked alternative,
which spans endpoint divergence well but leaves the two prompts differing only at one position and only
in a way the model itself considered plausible; pairs differing in earlier tokens, or in more than one,
are untested. JSD saturates at $\ln 2$ and 29–37% of mined pairs sit near that ceiling, so the
full-bank correlations in Table 2 understate the effect and Table 3 is the cleaner estimate. All
results are for one patch site (block 0) at one position (the final token) in two models of similar
size; whether patching deeper changes the picture is exactly the experiment the mechanism paragraph
above would need, and we did not run it. Finally, $d(\alpha)$ measures movement in raw logit space, so
a pair could hold its logit vector still while reordering low-probability tokens and this metric would
not see it.
