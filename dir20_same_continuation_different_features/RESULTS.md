# RESULTS — Do last-token activation interpolations induce plateaus?

> CURRENT-BEST ONLY. One row per experiment. No history (that lives in CHANGELOG.md).

## Headline

Interpolating a **single token's** early-layer activation between two prompts produces a
flat-then-abrupt ("plateau") logit response for most prompt pairs — 82% of 200 corpus-mined pairs in
gpt2-medium, 48% in pythia-410m, and 9 of 10 hand-picked model-pair cells including a deliberately
dissimilar control. Endpoint next-token similarity predicts plateau sharpness with the sign **opposite**
to the hypothesis: over the mined bank, Spearman $\rho$(JSD, $w_{TV}$) $= -0.55$ in gpt2-medium
(95% CI $[-0.66,-0.41]$, $p = 6\times10^{-17}$, $n=200$), i.e. the *more* differently the two prompts
predict the next token, the *sharper* the transition. Excluding pairs at the $\ln 2$ divergence
ceiling, both models agree: $\rho = -0.61$ (gpt2-medium, $n=142$) and $-0.45$ (pythia-410m, $n=127$),
both $p<10^{-7}$. Moving the patch site up the stack shows where the shape comes from: with only 3
blocks left below the patch, the median pair is no longer sharp in gpt2-medium (10% sharp, down from
82%) and pythia-410m has returned exactly to the linear response (0% sharp). A plateau here is
therefore not evidence that two prompts share a continuation — it is mostly a product of the depth
that processes the edit.

## Experiment 1 — five hand-picked pairs with a dissimilar control

All 5 pairs tokenized validly in both models (identical prefix, exactly one differing single final
token). Across all 1210 sweeps in this report, patching at $\alpha=0$ and $\alpha=1$ reproduced the
clean runs to $|d| \le 4\times10^{-4}$, so the interpolation harness is correct. Lower $w_{10-90}$ and
$w_{TV}$ mean a sharper transition; higher PF (plateau fraction) means more of the sweep sits at an
endpoint value. The linear-response reference is $w_{10-90}=0.8$, $w_{TV}=0.5$, $\mathrm{PF}=0.2$.

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

Nine of ten cells beat the linear baseline on the threshold-free statistic ($w_{TV} \le 0.27$ vs 0.5),
so plateaus are real. The two pairs the plan expected to plateau most strongly — the smallest endpoint
divergences, `Mary`/`her` at 0.033–0.068 nats and `four`/`4` at 0.056–0.138 nats — give the widest,
most nearly linear transitions in the table. The control, at the largest divergence measured, is
sharper than both of them in gpt2-medium on $w_{TV}$ and sharper than three of four test pairs in
pythia-410m. At $n=10$ cells only $|\rho| \ge 0.75$ would have been detectable, so these five pairs
motivate the powered test below rather than settling it.

The verdict rests first on the shape of the raw sweeps, so we show all ten before any summary
statistic. If plateaus tracked continuation similarity, the top two rows (lowest JSD) would be the
flattest and the bottom row (the control) the most linear.

![Relative distance versus interpolation position for five prompt pairs in two models](plots/final_logit_curves.png)

**Figure 1.** Final-logit response to interpolating one token's block-0 activation. x: interpolation
position $\alpha$ from prompt A (0) to prompt B (1); y: relative distance $d$ (0 = at A's logits,
1 = at B's logits). Solid curve with circles = measured $d(\alpha)$; gray dashed = the linear
reference $d=\alpha$. Rows are prompt pairs, columns are models; the bottom row (thick frame) is the
control pair, whose continuations differ most. Every panel except pythia-410m `four`/`4` bends well
away from the diagonal into a flat-then-jump shape, and the control bends as much as the test pairs.

## Experiment 2 — 200 corpus-mined pairs per model

Each mined pair is a WikiText-103 prefix (40 prefixes, 10–40 tokens) plus the model's own top-1 next
token versus a token at rank $r$, with $r$ log-uniform in $[1,5000]$, so endpoint JSD spans its whole
range (0.002–0.693 nats in gpt2-medium, 0.007–0.693 in pythia-410m). To ask how much information a
plateau carries at all, we first measure how often an arbitrary pair plateaus and where the
hand-picked pairs land inside that distribution.

![Distribution of transition sharpness over 200 mined prompt pairs per model, with the five hand-picked pairs marked](plots/bank_prevalence.png)

**Figure 2.** Plateaus are the norm for arbitrary prompt pairs. x: $w_{TV}$ at the final logits
(smaller = sharper); y: number of mined pairs per bin (gray hatched histogram, $n=200$ per model).
Gray dashed = linear response (0.5), dotted = sharpness threshold (0.25). The five markers on the
strip above each histogram are the hand-picked pairs of Table 1 at their $w_{TV}$ values (shape and
color per the legend, control with a thick black edge); their y position carries no meaning.

Median $w_{TV}$ is 0.080 in gpt2-medium (82% of pairs sharp) and 0.266 in pythia-410m (48% sharp);
median $w_{10-90}$ is 0.241 and 0.593. Every hand-picked pair sits inside the bulk of its model's
distribution, so a plateau observed for one chosen pair in gpt2-medium is close to uninformative —
four in five random pairs do the same.

With $n=200$ the association test detects $|\rho| \ge 0.14$, and it finds a clear effect running
against the hypothesis.

![Endpoint divergence against two sharpness statistics for 200 mined pairs per model, with fits](plots/bank_regression.png)

**Figure 3.** Endpoint divergence predicts sharpness, with the sign opposite to the hypothesis.
x (all panels): endpoint JSD in nats (larger = more different next-token predictions); the dash-dot
vertical line is the $\ln 2$ ceiling JSD attains for disjoint predictions. y: $w_{10-90}$ (top) and
$w_{TV}$ (bottom) at the final logits, smaller = sharper. Columns are models. Light circles = the 200
mined pairs; solid line = OLS fit; dashed line with squares = quintile means of JSD with $\pm1$ SE;
stars = the five hand-picked pairs (thick black edge = control). Gray dashed = linear response, dotted
= plateau threshold. Both models trend downward; pythia-410m's last two quintile means turn back up,
and both sit at the JSD ceiling.

Negative $\rho$ for a width and positive $\rho$ for PF both mean "more divergent endpoints → sharper
plateau", the opposite of what the hypothesis predicts. Intervals are 95% cluster bootstrap over the
40 prefixes.

| Model | statistic | $\rho$ | 95% CI | $p$ | OLS slope (per nat) |
|---|---|---|---|---|---|
| gpt2-medium | $w_{10-90}$ | $-0.47$ | $[-0.59, -0.33]$ | $1.4\times10^{-12}$ | $-0.64\ [-0.78, -0.49]$ |
| gpt2-medium | $w_{TV}$ | $-0.55$ | $[-0.66, -0.41]$ | $6.2\times10^{-17}$ | $-0.42\ [-0.54, -0.31]$ |
| gpt2-medium | PF | $+0.44$ | $[+0.31, +0.56]$ | $5.7\times10^{-11}$ | — |
| pythia-410m | $w_{10-90}$ | $-0.12$ | $[-0.30, +0.05]$ | $0.090$ | $-0.24\ [-0.35, -0.11]$ |
| pythia-410m | $w_{TV}$ | $-0.11$ | $[-0.30, +0.07]$ | $0.123$ | $-0.20\ [-0.31, -0.07]$ |
| pythia-410m | PF | $+0.12$ | $[-0.06, +0.31]$ | $0.090$ | — |

All three statistics agree on the direction in both models. In gpt2-medium the size is substantial:
going from identical to disjoint next-token predictions ($0 \to 0.69$ nats) shortens the transition by
$0.29$ of the sweep on $w_{TV}$, and the quintile means fall monotonically
($0.35 \to 0.13 \to 0.09 \to 0.07 \to 0.08$).

Pythia-410m looks weak above because 37% of its mined pairs sit at or above JSD 0.65, essentially at
the $\ln 2$ ceiling where JSD can no longer order them. Restricted to pairs below that ceiling, both
models show the effect strongly:

| Model | $n$ | $\rho$ ($w_{TV}$) | $p$ | $\rho$ ($w_{10-90}$) | $p$ |
|---|---|---|---|---|---|
| gpt2-medium | 142 | $-0.61$ | $1.5\times10^{-15}$ | $-0.54$ | $4.9\times10^{-12}$ |
| pythia-410m | 127 | $-0.45$ | $9.0\times10^{-8}$ | $-0.47$ | $2.3\times10^{-8}$ |

**Not explained by activation geometry.** Controlling for the block-0 angle $\Omega$ between the two
patched vectors, the partial Spearman correlation of JSD with $w_{TV}$ is $-0.55$ in gpt2-medium
(raw $-0.55$; $\rho(\Omega,\mathrm{JSD}) = 0.03$) and $-0.16$ in pythia-410m (raw $-0.11$;
$\rho(\Omega,\mathrm{JSD}) = 0.31$). $\Omega$ itself correlates only weakly with sharpness
($\rho = 0.16$ in both).

## Experiment 3 — where the sharpness accumulates

Sharpness has to originate somewhere in the network. Recomputing the same width at every block's
residual stream between the patch site and the output locates it.

![Transition width versus recording block for five prompt pairs in two models](plots/layerwise_widths.png)

**Figure 4.** The plateau is built up gradually across depth, not created at the patch site.
x: block whose `resid_post` is read out (patch is applied after block 0; the last x value is the final
logits); y: $w_{10-90}$ at that read-out point. One line per prompt pair (color, line style and marker
all vary together; see legend). Gray dashed = linear response (0.8), dotted = plateau threshold (0.5).
Every pair starts near 0.8 just after the patch and narrows monotonically with depth in both models;
the control (triangles, dash-dot) is among the fastest to sharpen.

One block after the patch every cell sits at $w_{10-90} \approx 0.79$–$0.81$ — an almost exactly
proportional response — and the width then falls monotonically over the following 23 blocks to
$0.12$–$0.76$ at the logits. The shape is supplied by the depth downstream of the patch, in every
condition alike, which is why it cannot by itself carry information about the prompt pair.

## Experiment 4 — take depth away and the plateau goes with it

Figure 4 only shows *where* sharpness accumulates; it cannot show that the downstream blocks *cause*
it, because reading out earlier is not the same as computing less. The causal version moves the patch
site instead: re-running the identical 200-pair bank with the interpolated vector inserted after block
12 and after block 20 leaves 11 and 3 blocks to process it instead of 23, with everything else fixed.

![Median transition width and JSD-sharpness correlation against patch site for two models](plots/depth_effect.png)

**Figure 5.** Removing downstream blocks removes the plateau, but not the divergence effect in
gpt2-medium. x (both panels): the patch site — the block whose `resid_post` at the final token is
replaced by the interpolated vector — labelled with the number of blocks remaining below it.
Left y: median $w_{TV}$ at the final logits over the 200 mined pairs (smaller = sharper), shaded band
= interquartile range, gray dashed = linear response (0.5), dotted = sharp threshold (0.25).
Right y: Spearman $\rho$ between endpoint JSD and $w_{TV}$ over the pairs below the $\ln 2$ ceiling
(JSD $< 0.65$; $n=142$ gpt2-medium, $n=127$ pythia-410m), error bars = 95% cluster bootstrap over the
40 prefixes, gray dashed = no association. gpt2-medium = circles, solid; pythia-410m = squares, dashed.

| Model | patch site | blocks below | median $w_{TV}$ | % sharp | median $w_{10-90}$ | monotonic | $\rho$(JSD, $w_{TV}$), JSD $<0.65$ | 95% CI | $p$ |
|---|---|---|---|---|---|---|---|---|---|
| gpt2-medium | block 0 | 23 | 0.080 | 82.0% | 0.241 | 7.5% | $-0.61$ | $[-0.70,-0.46]$ | $1.5\times10^{-15}$ |
| gpt2-medium | block 12 | 11 | 0.250 | 50.5% | 0.556 | 33% | $-0.53$ | $[-0.64,-0.39]$ | $1.1\times10^{-11}$ |
| gpt2-medium | block 20 | 3 | 0.383 | 10.0% | 0.701 | 72% | $-0.53$ | $[-0.66,-0.38]$ | $8.2\times10^{-12}$ |
| pythia-410m | block 0 | 23 | 0.266 | 47.5% | 0.593 | 98% | $-0.45$ | $[-0.63,-0.24]$ | $9.0\times10^{-8}$ |
| pythia-410m | block 12 | 11 | 0.419 | 2.5% | 0.749 | 100% | $-0.44$ | $[-0.62,-0.23]$ | $2.4\times10^{-7}$ |
| pythia-410m | block 20 | 3 | 0.509 | 0.0% | 0.808 | 100% | $+0.04$ | $[-0.11,+0.22]$ | $0.62$ |

**The plateau is manufactured by the layers below the patch.** Sharpness falls off monotonically as
depth is removed, in both models and on both width statistics. The clearest single number is
pythia-410m at block 20: median $w_{TV} = 0.509$ and median $w_{10-90} = 0.808$, against the linear
response's $0.5$ and $0.8$ — with 3 blocks left, not one of its 200 pairs is sharp and the average
response is proportional to the edit to within 2%. gpt2-medium is more resistant (10% of pairs still
sharp with 3 blocks below) but moves the same way, from 82% to 10%. The non-monotonic wiggles that
motivated $w_{TV}$ are also a deep-stack product: the fraction of monotonic gpt2-medium curves rises
from 7.5% to 72% as depth is removed.

**The divergence effect is not simply the same thing.** If more divergent pairs were sharper only
because deep stacks compress them harder, the correlation should decay along with the plateau. In
gpt2-medium it does not: $\rho$ is $-0.61$, $-0.53$, $-0.53$ at 23, 11 and 3 remaining blocks, flat
within the bootstrap intervals even where 90% of pairs no longer plateau. In pythia-410m it holds at
$-0.45$ and $-0.44$ and then disappears ($+0.04$, $p=0.62$) precisely at the site where the response
has become linear and there is no transition shape left to modulate. So the two findings are
separable: depth sets *how much* the response is compressed, while endpoint divergence sets *which*
pairs compress more, and the latter is already present in the last three blocks of gpt2-medium.
