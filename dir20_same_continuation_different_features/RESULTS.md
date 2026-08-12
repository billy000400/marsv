# RESULTS — Do internal feature differences explain transition width at matched successor JSD?

> CURRENT-BEST ONLY. One row per experiment. No history (that lives in CHANGELOG.md).

## Headline

Two prompts that differ in one token can have almost the same next-token prediction and still behave
completely differently when you interpolate between their internal activations: one flips from A's
answer to B's answer in a narrow window, the other drifts across smoothly. This report tests one
explanation for that difference and finds it holds.

**The prediction.** Among prompt pairs matched on how much their outputs differ, the pair that engages
more *different* downstream MLP neurons switches more sharply.

**The test, in two banks.** In GPT-2 Large we mined every eligible paragraph of the WikiText-103
**test** split — 1395 prefixes, 385020 candidate final-token pairs — and locked 101 within-prefix
contrasts before computing a single interpolation curve. Each contrast holds two pairs that share a
prefix, use four distinct final tokens, are matched on successor JSD and four endpoint-geometry
confounds, and differ in the feature difference $F$. The manifest was hashed (`2415f5ff…`) before any
width was measured. We call this the **amended analysis**: the plan had fixed the bank at 300 prefixes
and required us to stop with an underpowered verdict below 40 contrasts, and when 300 prefixes yielded
21 contrasts the bank was enlarged to 1395 rather than stopped. No width had been computed at the time,
so the enlargement was blind to the outcome, but it broke a rule frozen in advance. A second,
**pre-registered independent replication** was therefore run on the WikiText-103 **train** split — a
split untouched by any analysis here — with the bank size fixed at 1400 prefixes and a single-run
stopping rule written to disk before any of its data was scored.

**The result: supported in both banks, with a large effect.** In the amended analysis the high-$F$
member has the sharper switch in 83 of 101 contrasts; median transition width falls from
$w_{TV} = 0.203$ (low-$F$) to $0.098$ (high-$F$), a median paired difference of $\Delta w = -0.071$,
95% CI $[-0.087, -0.058]$, paired permutation $p < 10^{-4}$. The replication returns $-0.064$, 95% CI
$[-0.091, -0.043]$, 78.8% predicted sign, $p < 10^{-4}$ over 99 contrasts, and meets all four clauses
of the gate written before it was run ($n \ge 80$, median $\Delta w \le -0.05$, $\ge 60$% predicted
sign, CI below zero). The association is therefore a confirmed result; the amended analysis is the
better-powered estimate of its size. Confounds are matched to standardized mean differences of $0.03$
(JSD), $0.005$ (norm ratio) and $0.02$ (surprisal) in the amended bank, against $1.51$ for $F$ itself.

**And the same neurons carry the switch causally (amended bank only).** Forcing exactly the neurons that distinguish the
two endpoints to interpolate linearly — 1.7% of the MLP neurons below the patch, with both endpoints
left bit-identical — widens the median switch from $w_{TV} = 0.144$ to $0.471$, essentially the linear
response of 0.5. An equal-size control set matched on activation magnitude, endpoint gap and
output-weight norm moves it to $0.167$. The gap is $+0.275$ (95% CI $[0.251, 0.298]$) and every one of
the 202 pairs shows it.

| Stage | What it establishes | Headline number |
|---|---|---|
| S1 sanity | The harness reproduces the reported contrast | $w_{TV}$ 0.012 (`big`/`in`) vs 0.292 (`big`/`large`) |
| S2 locking (amended) | 101 matched contrasts, outcome-blind | $\Delta F$ median 0.095; JSD SMD +0.030 |
| S3 primary (amended) | Higher feature difference → sharper switch | median $\Delta w = -0.071$, CI $[-0.087, -0.058]$ |
| S3 robustness (amended) | Effect survives the residual imbalance | adjusted intercept $-0.085 \pm 0.013$ |
| S3R replication (pre-registered) | The same effect on an untouched corpus split | median $\Delta w = -0.064$, CI $[-0.091, -0.043]$, $n = 99$ |
| S4 causal (amended) | The differential neurons carry the switch | median $w_{TV}$ 0.471 (differential) vs 0.167 (control) |

**What it does not show.** $F$ counts top-scoring MLP neurons; those neurons are a proxy for features,
not proven semantic units. Both banks are one model (GPT-2 Large), one patch site (block-0
`resid_post` at the final token) and one interpolation rule. The primary calipers had to be relaxed
once, as pre-specified, in each bank (4 and 5 contrasts before relaxation, 101 and 99 after). The
replication is independent in data and protocol, not in personnel: same code, same authors, so a
systematic error shared by both banks would survive it, and replication by another group remains
outstanding. The causal experiment was run on the amended bank only and has no pre-registered
counterpart.

---

## S1 — the harness reproduces the reported contrast

Before mining anything we ran the two prompt pairs the phenomenon was originally reported on, through
the exact frozen protocol. The gate was fixed in advance: endpoint reconstruction error below $10^{-4}$,
and the plateau case must come out sharper than the smooth case. Both passed with three orders of
magnitude of margin on the error.

| Pair | successor JSD | $w_{TV}$ | $w_{10\text{-}90}$ | non-monotonicity | endpoint error |
|---|---|---|---|---|---|
| `The house was` + ` big` / ` in` | 0.663 | **0.012** | 0.044 | 0.000 | $3.5\times10^{-7}$ |
| `The house was` + ` big` / ` large` | 0.053 | **0.292** | 0.592 | 0.000 | $3.2\times10^{-7}$ |

The two curves are 24-fold apart in $w_{TV}$, so the measurement has ample dynamic range for the matched
test that follows. Figure 1 shows both the raw curves and the cumulative-variation construction that
$w_{TV}$ reads off.

![Two interpolation curves and their cumulative variation](plots/matthew_sanity.png)

**Figure 1.** The block-0 final-token interpolation in GPT-2 Large for the two originally reported
pairs. Left — x: interpolation position $\alpha \in [0,1]$; y: relative distance $d(\alpha)$, the
fraction of the way the final-token logit vector has moved from prompt A to prompt B. Right — x:
$\alpha$; y: cumulative variation $c(\alpha)$, the share of the curve's total up-and-down movement
spent by position $\alpha$; the two dotted horizontals are the 0.25 and 0.75 levels whose $\alpha$-gap
defines $w_{TV}$. Solid with circles = ` big`/` in` (the plateau case); dashed with squares =
` big`/` large` (the smooth case); dotted gray = the linear response $d(\alpha) = \alpha$.

---

## S2 — the locked matched bank (amended analysis)

The whole design rests on choosing the contrasts before seeing any width. We scored all 385020
candidate pairs on successor JSD, the feature difference $F$ and four confounds, applied the
pre-specified eligibility window, and only then searched within each prefix for one matched contrast.
The primary calipers yielded 4 contrasts, so the single pre-specified relaxation was applied, giving
101. The 1395 prefixes are the amendment: the plan fixed 300, which yielded 21 contrasts, below the
plan's own floor of 40 at which it required us to stop.

| Quantity | Value |
|---|---|
| Prefixes (WikiText-103 test, seed 31, 20–40 tokens) | 1395 |
| Candidate final-token pairs (24 candidates per prefix) | 385020 |
| Eligible pairs ($0.005 \le \mathrm{JSD} \le 0.20$, logit distance $>$ p10 $= 233.2$) | 26275 |
| Contrasts under the primary calipers | 4 |
| Contrasts under the single pre-specified relaxation (used) | **101** |
| $\Delta F$ across the locked contrasts (median, range) | 0.095 (0.080–0.187) |
| Confound distance across the locked contrasts (median, cap) | 0.62 (0.75) |
| Manifest sha256, recorded before any sweep | `2415f5ff6dfcf88f…` |

The eligibility window is the binding constraint, not the calipers: only 6.8% of candidate pairs have a
successor JSD below 0.20, because two arbitrary high-probability continuations of the same prefix
usually predict very different next tokens (bank median JSD 0.562). The feature difference is high and
narrow across the bank (median $F = 0.904$, 5th–95th percentile 0.723–0.954), which is why $\Delta F$
had to be measured in tenths and why the relaxation to $\Delta F \ge 0.08$ mattered.

Balance is what makes the comparison a contrast on $F$ alone, so we report it before any outcome. All
five matched variables sit on the diagonal in Figure 2 while $F$ is displaced by construction.

| Matched variable | high-$F$ mean | low-$F$ mean | standardized mean difference |
|---|---|---|---|
| successor JSD (nats) | 0.0967 | 0.0951 | **+0.030** |
| block-0 \|log norm ratio\| | 0.1147 | 0.1143 | **+0.005** |
| mean final-token surprisal (nats) | 5.218 | 5.175 | **+0.025** |
| final-logit L2 distance | 288.0 | 279.1 | +0.231 |
| block-0 endpoint angle (rad) | 1.0614 | 1.0267 | +0.252 |
| feature difference $F$ (the variable under test) | 0.8652 | 0.7622 | **+1.506** |

Three of the five confounds are matched to within 3% of a standard deviation. The final-logit distance
and block-0 angle carry a residual imbalance of about a quarter of a standard deviation, both in the
direction that would flatter the hypothesis; S3's robustness section removes it and the effect survives.

![Scatter of each matched variable, high-F member against low-F member](plots/matching_balance.png)

**Figure 2.** Balance of the 101 locked contrasts. Each panel is one variable; x: its value for the
low-$F$ member of the contrast, y: its value for the high-$F$ member; the dotted line is $y = x$.
Points hugging the diagonal mean the variable is matched. The five confound panels do; the sixth
panel, feature difference $F$, sits entirely above the diagonal because that is the variable the
contrast varies. Each panel title gives the standardized mean difference (SMD), the mean gap divided
by the pooled standard deviation.

---

## S3 — the matched prediction, and it holds (amended analysis)

Every locked contrast was swept identically: 101 interpolation points at block-0 `resid_post` of the
final token, readout at the final-token logits. Endpoint reproduction was exact to $9.2\times10^{-7}$
across all 202 sweeps, so the widths describe the model and not patching error.

| Summary | Value | Gate | Met |
|---|---|---|---|
| Contrasts $n$ | 101 | $\ge 80$ | yes |
| Median $\Delta w = w_{TV}(\text{high-}F) - w_{TV}(\text{low-}F)$ | **$-0.0708$** | $\le -0.05$ | yes |
| Prefix bootstrap 95% CI on the median | $[-0.0866, -0.0582]$ | below 0 | yes |
| Fraction with the predicted sign ($\Delta w < 0$) | **82.2%** (83/101) | $\ge 60$% | yes |
| Paired permutation $p$ (10000 sign flips) | $< 10^{-4}$ | — | — |
| Median $w_{TV}$, low-$F$ → high-$F$ | $0.203 \rightarrow 0.098$ | — | — |
| Median $w_{10\text{-}90}$, low-$F$ → high-$F$ | $0.512 \rightarrow 0.316$ | — | — |
| Verdict | **supported** | | |

The effect is not a small shift of a noisy statistic. Doubling the sharpness of the median pair — a
window of 20% of the interpolation range down to 10% — comes from a median $\Delta F$ of 0.095, i.e.
about a 10-point difference in how many of the top-scoring MLP neurons the two endpoints share. The
secondary width $w_{10\text{-}90}$ moves the same way, so the result is not an artifact of the
total-variation definition. Figure 3 shows the paired lines and the full $\Delta w$ distribution.

![Paired low-F to high-F width lines and the distribution of their differences](plots/matched_widths.png)

**Figure 3.** The primary result. Left — each thin line is one contrast; x: which member (low-$F$ at
left, high-$F$ at right); y: transition width $w_{TV}$ (smaller = sharper switch). Solid lines fall
(the predicted direction), dashed lines rise; the heavy black line with circles joins the two medians,
$0.203 \rightarrow 0.098$. Right — x: the paired difference $\Delta w$; y: number of contrasts. The
dotted vertical is zero, the dashed vertical is the median $-0.071$, and the shaded band is its
bootstrap 95% CI.

### Robustness to the residual confound imbalance (post-hoc)

Because the high-$F$ member carries a slightly larger final-logit distance and block-0 angle, the
question is whether those, and not $F$, produce the effect. Two checks answer it. Restricting to
contrasts where the high-$F$ member is *not* favoured on a confound keeps the effect; and a linear
model of $\Delta w$ on the five paired confound differences puts the effect at a matched contrast in
the intercept.

| Analysis | $n$ | median $\Delta w$ | 95% CI | fraction $< 0$ |
|---|---|---|---|---|
| All locked contrasts (primary) | 101 | $-0.0708$ | $[-0.0866, -0.0582]$ | 0.822 |
| Contrasts where high-$F$ has the smaller final-logit distance | 30 | $-0.0562$ | $[-0.0918, -0.0185]$ | 0.733 |
| Contrasts where high-$F$ has the smaller block-0 angle | 25 | $-0.0823$ | $[-0.1557, -0.0260]$ | 0.840 |
| Both at once | 5 | $-0.0253$ | $[-0.1992, +0.0185]$ | 0.800 |
| Covariate-adjusted intercept ($\pm$ s.e.) | 101 | $-0.0847 \pm 0.0131$ | $[-0.1104, -0.0590]$ | — |

The two single-confound subsets keep a significant negative effect at a quarter of the sample size, and
the covariate-adjusted estimate is slightly *larger* in magnitude than the raw one. The five paired
confound differences together explain 5.2% of the variance in $\Delta w$, so the residual imbalance is
not doing the work. The both-at-once cell has 5 contrasts and settles nothing on its own.

### Supporting cases and counterexamples

An aggregate median can hide a bimodal population, so we plot the extremes of the distribution
directly. Figure 4 shows the five contrasts with the most negative $\Delta w$ and the five with the
most positive, with each pair's tokens, $F$ and width printed on the panel.

![Ten interpolation curve pairs: five strongest supporting contrasts and five counterexamples](plots/example_curves.png)

**Figure 4.** Raw curves at the two extremes. Every panel is one contrast; x: interpolation position
$\alpha$; y: relative distance $d(\alpha)$; dotted gray is the linear response. Solid with circles =
the high-$F$ member, dashed with squares = the low-$F$ member; the legend gives each member's two final
tokens, its $F$ and its $w_{TV}$; the panel title gives the prefix index, the pair's successor JSD and
$\Delta w$. Top row: the five strongest supporting contrasts (down to $\Delta w = -0.387$). Bottom row:
the five strongest counterexamples (up to $\Delta w = +0.262$).

The counterexamples are real and instructive: in the worst one (prefix 863) the low-$F$ pair
`ices`/`ression` is already almost a perfect step ($w_{TV} = 0.080$), leaving no room to be sharper, so
the high-$F$ member can only lose. Contrasts in which the low-$F$ member is already near-maximally
sharp are where the prediction has no headroom, which is a floor effect and not a counter-mechanism.

---

## S3R — the pre-registered independent replication

S3's bank reached 101 contrasts only because the planned 300 prefixes were enlarged to 1395 after the
first count came in low, so S3 cannot be read as a confirmatory test however outcome-blind that
enlargement was. S3R fixes that. Before any of its data was scored we wrote to disk: the corpus (the
WikiText-103 **train** split, never analysed in this direction), the sampling seeds, a bank size of
**exactly 1400 prefixes**, an instruction to run the bank **once** with no enlargement, re-seeding,
re-drawing or second relaxation, an identical protocol in every other respect, and the same four-clause
decision rule. The bank was locked to `results/matched_pairs_rep.json` and hashed
(`ed1df0866f012b61…`) before its first interpolation curve existed.

| Quantity | Amended (S2/S3) | Replication (S3R) |
|---|---|---|
| Corpus split | WikiText-103 test | WikiText-103 train |
| Prefixes | 1395 (planned 300, enlarged) | 1400 (fixed in advance, run once) |
| Candidate final-token pairs | 385020 | 386400 |
| Eligible pairs | 26275 | 25321 |
| Contrasts: primary calipers → relaxation | 4 → **101** | 5 → **99** |
| Median $\Delta w$ | $-0.0708$ | **$-0.0641$** |
| Bootstrap 95% CI on the median | $[-0.0866, -0.0582]$ | $[-0.0908, -0.0426]$ |
| Fraction with the predicted sign | 82.2% (83/101) | 78.8% (78/99) |
| Paired permutation $p$ | $< 10^{-4}$ | $< 10^{-4}$ |
| Median $w_{TV}$, low-$F$ → high-$F$ | $0.203 \rightarrow 0.098$ | $0.173 \rightarrow 0.095$ |
| Gate ($n \ge 80$, median $\le -0.05$, $\ge 60$% sign, CI $< 0$) | met | **met, as pre-registered** |

The replication's balance matches S2's: standardized mean differences of $+0.03$ on successor JSD,
$-0.05$ on the block-0 log norm ratio and $+0.09$ on surprisal, against $+1.63$ on $F$, with the same
residual imbalance on final-logit distance ($+0.20$) and block-0 angle ($+0.29$). The effect sizes agree
— $-0.064$ against $-0.071$, each inside the other's confidence interval — and both banks show the
median width of the high-$F$ member at roughly half that of its matched low-$F$ partner. The
replication's interval is the wider of the two ($0.048$ across against $0.028$) and its upper end
$-0.043$ sits closer to zero, which is what 99 contrasts from a different corpus split should look
like. Figure 5 shows both estimates on one axis.

![Two horizontal confidence intervals and paired bars comparing the amended analysis with the replication](plots/replication_forest.png)

**Figure 5.** The amended analysis against the pre-registered independent replication. Left — x: median
paired difference $\Delta w = w_{TV}(\text{high-}F) - w_{TV}(\text{low-}F)$, negative meaning the
high-$F$ member switches more sharply; each horizontal bar is one bank's bootstrap 95% CI with its
median marked (circle = amended analysis, test split, $n = 101$; square = replication, train split,
$n = 99$); the dashed vertical is the gate threshold $-0.05$, the dotted vertical is zero. Right — x:
which member of the contrast; y: that group's median transition width $w_{TV}$; bars hatched `//` are
the amended analysis, `xx` the replication; the dotted horizontal at $0.5$ marks a perfectly
proportional response.

---

## S4 — the differential neurons carry the switch (amended analysis)

S3 establishes an association, so S4 asks whether the differential neurons are doing the work. For both
members of all 101 contrasts — 202 pairs — we re-ran the identical sweep twice more. In the
**differential** condition the post-GELU activations of the neurons in the symmetric difference of the
two endpoints' top-64-per-block sets (blocks 1–35) are overwritten at every $\alpha$ with the straight
line between their two endpoint values. In the **control** condition the same is done to an equal-size,
per-block set matched on mean contribution magnitude, endpoint activation gap and output-weight norm.
Both conditions leave the two endpoints bit-identical by construction, verified to $8.9\times10^{-7}$.

| Condition | median $w_{TV}$ | median change from unablated |
|---|---|---|
| Unablated (the S3 sweeps) | 0.144 | — |
| Control set linearized (1.7% of neurons) | 0.167 | $+0.019$ |
| Differential set linearized (1.7% of neurons) | **0.471** | $+0.308$ |
| Gap (differential $-$ control), median | | **$+0.275$**, 95% CI $[0.251, 0.298]$ |
| Pairs with the predicted sign | 202/202 (**100%**) | permutation $p < 10^{-4}$ |

Linearizing a median of 3063 neurons — 1.7% of the 179200 MLP neurons below the patch — takes the
median pair from a switch that completes in 14% of the interpolation range to one that is within 0.03
of a proportional response. The matched control touches the same number of neurons in the same blocks
with the same activation statistics and moves the median by 0.019, so the effect belongs to *which*
neurons were linearized. The two S3 groups converge under the intervention: the high-$F$ member goes
$0.098 \rightarrow 0.467$ and the low-$F$ member $0.203 \rightarrow 0.474$, ending at the same place.
That is what a mechanism looks like — the width difference S3 measured lives in the neurons that
distinguish the endpoints, and removing their nonlinearity removes the difference along with the
switch. Figure 6 shows that the effect is not confined to the median: all 202 pairs move the same way.

![Per-pair widths under three conditions and the distribution of the differential-minus-control gap](plots/causal_linearization.png)

**Figure 6.** The causal test. Left — each thin line is one of the 202 pairs across the three
conditions; x: condition (unablated, control linearized, differential linearized); y: transition width
$w_{TV}$; the heavy black line with circles joins the medians and the dotted horizontal marks the
linear response $w_{TV} = 0.5$. Right — x: the per-pair gap, how much more the differential
linearization widened the switch than the control did; y: number of pairs. The dotted vertical is
zero, the dashed vertical the median $+0.275$, and the shaded band its bootstrap 95% CI; the whole
distribution lies above zero.

---

## Verdict

The primary hypothesis is **supported**, and the support now has two tiers. Among GPT-2 Large prompt
pairs matched on successor JSD and endpoint geometry, the pair engaging more different downstream MLP
features has the sharper block-0 transition. The pre-registered replication S3R establishes this as a
confirmed result — 99 contrasts on an untouched corpus split, median $\Delta w = -0.064$, CI
$[-0.091, -0.043]$, 78.8% predicted sign, all four gate clauses met under a stopping rule fixed in
advance. The amended analysis S2/S3 gives the better-powered estimate of the same effect
($-0.071$, CI $[-0.087, -0.058]$, 82.2% predicted sign) and carries the two supporting findings: the
effect survives removing the residual confound imbalance, and forcing exactly the differential neurons
to interpolate linearly collapses the switch to the linear response while a matched control does not.
Those two rest on the amended bank alone and await a pre-registered replication of their own.
