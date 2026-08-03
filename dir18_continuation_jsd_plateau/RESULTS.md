# RESULTS — Does training-corpus next-token divergence predict transition sharpness?

> CURRENT-BEST ONLY. One row per experiment. No history, no superseded/weaker variants
> (those live in CHANGELOG.md). Full method definitions are in REPORT.md.

**Question.** Can a statistic computed from the training corpus tell you which input pairs a model
separates sharply? For each endpoint token we count what token comes **immediately** after it in
2.05B tokens of Pythia's actual released training stream, and take the base-2 Jensen-Shannon
divergence between two endpoints' next-token distributions as the predictor. The version estimated on
the **holdout** split, $\widehat J_{\mathrm{hold}}(u,v)$, is the predictor in every test; the
**selection** split's version $\widehat J_{\mathrm{sel}}(u,v)$ only defined the strata and picked the
pairs. The outcome is the **10%–90% transition width** $w = t(d=0.9) - t(d=0.1)$ of the
relative-logit coordinate $d(t)$ measured while interpolating the model's post-block-0 residual state
between the two endpoints. **Smaller $w$ = the output flips between the two words over a shorter
stretch of the path.** A negative correlation means higher corpus divergence predicts a sharper flip.

**Headline.** *Within a stratified bank of high-frequency, model-plausible **single-token word-start**
endpoint pairs, held-out corpus immediate-next-token JSD is associated with **narrower median
10%–90% relative-logit transitions**.* This is the prespecified verdict branch *"corpus JSD predicts
model-output JSD and smaller $w$; step 0 does not"*. The headline number is a **total association** —
observational, not causal. It is attenuated after adjustment, and the fully adjusted estimate is not
statistically significant.

## Primary result — prespecified top-256 bank

The primary bank is 60 endpoint-disjoint pairs (14/13/11/10/12 per $\widehat J_{\mathrm{sel}}$
quintile), 3 carrier contexts each, 50 interpolation positions, 180 raw curves per checkpoint. Only 60
pairs are possible because the top-256 filter leaves 123 eligible endpoints and forbidding endpoint
reuse permits at most 61 disjoint pairs; that removes *direct* dependence through a shared endpoint
without making the pairs fully independent. CIs are 95% from 10,000 bootstrap resamples over pairs.

| Result | Trained 1.4B (step143000) | Untrained 1.4B (step0) | 410M (step143000) |
|---|---|---|---|
| Spearman ρ($\widehat J_{\mathrm{hold}}$, $w$) — **primary** | **−0.525** [−0.701, −0.304], p=1.7e−5 | −0.056 [−0.314, +0.211], p=0.67 | −0.512 [−0.711, −0.272], p=2.9e−5 |
| Same with $\widehat J_{\mathrm{sel}}$ (split sensitivity) | −0.526 | −0.053 | −0.511 |
| Partial ρ (freq, entropy, surprisal, block-0 cos & dist adjusted) | −0.384 | −0.142 | −0.396 |
| Spearman ρ($\widehat J_{\mathrm{hold}}$, model output JSD) | **+0.751** [+0.615, +0.843], p=4.9e−12 | +0.145 [−0.122, +0.394], p=0.27 | +0.749 [+0.611, +0.838] |
| Median $w$ (IQR) | 0.541 (0.169) | 0.831 (0.006) | 0.640 (0.133) |
| Median $w$ by quintile Q1→Q5 | 0.619, 0.608, 0.462, 0.502, 0.479 | 0.831, 0.832, 0.833, 0.830, 0.828 | 0.723, 0.683, 0.610, 0.582, 0.578 |
| Median edge drift $E$ (0 = flat ends; 0.184 = no plateau) | **0.076** | 0.213 | 0.109 |
| Valid-curve rate under the strict criteria (and in every bin) | 1.000 | 1.000 | 1.000 |

Which corpus split supplies the predictor makes no difference:
ρ($\widehat J_{\mathrm{sel}}$, $\widehat J_{\mathrm{hold}}$) = 0.99972 on the bank, so the two are
interchangeable; the holdout version is reported because it played no part in ordering or choosing
the pairs. The 410M row is a **cross-scale robustness check** — same corpus estimates, same frozen
bank — and not an independent replication.

**Learned sharpening and mediation.** Two follow-up analyses on the same 60 pairs. (a) *Learned
sharpening*: predicting the **change** training produced in each pair's width,
$\Delta w = w(\text{trained}) - w(\text{step }0)$, which subtracts each pair's own untrained baseline.
(b) *Mediation*: the model's own output divergence $JSD_{\mathrm{out}}$ is the obvious pathway from
corpus statistics to transition shape, so we ask how much of the association is left after adjusting
for it. Adjusted p-values come from the residual rank correlation.

| Association with corpus $\widehat J_{\mathrm{hold}}$ | ρ | 95% CI | p |
|---|---|---|---|
| Trained width $w$ — **headline, unadjusted (total association)** | **−0.525** | [−0.701, −0.304] | 1.7e−5 |
| Learned sharpening $\Delta w$ | **−0.517** | [−0.694, −0.294] | 2.3e−5 |
| $w$, adjusted for the mediator $JSD_{\mathrm{out}}$ | −0.277 | [−0.509, −0.002] | 0.032 |
| $w$, adjusted for $JSD_{\mathrm{out}}$ + the 5 covariates | −0.204 | [−0.471, +0.080] | **0.119 (n.s.)** |
| $w$, adjusted for the 5 covariates only | −0.384 | [−0.623, −0.110] | 0.0024 |

Using $\Delta w$ in place of $w$ in the two adjusted rows gives −0.263 (p = 0.042) and −0.198
(p = 0.129) — the same picture. Median $\Delta w$ = −0.287: training narrows the typical transition by
about 0.29 of the path. **Read this as: the total association is strong and survives geometry
adjustment; adjusting for the model's own output separation attenuates it to −0.277 (still
significant), and the fully adjusted estimate is not significant.**

## Secondary result — 1,000 pairs with endpoint-clustered inference

Sixty matched pairs is a small bank, so we assayed a ten-times-larger one on the trained 1.4B model:
1,000 pairs drawn from the same 123 endpoints, 200 per selection-split quintile, at most 20 pairs per
endpoint, selected without consulting any transition curve. Endpoints necessarily recur, so these are
**not** 1,000 independent observations and no naive p-value is reported for them.

| Quantity | Value |
|---|---|
| Pairs / endpoints / uses per endpoint (min, median, max) | 1,000 / 123 / (1, 17, 20) |
| Spearman ρ($\widehat J_{\mathrm{hold}}$, $w$) | **−0.486** |
| Dyadic endpoint-bootstrap 95% CI (4,000 resamples) | [−0.603, −0.353] |
| Endpoint-label permutation p (4,000 relabellings) | **< 0.00025** (0 of 4,000 reached that magnitude) |
| Naive pair-bootstrap CI — invalid here, shown for contrast | [−0.533, −0.437] |
| Spearman ρ($\widehat J_{\mathrm{sel}}$, $w$) | −0.485 |
| Spearman ρ($\widehat J_{\mathrm{hold}}$, $JSD_{\mathrm{out}}$) | +0.729 |
| Median $w$ (IQR) / valid-curve rate over 3,000 curves | 0.555 (0.129) / 1.000 |
| **Same 1,000 pairs at step 0** — ρ, clustered CI, permutation p | **−0.008** [−0.126, +0.109], p = 0.86 |
| Same 1,000 pairs at step 0 — ρ with $JSD_{\mathrm{out}}$ / median $w$ (IQR) | +0.001 / 0.831 (0.005) |

Running the identical 1,000 pairs on the untrained step-0 network gives ρ = −0.008 with a clustered
CI of [−0.126, +0.109] — a null that is now tightly bounded rather than merely non-significant — so the
association requires training at this scale too. The same restricted-range caveat applies: untrained
widths span an IQR of 0.005. The trained point estimate is slightly smaller than the primary bank's, as expected for a bank that is not
matched pair-by-pair on frequency and surprisal and that fills in the crowded middle of the divergence
range. Treat it as an **endpoint-dependent robustness analysis** that confirms direction, magnitude
and monotonicity — not as 1,000 independent confirmations.

**Formation subset** — the same frozen 60-pair bank at six checkpoints of `pythia-1.4b-deduped`. The
plan expected the relationship to *strengthen* during training. It does not: it is already comparable
to later checkpoints at step 1000, the earliest checkpoint measured, and then fluctuates within
overlapping CIs. Transitions narrow through step 64000 and then show a **modest late reversal**:
median $w$ rises from 0.512 at 64k to 0.541 at the final checkpoint, 38 of 60 pairs end blunter
(paired Wilcoxon p = 0.0052, median per-pair Δ = +0.012).

| Training step | 0 | 1000 | 8000 | 32000 | 64000 | 143000 |
|---|---|---|---|---|---|---|
| Spearman ρ($\widehat J_{\mathrm{hold}}$, $w$) | −0.056 | −0.582 | −0.456 | −0.408 | −0.628 | −0.525 |
| 95% CI | [−0.31,+0.21] | [−0.77,−0.36] | [−0.66,−0.21] | [−0.62,−0.16] | [−0.77,−0.44] | [−0.70,−0.31] |
| Spearman ρ($\widehat J_{\mathrm{hold}}$, output JSD) | +0.145 | +0.791 | +0.721 | +0.766 | +0.750 | +0.751 |
| Median $w$ (IQR) | 0.831 (0.006) | 0.753 (0.107) | 0.601 (0.150) | 0.555 (0.131) | **0.512** (0.150) | 0.541 (0.169) |
| Median edge drift $E$ | 0.213 | 0.153 | 0.088 | 0.077 | 0.069 | 0.076 |
| Valid-curve rate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Validity and controls

Every gate the plan prespecified, plus the assay self-tests, is listed here with its measured value.

| Check | Value | Requirement |
|---|---|---|
| Reliability Spearman($\widehat J_{\mathrm{sel}}$, $\widehat J_{\mathrm{hold}}$), 10k pairs | 0.9998 | ≥ 0.90 ✔ |
| Same-word split-half noise / between-word JSD (ratio of medians) | 0.072 | < 0.25 ✔ |
| Strict-validity failures (span / monotone / single-crossing), 7,080 curves | 0 / 0 / 0 | shown, not assumed ✔ |
| Largest backslide on any curve (non-monotonicity) | 0.0000 | ≤ 0.02 ✔ |
| Calibration IQR($w$), 15 frozen pairs | 0.109 | ≥ 0.05 ✔ |
| Endpoint self-test ($t$=0,1 reproduce logits), max rel. err | 6.3e−5 | ≈ 0 ✔ |
| Reversal (swap $u$↔$v$), max Δ$w$ | 1.1e−5 | ≪ grid 0.0204 ✔ |
| Prefix block-0 residual difference within a pair | 0.0 | exactly 0 ✔ |
| Bank balance across bins (Kruskal-Wallis p) | 0.52 log-freq, 0.21 surprisal | no significant imbalance detected ✔ |
| Per-context ρ (3 carrier contexts) | −0.486, −0.411, −0.504 | consistent ✔ |
| Block scan (10 pairs, 1 context): median $w$ at blocks 0/6/12/18/23 | 0.599 / 0.661 / 0.741 / 0.805 / 0.804 | monotone ↑ ✔ |
| Word-fragment sensitivity: ρ after dropping the ` un`/` better` pair (n = 59) | −0.502, p = 5.2e−5 | unchanged conclusion ✔ |

**Secondary, post-hoc bank (top-512).** A larger bank (n = 75) built by relaxing the endpoint filter
from the prespecified top-256 to top-512. It is *not* a prespecified fallback; it is reported only to
show the conclusion does not depend on the filter. ρ($\widehat J_{\mathrm{hold}}$, $w$) = −0.419
[−0.587, −0.225] trained, −0.155 [−0.363, +0.071] at step 0, −0.320 [−0.533, −0.085] at 410M;
ρ($\widehat J_{\mathrm{hold}}$, output JSD) = +0.729 trained. Same direction, same verdict, slightly
weaker than the prespecified bank.

## Figures

Before trusting the predictor we check that a count-based divergence is stable across two disjoint
1.02B-token samples of the training stream, and that it sits far above the sampling-noise floor
(Figure 1).

![Selection-split JSD vs holdout-split JSD, and between-word vs same-word divergence histograms.](plots/jsd_reliability.png)

**Figure 1.** Both prespecified reliability gates pass. *Left:* x = $\widehat J_{\mathrm{sel}}(u,v)$
(bits, selection split), y = $\widehat J_{\mathrm{hold}}(u,v)$ (bits, holdout split); 10,000 word
pairs; dashed line is y = x; Spearman 0.9998. *Right:* x = JSD (bits), y = count. `//`-hatched =
between-word $\widehat J_{\mathrm{hold}}$ (median 0.673); `\\`-hatched = same-word split-half noise
floor (median 0.049). The ratio of medians is 0.072, far below the 0.25 gate.

$w$ is a summary the source post never defined, so the raw curves are the primary evidence — and
looking at them is also the audit for the validity criteria (Figure 2).

![Small multiples: all 180 raw d(t) curves at each of two checkpoints, one panel per divergence quintile, trained on top and untrained below.](plots/all_curves.png)

**Figure 2.** All 180 curves of the primary bank at each of the two checkpoints drawn here: trained
(top row) and untrained (bottom row). x = interpolation position $t$; y = relative-logit coordinate
$d(t)$. One panel per $\widehat J_{\mathrm{sel}}$ quintile (Q1 = most similar continuations); thin
lines are the 3 carrier contexts of every pair in that bin (line style distinguishes context), the
thick dark line is the bin's pointwise median, dotted horizontals mark d = 0.1 and 0.9. The validity
audit is wider than this figure: across all 1,080 curves of the six 60-pair checkpoint runs **and** the
6,000 curves of the two 1,000-pair runs, none backslides and none crosses a level twice. The untrained
network (bottom) is a straight line in every bin; the trained one (top) bends into an S, more so in
the higher-divergence bins.

Individual pairs are noisy, so the effect should be read as distributional rather than
pair-by-pair — the six pairs of Figure 3 make that concrete.

![Raw curves for the three lowest- and three highest-divergence pairs, all carrier contexts drawn separately.](plots/reference_curves.png)

**Figure 3.** The trend does not hold pair-by-pair. x = $t$; y = $d(t)$; solid with round/square/
triangle markers = the 3 lowest-$\widehat J_{\mathrm{hold}}$ pairs (` of`/` in`, ` on`/` with`,
` never`/` always`, 0.14–0.27 bits); dashed = the 3 highest (` out`/` your`, ` un`/` better`,
` extremely`/` happening`, 0.85–0.94 bits). All three carrier contexts of each pair are drawn
separately (no averaging). The two function-word pairs at the bottom of the divergence range are
indeed the widest, but ` never`/` always` is among the sharpest curves in the figure despite low
corpus divergence.

The primary test asks whether the corpus predictor tracks transition width, and whether the
relationship requires training.

![Transition width vs held-out corpus divergence for trained 1.4B, untrained step 0, and 410M.](plots/jsd_vs_width.png)

**Figure 4.** The relationship exists only after training. x = $\widehat J_{\mathrm{hold}}(u,v)$
(bits), y = $w$ (smaller = sharper). **Each dot is one endpoint pair**: its $w$ is the median over the
3 carrier contexts, and each context's $w$ comes from a 50-position curve — so each panel holds
exactly **60 pair-level observations**, and the three panels re-use the same 60 pair identities.
Marker shape and hue = the pair's $\widehat J_{\mathrm{sel}}$ **stratum**; the dashed `x`-marked line
is **five non-overlapping equal-count medians after re-binning the same 60 pairs by
$\widehat J_{\mathrm{hold}}$** — not extra observations, not a running median. **Y-ranges differ per
panel:** trained 1.4B spans 0.40–0.80, step 0 only 0.820–0.840, a restricted range just under the
linear-response ceiling, so the step-0 null is weaker evidence than it looks. Trained 1.4B ρ = −0.525;
step 0 ρ = −0.056; 410M ρ = −0.512.

Binning (Figure 5) shows how much of the trend survives aggregation and how far it is from a clean
monotone staircase.

![Box plots of w by divergence quintile for all three checkpoints.](plots/width_by_jsd_bin.png)

**Figure 5.** Lower width in the higher-divergence bins, but not monotonically at 1.4B.
x = $\widehat J_{\mathrm{sel}}$ quintile (Q1 = most similar continuations); y = $w$. Groups by hatch
and marker: `//` with round markers = trained 1.4B, `\\` with square markers = step-0 1.4B, `..` with
triangular markers = 410M. Boxes are the interquartile range with a median bar; points are individual
pairs. The 410M bin medians fall monotonically (0.723 → 0.578); at 1.4B, Q3 (0.462) dips below Q4 and
Q5 (0.502, 0.479), so the bin-level trend is real but noisy at n ≈ 12 per bin.

A smaller $w$ could mean a genuine plateau (flat regions then a jump) or just a steeper straight line,
so we measure endpoint flatness separately (Figure 6).

![Left: histogram of edge drift for the three checkpoints against the no-plateau reference. Right: edge drift vs transition width.](plots/edge_drift.png)

**Figure 6.** The trained curves really are plateau-shaped in the relative-logit coordinate, but
flatness and width are redundant. *Left:* x = edge drift $E$ (mean movement of $d$ away from its
endpoint value inside the outer 20% of the path; 0 = perfectly flat ends), y = number of pairs;
`//`-hatched = trained 1.4B (median 0.076), `\\`-hatched = untrained step 0 (0.213), `..`-hatched =
410M (0.109); the dashed vertical is the no-plateau reference $E$ = 0.184 for a straight line
$d(t) = t$. Trained curves sit far below it, untrained ones slightly above. *Right:* x = $w$, y = $E$,
round markers = trained, square = step 0; Spearman($w$, $E$) = +0.971, i.e. at the pair level the two
metrics say the same thing.

Corpus divergence is global and context-free, so we verify it predicts a distinction the model itself
makes in the specific carrier contexts — otherwise a width null would be uninterpretable (Figure 7).

![Model output JSD vs corpus divergence, 60 pairs.](plots/output_jsd_validation.png)

**Figure 7.** Corpus divergence strongly predicts the model's own output divergence. x =
$\widehat J_{\mathrm{hold}}(u,v)$ (bits, corpus); y = $JSD_{\mathrm{out}}$ (bits), the median over the
3 carrier contexts of the JSD between the two endpoints' model output distributions, restricted to the
50,060 corpus-observed target IDs; marker shape and hue = $\widehat J_{\mathrm{sel}}$ stratum.
ρ = +0.751 [+0.615, +0.843]. At step 0 the same correlation is +0.145 (p = 0.27).

If corpus divergence predicts width only *through* that learned output separation, then adjusting for
it should attenuate the association — so Figure 8 tests the within-pair change training produced, and
then strips the mediator away.

![Left: scatter of the training-induced width change against corpus divergence. Right: forest plot of the association before and after adjustment.](plots/mediation.png)

**Figure 8.** Corpus divergence predicts how much training narrowed each pair's transition, and the
association is attenuated by adjusting for the model's own output separation. *Left:* x =
$\widehat J_{\mathrm{hold}}(u,v)$ (bits); y = learned sharpening $\Delta w$ (more negative = training
narrowed the transition more); marker shape and hue = $\widehat J_{\mathrm{sel}}$ stratum; dotted
horizontal = no change. ρ = −0.517 [−0.694, −0.294]. *Right:* x = Spearman ρ with
$\widehat J_{\mathrm{hold}}$ (bars = 95% bootstrap CI), y = the four analyses listed top to bottom; a
filled marker means p < 0.05, an open marker p > 0.05; dotted vertical = zero. The total association
(−0.525) and the learned-sharpening version (−0.517) are strong; adjusting for the mediator
$JSD_{\mathrm{out}}$ leaves −0.277 (p = 0.032, still significant), and adjusting for the mediator plus
the five covariates leaves −0.204 (p = 0.119, not significant).

Figure 4 shows the relationship needs training; the intermediate checkpoints ask *when* it forms,
separate how well corpus divergence predicts width from how sharp the transitions actually are, and
show that the narrowing is not monotone to the end (Figure 9).

![Left: Spearman correlations vs training step. Middle: median transition width vs training step. Right: per-pair width at step 64000 against the final checkpoint.](plots/formation.png)

**Figure 9.** The predictor is already at full strength at the earliest measured step; transitions
narrow through step 64000 and then partly reverse. *Left and middle:* x = training step on a log
scale, step 0 drawn at the left edge, ticks at 0/1k/8k/32k/64k/143k. *Left:* y = Spearman ρ with
corpus $\widehat J_{\mathrm{hold}}$. Solid with round markers = ρ with $w$ (shaded `//`-hatched band =
95% bootstrap CI); dashed with square markers = ρ with model output JSD; dotted line = zero. Both jump
from ≈ 0 at step 0 to full magnitude by step 1000 (−0.582 and +0.791) and then move within overlapping
CIs; nothing here constrains what happened between step 0 and step 1000. *Middle:* y = median $w$,
`//`-hatched band = median ± IQR/2, dashed horizontal = the linear-response value 0.8. Median $w$ falls
0.831 → 0.512 by step 64000 and then rises to 0.541. *Right:* per-pair check that this rebound is
real. x = $w$ at step 64000, y = $w$ at step 143000, one point per pair; triangles = the 38 pairs that
end blunter, circles = the 22 that end sharper; dashed line = no change. Paired Wilcoxon p = 0.0052,
median Δ$w$ = +0.012.

The primary bank is the prespecified one; the larger relaxed bank and the fragment-dropped bank are
shown only to check the conclusion does not hinge on those choices (Figure 10).

![Spearman rho with 95% CIs for the top-256, top-512 and fragment-dropped banks at three checkpoints.](plots/bank_comparison.png)

**Figure 10.** No version of the bank changes the conclusion. x = checkpoint; y = Spearman
ρ($\widehat J_{\mathrm{hold}}$, $w$) with 95% bootstrap CI bars; round markers = prespecified top-256
bank (n = 60), square markers = post-hoc top-512 bank (n = 75), triangular markers = top-256 minus the
one pair whose endpoint ` un` is a word-start fragment rather than a whole word (n = 59); dotted line =
zero. All three agree at every checkpoint, with heavily overlapping CIs: trained 1.4B −0.525 / −0.419
/ −0.502.

The 60-pair bank is small and tightly matched, so the generality question is whether the association
survives on a much larger, endpoint-reusing bank and whether it stays monotone across the divergence
range (Figure 11).

![Left: transition width against held-out corpus divergence for 1,000 pairs with ten binned medians. Right: forest plot comparing the primary estimate with clustered and naive intervals.](plots/large_bank.png)

**Figure 11.** The association holds on a ten-times-larger bank and is monotone across the range.
*Left:* x = $\widehat J_{\mathrm{hold}}(u,v)$ (bits), y = $w$; each of the 1,000 small markers is one
endpoint pair (median $w$ over 3 carrier contexts), coloured and shaped by its
$\widehat J_{\mathrm{sel}}$ stratum. The dashed `x`-marked line is the median $w$ in **ten
non-overlapping equal-count $\widehat J_{\mathrm{hold}}$ bins** (100 pairs each; bars = interquartile
range) — summaries of the same 1,000 pairs, not extra observations. Bin medians fall 0.649 → 0.499
essentially monotonically, flattening above ≈ 0.75 bits; ρ = −0.486. *Right:* x = Spearman
ρ($\widehat J_{\mathrm{hold}}$, $w$) with 95% CI bars; y lists three estimates: the primary 60-pair
endpoint-disjoint bank (round marker, −0.525), the 1,000-pair bank under the dyadic endpoint bootstrap
(square marker, [−0.603, −0.353]), and the same 1,000 pairs under a naive pair bootstrap (triangular
marker, [−0.533, −0.437]) — shown only to quantify how badly ignoring endpoint reuse understates
uncertainty (2.6× narrower). The endpoint-label permutation test gives p < 0.00025: none of 4,000
relabellings reached ρ = −0.486 in magnitude.

Finally, a control on the assay (Figure 12): sharpness should depend on the blocks that still run
after the patch.

![Transition width vs patched block index for low- and high-divergence pairs.](plots/block_scan.png)

**Figure 12.** Width grows as fewer blocks follow the patch. x = patched block index $L$ (block 23 is
the last of 24, so almost no computation remains); y = $w$. Solid with round markers = median of the 5
lowest-$\widehat J_{\mathrm{hold}}$ pairs; dashed with square markers = median of the 5 highest; faint
lines are individual pairs. $w$ rises monotonically 0.599 → 0.804, converging on the linear-response
value of about 0.8. This scan covers only these 10 extreme pairs in one carrier context, so it is
consistent with a role for downstream computation without establishing that downstream blocks are
generally required.

## Interpretation

**In trained Pythia, how differently two words are continued in the training corpus predicts how
sharply the model flips between them (ρ = −0.525, p = 1.7e−5 on 60 endpoint-disjoint pairs; −0.486
with an endpoint-clustered CI of [−0.603, −0.353] on 1,000 pairs; −0.512 at 410M), and predicts even
more strongly how far apart it puts their output distributions (ρ = +0.751).** The untrained step-0
network shows neither (−0.056 and +0.145) — but its widths occupy a 2%-wide band just under the
linear-response ceiling, a restricted range, so that control is weaker than it looks. The same holds
for what training *changed*: corpus divergence predicts the per-pair narrowing $\Delta w$ at ρ = −0.517.

**The headline is a total association.** It attenuates to −0.384 after adjusting for endpoint
frequency, entropy, surprisal and block-0 geometry, to −0.277 (p = 0.032, still significant) after
adjusting for the model's own output divergence, and to **−0.204 (p = 0.119, not significant)** after
adjusting for both. So corpus divergence is a good *predictor* of transition width, and this design
gives no significant evidence that it explains width **independently** of the output separation the
model learned. Observational, not causal. Note also that only the JSD predictor itself is computed
from corpus statistics: endpoint filtering and covariate matching use trained-model probabilities and
surprisal.

**What this does not show.** $w$ is the width of the *entire* 10%→90% transition of one relative-logit
coordinate. A flat $d(t)$ means that coordinate barely moves — not that the full logit vector or the
output distribution stays put. The trained curves are genuinely plateau-shaped in that coordinate
(edge drift 0.076 versus 0.184 for a straight line), but edge drift and $w$ correlate at +0.971 across
pairs, so this experiment cannot separate "corpus divergence predicts flatter plateaus" from "corpus
divergence predicts narrower transitions". The honest claim is the second one.

**Unexpected:** contrary to the plan's expectation, the relationship does not strengthen during
training. It is already comparable to later checkpoints at the earliest step measured (step 1000,
ρ = −0.582) and then moves within overlapping CIs (−0.456, −0.408, −0.628, −0.525), even though
transitions go on narrowing through step 64000 (median $w$ 0.831 → 0.512) before a modest late reversal
to 0.541 at the final checkpoint (38/60 pairs blunter than at 64k, paired Wilcoxon p = 0.0052).
Narrowing continues after the corpus statistic has stopped explaining more of it, and then partly
undoes itself.

**Auditability.** Every raw $d(t)$ curve is committed — `results/curves_*.npy` plus a plain-text
`results/curves_*.csv.gz` export — so all width, flatness and validity numbers above can be recomputed
independently. (This direction carries its own `.gitignore` that un-ignores them from the repo-wide
`*.npy` / `*.gz` rules.)
