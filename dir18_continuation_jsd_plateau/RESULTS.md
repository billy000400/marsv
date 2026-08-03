# RESULTS — Does training-corpus continuation JSD predict transition sharpness?

> CURRENT-BEST ONLY. One row per experiment. No history, no superseded/weaker variants
> (those live in CHANGELOG.md). Full method definitions are in REPORT.md.

**Question.** Can the training corpus alone tell you which input pairs a model separates sharply? We
measure how differently two words are continued in 2.05B tokens of Pythia's actual released training
stream (base-2 Jensen-Shannon divergence, `JSD_B`), then interpolate the model's own internal state
between the two words and measure the **transition width** `w = t(d=0.9) − t(d=0.1)` of the resulting
output curve. **Smaller `w` = the model flips between the two words over a shorter stretch of the
path.** A negative correlation means higher corpus divergence predicts a sharper flip.

**Verdict (prespecified branch):** *corpus JSD predicts model-output JSD and smaller `w`; step 0 does
not.* Stated precisely: **corpus continuation divergence predicts (i) how far apart the trained model
puts the two words' output distributions and (ii) the overall width of the transition between them.**
It is an observational total association, not a causal claim, and — see "What this does not show" —
`w` measures the whole transition, so it does not isolate plateau flatness from overall width.

## Primary result — prespecified top-256 bank

n = 60 endpoint-disjoint pairs (14/13/11/10/12 per `JSD_A` quintile), 3 carrier contexts each, 50
interpolation points, 180 raw curves per checkpoint. CIs are 95% from 10,000 bootstrap resamples over
pairs.

| Result | Trained 1.4B (step143000) | Untrained 1.4B (step0) | 410M (step143000) |
|---|---|---|---|
| Spearman ρ(`JSD_B`, `w`) — **primary** | **−0.525** [−0.701, −0.304], p=1.7e−5 | −0.056 [−0.314, +0.211], p=0.67 | −0.512 [−0.711, −0.272], p=2.9e−5 |
| Partial ρ (freq, entropy, surprisal, block-0 cos & dist adjusted) | −0.384 | −0.142 | −0.396 |
| Spearman ρ(`JSD_B`, model output JSD) | **+0.751** [+0.615, +0.843], p=4.9e−12 | +0.145 [−0.122, +0.394], p=0.27 | +0.749 [+0.611, +0.838] |
| Median `w` (IQR) | 0.541 (0.169) | 0.831 (0.006) | 0.640 (0.133) |
| Median `w` by quintile Q1→Q5 | 0.619, 0.608, 0.462, 0.502, 0.479 | 0.831, 0.832, 0.833, 0.830, 0.828 | 0.723, 0.683, 0.610, 0.582, 0.578 |
| Median edge drift `E` (0 = flat ends; 0.184 = no plateau) | **0.076** | 0.213 | 0.109 |
| Valid-curve rate under the strict criteria (and in every bin) | 1.000 | 1.000 | 1.000 |

**Formation subset** — the same frozen bank at six checkpoints of `pythia-1.4b-deduped`. The plan
expected the relationship to *strengthen* during training. It does not: it is already at full
strength at step 1000 and then fluctuates within overlapping CIs.

| Training step | 0 | 1000 | 8000 | 32000 | 64000 | 143000 |
|---|---|---|---|---|---|---|
| Spearman ρ(`JSD_B`, `w`) | −0.056 | −0.582 | −0.456 | −0.408 | −0.628 | −0.525 |
| 95% CI | [−0.31,+0.21] | [−0.77,−0.36] | [−0.66,−0.21] | [−0.62,−0.16] | [−0.77,−0.44] | [−0.70,−0.31] |
| Spearman ρ(`JSD_B`, output JSD) | +0.145 | +0.791 | +0.721 | +0.766 | +0.750 | +0.751 |
| Median `w` (IQR) | 0.831 (0.006) | 0.753 (0.107) | 0.601 (0.150) | 0.555 (0.131) | **0.512** (0.150) | 0.541 (0.169) |
| Median edge drift `E` | 0.213 | 0.153 | 0.088 | 0.077 | 0.069 | 0.076 |
| Valid-curve rate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Validity and controls

| Check | Value | Requirement |
|---|---|---|
| Reliability Spearman(`JSD_A`, `JSD_B`), 10k pairs | 0.9998 | ≥ 0.90 ✔ |
| Same-word split-half noise / between-word JSD | 0.072 | < 0.25 ✔ |
| Strict-validity failures (span / monotone / single-crossing), 180 curves | 0 / 0 / 0 | shown, not assumed ✔ |
| Largest backslide on any curve (non-monotonicity) | 0.0000 | ≤ 0.02 ✔ |
| Calibration IQR(`w`), 15 frozen pairs | 0.109 | ≥ 0.05 ✔ |
| Endpoint self-test (`t`=0,1 reproduce logits), max rel. err | 4.6e−5 | ≈ 0 ✔ |
| Reversal (swap A↔B), max Δ`w` | 1.1e−5 | ≪ grid 0.0204 ✔ |
| Prefix block-0 residual difference within a pair | 0.0 | exactly 0 ✔ |
| Bank balance across bins (Kruskal-Wallis p) | 0.52 log-freq, 0.21 surprisal | large = balanced ✔ |
| Per-context ρ (3 carrier contexts) | −0.486, −0.411, −0.504 | consistent ✔ |
| Block scan: median `w` at blocks 0/6/12/18/23 | 0.599 / 0.661 / 0.741 / 0.805 / 0.804 | monotone ↑ ✔ |

**Secondary, post-hoc bank (top-512).** A larger bank (n = 75) built by relaxing the endpoint filter
from the prespecified top-256 to top-512. It is *not* a prespecified fallback; it is reported only to
show the conclusion does not depend on the filter. ρ(`JSD_B`, `w`) = −0.419 [−0.587, −0.225] trained,
−0.155 [−0.363, +0.071] at step 0, −0.320 [−0.533, −0.085] at 410M; ρ(`JSD_B`, output JSD) = +0.729
trained. Same direction, same verdict, slightly weaker than the prespecified bank.

## Figures

Before trusting the predictor we check that a count-based divergence is stable across two disjoint
1.02B-token samples of the training stream, and that it sits far above the sampling-noise floor.

![JSD from split A vs split B, and between-word vs same-word divergence histograms.](plots/jsd_reliability.png)

**Figure 1.** Both prespecified reliability gates pass. *Left:* x = `JSD_A` (bits, split A),
y = `JSD_B` (bits, split B); 10,000 word pairs; dashed line is y = x; Spearman 0.9998. *Right:*
x = JSD (bits), y = count. `//`-hatched = between-word `JSD_B` (median 0.673); `\\`-hatched = same-word
split-half noise floor (median 0.049). Ratio 0.072, far below the 0.25 gate.

`w` is a summary the source post never defined, so the raw curves are the primary evidence and every
one of them is shown — this is also the audit for the validity criteria.

![Small multiples: all 180 raw d(t) curves per checkpoint, one panel per divergence quintile, trained on top and untrained below.](plots/all_curves.png)

**Figure 2.** Every raw curve in the frozen bank, trained (top row) and untrained (bottom row).
x = interpolation position `t`; y = relative logit distance `d(t)`. One panel per `JSD_A` quintile
(Q1 = most similar continuations); thin lines are the 3 carrier contexts of every pair in that bin
(line style distinguishes context), the thick dark line is the bin's pointwise median, dotted
horizontals mark d = 0.1 and 0.9. All 180 curves per checkpoint pass the strict validity criteria —
none backslides, none crosses a level twice. The untrained network (bottom) is a straight line in
every bin; the trained one (top) bends into an S, more so in the higher-divergence bins.

Individual pairs are noisy, so the effect should be read as distributional rather than
pair-by-pair — these six pairs make that concrete.

![Raw curves for the three lowest- and three highest-divergence pairs, all carrier contexts drawn separately.](plots/reference_curves.png)

**Figure 3.** The trend does not hold pair-by-pair. x = `t`; y = `d(t)`; solid with round/square/
triangle markers = the 3 lowest-`JSD_B` pairs (`of`/`in`, `on`/`with`, `never`/`always`, 0.14–0.27);
dashed = the 3 highest (`out`/`your`, `un`/`better`, `extremely`/`happening`, 0.85–0.94). All three
carrier contexts of each pair are drawn separately (no averaging). The two function-word pairs at the
bottom of the divergence range are indeed the widest, but `never`/`always` is among the sharpest
curves in the figure despite low corpus divergence.

The primary test: does the corpus predictor track transition width, and does the relationship require
training?

![Transition width vs corpus divergence for trained 1.4B, untrained step 0, and 410M.](plots/jsd_vs_width.png)

**Figure 4.** The relationship exists only after training. x = `JSD_B` (bits), y = `w` (smaller =
sharper); marker shape and hue = `JSD_A` quintile; the dashed `x`-marked line is the running median in
5 equal-count bins. **Y-ranges differ per panel:** trained 1.4B spans 0.40–0.80, step 0 only
0.820–0.840 — the untrained network has essentially no variation in `w` to correlate with, so its
null is a floor effect as much as an absence of association. Trained 1.4B ρ = −0.525; step 0
ρ = −0.056; 410M ρ = −0.512.

Binning shows how much of the trend survives aggregation and how far it is from a clean monotone
staircase.

![Box plots of w by divergence quintile for all three checkpoints.](plots/width_by_jsd_bin.png)

**Figure 5.** Lower width in the higher-divergence bins, but not monotonically at 1.4B.
x = `JSD_A` quintile (Q1 = most similar continuations); y = `w`. Groups by hatch and marker: `//` with
round markers = trained 1.4B, `\\` with square markers = step-0 1.4B, `..` with triangular markers =
410M. Boxes are the interquartile range with a median bar; points are individual pairs. The 410M bin
medians fall monotonically (0.723 → 0.578); at 1.4B, Q3 (0.462) dips below Q4 and Q5 (0.502, 0.479),
so the bin-level trend is real but noisy at n ≈ 12 per bin.

`w` shrinking could mean a genuine plateau (flat regions then a jump) or just a steeper straight
line, so we measure endpoint flatness separately.

![Left: histogram of edge drift for the three checkpoints against the no-plateau reference. Right: edge drift vs transition width.](plots/edge_drift.png)

**Figure 6.** The trained curves really are plateau-shaped, but flatness and width are redundant.
*Left:* x = edge drift `E` (mean movement of `d` away from its endpoint value inside the outer 20% of
the path; 0 = perfectly flat ends), y = number of pairs; `//`-hatched = trained 1.4B (median 0.076),
`\\`-hatched = untrained step 0 (0.213), `..`-hatched = 410M (0.109); the dashed vertical is the
no-plateau reference `E` = 0.184 for a straight line `d(t) = t`. Trained curves sit far below it,
untrained ones slightly above. *Right:* x = `w`, y = `E`, round markers = trained, square = step 0;
Spearman(`w`, `E`) = +0.971, i.e. at the pair level the two metrics say the same thing.

Corpus JSD is global and context-free, so we verify it predicts a distinction the model itself makes
in the specific carrier context — otherwise a width null would be uninterpretable.

![Model output JSD vs corpus JSD, 60 pairs.](plots/output_jsd_validation.png)

**Figure 7.** Corpus divergence strongly predicts the model's own output divergence. x = `JSD_B`
(bits, corpus); y = model output JSD (bits) in the carrier context; marker shape and hue = `JSD_A`
quintile. ρ = +0.751 [+0.615, +0.843]. At step 0 the same correlation is +0.145 (p = 0.27).

Figure 4 shows the relationship needs training; the intermediate checkpoints ask *when* it forms —
and separate how well corpus divergence predicts width from how sharp the transitions actually are.

![Left: Spearman correlations vs training step. Right: median transition width vs training step.](plots/formation.png)

**Figure 8.** The predictor is fully formed by step 1000 while transitions keep sharpening. Both
panels: x = training step on a log scale, step 0 drawn at the left edge, ticks at 0/1k/8k/32k/64k/143k.
*Left:* y = Spearman ρ with corpus `JSD_B`. Solid with round markers = ρ with `w` (shaded `//`-hatched
band = 95% bootstrap CI); dashed with square markers = ρ with model output JSD; dotted line = zero.
Both jump from ≈ 0 at step 0 to full magnitude by step 1000 (−0.582 and +0.791) and then move within
overlapping CIs. *Right:* y = median `w`, `//`-hatched band = median ± IQR/2, dashed horizontal = the
linear-response value 0.8. Median `w` falls from 0.831 to ≈ 0.52 and stays there.

The primary bank is the prespecified one; the larger relaxed bank is shown only to check the
conclusion does not hinge on that choice.

![Spearman rho with 95% CIs for the top-256 and top-512 banks at three checkpoints.](plots/bank_comparison.png)

**Figure 9.** The relaxed bank gives the same conclusion, slightly weaker. x = checkpoint;
y = Spearman ρ(`JSD_B`, `w`) with 95% bootstrap CI bars; round markers = prespecified top-256 bank
(n = 60), square markers = post-hoc top-512 bank (n = 75); dotted line = zero. The two banks' CIs
overlap heavily at every checkpoint, so the difference between −0.525 and −0.419 is not itself a
finding.

Finally, a control on the assay: sharpness should depend on the blocks that still run after the patch.

![Transition width vs patched block index for low- and high-divergence pairs.](plots/block_scan.png)

**Figure 10.** Sharpness is produced downstream of the patch. x = patched block index `L` (block 23 is
the last of 24, so almost no computation remains); y = `w`. Solid with round markers = median of the 5
lowest-`JSD_B` pairs; dashed with square markers = median of the 5 highest; faint lines are individual
pairs. `w` rises monotonically 0.599 → 0.804, converging on the linear-response value of about 0.8.

## Headline

**In trained Pythia, how differently two words are continued in the training corpus predicts how
sharply the model flips between them (ρ = −0.525, p = 1.7e−5 at 1.4B; −0.512 at 410M), and predicts
even more strongly how far apart it puts their output distributions (ρ = +0.751).** The untrained
step-0 network shows neither (−0.056 and +0.145) — but it also has almost no variation in width to
predict (IQR 0.006), so that control is partly a floor effect. The association attenuates to −0.384
after adjusting for endpoint frequency, entropy, surprisal and block-0 geometry, so we report a total
association and do **not** claim corpus divergence explains sharpness beyond learned endpoint
geometry. Observational, not causal.

**What this does not show.** `w` is the width of the *entire* 10%→90% transition. The trained curves
are genuinely plateau-shaped in level terms (edge drift 0.076 versus 0.184 for a straight line), but
edge drift and `w` correlate at +0.971 across pairs, so this experiment cannot separate "corpus
divergence predicts flatter plateaus" from "corpus divergence predicts narrower transitions". The
honest claim is the second one.

**Unexpected:** contrary to the plan's expectation, the relationship does not strengthen during
training. It is already at full strength at the earliest checkpoint we ran (step 1000, ρ = −0.582)
and then moves within overlapping CIs (−0.456, −0.408, −0.628, −0.525), even though transitions keep
sharpening throughout (median `w` 0.831 → ≈ 0.52). Sharpening continues after the corpus statistic
has stopped explaining more of it.
