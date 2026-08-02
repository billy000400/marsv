# RESULTS — Does training-corpus continuation JSD predict plateau strength?

> CURRENT-BEST ONLY. One row per experiment. No history, no superseded/weaker variants
> (those live in CHANGELOG.md). Full method definitions are in REPORT.md.

**Question.** Can the training corpus alone tell you which input pairs a model separates sharply? We
measure how differently two words are continued in 2.05B tokens of Pythia's actual released training
stream (base-2 Jensen-Shannon divergence, `JSD_B`), then measure how sharply the trained model flips
between them — the **transition width** `w = t(d=0.9) − t(d=0.1)` of the block-0-to-logit
interpolation curve. **Smaller `w` = sharper plateau.** A negative correlation means higher corpus
divergence predicts sharper plateaus.

**Verdict (prespecified):** *corpus JSD predicts model-output JSD and smaller `w`; step 0 does not* —
**predictive divergence is associated with learned plateau sharpening.**

## Metrics

n = 75 endpoint-disjoint pairs (15 per JSD quintile), 3 carrier contexts each, 50 interpolation
points. CIs are 95% from 10,000 bootstrap resamples.

| Result | Trained 1.4B (step143000) | Untrained 1.4B (step0) | 410M (step143000) |
|---|---|---|---|
| Spearman ρ(`JSD_B`, `w`) — **primary** | **−0.419** [−0.585, −0.222], p=1.8e−4 | −0.155 [−0.368, +0.068], p=0.18 | −0.320 [−0.526, −0.087], p=5.1e−3 |
| Partial ρ (freq, entropy, surprisal, block-0 cos & dist adjusted) | −0.267 | −0.146 | −0.251 |
| Spearman ρ(`JSD_B`, model output JSD) | **+0.729** [+0.599, +0.818], p=1.2e−13 | −0.144 [−0.363, +0.085], p=0.22 | +0.717 [+0.584, +0.808] |
| Median `w` (IQR) | 0.562 (0.111) | 0.831 (0.004) | 0.655 (0.075) |
| Median `w` by quintile Q1→Q5 | 0.611, 0.568, 0.532, 0.516, 0.516 | 0.832, 0.831, 0.831, 0.831, 0.829 | 0.699, 0.679, 0.648, 0.647, 0.629 |
| Valid-curve rate (and in every bin) | 1.000 | 1.000 | 1.000 |

**Formation subset** — the same frozen bank on `pythia-1.4b-deduped` at six checkpoints. The plan
expected the relationship to *strengthen* during training; it does the opposite.

| Training step | 0 | 1000 | 8000 | 32000 | 64000 | 143000 |
|---|---|---|---|---|---|---|
| Spearman ρ(`JSD_B`, `w`) | −0.155 | **−0.660** | −0.605 | −0.524 | −0.539 | −0.419 |
| 95% CI | [−0.363,+0.068] | [−0.779,−0.496] | [−0.734,−0.433] | [−0.674,−0.323] | [−0.678,−0.355] | [−0.586,−0.219] |
| Spearman ρ(`JSD_B`, output JSD) | −0.144 | +0.779 | +0.693 | +0.726 | +0.714 | +0.729 |
| Median `w` (IQR) | 0.831 (0.004) | 0.758 (0.087) | 0.624 (0.088) | 0.582 (0.098) | **0.541** (0.114) | 0.562 (0.111) |
| Valid-curve rate | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

| Validity check | Value | Requirement |
|---|---|---|
| Reliability Spearman(`JSD_A`, `JSD_B`), 10k pairs | 0.9998 | ≥ 0.90 ✔ |
| Same-word split-half noise / between-word JSD | 0.072 | < 0.25 ✔ |
| Calibration IQR(`w`), 15 frozen pairs | 0.115 | ≥ 0.05 ✔ |
| Endpoint self-test (`t`=0,1 reproduce logits), max rel. err | 4.7e−5 | ≈ 0 ✔ |
| Reversal (swap A↔B), max Δ`w` | 1.1e−5 | ≪ grid 0.0204 ✔ |
| Prefix block-0 residual difference within a pair | 0.0 | exactly 0 ✔ |
| Bank balance across bins (Kruskal-Wallis p) | 0.92 log-freq, 0.81 surprisal | large = balanced ✔ |
| Per-context ρ (3 carrier contexts) | −0.326, −0.398, −0.437 | consistent ✔ |
| Block scan: median `w` at blocks 0/6/12/18/23 | 0.549 / 0.646 / 0.726 / 0.796 / 0.805 | monotone ↑ ✔ |

## Figures

Before trusting the predictor we checked that a count-based divergence is stable across two disjoint
1.02B-token samples of the training stream, and that it sits far above the sampling-noise floor.

![JSD from split A vs split B, and between-word vs same-word divergence histograms.](plots/jsd_reliability.png)

**Figure 1.** Both prespecified reliability gates pass. *Left:* x = `JSD_A` (bits, split A),
y = `JSD_B` (bits, split B); 10,000 word pairs; dashed line is y = x; Spearman 0.9998. *Right:*
x = JSD (bits), y = count. `//`-hatched = between-word `JSD_B` (median 0.673); `\\`-hatched = same-word
split-half noise floor (median 0.049). Ratio 0.072, far below the 0.25 gate.

`w` is a summary the source post never defined, so the raw curves are shown as primary evidence.

![Six d(t) curves for the three lowest- and three highest-divergence pairs.](plots/reference_curves.png)

**Figure 2.** High-divergence pairs flip more abruptly. x = interpolation position `t` along the
block-0 residual SLERP path; y = relative logit distance `d(t)`. Solid lines with round/square markers
= the 3 lowest-`JSD_B` pairs (`making/getting`, `later/done`, `nothing/someone`, 0.38–0.41); dashed
lines with diamond/triangle markers = the 3 highest (`un/before`, `gonna/happening`, `ra/okay`,
0.93–0.97). Dotted horizontals mark d = 0.1 and 0.9; their horizontal gap is `w`.

The primary test: does the corpus predictor track sharpness, and does the relationship require
training?

![Transition width vs corpus divergence for trained 1.4B, untrained step 0, and 410M.](plots/jsd_vs_width.png)

**Figure 3.** The relationship exists only after training. x = `JSD_B` (bits), y = `w` (smaller =
sharper); marker shape and hue = `JSD_A` quintile; the dashed `x`-marked line is the running median in
5 equal-count bins. **Y-ranges differ per panel:** trained 1.4B spans 0.41–0.74, step 0 only
0.820–0.837. Trained 1.4B ρ = −0.419; step 0 ρ = −0.155 (CI includes 0, and there is essentially no
plateau structure to predict); 410M ρ = −0.320.

Binning shows the trend is monotone rather than outlier-driven, and makes step 0's flatness explicit.

![Box plots of w by divergence quintile for all three checkpoints.](plots/width_by_jsd_bin.png)

**Figure 4.** Monotone sharpening across all five bins in both trained models, absent at step 0.
x = `JSD_A` quintile (Q1 = most similar continuations); y = `w`. Groups by hatch and marker: `//` with
round markers = trained 1.4B, unhatched with square markers = step-0 1.4B, `..` with triangular
markers = 410M. Boxes are the interquartile range with a median bar; points are individual pairs. All
75 curves valid in every bin at every checkpoint.

Corpus JSD is global and context-free, so we verify it predicts a distinction the model itself makes
in the specific carrier context — otherwise a plateau null would be uninterpretable.

![Model output JSD vs corpus JSD, 75 pairs.](plots/output_jsd_validation.png)

**Figure 5.** Corpus divergence strongly predicts the model's own output divergence. x = `JSD_B`
(bits, corpus); y = model output JSD (bits) in the carrier context; marker shape and hue = `JSD_A`
quintile. ρ = +0.729 [+0.599, +0.818]. At step 0 the same correlation is −0.144.

Figure 3 shows the relationship needs training; the intermediate checkpoints ask *when* it forms —
and separate how well corpus divergence predicts sharpness from how sharp the plateaus actually are.

![Left: Spearman correlations vs training step. Right: median transition width vs training step.](plots/formation.png)

**Figure 6.** The predictor peaks early while plateaus keep sharpening. Both panels: x = training step
on a log scale, step 0 drawn at the left edge, ticks at 0/1k/8k/32k/64k/143k. *Left:* y = Spearman ρ
with corpus `JSD_B`. Solid with round markers = ρ with `w` (shaded band = 95% bootstrap CI); dashed
with square markers = ρ with model output JSD; dotted line = zero. Both jump from ≈0 at step 0 to full
magnitude by step 1000 (−0.660 and +0.779); the sharpness correlation then *weakens* to −0.419 while
the output correlation stays flat. *Right:* y = median `w`, `//`-hatched band = median ± IQR/2, dashed
horizontal = linear-response value 0.8. Median `w` falls monotonically 0.831 → 0.562.

Finally, a control on the assay: sharpness should depend on the blocks that still run after the patch.

![Transition width vs patched block index for low- and high-divergence pairs.](plots/block_scan.png)

**Figure 7.** Sharpness is produced downstream of the patch. x = patched block index `L` (block 23 is
the last of 24, so almost no computation remains); y = `w`. Solid with round markers = median of the 5
lowest-`JSD_B` pairs; dashed with square markers = median of the 5 highest; faint lines are individual
pairs. `w` rises monotonically 0.549 → 0.805, converging on the linear-response value of about 0.8.

## Headline

**Corpus continuation divergence predicts activation-plateau sharpness in trained Pythia (ρ = −0.419,
p = 1.8e−4 at 1.4B; −0.320 at 410M), and the relationship is absent in the untrained step-0 network,
which has almost no plateau structure at all (median w = 0.831, IQR 0.004).** The predictor is
validated — it also tracks the model's own output divergence at ρ = +0.729 — but the association
attenuates to −0.267 after adjusting for endpoint frequency, entropy, surprisal, and block-0
geometry, so we report a total association and do **not** claim corpus divergence explains sharpness
beyond learned endpoint geometry. This is observational, not causal.

**Unexpected:** contrary to the plan's expectation, the relationship does not strengthen during
training. It is fully formed and *strongest* at the earliest checkpoint we ran (step 1000,
ρ = −0.660) and decays to −0.419 by step 143000, even though plateaus keep sharpening throughout
(median `w` 0.831 → 0.562). Plateau sharpening continues; the share of it explained by a
context-free corpus statistic shrinks.
