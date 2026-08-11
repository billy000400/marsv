# CHANGELOG — Direction: TODO — describe this direction

Append-only ledger of changes to RESULTS.md / REPORT.md. One dated entry per change: what changed,
why, and — if a result was superseded — the old -> new numbers. This is the ONLY place history lives;
RESULTS.md and REPORT.md themselves stay current-best with no history.

---

## 2026-08-10 — first deliverables: the leftover after corpus JSD is a per-token additive effect

Direction created. RESULTS.md and REPORT.md written from placeholder to a full current-best report
(nothing superseded — this is the first content).

**What is new.** Re-analysis of `dir18`'s stored 1,000-pair artifacts (`pythia-1.4b-deduped`
@ `step143000`, residual after block 0), no new model inference.

- **S3 endpoint-movement gate.** `min`-over-frames model-output JSD >= 0.2 bits keeps 929/1000 pairs.
  Gated-out pairs are not noisier (median across-frame `w` spread 0.02-0.04, same as the rest) but are
  systematically wide (median `w` 0.687 vs 0.545) and low-JSD (0.401 vs 0.665 bits). Headline
  correlation on the gated subset: rho(J,w) = -0.409 (p = 1.0e-38), vs -0.486 (p = 2.6e-60) on all
  1,000 pairs.
- **S2 matched contrasts.** 1,529 narrow-vs-wide contrasts matched on corpus JSD (<= 0.02 bits) and on
  endpoint output movement (<= 0.05 bits) with dw >= 0.15 consistent in all three frames; only 21 share
  a token. Largest: ` her`/` when` w=0.34 vs ` kind`/` wrong` w=0.77 at J = 0.70 bits.
- **Main result (new).** Held-out R^2 for `w` (5-fold over pairs): corpus JSD 0.149; quadratic 0.165;
  model-output JSD 0.187; 5 pair covariates + J 0.399; **token-additive 0.365; token-additive + J
  0.578**; + model-output JSD 0.648; + block-0 geometry 0.723. Reproducibility ceiling 0.934
  (Spearman-Brown on across-frame agreement, mean r = 0.825).
- **What the token effect is not.** a_u vs corpus log-frequency rho = -0.33 (p = 2.9e-4), vs
  continuation entropy -0.24 (p = 0.008), vs model surprisal +0.26 (p = 0.004).
- **Refuted alternative.** Path-length normalisation: CV(w) = 0.158 but CV(w * d0) = 0.216, and
  rho(d0, w) = +0.17 — the wrong sign for a fixed-size transition on a longer path.
- **Residual pair structure.** Additive+J residuals agree across sentence frames at r = 0.67
  (variance share 0.86 of what remains); adding output JSD and block-0 geometry lowers it to r = 0.54.

**Figures added (all embedded in both deliverables):** `scatter_and_gate.png`, `contrast_curves.png`,
`cv_r2.png`, `prediction.png`, `token_effects.png`.

`check_render.py REPORT.md RESULTS.md` passes (10 display eqs, 5 embeds + captions each, 0 problems).

## 2026-08-10 (same iteration, second step) — the per-token effect measured, not just fitted

Two new GPU probes on `pythia-1.4b-deduped` @ `step143000` at the same block-0 hook point, using six
**anchor tokens** (` and`, ` significant`, ` close`, ` playing`, ` bigger`, ` buried`) that appear in
none of the 1,000 pairs. ~220k forward passes, ~10 min.

- **Anchor width** (`experiments/anchor_width.py`): dir18's interpolation protocol between each of the
  123 endpoint tokens and each anchor; per-token median over 6 anchors x 3 frames (18/18 curves valid
  for every token). **rho(anchor width, fitted a_u) = +0.70 (p = 4.6e-19)**, +0.67 with output entropy
  partialled out. At the pair level, 2 free parameters on the measured sum give held-out R^2 = 0.350 vs
  0.365 for the 123-parameter fitted model, and 0.452 with corpus JSD added.
- **Basin radius** (`experiments/basin_probe.py`): angle of great-circle travel from a token's block-0
  state before its output distribution moves tau bits. Random directions: rho = -0.02 (p = 0.87);
  anchor directions (tau = 0.2): rho = +0.39 (p = 1.1e-5) but with the sign OPPOSITE to the basin
  prediction. Pair-level radius sum + J = 0.299. The basin mechanism is therefore not supported and has
  been dropped from the hypotheses.

**Deliverable changes.** REPORT.md gains a Methods subsection for the two probes (with equations for
the great-circle sweep, the output-movement M_u(theta) and the anchor width), Results patterns 6 and 7,
and Figure 6 (`plots/transfer.png`); old patterns 6-7 renumbered to 8-9. Hypotheses rewritten: H1 is now
"width is a per-token trait" (supported by the transfer test) instead of the basin-radius version;
H2 folded output entropy and logit norm into the predictability story; H3 unchanged. Recommended next
experiment changed from the basin probe (now run) to a **forward screen on 40 unseen tokens / 200 fresh
pairs**. Summary and Conclusion rewritten around the transfer result. RESULTS.md gains the measured-
predictor table, the transfer/basin rows, and Figure 6.

No previously reported number was superseded — all 2026-08-10 first-step numbers stand unchanged.
`check_render.py` passes (13 display eqs, 6 embeds + captions per file, 0 problems).

## 2026-08-10 (same iteration, third step) — the forward screen: prediction on unseen tokens

Ran the experiment the second step recommended (`experiments/forward_screen.py`), so the report's
"recommended next experiment" advanced again.

- **Setup.** 40 tokens that appear in none of the 1,000 bank pairs and are not anchors
  (` re`, ` do`, ` time`, ` life`, ` maybe`, ` delicious`, ` extraordinary`, ` awkward`, ...). Measured
  their anchor widths only, then predicted all 780 pairs among them with
  w_hat = -0.0275 + 0.5242 * (w_hat_u + w_hat_v), slope and intercept fitted on the 929 bank pairs and
  frozen. Then ran the 780 pairs through the full protocol and scored. 718 survive the curve-validity
  criteria and the 0.2-bit gate.
- **Result. Forward R^2 = 0.397, Spearman rho = +0.656 (p = 1.5e-89), MAE = 0.047** width units on an
  observed range 0.34-0.78. Median observed w by predicted tercile: 0.50 / 0.57 / 0.62. Baseline on the
  same pairs: model-output JSD rho = -0.507 (and it needs both endpoints of every pair, where the screen
  needs 40 per-token measurements to cover 780 pairings). Corpus JSD could not be scored here — dir18's
  corpus count arrays are no longer on disk — stated in the deliverables.

**Deliverable changes.** REPORT.md: new Methods subsection "The forward screen" (frozen-coefficient
protocol + forward R^2 equation), new Results pattern 7 with **Figure 7** (`plots/forward_screen.png`);
patterns 7-9 renumbered to 8-10 and all cross-references updated; Summary and Conclusion now carry the
forward result; Limitations rewritten (the "screen untested outside the 123 tokens" caveat is retired,
the anchor-set-composition caveat added). Recommended next experiment changed from the forward screen
(now run) to **H1's anchor-set swap**, with a layer sweep as follow-on. RESULTS.md: new forward-screen
table, Figure 7, headline paragraph, and next-experiment section.

No earlier number superseded. `check_render.py` passes (15 display eqs, 7 embeds + captions per file).

## 2026-08-10 (same iteration, fourth step) — anchor-set swap: the trait is real, the measuring stick is not neutral

Ran the control the third step recommended (`experiments/anchor_swap.py`): the anchor width of all 123
endpoint tokens recomputed against two further anchor sets, disjoint from each other and from the
original — six function words (` he`, ` it`, ` we`, ` but`, ` they`, ` them`) and six rare content words
(` surreal`, ` creepy`, ` unbelievable`, ` disgusting`, ` ironic`, ` tempting`, chosen as the highest-id
alphabetic pool tokens, GPT-NeoX BPE ids being frequency-ordered).

- **Result.** The two disjoint sets rank the 123 tokens at rho = +0.46 (p = 1.0e-7) — above chance, well
  below identity. Each still recovers the fitted token effect: rho = +0.57 (function) and +0.61 (rare
  content) against +0.70 for the original mixed set. As pair-level predictors: held-out R^2 = 0.146
  (function), 0.265 (rare content), 0.350 (mixed), 0.318 (function + rare content together).
  Conclusion: a common per-token trait exists that any anchor set finds, plus an anchor-specific
  component, so the anchor set is part of the method and a mixed set of common words is the better
  measuring stick.
- A first run of this control used only ONE valid function-word anchor (the intended list barely
  intersected the eligible pool); it was rerun with a corrected six-token set and only that arm was
  recomputed. The one-anchor numbers are not reported anywhere.

**Deliverable changes.** REPORT.md: Methods subsection "The anchor-set swap", Results pattern 8 with
**Figure 8** (`plots/anchor_swap.png`), patterns 8-10 renumbered to 9-11 with cross-references updated,
Summary now states three qualifications instead of two, H1's alternative reading updated (partly
adjudicated) and its remaining test changed to a single-anchor-many-frames measurement, Conclusion and
Limitations updated (the "anchor-set swap not run" caveat retired, replaced by "w_hat_u means width
against this anchor set"). Recommended next experiment changed from the anchor-set swap (now run) to a
**layer sweep of the anchor width** at blocks 6/12/18. Also added a fold-seed stability note to Methods
(five fold seeds move each held-out R^2 by <= 0.01). RESULTS.md: anchor-set table, Figure 8, headline
paragraph, next-experiment section.

No earlier number superseded. `check_render.py` passes (15 display eqs, 8 embeds + captions per file).

## 2026-08-10 (same iteration, fifth step) — layer sweep: the ranking is early, the sharpening is deep

Ran the experiment the fourth step recommended (`experiments/layer_sweep.py`): the anchor-width
measurement repeated with the interpolation site at blocks 6, 12 and 18 (same 6 mixed anchors, same
protocol, 100% curve validity at every site).

| site | rho vs block-0 w_hat | rho vs fitted a_u | median w_hat | IQR across tokens | held-out R^2 for pair w |
|---|---|---|---|---|---|
| block 0 | 1.00 | +0.70 | 0.553 | 0.102 | 0.350 |
| block 6 | +0.92 | +0.59 | 0.621 | 0.086 | 0.284 |
| block 12 | +0.84 | +0.52 | 0.728 | 0.065 | 0.214 |
| block 18 | +0.72 | +0.35 | 0.800 | 0.020 | 0.146 |

The token ranking survives the move (rho = +0.72 at block 18) while the transitions themselves flatten
to the proportional-response value w = 0.8 and the token-to-token spread collapses fivefold. So which
tokens are narrow is settled early, and how sharp any transition gets is produced by the blocks below
the interpolation site — the sharpening-by-downstream-MLPs picture from the plateau literature.

**Deliverable changes.** REPORT.md: Methods subsection "The layer sweep", Results pattern 9 with
**Figure 9** (`plots/layer_sweep.png`), patterns 9-11 renumbered to 10-12 with cross-references updated,
a new Summary paragraph and Conclusion paragraph on depth, and a Limitations note that the block-18
correlations are attenuated by an IQR of 0.02. Recommended next experiment changed from the layer sweep
(now run) to **an embedding-level probe** (interpolate at the input embedding; fit a linear probe from
static embedding to w_hat_u on 80 tokens, test on 43). RESULTS.md: layer-sweep table, Figure 9, and the
new next-experiment section.

No earlier number superseded. `check_render.py` passes (15 display eqs, 9 embeds + captions per file).

## 2026-08-10 (iteration 2) — the embedding probe: the per-token number can be looked up, not measured

The previous iteration's session ended immediately after `experiments/embed_probe.py` wrote
`results/embed.json` (21:55), so that experiment had never been plotted or folded into the
deliverables. This iteration completed it and added the lookup's forward test.

- **Embedding site.** Anchor widths measured with the interpolation site at the *input embedding*
  (below block 0; 123 tokens x 3 frames x 6 anchors, 99.9% curve validity) agree with the block-0
  values at rho = +0.79 (p = 2.0e-27) and with the fitted token effect at rho = +0.60.
- **Probe.** A ridge probe from the static embedding row W_E[u] to the block-0 anchor width, fitted on
  80 of the 123 tokens and tested on 43 over 50 random splits: held-out rho = +0.764 +- 0.045,
  R^2 = 0.514 +- 0.073, positive in 50/50 splits. Shuffled-target control rho = -0.201. Probe to the
  fitted effect a_u: rho = +0.505.
- **New baseline run this iteration** (`experiments/embed_forward.py`): embedding norm alone, same
  splits, rho = +0.597 +- 0.071, R^2 = 0.190. So norm (a frequency proxy) carries a large part of the
  probe's signal and the rest of the embedding carries more. A first attempt added this baseline inside
  `embed_probe.py`, but re-running that script means redoing six 2048-feature ridge probes (>15 min
  under 4-way CPU contention), so the baseline was moved into `embed_forward.py`, which already loads
  the embedding matrix; `embed_probe.py` is back to its original form and `results/embed.json` is the
  file it wrote at 21:55, unmodified.
- **Zero-forward-pass screen** (`experiments/embed_forward.py`, new): probe fitted on the 123 bank
  tokens, applied to the 40 tokens outside the bank (their measured anchor widths recovered at
  rho = +0.66, p = 3.4e-6), pair-level slope/intercept re-estimated on bank pairs from out-of-fold probe
  features and frozen (w = -0.1019 + 0.5977 * (w_u + w_v)). On the same 718 gated pairs as the measured
  screen: R^2 = 0.213, rho = +0.526, MAE 0.055, tercile medians 0.508 / 0.567 / 0.610 — against the
  measured screen's 0.397 / +0.656 / 0.047. The gap is ridge shrinkage of the predicted range; the
  lookup still ranks unseen pairs about as well as model-output JSD (rho = -0.51) with no forward pass.

**Deliverable changes.** REPORT.md: new Methods subsection "The embedding probe: can the per-token
number be looked up instead of measured?" (probe equation, the two controls, the lookup screen); new
Results pattern 10 with **Figure 10** (`plots/embed_probe.png`, 4 panels); old patterns 10-12 renumbered
11-13 with all cross-references updated; a new Summary paragraph and Conclusion paragraph on the
lookup; forward-pass count 900k -> ~1M; Limitations gained the pool-scope caveat (every token used
anywhere here is a common single-token alphabetic word) and "four sites" -> "five sites". Recommended
next experiment changed from the embedding probe (now run) to **testing the lookup on the vocabulary at
large** (~30 tokens spanning the probe's predicted range, including subword fragments, punctuation,
numerals, capitalised names). RESULTS.md: headline paragraph, two new tables (probe vs its baselines;
measured screen vs lookup screen), Figure 10, and the new next-experiment section.

No earlier number superseded. `check_render.py` passes (16 display eqs, 10 embeds + captions per file).

## 2026-08-10 (iteration 2, second step) — the lookup tested on the vocabulary at large

Ran the experiment the first step recommended (`experiments/vocab_probe.py`): the probe fitted on the
123 bank tokens applied to all 50,304 embedding rows, then 32 tokens selected from four classes
`dir18`'s pool excludes or under-samples — ordinary words outside the pool, subword fragments,
punctuation/numerals, capitalised names — eight per class, spaced evenly over that class's predicted
range, and their anchor widths measured at block 0 (same six anchors, same three frames, 576/576 curves
valid).

| class (8 tokens each) | rho(predicted, measured) | median measured w_hat |
|---|---|---|
| all 32 together | +0.60 (p = 3.0e-4) | - |
| ordinary words outside the pool | +0.57 | 0.632 |
| subword fragments | +0.31 | 0.569 |
| punctuation and numerals | +0.24 | 0.529 |
| capitalised names | +0.83 | 0.527 |
| (reference: the 123 pool tokens) | - | 0.549 |

MAE 0.046; measured widths span 0.367-0.686 against the pool's 0.361-0.660; predicted sd 0.047 vs
measured sd 0.073 (ridge shrinkage, same as pattern 10). The ranking transfers outside the pool and no
class inverts it, though per-class estimates at n = 8 are indicative only.

**Deliverable changes.** REPORT.md: new Methods subsection "The vocabulary-wide test"; new Results
pattern 11 with **Figure 11** (`plots/vocab_probe.png`); old patterns 11-13 renumbered 12-14 with
cross-references updated; Conclusion paragraph extended with the vocabulary check; Limitations reworded
(the "validated only on common alphabetic words" caveat is replaced by the weaker, accurate "eight
tokens per class, indicative"); forward-pass accounting extended. Recommended next experiment changed
from the vocabulary test (now run) to **re-measuring anchor widths in structurally different contexts**
(mid-sentence, interrogative, list, code-like), since every result so far shares one frame shape.
RESULTS.md: vocabulary table, Figure 11, headline sentence, and the new next-experiment section.

No earlier number superseded. `check_render.py` passes (16 display eqs, 11 embeds + captions per file).

## 2026-08-10 (iteration 2, third step) — frame-shape control: the ranking is the token's, the level is the context's

Ran the experiment the second step recommended (`experiments/frame_control.py`): the anchor-width
measurement for the same 123 tokens and the same six anchors repeated in four contexts of deliberately
different shape, and each context's token ranking correlated with the original one. The reference is
the agreement among the three ORIGINAL frames, each summarised the same way (median over 6 anchors
within one frame), which is what two measurements of the same shape achieve.

| context | rho with the original ranking | median w_hat | IQR | curve validity |
|---|---|---|---|---|
| (reference: two of the three original frames) | +0.822 | 0.549 | 0.102 | - |
| `She kept walking because everything felt` | +0.844 | 0.599 | 0.123 | 1.000 |
| `Is it really` | +0.770 | 0.623 | 0.107 | 0.996 |
| `The report mentions the following:` | +0.735 | 0.530 | 0.118 | 1.000 |
| `def solve(x): / return` (code) | +0.501 | 0.705 | 0.049 | 1.000 |

The nearest context matches the within-shape reference (+0.844 vs +0.822), and even the code prefix
stays above the two-disjoint-anchor-set agreement (+0.46). The level moves a lot (median 0.530-0.705),
and the code context also compresses the token spread (IQR 0.049). Mean agreement among the four new
contexts: +0.624.

**Deliverable changes.** REPORT.md: new Methods subsection "The frame-shape control"; new Results
pattern 12 with **Figure 12** (`plots/frame_control.png`); old patterns 12-14 renumbered 13-15 with
cross-references updated; new Summary paragraph; Conclusion extended; **Limitations rewritten** — the
"three sentence frames shared by every measurement" caveat is now narrowed to the pair-level results,
since the per-token measurement has been checked in four other context shapes. Forward-pass accounting
1M -> 1.1M. Recommended next experiment changed from the frame-shape control (now run) to an
**embedding-space intervention**: add +-epsilon times the probe's unit direction to a token's embedding
row, re-measure its anchor width, with a matched-norm random-direction control — the first test of
whether the probe direction is causal rather than correlational. RESULTS.md: frame-control table,
Figure 12, headline paragraph, setting line, and the new next-experiment section.

No earlier number superseded. `check_render.py` passes (16 display eqs, 12 embeds + captions per file).

## 2026-08-10 (iteration 2, fourth step) — the embedding intervention: a null, and the first causal test

Ran the experiment the third step recommended (`experiments/embed_intervene.py`). The probe's gradient
with respect to the raw embedding row, g_j = beta_j / sd_j, gives a step delta = (Delta / ||g||^2) g
that changes the probe's OWN prediction by exactly Delta. For 16 tokens spread over the measured-width
range, that step was added to the token's embedding row for Delta in {-0.05, -0.025, +0.025, +0.05}
width units, the token's anchor width re-measured, the row restored; controls were a random direction
of the same step norm and the JSD shift of the token's next-token distribution.

| edit along | slope of measured vs requested dw | mean abs dw | sign agreement | output shift |
|---|---|---|---|---|
| (a causal direction would give) | 1.0 | 0.0375 | 1.00 | - |
| probe direction | -0.023 | 0.0027 | 0.39 | 0.0001 bits |
| random direction, same norm | +0.000 | 0.0008 | 0.50 | 0.0000 bits |

Per-token slopes scatter from -0.13 to +0.15 with no relation to the token's base width
(rho = -0.115, p = 0.67), so the "perturbation compresses width toward the middle" reading was checked
and is not supported either. Step norm 0.053 against a median embedding-row norm of 0.984 (~5%).
**Conclusion reported:** the embedding direction that predicts width does not set it — but the edits
shifted the model's output by only 0.0001 bits, so this establishes that the probe direction is not an
efficient lever, not that no lever exists.

**Deliverable changes.** REPORT.md: new Methods subsection "The embedding intervention" (with the step
equation); new Results pattern 13 with **Figure 13** (`plots/intervene.png`); old patterns 13-15
renumbered 14-16 with cross-references updated; Conclusion extended with the null and its caveat;
forward-pass accounting 1.1M -> 1.2M. Recommended next experiment changed from the intervention (now
run) to **repeating it with the step calibrated on the model's behaviour** (grow the step until the
token's output moves 0.05 / 0.1 / 0.2 bits), with the fallback target named (block 0's attention and
MLP response) if both directions then move width equally. RESULTS.md: intervention table, Figure 13,
setting line, and the new next-experiment section.

No earlier number superseded. `check_render.py` passes (17 display eqs, 13 embeds + captions per file).

## 2026-08-11 (iteration 3) — the behaviour-calibrated intervention: the trait is fragile, not steerable

Ran the experiment the previous iteration recommended (`experiments/embed_intervene2.py`,
`experiments/plot_intervene2.py`). For 12 tokens spanning the measured-width range, the embedding edit
is no longer sized by the probe's own prediction but grown along a unit direction until the token's
next-token distribution has moved a fixed number of bits (0.05 / 0.1 / 0.2), in BOTH signs, with a
random unit direction calibrated to the SAME output movement as the control — 144 re-measurements.
Calibration fidelity: achieved / requested output movement median 1.00 (IQR 0.91-1.05).

| edit along | mean abs dw at 0.05/0.1/0.2 bits | signed dw, + / - step (0.05 bits) | median step norm | edits that widen |
|---|---|---|---|---|
| probe direction | 0.103 / 0.130 / 0.148 | +0.118 / +0.088 | 1.01 | 72/72 |
| random, matched on output movement | 0.109 / 0.125 / 0.135 | +0.109 / +0.109 | 1.62 | 72/72 |

Probe vs random on matched edits 0.127 vs 0.123 (Wilcoxon p = 0.47, probe larger in 53%); regression of
measured on probe-predicted dw: slope -0.002, rho = +0.06 (p = 0.61). After a 0.2-bit edit the 12
tokens land at mean w_hat 0.691 (probe) / 0.678 (random) with sd across tokens 0.022 / 0.015, against
mean 0.543 sd 0.083 before; rho(base w_hat, dw) = -0.78 (probe) / -0.94 (random). Reaching a given
output movement takes a smaller step along the probe direction (norm ratio 1.54 / 1.66 / 1.76), and the
larger random displacements (median edited row norm 1.90 vs 1.40) still land at the same width, so the
collapse is indexed by output movement rather than by how far the row moved.

**Conclusion reported.** The previous iteration's null WAS a step-size null — width moves fifty times
more once the edit is behaviourally real (0.003 -> 0.10-0.15 width units) — but the specificity test
fails completely: no single embedding direction is a lever, and every behaviourally sized edit destroys
the token's width trait, leaving it at a generic ~0.68. Narrow transitions are a fragile property of the
exact embedding training produced.

**Deliverable changes.** REPORT.md: new Methods subsection "The behaviour-calibrated intervention"
(with the calibration equation); new Results pattern 14 with **Figure 14** (`plots/intervene2.png`); old
patterns 14-16 renumbered 15-17 with cross-references updated; Summary paragraph on the open mechanism
rewritten around the collapse result; Conclusion rewritten in the same place; forward-pass accounting
1.2M -> 1.3M. Recommended next experiment changed from the behaviour-calibrated intervention (now run)
to a **behaviour-preserving displacement**: step the embedding row by the norm a 0.2-bit edit needed but
along directions whose output shift stays under 0.005 bits, and see whether w_hat survives — deciding
whether the trait is behavioural or geometric, which also decides what the free vocabulary lookup is
reading. RESULTS.md: two new tables (calibrated intervention, and where the edited tokens land),
Figure 14, a new headline paragraph, the setting line, and the new next-experiment section.

No earlier number superseded; the probe-calibrated intervention is retained as the test of the probe's
own quantitative claim, explicitly framed as the loophole this experiment closes. `check_render.py`
passes (18 display eqs, 14 embeds + captions per file).

## 2026-08-11 (iteration 3, second step) — the fixed-displacement test: the collapse follows the move, not the model's response

Ran `experiments/embed_quiet.py` + `experiments/plot_quiet.py` (Figure 15) to separate the two readings
of the collapse. For each of the same 12 tokens, 48 random directions were probed at a step of 0.05
(the linear regime) and the logit responses stacked into a matrix whose left singular vectors give the
loudest and quietest combinations; the token was then displaced by the SAME norm its own 0.2-bit edit
required (median 1.84) along the quietest, the loudest and one plain random direction.

| direction, same displacement norm per token | output movement produced | mean dw | mean w_hat after | sd across tokens |
|---|---|---|---|---|
| (before any edit) | - | - | 0.543 | 0.083 |
| quietest of 48 combinations | 0.181 bits | +0.132 | 0.675 | 0.019 |
| loudest of 48 combinations | 0.261 bits | +0.105 | 0.648 | 0.039 |
| plain random direction | 0.165 bits | +0.132 | 0.675 | 0.026 |

Across the 36 edits, rho(output movement, dw) = +0.074 (p = 0.67) over a 0.03-0.77 bit range, while the
collapse reproduces in all three directions. Post-edit widths still correlate with the originals at
rho = +0.62 (p = 0.03). **Reported limitation:** the "quiet" construction is not quiet at this
displacement (0.181 vs 0.165 bits for a random direction) — the linear response at eps = 0.05 does not
survive a step of norm 1.84 — so the behaviour-preserving version of the test was not achieved.

**Superseded numbers.** The previous entry's claim that the collapse "is indexed by output movement
rather than by how far the row moved" (inferred from probe steps of norm 1.40 landing at 0.691 against
random steps of norm 1.90 landing at 0.678) is **withdrawn**: at fixed displacement norm the width
change does not track output movement at all (rho = +0.07, p = 0.67). The corrected statement is that
the landing width is insensitive to what the edit does to the model over this range, and what matters
is that a displacement of this size happened. Also corrected: "destroys the trait" -> "compresses the
trait", since the post-edit ranking still agrees with the original (rho = +0.73 / +0.85 after a
0.05-bit edit, +0.57 / +0.36 after a 0.2-bit one, +0.62 in this run).

**Deliverable changes.** REPORT.md: new Methods subsection "The fixed-displacement test" (with the
response-matrix equation); new Results pattern 15 with **Figure 15** (`plots/quiet.png`); patterns
15-17 renumbered 16-18 with cross-references updated; pattern 14, the Summary and the Conclusion
corrected as above; Limitations extended with the 12-token sample size and the failed quiet
construction; forward-pass accounting 1.3M -> 1.4M. Recommended next experiment changed to **a ladder
of displacement norms (0.1 to 1.0) with the quiet/loud combinations rebuilt at each**, to find the
norm at which the two separate. RESULTS.md: fixed-displacement table, Figure 15, headline paragraph
and next-experiment section updated the same way. `check_render.py` passes (19 display eqs, 15 embeds
+ captions per file).

## 2026-08-11 (iteration 4) — the displacement-norm ladder: behaviour destroys the trait, displacement only compresses it

Ran `experiments/norm_ladder.py` + `experiments/plot_ladder.py` (Figure 15, `plots/ladder.png`), the
experiment the previous entry recommended. For each of the same 12 tokens and each rung of a
displacement ladder (norms 0.15 / 0.4 / 0.9 / 1.8, against a median embedding-row norm of 0.98), 24
random unit directions were displaced by that norm and their **actual** output movement measured there;
the argmin became the quiet direction and the argmax the loud one, with a fixed random direction as
control. Anchor width was re-measured for all three at every rung (144 re-measurements).

| displacement norm | quiet: bits / mean w_hat / rho(before, after) | loud: bits / mean w_hat / rho(before, after) | paired p |
|---|---|---|---|
| (before any edit) | - / 0.543 / - | - / 0.543 / - | - |
| 0.15 | 0.0001 / 0.544 / +1.00 | 0.0003 / 0.546 / +1.00 | 0.09 |
| 0.40 | 0.0006 / 0.552 / +0.99 | 0.0027 / 0.562 / +0.99 | 0.02 |
| 0.90 | 0.0053 / 0.589 / +0.91 | 0.0221 / 0.620 / +0.87 | 0.0005 |
| 1.80 | 0.0489 / 0.656 / +0.94 | 0.4023 / 0.683 / +0.08 | 0.09 |

**Superseded result and reversed conclusion.** This replaces the fixed-displacement test (previous
entry, `plots/quiet.png`, 12 tokens at per-token norm ~1.84 with quiet/loud built from the SVD of
linear-regime logit responses), whose reported limitation was that the construction produced no
genuinely quiet direction there (0.181 bits quiet vs 0.165 random). Selecting by measured response at
each rung fixes that: 0.049 vs 0.402 bits at norm 1.8, an 8x separation. With a real contrast the
earlier reading **reverses**. Old: "the width change is flat in output movement (rho = +0.07, p = 0.67)
... what erases the trait is the displacement itself rather than what the displacement does to the
model's output." New: at identical displacement the quiet edit preserves the token ordering
(rho = +0.94, p = 4e-6) and the loud edit erases it (rho = +0.08, p = 0.80); the quiet direction widens
less than the loud one in the paired test at every rung (p = 0.0005 at norm 0.9, p = 0.02 at 0.4). The
*level* still follows the displacement (0.543 -> 0.656 even for the quietest direction at norm 1.8, sd
across tokens 0.083 -> 0.038), so the corrected statement is: displacement compresses the level,
behaviour destroys the ordering. Consequence for the deliverable's story: the vocabulary-wide lookup
reads a behavioural property, not a geometric accident — the caveat the previous entry flagged as
unresolved is resolved in the favourable direction.

**Deliverable changes.** REPORT.md: Methods subsection "The fixed-displacement test" replaced by "The
displacement-norm ladder" (with the per-rung response equation and the level/ordering split stated as
the two discriminating quantities); pattern 15 rewritten with the rung table and **Figure 15** now
`plots/ladder.png` (`plots/quiet.png` retired from both deliverables, PNG kept on disk); Summary and
Conclusion paragraphs on fragility rewritten from "displacement erases it" to "behaviour erases the
ordering, displacement compresses the level"; Limitations updated (quiet = quietest of 24 draws, and
0.049 bits is quiet-relative-to-loud, not silent); forward-pass accounting 1.4M -> 1.5M. Recommended
next experiment changed from the ladder itself to **decomposing the loud edit's output change by
successor token** (top-mass successors vs tail, at matched total output movement), which asks which
part of the token's behaviour carries the trait and whether it reconnects to corpus successor JSD.
RESULTS.md: headline fragility paragraph, ladder table, Figure 15 and next-experiment section updated
the same way. `check_render.py` passes (19 display eqs, 15 embeds + captions per file).

## 2026-08-11 — iteration 5: the mode split (which successors a disruptive edit moves)

**New experiment.** `experiments/mode_split.py` (+ `experiments/plot_mode.py`, `results/mode_split.json`,
`results/mode_split.log`) runs the experiment the previous entry recommended. For the same 12 tokens as
the ladder, every edit's output change is partitioned by successor token (JSD is a sum of non-negative
per-successor terms) into the share $S$ landing on the token's 32 most likely successors and the rest.
24 random directions per token at displacement norm 1.8 give the descriptive half; the most top-heavy
and most tail-heavy of them, each rescaled by a log-log calibration scan to the same 0.4 bits of output
movement, give the causal half. 12 tokens x (24 scan directions + 2 calibrated re-measurements).

**New numbers.** Base mass in the top 32 successors 0.707, but the loudest random direction puts only
**0.389** of its divergence there — the disturbance is tail-weighted relative to mass — and louder
directions are more tail-weighted still (median-token $\rho(B_j, S_j) = -0.36$). At matched movement:
top-heavy 0.410 bits, $S = 0.408$, mean $\hat w_u$ 0.666, $\rho$(before, after) $= -0.08$ ($p = 0.81$);
tail-heavy 0.453 bits, $S = 0.355$, mean $\hat w_u$ 0.651, $\rho = -0.37$ ($p = 0.24$); pre-edit
0.543 +- 0.083. Both destroy the ordering. One paired difference survives: the top-heavy edit widens
more (mean $\Delta\hat w_u$ +0.124 vs +0.108, Wilcoxon $p = 0.009$) while moving the output less.

**Nothing superseded.** This is a new result, not a replacement; no earlier number changed. The
steering half is a null with a stated limit (random directions span only $S = 0.36$–$0.56$ and never
reach the mass-proportional 0.71), reported as such.

**Deliverable changes.** REPORT.md: new Methods subsection "The mode split: which successors does a
disruptive edit move?" (per-successor JSD partition and top-mass share $S$ defined in a display
equation, with the base-mass reference point and the matched-movement rescaling); new Results
**pattern 16** with **Figure 16** = `plots/mode_split.png`; previous patterns 16/17/18 renumbered to
17/18/19 and the two "(pattern 16)" cross-references to the basin result updated to "(pattern 17)";
"Recommended next experiment" rewritten from "decompose the loud edit by successor token" (now done) to
**constructing** top-heavy and tail-heavy directions from the top successors' unembedding rows or a
subspace search, since random draws give too little contrast in $S$. RESULTS.md: mode-split table and
Figure 16 added, headline fragility paragraph extended with the tail-weighting number, next-experiment
section rewritten to match, forward-pass accounting and setting line updated. `check_render.py` passes
(REPORT 20 display eqs / 16 embeds, RESULTS 16 embeds, 0 problems).

## 2026-08-11 — iteration 6: constructing the top-heavy / tail-heavy directions

**New experiment.** `experiments/mode_construct.py` (+ `experiments/plot_construct.py`,
`results/mode_construct.json`, `results/mode_construct.log`) runs the experiment the previous entry
recommended, in its cheap form. For each of the same 12 tokens, 24 random probe directions are applied
at displacement norm 0.6 and their logit responses are turned into a 24x24 generalised eigenproblem
whose Rayleigh quotient IS the top-mass share $S$; the extreme eigenvectors give the $S$-maximising and
$S$-minimising combinations in closed form (no Jacobian, ~40x cheaper than the projection version). Both
are rescaled to the same 0.4 bits and anchor width is re-measured.

**New numbers.** Predicted $S$ (small-step regime) 0.856 vs 0.179 — a 0.68 separation, 3x what 24 random
draws span (0.21), with the top-heavy end past the mass-proportional 0.71. Achieved $S$ once rescaled to
0.4 bits: **0.369 vs 0.390** — indistinguishable, both below the base mass, paired difference
$p = 0.09$ and in the *wrong* direction. Output movement 0.422 vs 0.419 bits; mean $\hat w_u$ 0.666 vs
0.672 (pre-edit 0.543 +- 0.083), sd across tokens 0.023 / 0.020; $\rho$(before, after) $= -0.16$
($p = 0.62$) and $-0.28$ ($p = 0.38$); paired $\Delta\hat w_u$ difference $p = 0.38$.

**Nothing superseded.** New result; no earlier number changed. It converts pattern 16's underpowered
steering null into a *bounded* one — the instrument now exists and still cannot separate the arms —
and adds the mechanistic point that the tail-weighting of a large embedding edit is set by the step
size, not by the direction.

**Deliverable changes.** REPORT.md: new Methods subsection "Can the split be steered on purpose?" (the
Rayleigh-quotient construction with $A$/$B$ defined in a display equation, probe norm 0.6, predicted-vs-
achieved $S$ as a test of the method); new Results **pattern 17** with **Figure 17** =
`plots/mode_construct.png`; previous patterns 17/18/19 renumbered to 18/19/20 and the four cross-
references updated; "Recommended next experiment" rewritten from "build the two directions" (now done,
negative) to **component ablation in blocks 0–5** scored by how much of the across-token spread in
$\hat w_u$ each component destroys — the first intervention that is not an embedding edit; forward-pass
accounting 1.5M -> 1.6M and runtime 4h -> 4.5h. RESULTS.md: headline fragility paragraph extended with
the constructed-direction result, Figure 17 and its table added, next-experiment section rewritten to
match, setting line updated. `check_render.py` passes (REPORT 21 display eqs / 17 embeds, RESULTS 17
embeds, 0 problems).

## 2026-08-11 — iteration 7: component ablation localises the trait to the block-0 MLP (Figure 18)

**What ran.** New `experiments/ablate.py` + `experiments/plot_ablate.py` (outputs `results/ablate.json`,
`results/ablate_summary.json`, `results/ablate.log`, `plots/ablate.png`) runs the experiment the previous
entry recommended: the first intervention in this direction that is not an embedding edit. Each of the
102 early components — the 16 attention heads and the MLP of every block 0–5 — is mean-ablated one at a
time at the final token position (replacement = that component's mean output at that position over the
18 endpoint prompts, measured unablated), endpoints and interpolation bank are recomputed with the
ablation live, and the per-token anchor width is re-measured for the 12 intervention tokens against the
6 anchors in the first frame. 470 s on one GPU.

**New numbers.** Unablated: mean $\hat w_u$ 0.565, sd across tokens 0.084. **Block-0 MLP ablated: mean
0.822, sd 0.018, $\rho$(before, after) $= -0.10$, output movement 0.451 bits.** Every other component:
median $\rho = +0.99$ and median sd 0.084 across the 102; worst of the 96 attention heads $\rho = +0.97$
(sd 0.076, $\le 0.0004$ bits); worst of the five MLPs above block 0 $\rho = +0.90$ (sd 0.091,
$\le 0.007$ bits). $\rho$(output movement, $\rho_c$) $= -0.46$ across components.

**Nothing superseded.** New result; no earlier number changed. It is the first positive mechanistic
localisation in the direction, and it is reported with its confound stated: 0.451 bits is essentially
the 0.4-bit rung at which pattern 15's ladder showed any disturbance flattens the ordering, so ablation
alone cannot say whether the block-0 MLP computes the trait or is merely the only early component large
enough to reach that regime.

**Deliverable changes.** REPORT.md: new Methods subsection "Component ablation: which early computation
carries the trait?" (mean-ablation protocol, head-slice vs MLP-output hook points, and a display
equation defining the three per-component scores $\mathrm{sd}_c$, $\rho_c$, $B_c$); new Results
**pattern 21** with **Figure 18** = `plots/ablate.png` and its table, appended after pattern 20 so
figure order stays sequential; Summary gains a paragraph on the localisation and its confound;
"Recommended next experiment" rewritten from component ablation (now done) to the **block-0 MLP
dose–response curve against an output-movement-matched random control**. RESULTS.md: headline gains the
localisation paragraph, new ablation table + Figure 18 with the flat-profile reading spelled out, next-
experiment section rewritten to match. `check_render.py` passes (REPORT 22 display eqs / 18 embeds,
RESULTS 18 embeds, 0 problems).

## 2026-08-11 — iteration 8: dose–response breaks the block-0 MLP confound (pattern 22)

**New experiment.** `experiments/dose.py` + `experiments/plot_dose.py` → `results/dose.json`,
`plots/dose.png`. The block-0 MLP's final-position output is blended toward its mean with weight
alpha = 0.1 … 1; at each dose a fixed random direction added to the same residual stream is
bisection-scaled to move the model's output by the same number of bits. Both arms are scored by
rho(before, after) of the per-token anchor width and by the across-token sd, for the same 12 tokens,
6 anchors and 1 frame as the ablation sweep. 56 s.

**Result (new, supersedes nothing).** The arms separate where it matters. Over 0.007–0.103 bits the MLP
dose is below its matched control at every rung (rho +0.84/+0.99, +0.64/+0.91, +0.62/+0.79,
+0.25/+0.61); the MLP arm crosses rho = 0.6 at ~0.03 bits and the control at ~0.10, i.e. a random
disturbance needs ~3.5x more output movement for the same damage. The across-token sd collapses
identically in both arms (0.069/0.067, 0.055/0.053, 0.027/0.026). Above 0.25 bits both arms are at
noise (+0.74/−0.32, −0.10/−0.76; SE(rho) ≈ 0.3 at n = 12) and are reported but not interpreted.

**Deliverable changes.**
- RESULTS.md — summary bullet on the ablation updated: the confound sentence ("which is both the
  finding and its caveat") replaced by the dose–response verdict. New subsection "Dose–response: is the
  block-0 MLP special, or merely loud?" with the matched-bits table and **Figure 19**
  (`plots/dose.png`). "Next experiment" rewritten: old ask (run the dose–response) → new ask (probe and
  transplant the block-0 MLP output vector m_u).
- REPORT.md — Summary paragraph on the ablation updated with the dose–response verdict; new Methods
  subsection "Separating a carrier from a loud component: the matched-bits dose–response" defining
  m_u^mlp(alpha), m_u^ctrl(c) and the bisection condition B(ctrl) = B(mlp) in a ```math fence; new
  Results pattern 22 with the table and **Figure 19**; "Recommended next experiment" rewritten from the
  dose–response ask to the probe/transplant ask.
- Figure count 18 → 19 in both files. `check_render.py` passes on both (23 display eqs, 19 embeds).

**Interpretation recorded.** The direction's first positive mechanistic localisation: the per-token
width trait is realised in the block-0 MLP's contribution to the final-position residual stream, and
level (spread) versus ordering are separate channels — disturbance of any kind compresses the level,
only the block-0 MLP erases the ordering.

## 2026-08-11 — iteration 9 (final): Conclusion brought up to the current-best story; direction closed

**No new experiment.** Time budget exhausted; this iteration is finalization only. All plan stages
S1–S5 are complete and every experiment S5 recommended has been run through the block-0 MLP
dose–response (iteration 8). Zero unaddressed `human_feedback*.md` / `*REVIEW*` files in this
direction (checked at iteration start), so writing `STOP` is permitted under CLAUDE.md rule 11.

**Deliverable changes (REPORT.md only; RESULTS.md was already current-best and is unchanged).**
- REPORT.md Conclusion contained a stale claim that iteration 7–8's results had superseded:
  "What each token carries is the shape of its transition, and we cannot yet say what in the network
  produces it." The Summary and Results already carried the ablation + dose–response localisation, so
  the Conclusion contradicted the rest of the report. Old → new: that clause is cut, and a new
  Conclusion paragraph states the localisation — 101 of 102 early components leave the token ordering
  intact (median rho = +0.99, every head >= +0.97); only the block-0 MLP collapses the spread
  (0.084 → 0.018) and erases the ordering (rho = −0.10); the matched-bits dose–response breaks the
  loudness confound (at 0.014 bits, rho = +0.64 MLP vs +0.91 control; ~3.5x more output movement
  needed by the control for equal damage), with the across-token spread collapsing identically in both
  arms. Level and ordering named as separate channels; the probe/transplant follow-up named.
- REPORT.md Limitations gains the dose–response caveat: 12 tokens, one frame, one random-control seed,
  and both arms at noise above 0.25 bits (SE(rho) ≈ 0.3 at n = 12), so the localisation rests on the
  four rungs between 0.007 and 0.103 bits.
- No figure added or removed; figure count stays 19 in both files, numbering unchanged.

**Verification.** `python3 experiments/check_render.py REPORT.md RESULTS.md` → ALL CHECKS PASS
(REPORT 23 display eqs / 522 inline eqs / 19 embeds / 0 problems; RESULTS 319 inline eqs / 19 embeds /
0 problems).

**Final state of the deliverable.** Headline chain, current-best: corpus successor JSD explains
held-out R^2 = 0.149 of transition width against a 0.934 reproducibility ceiling; a per-token additive
term takes it to 0.578 and alone reaches 0.365; the per-token number is measurable from six unpaired
anchors (rho = +0.70 with the fitted effect, 2 parameters matching 123); frozen into a screen it
predicts 718 pairs of 40 unseen tokens at R^2 = 0.397 / rho = +0.66 / MAE 0.047; it is readable from
the static embedding (probe rho = +0.76, zero-forward-pass screen R^2 = 0.213) and holds outside the
curated pool (rho = +0.60 on 32 tokens from four excluded classes) and across four context shapes
(rho +0.84 … +0.50 vs +0.82 for two original frames); the trait is behavioural, not positional
(quiet vs loud edits at displacement 1.8: rho +0.94 vs +0.08); and it is realised in the block-0 MLP's
contribution to the final-position residual stream (dose–response vs an output-matched control).
Negative results retained: basin radius refuted, path-length normalisation refuted, probe direction is
not a width lever, top-mass steering untestable by embedding edits.

**STOP written.** Plan complete, no unaddressed feedback.
