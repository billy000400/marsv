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
