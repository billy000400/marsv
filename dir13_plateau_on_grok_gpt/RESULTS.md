# RESULTS — Do Grokking and Matthew-style activation plateaus emerge together?

> CURRENT-BEST ONLY. History lives in CHANGELOG.md. Read before rewriting.

## Question & verdict

**Question.** In a 12-layer character-level Shakespeare GPT — the architecture of *Deep Networks
Always Grok* (Figure 9) — do **Matthew-style activation plateaus** (Shinkle & Hex, *Activation
Plateaus: Where and How They Emerge*) appear or sharpen during the same training window as the
Figure-9 grokking signature (a **second local-complexity descent** plus **delayed `ε=0.03`-PGD
adversarial robustness**)? A "plateau" is this: interpolate between two natural inputs' final-position
activations and the downstream output stays locked near endpoint A, snaps across a narrow boundary,
then locks near endpoint B — a plateau–boundary–plateau `d(t)` curve, not a straight diagonal.

**Bounded verdict = PLAN case 5, "primary relationship not testable," refined by a secondary
temporal observation.**

1. **Grokking gate: all three trained models FAIL.** The existing 3,500-step pilot, a fresh 30k-step
   character run, and a fresh BPE run each show the *first* local-complexity (LC) descent and emerging
   `ε=0.03`-PGD robustness, but **no second LC descent** within budget — the ordering that *defines*
   grokking is absent. In particular the **BPE** model, required to carry Matthew's exact `big/in`,
   `big/large` single tokens, fails the gate. So the **primary** Matthew-exact Grokking↔plateau
   relationship is **not testable** on a grokking model here.

2. **Plateaus are present and we can time their emergence.** Using Matthew's exact config path
   (`shared_context = "The house was"`, 50-step slerp grid, full interpolation-layer sweep) with the
   two preregistered single-token **character controls** `b↔i` and `b↔l`, plateaus **emerge during
   training**: at initialization the `d(t)` curve is the diagonal (width ≈ 0.80, no plateau); by
   **step ~831** it is already a sharp sigmoid (width ≈ 0.33–0.35) and stays there to step 30k. This
   emergence coincides with the **first** LC descent and the initial clean-accuracy rise, and is
   **fully formed before** adversarial robustness saturates — so, in this non-grokking model, the
   plateau tracks *initial fit*, not the (absent) grokking transition.

3. **Bounded relationship claim.** We **cannot** claim plateaus sharpen *during* a second-descent /
   delayed-robustness window, because no run ever opens that window. The secondary evidence we *do*
   have points the other way: the plateau is a property of the trained-and-memorising downstream
   stack that appears early, so there is **no visible temporal coupling to the grokking signature**
   (which never occurs). This is evidence about temporal association only, not causation.

4. **The plateau shape is general, but graded — and it survives a change of context.** Holding one
   endpoint fixed at the **comma** and interpolating to **all 64 other characters**, every pair bends
   away from the straight line — none is linear — yet only 1/64 meets the strict frozen plateau rule
   (transition width ≤ 0.25; median width 0.340 against 0.80 for a straight line). Repeating the whole
   sweep in **8 further contexts** taken from held-out text (**576 pairs**) replicates this exactly:
   **0/576** curves are near-linear and per-context median widths stay in a narrow band (0.313–0.436).
   The claim that sharpness tracks how likely the model thinks the other character is holds in
   **direction everywhere** (all 9 contexts give a negative rank correlation, sign test p = 0.004) but
   its **size is context-dependent** (median ρ = −0.41, range −0.05 to −0.74).

## Models actually tested

| Model | Tokenizer | Params | Trained to | Role |
|---|---|---|---|---|
| Fresh character GPT | char (vocab 65) | 8.38M | 30,000 steps, val acc 0.554 (peak 0.568) | Figure-9 control + **Matthew char-control assay** |
| Fresh BPE GPT | GPT-2 BPE (vocab 50257) | — | 10,000 steps (killed; overfit) | primary Matthew bridge (`big/in`, `big/large`) — **gate FAILs** |
| Pilot character GPT | char (vocab 65) | 8.38M | 3,500 steps, val acc 0.560 | pilot only |

All are 12-layer/12-head GeLU GPTs (`d_model=240`, context 128). Provenance, seeds, corpus SHA-256 in
`results/train_meta*.json`; confirmed-vs-reconstructed fields in `MODEL_SPEC.md`. The paper's exact
GPT code/checkpoint is **not public** (repo audited 2026-07-15), so these are faithful reconstructions.

![Pilot character-model training curves. Left: cross-entropy loss in nats (y) vs training step (x) for the train split (solid) and validation split (dashed, squares) — validation bottoms at ≈1.49. Right: validation next-character accuracy (y) vs training step (x), rising to 0.56.](plots/training_curves.png)

## Figure-9 grokking gate — all three models FAIL

PLAN makes a **validity gate** mandatory before any *joint* claim: the model must qualitatively
reproduce *Deep Networks Always Grok* Fig. 9 — a **second LC descent** beginning before the
test-accuracy peak, plus **delayed adversarial robustness** (`ε=0.03` `ℓ∞`-PGD accuracy rising while
that second descent continues). LC is the paper's sign-crossing count summed over the 12 GeLU layers
on 1,024 train/test/random points (`r=0.005`, `P=25`, 99% CIs); pipeline source-locked to the official
repo (`experiments/fig9.py`, 0.0 logit-reimplementation error). Preregistered PASS/FAIL rule in
`experiments/fig9_verdict.py`.

| Figure-9 quantity | Pilot char (3.5k) | **Fresh char (30k)** | **Fresh BPE (10k)** |
|---|---|---|---|
| checkpoints evaluated | 13 | 14 | 10 |
| clean acc (peak / final) | 0.564 / 0.564 | 0.568@4994 / 0.554 | 0.299@831 / 0.274 |
| `ε=0.03` PGD adv acc (final) | 0.327 | **0.528** | **0.187** |
| test LC (first → min → final) | 1940 → 68 → 68 | 1940 → **8.1** → 8.1 | 2182 → **95** → 95 |
| LC minimum at… | last ckpt | last ckpt | last ckpt |
| second LC descent? | No | No | No |
| delayed robustness emerged? | Yes | Yes | Yes |
| **preregistered verdict** | **FAIL** | **FAIL** | **FAIL** |

The fresh char run reaches *higher* adversarial accuracy than the pilot (0.53 vs 0.33), so robustness
clearly emerges — but test LC descends **monotonically** to its minimum (8.1) at the last checkpoint,
never rising to produce a second descent. Both fresh runs **overfit** (val loss bottoms early — char
≈step 3,750, BPE ≈step 750 — then rises while train loss falls), the opposite of grokking's delayed
val-loss recovery. Robustness here is a property of the first fold-collapse of a memorising network,
not the grokking second-descent event.

![Pilot char (3.5k) Figure-9 curves. Left y-axis: local complexity (sign-crossings summed over 12 GeLU layers) for the train (solid), test (dashed) and random (dash-dot) base-point sets, with 99% CI bands. Right y-axis: next-token accuracy — black with circle markers = clean test accuracy, black dotted with square markers = ε=0.03 PGD adversarial accuracy. x-axis: training step (log scale, step 0 at 1). LC monotone to the 3,500-step horizon (no second descent); adv accuracy climbs to 0.33; verdict FAIL.](plots/grokking_pilot_char.png)
![Fresh char (30k) Figure-9 gate. Same axes and line styles as the pilot panel. LC monotone to 8.1 while adversarial accuracy climbs to 0.53; no second descent → FAIL.](plots/grokking_fresh_char.png)
![Fresh BPE (10k) Figure-9 gate. Same axes and line styles. LC monotone to 95 while adversarial accuracy climbs to 0.19; no second descent → FAIL.](plots/grokking_fresh_bpe.png)

**Joint timeline (S7).** On one training-step axis the Grokking side is uniform across runs — LC falls
monotonically and PGD robustness rises, no second descent anywhere. Because the BPE bridge to Matthew's
exact tokens does not reproduce Figure 9 (nor does either character run), the primary Matthew-exact
relationship is not testable: that window never opens.

![Joint checkpoint timeline. Left: test local complexity (y) vs training step (x, log) for the pilot char run (dotted, triangles), fresh char run (solid, circles) and fresh BPE run (dashed, squares); the legend gives each run's Figure-9 verdict. Middle: ε=0.03 PGD adversarial accuracy (y) vs step (x, log), same three line styles; the horizontal dashed line is the 0.05 robustness threshold. Right: text summary of the three FAIL verdicts and the bounded relationship verdict. No second LC descent in any run → primary relationship not testable.](plots/joint_timeline.png)

## Primary plateau evidence — Matthew-faithful char-token controls across training (S6)

This is the **plateau half** of the joint question, run through Matthew's own code path
(`experiments/run_matthew_ckpts.py`, config `configs/matthew_char_control.yaml`): shared context
`"The house was"`, **exactly 50** evenly spaced interpolation values, `slerp_rescale` (spherical
direction + linear norm), patch **only the final position**, and sweep **every** interpolation layer
(resid_post blocks 0–11) recording Matthew's downstream hooks + final logits. The two single-token
pairs are the preregistered controls `b↔i` and `b↔l` (each a single character token; the BPE
`big/in`, `big/large` bridge is non-decisive because the BPE model fails the gate). We evaluate the
**6 frozen phases** (`results/frozen_phases_char.json`, chosen from the Figure-9 curve **before**
inspecting any plateau curve): steps **0, 56, 831, 7819, 17500, 30000**.

**The plateau is absent at init and emerges during the first LC descent, then is stable.** At
interpolation block 0 (Matthew's default), the final-logit transition width `w_10→90` per pair:

| training step | `b↔i` width | `b↔l` width | plateau? |
|---:|---:|---:|---|
| 0 (init) | 0.802 | 0.802 | no — diagonal |
| 56 | 0.771 | 0.814 | no — diagonal |
| 831 | 0.348 | 0.674 | forming |
| 7,819 | 0.364 | 0.326 | **yes** |
| 17,500 | 0.336 | 0.338 | **yes** |
| 30,000 | 0.331 | 0.330 | **yes** |

Diagonal (no-plateau) reference = 0.80. The width collapses from the diagonal to ≈0.33 between steps
56 and 831 — i.e. **during the first LC descent and initial clean-accuracy rise**, and **fully formed
before** `ε=0.03` robustness saturates (which happens over steps ~10³–10⁴; see the timeline below).
The plateau then holds flat to step 30k. This is the concrete temporal read: in a model that never
groks, the plateau appears with *initial fit*, not with a second descent.

![Matthew-faithful char-control d(t) (y) vs interpolation step t (x), interpolation block 0, final logits, one panel per frozen checkpoint (steps 0→30000). The b↔i pair is the solid line with circle markers, b↔l the dashed line with square markers, and the gray dashed straight line is the diagonal d = t. Curves lie on the diagonal at init and step 56 and become sharp plateau–boundary–plateau sigmoids by step 831, stable thereafter.](plots/matthew_char_ctrl_by_checkpoint.png)

**Same timeline, both phenomena overlaid.** Top: Figure-9 grokking metrics (LC train/test/random +
clean/adv accuracy) for the fresh char run. Bottom: the plateau transition width for both controls.
The width drops to its floor by step ~831 while LC is still in its *first* descent and robustness has
not yet risen — no temporal coupling to a (non-existent) second descent.

![Grokking metrics vs plateau width on one timeline (fresh char run). Top: left y = local complexity for the train (solid), test (dashed) and random (dash-dot) base-point sets with 99% CI bands; right y = next-token accuracy, black with circles = clean, black dotted with squares = ε=0.03 PGD adversarial; x = training step (log). Bottom: transition width w_10→90 (y) for b↔i (solid, circles) and b↔l (dashed, squares) vs step (log); the gray dashed line is the diagonal 0.8, the black dotted line the plateau bar 0.25. Width collapses to its floor by step ~831 — during the first LC descent, before robustness rises.](plots/joint_timeline_char_ctrl.png)

**Depth control also holds for the Matthew-faithful assay.** Within the final checkpoint, moving the
interpolation point later (fewer downstream layers) widens the boundary back toward the diagonal — e.g.
`b↔i` at step 30000: width 0.33 (block 0) → 0.72 (block 3) → 0.80 (block 11). So the plateau is built
by the downstream stack, matching Matthew's layerwise prediction. Raw `d(t)` for every (step, pair,
interp-layer, hook) in `results/matthew_char_ctrl_raw.npz`; widths in
`results/matthew_char_ctrl_summary.json`.

## Comma vs every other character — 64 pairs from one endpoint

An operator asked whether the plateau survives when we hold one endpoint fixed at the **comma** and
interpolate to **every other character in the vocabulary**. Same code path and same settings as the
`b↔i`/`b↔l` controls above (context `"The house was "`, 50 interpolation steps, `slerp_rescale`,
final-position patch, final-logit `d(t)`): endpoint A is always `"The house was ,"`, endpoint B is
`"The house was "` + one of the other **64** characters. Script `experiments/comma_sweep.py`; raw
curves in `results/comma_sweep_raw.npz`, widths in `results/comma_sweep_summary.json`.

**Answer: yes, but the sharpness is graded, and only 1 of 64 pairs meets the strict bar.** At the
final checkpoint (step 30,000, interpolation block 0):

| quantity (64 comma→character pairs) | value |
|---|---|
| median transition width | **0.340** (inter-quartile range 0.305–0.409) |
| narrowest / widest | 0.245 (`c`) / 0.665 (`3`) |
| straight-line reference (no plateau) | 0.80 |
| pairs meeting the strict rule (width ≤ 0.25 + rests near both endpoints) | **1 / 64** |
| pairs with width ≤ 0.35 / ≤ 0.45 | 33 / 64 and 52 / 64 |
| pairs that are near the straight line (width ≥ 0.7) | **0 / 64** |
| pairs whose curve rises without wiggling (isotonic deviation = 0) | 64 / 64 |
| median start / end of the transition | t = 0.252 / t = 0.603 |

Every one of the 64 curves is S-shaped: it stays near the comma prompt's output for roughly the first
quarter of the path, rises steeply, then stays near the other character's output for the last ~40%.
None is a straight line. But most transitions occupy about a third of the path rather than the ≤ 25%
demanded by the strict frozen rule, so "is there a plateau?" is a question of degree here, not a
yes/no. The two preregistered controls `b↔i` (0.331) and `b↔l` (0.330) sit right at this sweep's
median — they were typical pairs, not lucky ones.

![All 64 comma→character curves at step 30,000. Left: relative distance d(t) (y, 0 = output still looks like the comma prompt, 1 = looks like the other-character prompt) vs interpolation step t (x); one thin line per pair, shaded on the viridis scale by that pair's transition width (colour bar); the thick black line is the median over the 64 pairs and the gray dashed line the straight line d = t expected with no plateau. Right: histogram of transition width (x) vs number of pairs (y); the black dotted vertical line is the strict plateau rule 0.25, the gray dashed line the straight-line value 0.80, the thick black line the median 0.34.](plots/comma_all_chars_curves.png)

**Which character sits at the other end matters a lot.** Sorting the 64 pairs by width splits them by
character type: lower-case letters give the sharpest transitions (median 0.313, n = 26), upper-case
letters next (0.355, n = 26), space and newline in between (0.336, n = 2), and punctuation or the
digit `3` are clearly the flattest (0.564, n = 10).

![Transition width (y) for each comma→character pair (x, one bar per character, sorted from sharpest to flattest; ␣ = space, \n = newline), final checkpoint, interpolation block 0, final logits. Each character type has its own bar hatch as well as its own colour: lower-case letter (//), upper-case letter (\\), space/newline (xx), punctuation or digit (..). The black dotted horizontal line is the strict plateau rule 0.25; the gray dashed line the straight-line value 0.80.](plots/comma_width_by_char.png)

**What predicts the width.** The single best predictor we found is how likely the model thinks that
character is in this context: transition width falls as the model's probability for the character
after `"The house was "` rises (Spearman rank correlation **ρ = −0.74**, p = 2.7e-12, n = 64). How
far apart the two endpoint outputs are explains less (ρ = −0.48, p = 5.6e-5) — and in the direction
that rules out a trivial artifact: *wider*-separated endpoints give *sharper*, not flatter,
transitions. Note the comma itself is an unlikely continuation here (model probability 1.0e-7), so
the sharp cases are not "both endpoints are common inputs" — what varies is the other character. The
context control below shows this correlation is the strongest of the nine contexts we measured, so
−0.74 should be read as the top of a range, not a typical value.

![Left: transition width (y) vs the model's probability of the other character after "The house was " (x, log scale), one point per pair with a distinct marker per character type — circle = lower-case letter, square = upper-case letter, triangle = space/newline, diamond = punctuation or digit; Spearman ρ = −0.74. Right: transition width (y) vs the L2 distance between the two endpoints' final-logit vectors (x), same markers; Spearman ρ = −0.48. The black dotted horizontal line is the strict plateau rule 0.25; the gray dashed line the straight-line value 0.80.](plots/comma_width_vs_endpoints.png)

**Both structural controls replicate at n = 64.** Moving the interpolation point deeper (fewer layers
left to act) flattens the curve back to the straight line — median width 0.34 (block 0), 0.51, 0.65,
0.72, 0.77, 0.79, then ≈0.80 for blocks 6–11. Across training the transition narrows early and then
stops changing: median width 0.799 (init) → 0.751 (step 56) → 0.524 (831) → 0.328 (7,819) → 0.367
(17,500) → 0.340 (30,000). Both match the `b↔i`/`b↔l` result with 32× more pairs.

![Left: median transition width over the 64 pairs (y, solid line with circle markers) vs interpolation block (x, 0–11: the residual stream after this block is replaced); the hatched band is the inter-quartile range; the gray dashed line is the straight line 0.80 and the black dotted line the strict plateau rule 0.25. Right: median transition width (y, dashed line with square markers) vs training step (x, log scale, step 0 drawn at 1) at interpolation block 0; hatched band = inter-quartile range; same two reference lines.](plots/comma_depth_and_training.png)

### Discussion of the comma sweep

1. **A plateau-shaped response is the rule, not the exception.** Fixing one endpoint and sweeping all
   64 alternatives, no pair behaves like a straight line. The downstream stack always resists the
   interpolation for a while, then switches. That the effect survives an exhaustive sweep — rather
   than only for hand-picked pairs — is the main thing this experiment adds.
2. **But sharpness is a continuum, and the strict bar is close to the edge of it.** Only 1/64 pairs
   passes width ≤ 0.25, while 33/64 pass at ≤ 0.35. Any headline count of "how many plateaus" in this
   model is therefore mostly a statement about the threshold. We report the whole distribution for
   that reason.
3. **The transition is sharpest for characters the model actually expects.** The rank correlation
   with the model's own next-character probability (ρ = −0.74) is strong for n = 64. A plain reading:
   when the second endpoint is a continuation the model has a confident, well-practised output for,
   the downstream layers snap between two familiar outputs; when it is a character the model
   essentially never predicts there (`3`, `&`, `!`, `:`, `z`), the output drifts more gradually. This
   is a correlation across 64 characters in one context, not a causal test.
4. **It is not an artifact of endpoint geometry.** If wide transitions were just "endpoints too close
   to separate", width would grow with smaller endpoint separation *and* that would be the dominant
   effect. Separation correlates only −0.48, and the sign says well-separated endpoints transition
   *faster*.
5. **It does not change the grokking verdict.** These 64 pairs are measured on the same non-grokking
   character run, at the same frozen checkpoints. They confirm the plateau appears with initial fit
   (already at its floor by step ~7,800) and stays flat, so the bounded relationship verdict stays
   PLAN case 5.
6. **Caveats.** One model, one interpolation position (final token), and single characters as
   endpoints. Widths near 0.33 are "sharper than linear" but not step-like. The two context-related
   caveats — that all of this came from one shared context, and that the comma endpoint is itself an
   implausible input — are tested directly in the next section.

## Does the plateau depend on the context? — 8 further contexts, 576 pairs

Every plateau number above comes from the single shared context `"The house was "`, and the fixed
comma endpoint is an implausible continuation there (model probability 1.0e-7). Either could be
driving the result. To test both at once we repeat the entire comma sweep in **8 additional
contexts** of 64 characters each, drawn from held-out validation text and chosen to span the model's
own probability of a comma in that slot — from 5e-20 ("a comma is impossible here") to 0.997 ("a
comma is almost certainly next"). Everything else is unchanged (step 30,000, interpolation block 0,
50 steps, `slerp_rescale`, final-logit `d(t)`), giving 9 contexts × 64 pairs = **576 pairs**. Script
`experiments/context_sweep.py`; raw curves `results/context_sweep_raw.npz`, summary
`results/context_sweep_summary.json`.

**Answer: the shape claim replicates everywhere; the "expected characters switch sharper" claim
replicates in direction but not in size.**

| quantity (576 pairs, 9 contexts) | value |
|---|---|
| curves near the straight line (width ≥ 0.70) | **0 / 576** |
| per-context median width | 0.313 – 0.436 (reference context 0.340) |
| pooled median width | 0.381 |
| pairs meeting the strict rule (width ≤ 0.25) | 11 / 576 |
| pairs with width ≤ 0.35 | 198 / 576 |
| within-context ρ (width vs the model's probability of the target character) | all 9 negative; median **−0.41**, range −0.05 … −0.74; p < 0.05 in 7/9 |
| sign test on those 9 correlations | p = 0.004 |
| median width vs the model's probability of a **comma** in that context | ρ = −0.32, p = 0.41 (n = 9) — no effect |

To show the shape claim does not depend on the chosen context, and that the fixed endpoint's own
plausibility does not set the sharpness, we plot the width distribution for each context and each
context's median against its comma probability.

![Left: transition width w_10→90 (y) for the 64 comma→character pairs of each context (x, one box per context ordered by the model's probability of a comma there, printed under each box; "ref" = "The house was ", the context behind every earlier number, drawn with a cross hatch, the held-out contexts with a diagonal hatch). Boxes show the inter-quartile range with the median as a horizontal bar; whiskers 1.5×IQR; dots are outliers. Gray dashed = straight line 0.80, black dotted = strict plateau rule 0.25. Right: median transition width (y) vs the model's probability of a comma in that context (x, log scale); circles = held-out contexts, diamond = the reference context; same two reference lines.](plots/context_widths.png)

Not one of the 576 curves is near-linear, and the per-context medians sit in a narrow band well below
the straight-line value — the plateau shape is a property of this model, not of the one context we
started with. The context in which a comma is essentially certain (probability 0.997) gives median
width 0.330, statistically indistinguishable from the reference context's 0.340, and across the nine
contexts the comma's plausibility does not predict sharpness at all (ρ = −0.32, p = 0.41). That
**retires the caveat** that the sweep was made artificially hard by an implausible fixed endpoint.

To check whether the predictor we reported for one context is a general rule, we plot each context's
rank correlation and pool all 576 pairs.

![Left: Spearman ρ between transition width and the model's probability of the target character (x) for each context (y, ordered by the context's comma probability; the reference context "The house was " has a cross hatch, held-out contexts a diagonal hatch); the dash-dot vertical line is the median over contexts (−0.41). Right: transition width (y) vs the model's probability of the target character in its context (x, log scale) for all 576 pairs; circles = the 8 held-out contexts, diamonds = the reference context; gray dashed = straight line 0.80, black dotted = strict plateau rule 0.25.](plots/context_rho.png)

The correlation is negative in **all nine** contexts (sign test p = 0.004), so "the switch is sharper
for characters the model expects" is a real, repeatable tendency. But its strength swings from −0.05
to −0.74, and the context we happened to report first is the strongest one; the median context gives
−0.41 and the pooled correlation over all 576 pairs is −0.23. The honest summary is a consistent but
modest effect, not the tight relationship a single context suggested.

## Standalone exploratory evidence — 40 natural minimal pairs (character model, final checkpoint)

> **Clearly labelled as exploratory and out of the headline** (PLAN out-of-scope forbids a new
> 40-pair minimal-pair dataset in the *primary* analysis). It is retained because its **layerwise**
> and **depth** controls corroborate the Matthew-faithful result above with much larger `n`, using the
> same slerp/patch machinery on 40 natural Shakespeare minimal pairs (`prefix + char_A` vs
> `prefix + char_B`) frozen before any curve was inspected (`results/prompt_pairs.json`).

- **14/40** pairs meet the strict frozen plateau rule (width ≤ 0.25, rests near each endpoint,
  near-monotone); 24/40 have width ≤ 0.35; only 2/40 are near-diagonal; **0/40** non-monotone. Median
  width **0.309** vs diagonal 0.80.
- **Boundary sharpens with depth:** median width falls monotonically 0.777 (block 1) → 0.445 (block 11)
  → 0.309 (logits); strict rule passed only at the logits.
- **Later interpolation weakens the plateau:** median width 0.309, 0.564, 0.647, 0.733, 0.757, 0.802
  for interpolation blocks 0, 2, 4, 6, 8, 10 — reaching the diagonal when one block remains.

![Exploratory 40-pair result: raw d(t) (y) vs interpolation step t (x) in final-logit space, one panel per frozen pair (title = pair ID, endpoint chars, width w). Gray dashed = diagonal. Most curves hug d≈0, cross rapidly near t≈0.5, then hug d≈1.](plots/pair_curves_logits.png)
![Exploratory 40-pair layerwise emergence for four fixed pairs (IDs 0–3): d(t) (y) vs interpolation step t (x); thin lines are recording blocks on the cividis scale (dark early → light late); the thick black line is the final logits and the gray dashed line the diagonal. Curves start near-diagonal and sharpen into plateaus by the logits.](plots/layerwise_emergence.png)
![Exploratory 40-pair depth comparison. Left: median final-logit d(t) (y) vs interpolation step t (x) per interpolation block, cividis scale (dark = block 0 → light = block 10) as labelled in the legend; the block-0 curve is sigmoid, later blocks approach the gray dashed diagonal. Right: median width w_10→90 (y, inter-quartile-range bars, solid line with circle markers) vs interpolation block (x); the black dotted line is the plateau bar 0.25, the gray dashed line the diagonal 0.8.](plots/interpolation_layer_comparison.png)

## Implementation checks (all passed)

- `t=0` / `t=1` patched forwards reproduce the direct unpatched endpoint forwards (max logit error
  < 1e-3); `d(0) < 1e-4`, `d(1) > 1 − 1e-4` for every pair/checkpoint.
- Prefix positions differ only at the final character; all earlier-position activations of A and B
  match at every block (max abs diff < 1e-4; `prefix_err` logged per checkpoint).
- Batched interpolation matches a single-example reference to < 1e-5.
- Synthetic step path detected (w = 0.089); synthetic linear path rejected (w = 0.800).
- Slerp endpoints exact; norms interpolate linearly; documented near-collinear fallback.
- Both deliverables are checked to render on GitHub by `experiments/check_render.py` (KaTeX-compiles
  every equation, rejects macros GitHub's math renderer blocks such as `\operatorname`, and confirms
  every figure is an embedded image rather than a bare path).

## Headline

No model we trained reproduces *Deep Networks Always Grok* Figure 9 (all three FAIL the mandatory
second-descent gate; the BPE bridge to Matthew's exact tokens fails too), so the **primary
Grokking↔plateau relationship is not testable** (PLAN case 5). But the Matthew-faithful char-token
controls (`b↔i`, `b↔l`) let us time the plateau: it is **absent at initialization**, **emerges during
the first LC descent** (width 0.80 → 0.33 by step ~831), and is **fully formed before** adversarial
robustness saturates — so, in this non-grokking model, the plateau is an early property of the trained
downstream stack with **no visible temporal coupling** to the grokking signature (which never occurs).
Sweeping the comma against all 64 other characters — and repeating that sweep in 8 further held-out
contexts, 576 pairs in all — shows the same shape everywhere (**0/576** curves near the straight line,
per-context median widths 0.313–0.436) while making clear that sharpness is graded: only 11/576 clear
the strict ≤ 0.25 bar. The sharpest switches do go to the characters the model expects next, in all
nine contexts, but the strength of that link varies (median ρ = −0.41, range −0.05 … −0.74).
