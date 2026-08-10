# RESULTS — Do Grokking and Matthew-style activation plateaus emerge together?

> CURRENT-BEST ONLY. History lives in CHANGELOG.md. Read before rewriting.

## Question & verdict

**Question.** In a 12-layer character-level Shakespeare GPT — the architecture of *Deep Networks
Always Grok* (Figure 9 of that paper) — do **Matthew-style activation plateaus** (Shinkle & Hex,
*Activation Plateaus: Where and How They Emerge*) appear or sharpen during the same training window as
the grokking signature (a **second local-complexity descent** plus **delayed `ε=0.03`-PGD adversarial
robustness**)? A "plateau" is this: interpolate between two natural inputs' final-position activations
and the downstream output stays locked near endpoint A, snaps across a narrow boundary, then locks near
endpoint B — a plateau–boundary–plateau `d(t)` curve, not a straight diagonal.

**Bounded verdict = PLAN case 5, "primary relationship not testable," for Matthew's exact tokens;
PLAN case 1, "temporally associated," for the character analogues.**

1. **Grokking gate: both character runs PASS, the BPE run FAILs.** Test local complexity in the fresh
   30k character run falls 1940 → 491 (step 15), turns back **up** to 989 (step 36; a 498-unit rise
   against a ±3 99% CI), then descends for the rest of training to 8.1 — Fig. 9's first-descent →
   rise → **second descent** shape. On a grid densified to 24 checkpoints the rise is resolved by
   three points (steps 23, 36, 56), not one. The onset (step 36) precedes the clean-accuracy peak
   (step 4,994) and `ε=0.03`-PGD robustness rises from 0.001 there to 0.53, still climbing after clean
   accuracy saturates. The 3,500-step pilot shows the same ordering (484 @ 19 → 1043 @ 33 → 68). The **BPE**
   run — the only tokenizer under which Matthew's exact `big/in`, `big/large` completions are single
   tokens — shows only a 30-unit upturn (1.4% of range, inside the preregistered 5% tolerance) and
   **FAILs**. So the **primary** Matthew-exact relationship is **not testable** here, while the
   character analogues now sit on a run that does reproduce Figure 9.

2. **Plateaus are present and we can time their emergence.** Using Matthew's exact config path
   (`shared_context = "The house was"`, 50-step slerp grid, full interpolation-layer sweep) with the
   two preregistered single-token **character controls** `b↔i` and `b↔l`, plateaus **emerge during
   training**: at initialization the `d(t)` curve is the diagonal (width ≈ 0.80, no plateau); by
   **step ~831** it is already a sharp sigmoid (width ≈ 0.33–0.35) and stays there to step 30k. That
   sharpening happens **inside** the fresh char run's second-descent window (steps 36 → 30,000) and
   straddles its sustained robustness onset (step 531).

3. **Bounded relationship claim.** For the character analogues this is **PLAN case 1, temporally
   associated**: the plateau sharpens in the same checkpoint interval as the second LC descent and the
   emergence of delayed robustness. Three limits keep it bounded — one training run cannot show
   causation; the second descent opens at step 36, early enough that its window also contains ordinary
   initial fitting, so "sharpens with grokking" and "sharpens with initial fit" are not separated here;
   and the plateau is complete by step ~831 while robustness keeps rising to ~7,819, so the plateau is
   not waiting on the grokking transition. Matthew's *exact* tokens remain untested (case 5).

4. **The plateau shape is general, but graded — and it survives a change of context.** Holding one
   endpoint fixed at the **comma** and interpolating to **all 64 other characters**, every pair bends
   away from the straight line — none is linear — yet only 1/64 meets the strict frozen plateau rule
   (transition width ≤ 0.25; median width 0.340 against 0.80 for a straight line). Repeating the whole
   sweep in **8 further contexts** taken from held-out text (**576 pairs**) replicates this exactly:
   **0/576** curves are near-linear and per-context median widths stay in a narrow band (0.313–0.436).
   The claim that sharpness tracks how likely the model thinks the other character is holds in
   **direction everywhere** (all 9 contexts give a negative rank correlation, sign test p = 0.004) but
   its **size is context-dependent** (median ρ = −0.41, range −0.05 to −0.74).

5. **Every character owns a basin, and the basin is the model's next-character decision.** The
   exhaustive **all-pairs sweep** (all **2,080** character pairs) settles what the plateaus *are*:
   every one of the 65 characters is left only after the path has spent at least 10% of its length
   locked to it (`flat_frac` ≥ 0.86 for all 65), **78%** of the variance in how sharply a basin is
   left is explained by per-character terms alone, and **91%** of all next-character prediction changes
   along a path fall inside the transition window. The basins are **learned** — at initialization all
   2,080 paths are straight lines (median width 0.803 → 0.355 trained) — and are built by the
   **shallow blocks** (median width 0.34 patching at block 0 vs 0.81 at block 8). That site is
   contingent rather than necessary: seven retraining runs, each matching or beating the reference's
   validation accuracy, put the sharpening wherever blocks remain trainable — freeze 1–4 and it moves to
   blocks 5–8 (width 0.471), freeze 1–7 and it moves to 8–11 (0.558), freeze 5–11 and it moves back to
   1–4 (0.626), freeze 0–3 and 9–11 and it sits in the middle window 4–8 (**0.331**), freeze 0–1 and
   7–11 and it moves to 2–6 (**0.355**), freeze everything
   but blocks 5–7 and it packs 96% of the sharpening into that three-block window (**0.427**). The sharp
   transition is a **relocatable** computation this architecture and objective
   install wherever there is room; what freezing costs is *how* sharp it gets, and that cost is set by
   **where** the surviving trainable blocks sit, not how many there are — the same five trainable blocks
   give 0.365 at blocks 4–8, 0.365 again at 2–6 and 0.363 at 1–5, against 0.56–0.59 beside the readout
   and 0.63 at the bottom; all three mid-stack windows are sharper than the untouched 12-block
   reference; and even a three-block window at 5–7 with 74.6%
   of the parameters frozen ties that reference (0.446 vs 0.443, p = 0.17). The clearest single fact is
   that *training fewer blocks can help*: blocks 1–5 alone (0.363) beat blocks 0–7 (0.500), a strict
   superset of them, by 0.118 with 4.7% of pairs going the other way (p = 2e-25). An eighth run pushes
   trainable depth to its limit —
   freeze ten of twelve blocks so only block 11 remains usable — and there the plateau finally breaks
   down (0.726, just 17% of the reference sharpening, boundary no longer locked to the prediction flip).

## Models actually tested

Three models carry every number below, and which one carries which matters for how far the claims
reach: the fresh **character** GPT is the only model that both passes the grokking gate and supports
the plateau assays, so it does all the mechanistic work; the **BPE** GPT is the only one that can hold
Matthew's exact words as single tokens, and it fails the gate, which is precisely why the primary
Grokking↔plateau question stays untestable; the **pilot** run is kept only as a sanity check that
the recipe trains.

| Model | Tokenizer | Params | Trained to | Role |
|---|---|---|---|---|
| Fresh character GPT | char (vocab 65) | 8.38M | 30,000 steps, val acc 0.554 (peak 0.568) | Figure-9 control + **Matthew char-control assay** + all-pairs sweep |
| Fresh BPE GPT | GPT-2 BPE (vocab 50257) | — | 10,000 steps (killed; overfit) | primary Matthew bridge (`big/in`, `big/large`) — **gate FAILs** |
| Pilot character GPT | char (vocab 65) | 8.38M | 3,500 steps, val acc 0.560 | pilot only |

All are 12-layer/12-head GeLU GPTs (`d_model=240`, context 128). Provenance, seeds, corpus SHA-256 in
`results/train_meta*.json`; confirmed-vs-reconstructed fields in `MODEL_SPEC.md`. The paper's exact
GPT code/checkpoint is **not public** (repo audited 2026-07-15), so these are faithful reconstructions.

![pilot training curves](plots/training_curves.png)

**Figure 1.** Pilot character-model training. Left: cross-entropy loss in nats (y) vs training step
(x) for the train split (solid) and validation split (dashed, squares); validation bottoms at ≈1.49.
Right: validation next-character accuracy (y) vs training step (x), rising to 0.56. The model is
clearly trained, not random.

## Figure-9 grokking gate — both character runs PASS, the BPE run FAILs

PLAN makes a **validity gate** mandatory before any *joint* claim: the model must qualitatively
reproduce *Deep Networks Always Grok* Fig. 9 — a **second LC descent** beginning before the
test-accuracy peak, plus **delayed adversarial robustness** (`ε=0.03` `ℓ∞`-PGD accuracy rising while
that second descent continues). LC is the paper's sign-crossing count summed over the 12 GeLU layers
on 1,024 train/test/random points (`r=0.005`, `P=25`, 99% CIs); pipeline source-locked to the official
repo (`experiments/fig9.py`, 0.0 logit-reimplementation error). Preregistered PASS/FAIL rule in
`experiments/fig9_verdict.py`.

| Figure-9 quantity | Pilot char (3.5k) | **Fresh char (30k)** | **Fresh BPE (10k)** |
|---|---|---|---|
| checkpoints evaluated | 13 | 24 | 10 |
| clean acc (peak / final) | 0.564 / 0.564 | 0.568@4994 / 0.554 | 0.299@831 / 0.274 |
| `ε=0.03` PGD adv acc (final) | 0.327 | **0.528** | **0.187** |
| test LC (first → 1st local min → local max → final) | 1940 → 484 @ 19 → 1043 @ 33 → 68 | 1940 → 491 @ 15 → **989 @ 36** → **8.1** | 2182 → — → — → **95** |
| points resolving the LC local maximum | 1 (step 33) | **3** (steps 23, 36, 56) | — |
| LC rise vs 5%-of-range tolerance | 558 ≫ 94 | **498 ≫ 96.8** | 30 < 104 → rejected |
| second LC descent? | **Yes**, onset step 33 | **Yes**, onset step 36 | No |
| onset before clean-accuracy peak? | Yes (33 < 3500) | Yes (36 < 4994) | n/a |
| adv acc at onset → max at/after onset | 0.000 → 0.327 | 0.001 → 0.530 | n/a |
| sustained robustness onset (adv ≥ 0.05 thereafter) | step 1,091 | step 531 | step 217 |
| **preregistered verdict** | **PASS** | **PASS** | **FAIL** |

The fresh char run is the cleanest case: LC 491.2 ± 2.7 at step 15 → 989.1 ± 4.5 at step 36 (5.1× the
96.8-unit tolerance, ~110× the CI) → 8.1 at step 30,000 without rebound, well below the first minimum;
robustness crosses 0.05 for good at step 531 and keeps rising after clean accuracy saturates (0.55 by
step 2,038). Its grid was **densified from 14 to 24 checkpoints** this iteration (10 already-saved but
never-measured checkpoints at steps 1, 2, 6, 9, 23, 36, 88, 138, 339, 531 run through the identical
pipeline) specifically to test whether the turnaround was an artifact of one log-spaced point. It was
not: the rise is now traced by three points above the minimum (988 @ 23, 989 @ 36, 769 @ 56) and both
its height (278 → 498 units) and the verdict's margin grew. The BPE run's only upturn (459.5 @ 56 →
489.2 @ 217) is 1.4% of its LC range, inside the tolerance fixed in advance, so it is scored as no
second descent → **FAIL**. Two honest caveats remain on the passing runs: the turnaround happens very
early (steps 15–56) rather than long after saturation as in the paper, and in the **pilot** run the
local maximum is still resolved by a single log-spaced checkpoint. Both fresh
runs also **overfit** in ordinary validation loss (char ≈step 3,750, BPE ≈step 750), so classic delayed
val-loss recovery is absent even where the LC/robustness ordering passes. Figures 2–4 show the three
gate curves with the detected landmarks annotated.

![pilot char Figure-9 gate](plots/grokking_pilot_char.png)

**Figure 2.** Pilot char (3.5k) Figure-9 gate. Left y-axis: local complexity (sign-crossings summed
over the 12 GeLU layers) for the train (solid), test (dashed) and random (dash-dot) base-point sets,
with 99% CI bands. Right y-axis: next-token accuracy — black with circles = clean test accuracy,
black dotted with squares = `ε=0.03` PGD adversarial accuracy. x-axis: training step (log scale, step
0 drawn at 1). Grey vertical rules mark the detector's landmarks, labelled along the top: the first LC
local minimum (dash-dot, ▽), the local maximum opening the second descent (dashed, △), the sustained
robustness onset (dotted) and the clean-accuracy peak. LC falls to 484 at step 19, rises to 1043 at
step 33, then descends to 68 while adversarial accuracy climbs to 0.33 → **PASS**.

![fresh char Figure-9 gate](plots/grokking_fresh_char.png)

**Figure 3.** Fresh char (30k) Figure-9 gate on the densified 24-checkpoint grid, same axes, line
styles and landmark rules as Figure 2. The LC turnaround at steps 15 → 36 (491 → 989, CIs ±3 and ±4)
is the V-then-Λ notch between the ▽ and △ markers, now traced by three measured points above the
minimum (steps 23, 36, 56) rather than one; LC then descends to 8.1 while adversarial accuracy reaches
0.53, crossing 0.05 for good at step 531 and still rising after clean accuracy saturates → **PASS**.
This is the clearest of the three gate curves.

![fresh BPE Figure-9 gate](plots/grokking_fresh_bpe.png)

**Figure 4.** Fresh BPE (10k) Figure-9 gate, same axes and line styles as Figure 2 (only the
robustness-onset and accuracy-peak rules appear — no significant LC minimum/maximum was found). The
step-56 → step-217 upturn is 30 units, inside the 104-unit tolerance, so LC counts as descending to 95
while adversarial accuracy climbs to 0.19 → **FAIL**. This is the run that would
have carried Matthew's exact `big/in`, `big/large` tokens, so its failure is what makes the primary
relationship untestable.

**Joint timeline (S7).** On one training-step axis the two questions separate. The BPE bridge to
Matthew's exact tokens does not reproduce Figure 9, so the **primary** Matthew-exact relationship stays
untestable (PLAN case 5). On the fresh character run, which does pass, the second descent spans steps
36 → 30,000 and sustained robustness begins at step 531 — and the `b↔i`/`b↔l` plateau collapses from
width ≈ 0.80 (steps 0 and 56) to ≈ 0.33 by step 831, i.e. **inside** that window and straddling the
robustness onset. That is **PLAN case 1 (temporally associated)** for the character analogues, with two
limits: one run cannot show causation, and the window opens so early (step 36) that it also contains
ordinary initial fitting. Figure 5 puts all three runs side by side.

![joint checkpoint timeline for the three runs](plots/joint_timeline.png)

**Figure 5.** Joint checkpoint timeline. Left: test local complexity (y) vs training step (x, log) for
the pilot char run (dotted, triangles), fresh char run (solid, circles) and fresh BPE run (dashed,
squares); the legend gives each run's Figure-9 verdict. Middle: `ε=0.03` PGD adversarial accuracy (y)
vs step (x, log), same three line styles; the horizontal dashed line is the 0.05 robustness threshold
used by the verdict rule. Right: text summary of the three gate verdicts (PASS / PASS / FAIL) and the
bounded relationship verdict — the BPE FAIL is what keeps the Matthew-exact relationship untestable.

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
before** `ε=0.03` robustness saturates. The plateau then holds flat to step 30k. Figure 6 shows the
raw curves that this table summarises.

![Matthew char-control curves by checkpoint](plots/matthew_char_ctrl_by_checkpoint.png)

**Figure 6.** Matthew-faithful char-control `d(t)` (y) vs interpolation position `t` (x), at
interpolation block 0 in final-logit space, one panel per frozen checkpoint (steps 0→30000). The `b↔i`
pair is the solid line with circle markers, `b↔l` the dashed line with square markers; the gray dashed
straight line is the diagonal `d = t` expected with no plateau. Curves lie on the diagonal at init and
step 56 and are sharp plateau–boundary–plateau sigmoids by step 831, stable thereafter.

To read the plateau's emergence against the grokking metrics directly, Figure 7 puts both on one
training-step axis: the width reaches its floor between steps 56 and 831 — inside the second-descent
window (36 → 30,000) and straddling the sustained robustness onset (531), but long before robustness saturates
(~7,819). This is the association behind the case-1 verdict, and also its limit.

![grokking metrics and plateau width on one timeline](plots/joint_timeline_char_ctrl.png)

**Figure 7.** Both phenomena on one timeline (fresh char run). Top: left y = local complexity for the
train (solid), test (dashed) and random (dash-dot) base-point sets with 99% CI bands; right y =
next-token accuracy, black with circles = clean, black dotted with squares = `ε=0.03` PGD adversarial;
x = training step (log). Bottom: transition width `w_10→90` (y) for `b↔i` (solid, circles) and `b↔l`
(dashed, squares) vs step (log); the gray dashed line is the diagonal 0.80, the black dotted line the
strict plateau bar 0.25. Width collapses to its floor by step ~831 — during the first LC descent,
before robustness rises.

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
median — they were typical pairs, not lucky ones. Figure 8 shows all 64 raw curves, which are the
primary evidence, next to their width distribution.

![all 64 comma-to-character curves and their width histogram](plots/comma_all_chars_curves.png)

**Figure 8.** All 64 comma→character curves at step 30,000. Left: relative distance `d(t)` (y; 0 =
output still looks like the comma prompt, 1 = looks like the other character's prompt) vs
interpolation position `t` (x); one thin line per pair, shaded on the viridis scale by that pair's
transition width (colour bar); the thick black line is the median over the 64 pairs and the gray
dashed line the straight line `d = t`. Right: histogram of transition width (x) against number of
pairs (y); the black dotted vertical line is the strict rule 0.25, the gray dashed line the
straight-line value 0.80, the thick black line the median 0.34.

**Which character sits at the other end matters a lot.** Sorting the 64 pairs by width splits them by
character type: lower-case letters give the sharpest transitions (median 0.313, n = 26), upper-case
letters next (0.355, n = 26), space and newline in between (0.336, n = 2), and punctuation or the
digit `3` are clearly the flattest (0.564, n = 10). Figure 9 shows that per-character ordering.

![width per comma-to-character pair, sorted](plots/comma_width_by_char.png)

**Figure 9.** Transition width (y) for each comma→character pair (x, one bar per character, sorted
sharpest to flattest; ␣ = space, `\n` = newline) at the final checkpoint, interpolation block 0, final
logits. Each character type carries its own bar hatch as well as its own colour: lower-case letter
(`//`), upper-case letter (`\\`), space/newline (`xx`), punctuation or digit (`..`). Black dotted
horizontal line = strict rule 0.25; gray dashed line = straight-line value 0.80.

**What predicts the width.** The single best predictor we found is how likely the model thinks that
character is in this context: transition width falls as the model's probability for the character
after `"The house was "` rises (Spearman rank correlation **ρ = −0.74**, p = 2.7e-12, n = 64). How
far apart the two endpoint outputs are explains less (ρ = −0.48, p = 5.6e-5) — and in the direction
that rules out a trivial artifact: *wider*-separated endpoints give *sharper*, not flatter,
transitions. Note the comma itself is an unlikely continuation here (model probability 1.0e-7), so
the sharp cases are not "both endpoints are common inputs" — what varies is the other character. The
context control below shows this correlation is the strongest of the nine contexts we measured, so
−0.74 should be read as the top of a range, not a typical value. Figure 10 shows both candidate
predictors side by side.

![width vs next-character probability and vs endpoint separation](plots/comma_width_vs_endpoints.png)

**Figure 10.** Left: transition width (y) vs the model's probability of the other character after
`"The house was "` (x, log scale), one point per pair, with a distinct marker per character type —
circle = lower-case letter, square = upper-case letter, triangle = space/newline, diamond =
punctuation or digit; Spearman ρ = −0.74. Right: transition width (y) vs the L2 distance between the
two endpoints' final-logit vectors (x), same markers; Spearman ρ = −0.48. In both panels the black
dotted horizontal line is the strict rule 0.25 and the gray dashed line the straight-line value 0.80.

**Both structural controls replicate at n = 64.** Moving the interpolation point deeper (fewer layers
left to act) flattens the curve back to the straight line — median width 0.34 (block 0), 0.51, 0.65,
0.72, 0.77, 0.79, then ≈0.80 for blocks 6–11. Across training the transition narrows early and then
stops changing: median width 0.799 (init) → 0.751 (step 56) → 0.524 (831) → 0.328 (7,819) → 0.367
(17,500) → 0.340 (30,000). Both match the `b↔i`/`b↔l` result with 32× more pairs; Figure 11 shows
both controls.

![depth and across-training controls for the comma sweep](plots/comma_depth_and_training.png)

**Figure 11.** Left: median transition width over the 64 pairs (y, solid line with circle markers) vs
interpolation block (x, 0–11: the residual stream after this block is the one replaced); the hatched
band is the inter-quartile range; gray dashed = straight line 0.80, black dotted = strict rule 0.25.
Right: median transition width (y, dashed line with square markers) vs training step (x, log scale,
step 0 drawn at 1) at interpolation block 0; hatched band = inter-quartile range; same two reference
lines.

### Discussion of the comma sweep

1. **A plateau-shaped response is the rule, not the exception.** Fixing one endpoint and sweeping all
   64 alternatives, no pair behaves like a straight line. The downstream stack always resists the
   interpolation for a while, then switches.
2. **But sharpness is a continuum, and the strict bar is close to the edge of it.** Only 1/64 pairs
   passes width ≤ 0.25, while 33/64 pass at ≤ 0.35. Any headline count of "how many plateaus" in this
   model is therefore mostly a statement about the threshold. We report the whole distribution for
   that reason.
3. **The transition is sharpest for characters the model actually expects.** A plain reading: when the
   second endpoint is a continuation the model has a confident, well-practised output for, the
   downstream layers snap between two familiar outputs; when it is a character the model essentially
   never predicts there (`3`, `&`, `!`, `:`, `z`), the output drifts more gradually.
4. **It is not an artifact of endpoint geometry.** If wide transitions were just "endpoints too close
   to separate", width would grow with smaller endpoint separation *and* that would be the dominant
   effect. Separation correlates only −0.48, and the sign says well-separated endpoints transition
   *faster*.
5. **It does not change the grokking verdict.** These 64 pairs are measured on the character
   run at one checkpoint, so they add nothing to the checkpoint-aligned verdict.

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
plausibility does not set the sharpness, Figure 12 plots the width distribution for each context and
each context's median against its comma probability.

![width distribution per context](plots/context_widths.png)

**Figure 12.** Left: transition width `w_10→90` (y) for the 64 comma→character pairs of each context
(x, one box per context ordered by the model's probability of a comma there, printed under each box;
"ref" = `"The house was "`, the context behind every earlier number, drawn with a cross hatch; the
held-out contexts use a diagonal hatch). Boxes show the inter-quartile range with the median as a
horizontal bar; whiskers 1.5×IQR; dots are outliers. Gray dashed = straight line 0.80, black dotted =
strict rule 0.25. Right: median transition width (y) vs the model's probability of a comma in that
context (x, log scale); circles = held-out contexts, diamond = the reference context; same two
reference lines.

Not one of the 576 curves is near-linear, and the per-context medians sit in a narrow band well below
the straight-line value — the plateau shape is a property of this model, not of the one context we
started with. The context in which a comma is essentially certain (probability 0.997) gives median
width 0.330, statistically indistinguishable from the reference context's 0.340, and across the nine
contexts the comma's plausibility does not predict sharpness at all (ρ = −0.32, p = 0.41). That
**retires the caveat** that the sweep was made artificially hard by an implausible fixed endpoint.

To check whether the predictor we reported for one context is a general rule, Figure 13 plots each
context's rank correlation and pools all 576 pairs.

![per-context rank correlations](plots/context_rho.png)

**Figure 13.** Left: Spearman ρ between transition width and the model's probability of the target
character (x) for each context (y, ordered by the context's comma probability; the reference context
has a cross hatch, held-out contexts a diagonal hatch); the dash-dot vertical line is the median over
contexts (−0.41). Right: transition width (y) vs the model's probability of the target character in
its context (x, log scale) for all 576 pairs; circles = the 8 held-out contexts, diamonds = the
reference context; gray dashed = straight line 0.80, black dotted = strict rule 0.25.

The correlation is negative in **all nine** contexts (sign test p = 0.004), so "the switch is sharper
for characters the model expects" is a real, repeatable tendency. But its strength swings from −0.05
to −0.74, and the context we happened to report first is the strongest one; the median context gives
−0.41 and the pooled correlation over all 576 pairs is −0.23. The honest summary is a consistent but
modest effect, not the tight relationship a single context suggested.

## All pairs of characters — is every character in its own plateau? (2,080 pairs)

The sweeps above always hold one endpoint fixed. An operator asked the general version: interpolate
from **every** character to **every** other, and say whether each character sits in a plateau of its
own and what the plateaus correspond to. That is all **C(65,2) = 2,080** unordered pairs, run through
the same frozen code path (`experiments/allpairs_sweep.py`, analysis `experiments/analyze_allpairs.py`):
context `"The house was "`, 50 evenly spaced interpolation values, `slerp_rescale`, patch the final
position only, `d(t)` read in final-logit space, at interpolation block 0 of the step-30,000 character
checkpoint. Raw curves and the per-`t` prediction trace are in `results/allpairs_raw.npz`; every
statistic below is in `results/allpairs_summary.json`.

**Companion analyses of this same sweep live in `REPORT_followup.md`** (operator feedback #5): width
against each character's training frequency (Spearman ρ = −0.78 over 65 characters; median width 0.320
for the 1,378 pairs of well-trained characters vs 0.482 for the 702 pairs touching a character seen
fewer than 1,000 times), the shape asymmetry of individual curves, the ordering of width by the
partner's character class (Kendall W = 0.42 across 43 letters), and a table of the prompt context and
per-cell sample count behind every character-level figure.

**All diagnostics pass and the sweep is exactly symmetric.** No pair is dropped: the largest `d(0)`
over all 2,080 pairs is 3e-6, the smallest `d(1)` is 0.999998, the largest endpoint-reproduction error
is 1.7e-5, prefix activations of the two endpoints match exactly (error 0.0), and **every** curve is
exactly monotone (isotonic deviation 0.0 for all 2,080). Re-running 100 randomly chosen pairs with the
endpoints **swapped** changes the width by a median — and a maximum — of **0.000**. That is not luck:
swapping endpoints maps `d(t)` to `1 − d(1 − t)`, and our `t` grid is symmetric, so the width is
invariant by construction; the check confirms the implementation does what the algebra says, and it
licenses drawing the width matrix as a symmetric heatmap.

| quantity (2,080 pairs, step 30,000, block 0) | value |
|---|---|
| median transition width | **0.355** (inter-quartile range 0.298–0.444) |
| straight-line reference (no plateau) | 0.80 |
| pairs meeting the strict rule (width ≤ 0.25 + rests near both endpoints) | **182 / 2,080 (8.8%)** |
| pairs near the straight line (width ≥ 0.70) | 20 / 2,080 (1.0%) |
| pairs that are exactly monotone | 2,080 / 2,080 |
| per-character median width, range over the 65 characters | 0.264 (`o`) – 0.590 (`3`) |
| characters whose paths rest ≥ 10% of the way on them (`flat_frac`) | **65 / 65** (min 0.86, median 1.00) |
| variance in width explained by per-character terms alone | **78.2%** (adjusted 77.6%; chance level 3.0%) |

Figure 14 is the whole result in one image: the pairwise width matrix, with characters grouped by
class. The visible row/column stripes — not a checkerboard — are the first sign that width is carried
by the individual characters rather than by pair-specific chemistry.

![65x65 matrix of transition widths for all character pairs](plots/allpairs_width_matrix.png)

**Figure 14.** Transition width `w_10→90` for all 2,080 character pairs at interpolation block 0 of
the step-30,000 character GPT. x-axis: character B; y-axis: character A; both ordered by character
class (space/newline, punctuation & digits, upper case, lower case), with white lines separating the
classes and the diagonal masked (a character against itself is not a pair). Colour = width on the
viridis scale (dark = sharp switch, bright = close to the straight-line value 0.80); the matrix is
symmetric because swapping endpoints leaves the width unchanged. Bright rows/columns (`3`, `&`, `$`,
`X`, `Z`, `z`, `x`) are characters that are left gradually from *every* partner.

### Is each character in its own plateau?

We make this decidable with three per-character statistics over each character's 64 partners:
`med_w(c)` (median width — how sharply `c` is left), `flat_frac(c)` (the fraction of partners for
which the path rests within 0.1 of `c`'s output for at least 10% of its length — "`c` has a basin of
its own"), and `strict_frac(c)` (the fraction passing the frozen ≤ 0.25 plateau rule). Figure 15
answers the operator's question directly.

![per-character width distributions with flat_frac overlay](plots/allpairs_width_by_char.png)

**Figure 15.** Every character has a basin, but how sharply it is left varies by character. x-axis:
the 65 characters, sorted by median width (␣ = space, `\n` = newline). Left y-axis: the distribution
of `w_10→90` over that character's 64 partners as a box (box = inter-quartile range, bar = median,
whiskers 1.5×IQR, outliers hidden); each box's hatch gives the character class (`//` space/newline,
`\\` punctuation & digits, `xx` upper case, `..` lower case) as the legend below the axis states.
Right y-axis (diamonds): `flat_frac(c)`, the fraction of partners whose path rests on `c` for ≥ 10%
of the way — it is 1.00 for 59 of 65 characters and never below 0.86. Gray dashed = straight-line
value 0.80; black dotted = strict rule 0.25.

**Verdict: case (i) — every character has a basin of its own — with the sharpness graded, not
knife-edge.** `flat_frac` is 1.00 for 59 of the 65 characters and never falls below 0.86, so on
essentially every path the output stays locked to each endpoint for at least a tenth of the way before
switching. What differs between characters is *how sharply* the basin is left: median widths run from
0.264 (`o`) to 0.590 (`3`), and no character is a plateau by the strict ≤ 0.25 rule for the majority
of its partners (`strict_frac` ≥ 0.5 for 0 of 65; ≥ 0.25 for 6 — `o`, `s`, `a`, `I`, `\n`, `e`).

**It is a property of the character, not of the pair.** Fitting the additive model
`w_ij ≈ μ + a_i + a_j` by least squares over all 2,080 widths explains **78.2%** of the variance
(adjusted 77.6%); a permutation null with the same 65 free parameters explains only 3.0% (99th
percentile 4.1%). Only **21.8%** is pair-specific residual. That rules out PLAN case (iii)
("sharpness lives in the pair") and case (ii) ("only a subset of characters has a basin"): each
character carries its own transition sharpness into every pairing it appears in. Figure 16 shows the
raw curves behind this for six representative characters — the raw `d(t)` curves remain the primary
evidence, and they are visibly bundled per character.

![raw d(t) curves for six representative characters](plots/allpairs_curves_small_multiples.png)

**Figure 16.** Raw `d(t)` for six characters against all 64 of their partners. Each panel: relative
distance `d(t)` (y, 0 = output looks like the named character's prompt, 1 = looks like the partner's)
vs interpolation position `t` (x); one thin line per partner, all oriented so the named character is
at `t = 0`; the gray dashed line is the straight-line reference `d = t`. Panels are the sharpest
character (`o`), the flattest (`3`), and one typical member of each character class; titles give that
character's median width and `flat_frac`. Every bundle leaves the endpoint flat, turns over once, and
flattens again — and the bundles are tight, which is the per-character effect of Figure 15 seen in
raw form.

### What do the plateaus correspond to?

Two measurements distinguish the obvious candidate explanations. The first asks whether a plateau is
simply **the set of residual states that decode to the same next character**: along every path we also
record the model's `argmax` prediction at each `t`, and compare where the `d(t)` curve crosses its
midpoint (`t*`) with where the prediction first changes (`t_flip`). The second asks **where the
sharpness is generated**, by re-patching at deeper blocks. Figure 17 answers a preliminary question the
first test needs — does the boundary sit where the two characters become equally likely?

![midpoint crossing vs relative endpoint plausibility](plots/allpairs_boundary_vs_logp.png)

**Figure 17.** Where the switch happens versus which endpoint the model prefers. x-axis:
`log10 p(A | context) − log10 p(B | context)`, the model's log-probability preference between the two
endpoint characters (positive = it prefers A). y-axis: the midpoint crossing `t*`, the interpolation
position at which the isotonic `d(t)` reaches 0.5. One marker per pair, shaped and coloured by the
class of endpoint A (circle = space/newline, square = punctuation & digits, triangle = upper case,
diamond = lower case). Black dotted horizontal line = the symmetric position `t* = 0.5`; gray dashed
vertical line = equal plausibility. Spearman ρ = 0.27: the more likely endpoint keeps a slightly
*larger* share of the path, so basin size tracks plausibility — but weakly, and `t*` stays within
0.30–0.72 throughout.

Figure 18 is the readout-decision test itself.

![readout decision test panels](plots/allpairs_readout_decision.png)

**Figure 18.** The plateau boundary is the model's next-character decision boundary. Left: histogram
of `t* − t_flip` (x), the offset between the `d(t)` midpoint and the first change in the model's
predicted next character; y = number of pairs; black dotted line at 0. Median `|t* − t_flip|` = 0.045,
i.e. 2.2 steps of the 50-point grid. Middle: how many distinct next-character predictions a path
visits (x) against number of pairs (y) — median 3, and 32% visit exactly 2. Right: three summary
fractions (y) — the mean fraction of prediction changes falling inside the transition window
`[t_lo, t_hi]` (0.91), the fraction of pairs whose prediction changes *all* fall inside it (0.79), and
the fraction of pairs whose two flat arms are each a single prediction (0.80). Bars carry distinct
hatches as well as colours.

The decision reading survives the sharper form of the test. Paths do not simply flip once: the median
path visits **3** distinct next-character predictions, and only 32% visit exactly 2, so there are
usually one or two short-lived intermediate predictions. But those changes are **not spread over the
plateaus** — **91%** of all prediction changes fall inside the transition window, **79%** of pairs
have every change inside it, and **80%** of pairs have flat arms that are each a single prediction.
In other words, the flat parts of `d(t)` are regions of constant model output and the boundary is
where the output changes; the transition is a short scramble between two decisions rather than an
instantaneous flip.

Finally, Figure 19 gives the two mandatory controls: is the structure learned, and which layers build
it?

![init-vs-final width distributions and width by interpolation block](plots/allpairs_controls.png)

**Figure 19.** Controls. Left: distribution of `w_10→90` (x) against number of pairs (y) for the same
2,080 pairs at step 0 (initialization) and step 30,000 (final); the two histograms carry distinct
hatches and their medians are in the legend. Gray dashed = straight line 0.80, black dotted = strict
rule 0.25. Right: median `w_10→90` (y) against the interpolation block at which the patch is applied
(x = 0, 4, 8, 11), on a fixed 200-pair random subsample of the final checkpoint; bars are the
inter-quartile range; gray dashed = straight-line reference 0.80. Block 11 leaves only the final layer
norm and the unembedding downstream, so it is the near-linear readout reference.

**Both controls are decisive.** *Learned, not architectural:* at initialization **all 2,080** paths
are straight lines (median width **0.803**, inter-quartile range 0.800–0.806, **100%** at width ≥ 0.70,
**0** strict plateaus), against median **0.355** and 8.8% strict after training (Mann–Whitney
p < 1e-300). The basin structure is entirely trained in. *Built by the shallow blocks:* patching later
destroys it — median width 0.344 at block 0, **0.763** at block 4, 0.806 at block 8 and 0.806 at
block 11, which is the straight-line value. So essentially all of the sharpness is produced by blocks
1–4; the unembedding geometry contributes none of it.

**The plausibility confound is real but does not subsume the effect.** Width falls as the more likely
of the two endpoints becomes more likely (Spearman ρ = **−0.46** against `max(p(A), p(B))`, n = 2,080)
and also as the endpoints' logit vectors move further apart (ρ = **−0.46**). These two predictors are
themselves correlated, so we take partial rank correlations: controlling for endpoint separation,
width vs `max(p(A), p(B))` is **−0.59**; controlling for plausibility, width vs separation is
**−0.59**. Both survive, so neither explains the other away. The per-character version is stronger
still: a character's median width against its own log-probability in this context gives ρ = **−0.60**
(n = 65, p = 1.2e-7). Note the direction rules out the trivial artifact once more — *better-separated*
endpoints switch *faster*, not slower.

### The hypothesis

**A plateau in this model is the set of final-position residual states that decode to the same
next-character prediction, one basin per character — a shape that the MLPs of blocks 1–4 build and
everything downstream merely reads.** The evidence: 91% of all prediction changes along a path fall
inside the transition window and 80% of paths have single-prediction flat arms (Figure 18), every
character retains its own basin against every partner (`flat_frac` ≥ 0.86 for all 65) with 78% of the
width variance explained by per-character terms alone (Figures 14–15), the structure is absent at
initialization (Figure 19), and deleting the block-1–4 MLPs returns the width to that untrained value
while amplifying them sharpens it further (0.80 → 0.35 → 0.31, Figure 21). That "decodes to the same
prediction" clause is a **description, not the mechanism**: the decision survives the ablation that
flattens `d(t)` (80.7% of pairs still predict different characters at their endpoints, Figure 22), and
the leading alternative — that the basin is carved by endpoint *plausibility* — still predicts which
pairs are sharp (partial ρ = −0.59) even though it does not mediate the intervention
(ρ(Δw, Δmax_p) = +0.22). The "blocks 1–4 build it" clause survives only as a statement about *this*
trained network: six retraining runs show the sharpening simply **relocates** into whichever blocks
are left trainable — freeze 1–4 and it moves to 5–8 (width 0.471), freeze 1–7 and it moves to 8–11
(0.558), freeze 5–11 and it moves back into 1–4 (0.626), freeze 0–3 and 9–11 and it sits in the middle
window 4–8 (**0.331**), freeze all but blocks 5–7 and 96% of it lands in that three-block window
(**0.427**), always at or above the reference's validation
accuracy (Figure 23) — so the site is contingent and what freezing costs is sharpness, governed mainly
by *where* the trainable blocks sit. A seventh run tested the trainable-depth reading
at its limit and **confirmed** it there: freezing ten blocks so that only block 11 is both trainable and
downstream of the injection lands at **0.726**, matching the ≈0.70 trainable-depth prediction and
excluding the ≈0.56 "one block beside the readout suffices" alternative. An eighth run separated depth
from capacity: retrained at 12 trainable blocks but only 5.38M parameters (`n_embd` 192 instead of
240 — 4% *fewer* trainable parameters than freezing blocks 1–4 leaves), it lands at **0.397**, the
depth account's value rather than the capacity account's ≈0.47, and the second seed of each end of that
comparison leaves the two groups disjoint. The middle-window run then falsified what remained of the
count-first reading: five trainable blocks in mid-stack suffice for the full plateau, shrinking
that window to *three* (freeze 0–4 and 8–11) still lands level with
the 12-block reference at **0.446**, and sliding it one step off centre (blocks 2–6) reproduces
**0.365** exactly. Two further runs then killed the two geometric rules that had fit
the series. A window at blocks 1–5 was predicted above **0.47** because it touches the first block
after the injection, and landed at **0.363**; the coverage description that replaced it — sharp
exactly when the window covers mid-stack block 5 — required **0.55 or above** from a window at blocks
6–10, which excludes block 5, and that window landed at **0.342**, the sharpest matched-accuracy width
of any model in the study. Ten runs support no geometric summary at all. What they do support are two
direct network-to-network facts that need no rule: blocks 1–5 alone are 0.118 sharper than blocks 0–7
which contain them, and blocks 6–10 alone — five trainable blocks, 58% of the parameters frozen at
their random initialization — are 0.072 sharper than the full 12-block reference at the same accuracy
(p = 8.5e-18). The second of those was pre-registered and retrained from a fresh initialization, which
reproduced it to within 0.002 of width.

### The readout-rebalancing intervention — the plateau is upstream of the decision

The hypothesis above says a plateau is a set of states that *decode* to the same prediction. If that
is causal, then moving the readout's decision boundary should move the plateau boundary with it. We
tested this on all 1,873 of the 2,080 pairs whose two endpoints predict different next characters
(207 pairs predict the same character at both ends and have no boundary to move). The intervention
adds a constant to one row of the unembedding output — a pure readout bias — so every residual-stream
activation along the path is bit-identical. Two bias sizes, both fixed before looking at the result:
one that makes the two endpoint predictions score symmetrically (**equalised**, median 2.44 nats), and
one that forces the decision boundary exactly to the path midpoint (**midpoint-forced**, median 5.28
nats). Figure 20 shows where the boundary lands.

![histograms of decision-boundary position under three readouts, and boundary shift versus bias size](plots/rebalance_readout.png)

**Figure 20.** Readout rebalancing on 1,873 character pairs, interpolation block 0, step 30000.
Left (a): number of pairs (y) against position along the path `t` (x) for the plateau midpoint `t*`
(solid) and for the decision boundary `t_gap` under the unmodified readout (dashed), the equalised
bias (dash-dot) and the midpoint-forced bias (dotted, a spike at 0.5 by construction). Right (b):
shift of the decision boundary `t_gap^c − t_gap` (y) against the bias applied as a fraction of the
endpoint logit-gap span (x); circles = midpoint-forced, triangles = equalised. The inset states the
measured invariance of `d(t)`.

Two findings, one of them algebraic and one empirical.

- **The plateau cannot be moved by the readout at all.** `d(t)` is a ratio of distances *between*
  logit vectors, so adding the same bias vector to every point on the path cancels exactly. Measured
  deviation between the biased and unbiased `d(t)` is **1.3 × 10⁻⁶** (float32 noise), so the width
  `w_10→90` and the midpoint `t*` are **exactly invariant** to any additive readout bias, of any size.
  This also means the intervention cannot test the plausibility account's prediction that the *width*
  would change: no readout-level change of endpoint plausibility can alter `d(t)`. Plausibility, if it
  acts at all, must act through the learned weights of blocks 1–11 — consistent with those blocks being
  where the sharpness is built (Figure 19).
- **The decision boundary barely moves either — it is pinned to the residual-stream transition.** The
  logit gap swings a median **21.9 nats** across the path, so it is very steep. A 2.44-nat equalising
  bias moves the boundary by a median of only **0.020** in `t` (80% of pairs move less than 0.05), and
  even the 5.28-nat bias needed to force the boundary to the midpoint moves it a median **0.052**.
  Boundary and plateau midpoint stay aligned throughout: median `|t* − t_gap|` = **0.025** unmodified,
  **0.015** equalised, **0.035** midpoint-forced.

**What this changes in the hypothesis.** The tight `t* ≈ t_gap` alignment reported above is *not*
evidence that the decision creates the plateau — the causal arrow runs the other way. The prediction
flip and the `d(t)` transition coincide because both are driven by the same sharp change in the
residual stream, produced by blocks 1–4; the readout is a steep but passive reader of it. The wording
"a plateau is the set of states that decode to the same prediction" therefore stands as a *description*
of the basins, not as a mechanism, and the mechanism sits upstream of the unembedding.

### The MLP-gain intervention — blocks 1–4 causally set the sharpness

The readout probe pushed the mechanism upstream but could not say *which* upstream computation makes
the transition sharp. Experiment 5's depth control pointed at blocks 1–4 (width 0.34 patching at
block 0 vs 0.76–0.81 patching at blocks 4/8/11), but that is an observation about where the patch is
injected, not an intervention on the model. So we intervene: we multiply the MLP-branch output of a
group of blocks by a gain `g` (attention, LayerNorms and every other block untouched) and re-run the
identical assay, with the endpoints recomputed under the modified model. `g = 1` is the unmodified
model, `g = 0` deletes those MLPs, `g = 1.5` amplifies them. We do this for the **early** group
(blocks 1–4) and, as a specificity control, for a **late** group (blocks 8–11) that the depth
measurement says contributes almost nothing. 150 randomly chosen pairs (fixed seed) of the 2,080, at
interpolation block 0 of the step-30000 checkpoint, so widths are directly comparable to Experiment 5.

![transition width versus MLP gain for early and late blocks, and paired per-pair width changes](plots/mlp_gain_intervention.png)

**Figure 21.** MLP-gain intervention, 150 character pairs, interpolation block 0, step 30000. Left
(A): median transition width `w_10→90` (y) against the MLP-branch gain `g` (x; 1.0 = unmodified
model), band = interquartile range; solid/circles = gain applied to blocks 1–4, dashed/squares =
blocks 8–11. The dashed horizontal reference is the untrained (step-0) median width 0.803 and the
dotted one is the strict plateau threshold `w ≤ 0.25`. Right (B): paired per-pair change in width
`Δw` relative to the unmodified model, one box per condition (boxes hatched `//` = blocks 1–4,
`..` = blocks 8–11); boxes are the interquartile range, whiskers 1.5×IQR, outliers hidden.

- **Deleting the early MLPs destroys the plateau completely.** Median width goes 0.351 (unmodified) →
  0.533 at `g = 0.5` → **0.796** at `g = 0`, i.e. back to the straight-line/untrained value 0.803, and
  the strict plateau rule is passed by **0/150** pairs instead of 15/150. Every single pair widens
  (fraction with `Δw > 0` = **1.00**, median `Δw` = **+0.433**).
- **Amplifying them sharpens it further.** At `g = 1.5` the median width falls to **0.305** and the
  strict-rule pass rate *triples*, 10% → **30%**. The dose–response is monotone across all four gains,
  which is what a "these MLPs build the sharpness" account predicts and a nuisance-side-effect account
  does not.
- **The late blocks do almost nothing.** The same gains on blocks 8–11 move the median width only
  0.337 / 0.333 / 0.380 for `g = 0 / 0.5 / 1.5` — median paired `|Δw| ≤ 0.025`, a 17× smaller effect at
  `g = 0` than the early group. Deleting four whole MLPs at the top of the stack barely registers.
- **The boundary stays put.** Median `|Δt*|` is 0.074 at `g = 0` and ≤ 0.024 elsewhere: the
  intervention changes how *sharp* the transition is, not where it sits.

This is the first causal statement in this series: the plateau's sharpness is manufactured by the
MLPs of blocks 1–4 and merely read out downstream. It also sharpens the live alternative rather than
killing it — the plausibility account can only act through these same weights.

### The per-block scan — the sharpness is distributed, and tracks neither plausibility nor the decision

The gain intervention left two questions open: *which* of blocks 1–4 carries the sharpness, and does
the width change it produces track the **plausibility** confound or the **decision** structure — the
two accounts still standing. We answer both from the same runs. We delete each early block's MLP on
its own (`g = 0`, one block at a time) on the identical 150-pair subsample, and under every condition
we re-measure the two candidate mediators for each pair: the endpoint plausibility
`max_p = max(p(A | context), p(B | context))` (Experiment 5's confound) and the decision structure
(whether the two endpoints still predict different next characters, how many distinct `argmax`
regions the path visits, and the gap `|t* − t_flip|` between the plateau midpoint and the prediction
flip). Deleting all four MLPs is re-run in the same script as an in-run reference.

![median width per single-block MLP deletion, width change versus plausibility change, and decision-structure survival](plots/mlp_block_scan.png)

**Figure 22.** Per-block MLP ablation, 150 character pairs, interpolation block 0, step 30000.
**A** (left): median transition width `w_10→90` (y; bars = interquartile range) for each condition
(x: unmodified model, each single early block's MLP deleted, then all four). The dashed horizontal
reference is the untrained (step-0) median width 0.803 and the dotted one is the unmodified model's
0.351; the percentage above each point is that block's median width change as a fraction of the
all-four effect. **B** (middle): the mediation test — per-pair change in width `Δw` (y) against the
per-pair change in endpoint plausibility `Δmax_p` (x), one point per pair, for the all-four deletion;
dashed horizontal line = no width change, dotted vertical = no plausibility change. **C** (right):
decision structure per condition (x as in A) — solid line with circles (left y) = fraction of the 150
pairs whose two endpoints still predict different next characters; dashed line with squares (right y)
= median `|t* − t_flip|`.

- **No single block carries it; the contribution is graded and front-loaded.** Deleting one block's
  MLP widens the median transition to 0.541 (block 1), 0.478 (block 2), 0.446 (block 3), 0.402
  (block 4), against 0.351 unmodified and 0.796 for all four. As a fraction of the all-four effect
  that is **41% / 28% / 18% / 11%** — monotonically decreasing with depth, and summing to 98%, so the
  four contributions are close to additive. The largest single block recovers under half the effect,
  and every single-block deletion widens almost every pair (fraction with `Δw > 0` = 0.99 / 0.96 /
  1.00 / 0.95) while dropping the strict plateau rate from 10% to 0–3%.
- **The widening does not track plausibility.** Two facts, and they point the same way. First, the
  plausibility association itself *survives* every ablation essentially unchanged: the partial
  Spearman correlation between width and `max_p`, controlling for endpoint separation, is **−0.634**
  in the unmodified model (reproducing Experiment 5's −0.587 on all 2,080 pairs) and stays between
  **−0.45 and −0.64** in all five ablated models. So plausibility keeps explaining *which pairs* are
  sharp. Second, it does not explain the *ablation effect*: the per-pair widening is essentially
  uncorrelated with the per-pair plausibility change (Spearman `ρ(Δw, Δmax_p)` = +0.11, +0.15, −0.01,
  +0.02 for the four single blocks and **+0.22** for all four — Figure 22B), and the plausibility
  landscape barely moves at all (median `|Δmax_p| ≤ 0.0007`) while the width moves by up to +0.433.
  Where plausibility *does* move, it moves the wrong way: deleting all four MLPs raises median `max_p`
  from 0.0034 to 0.0136, and higher plausibility is associated with *narrower* plateaus, yet these
  plateaus vanish.
- **The decision structure survives the ablation that destroys the plateau.** With all four early
  MLPs deleted, **80.7%** of pairs still predict different characters at their two endpoints (86.7%
  unmodified) and the median number of distinct `argmax` regions along the path is **3**, unchanged in
  every condition. So the path still crosses a decision boundary — but `d(t)` is now a straight line
  (0.796 ≈ the untrained 0.803). The two also come apart in position: median `|t* − t_flip|` grows
  from **0.043** unmodified to **0.214** with all four deleted, a 5× decoupling.

**What this settles.** The answer to PLAN's question is *neither*. A plateau is not the decision
region — you can keep the decision and lose the plateau — and the widening is not mediated by
endpoint plausibility, which stays put (and shifts against the predicted direction) while the width
doubles. What blocks 1–4 build is a sharp change in the residual stream that is upstream of, and
separable from, both: the decision and the plausibility ranking are things the readout computes
*from* that geometry, not things that create it. The plausibility association therefore survives as a
description of which pairs get sharp basins, but is now excluded as the mechanism that makes them
sharp.

**Caveats.** 150 pairs, one shared context, one checkpoint, one model. Deleting four MLPs is a large
perturbation that degrades the model broadly; the decision structure being largely preserved is what
licenses the comparison, but it is not preserved perfectly (86.7% → 80.7%). The near-additivity of
the four per-block fractions is descriptive — single-block ablations need not compose linearly, and
we did not test pairs or triples of blocks.

### The frozen-block training test — the sharpening relocates into whatever blocks stay trainable

Every intervention above cuts into an already-trained network, which can only show that a trained
component is load-bearing *at inference*. The hypothesis made a stronger, training-time claim, and we
tested it directly: retrain from scratch with a block group held at its step-0 weights for the whole
run, everything else identical to the reference character run (same corpus SHA, seeds, optimizer,
30,000-step schedule, batch, checkpoint grid). Seven runs, each with its prediction fixed beforehand,
each one written after seeing the previous. The sixth is the one that matters most: it was
predicted to land between its two positional siblings and instead beat every model in the study, which
is what moved the conclusion below from "how many blocks can train" to "where they sit". The seventh
then tested that new reading and confirmed it.

| run | blocks frozen at init | prediction on record | outcome |
|---|---|---|---|
| **frozen-early** | 1–4 (the group the ablations implicate) | stays near the untrained width 0.80 | **falsified** — 0.471 |
| **frozen-late** | 8–11 (specificity control, same *number* of blocks) | sharpens like the reference (≈0.35) | **falsified** — 0.484, same as early |
| **frozen-deep** | 1–7 (58% of the blocks; only 0 and 8–11 trainable) | still sharpens well below 0.80, with the width drop between injection blocks 8 and 11 | **confirmed** — 0.558, drop entirely inside blocks 8–11 |
| **frozen-mirror** | 5–11 (mirror image: same 58% frozen, same 5 trainable blocks, but at the *bottom*) | if only the *count* of trainable blocks matters, ≈0.558 again with the drop between injection blocks 0 and 4 | **split** — the drop is exactly where predicted, but the width is **0.626**, markedly straighter |
| **frozen-two** | 1–10 (82.9% of the parameters; only blocks 0 and 11 trainable) | ≈0.70 if trainable depth is the first-order term; ≈0.56 if one trainable block beside the readout suffices | **confirmed** — **0.726**, decisively on the trainable-depth side |
| **frozen-mid** | 0–3 and 9–11 (the *same* 58% frozen and the same 5 trainable blocks as deep/mirror, but in the *middle*: blocks 4–8) | between the two known five-block values, ≈0.58–0.60, if position is a second-order correction favouring the readout end | **falsified** — **0.365**, far below *both*, below every 8-block run, and below the 12-block reference |
| **frozen-mid3** | 0–4 and 8–11 (74.6% of the parameters; the mid-stack window shrunk to 3 trainable blocks, 5–7) | ≈0.40–0.50, still beating 5 blocks at either end, if position dominates window size; ≥0.558 if the block count returns as the leading term | **confirmed** — **0.446**, level with the 12-block reference (p = 0.17) and 0.09–0.18 clear of every 5-block end window |
| **frozen-mid-off** | 0–1 and 7–11 (same 58% frozen and same 5 trainable blocks as deep/mirror/mid, but one step off centre: blocks 2–6) | ≈0.40–0.45, between frozen-mid and frozen-deep, if the cost tracks how the frozen blocks are distributed around the window | **falsified** — **0.365**, identical to frozen-mid (p = 0.06) despite five frozen blocks below the window instead of three |

All eight runs completed the full 30,000 steps and none lost any task performance — in fact each ended
*above* the reference: final validation next-character accuracy **0.5625** (frozen-early), **0.5622**
(frozen-late), **0.5742** (frozen-deep), **0.5744** (frozen-mirror), **0.5728** (frozen-mid),
**0.5711** (frozen-mid3), **0.5744** (frozen-mid-off) and
**0.5668** (frozen-two) against
the reference run's **0.5502**, reaching the reference's final accuracy at steps 2,750 / 2,500 / 3,000 /
2,750 / 3,750 / **7,000** / 3,500 / **7,000**. So every comparison below is between networks that predict held-out
Shakespeare at least as well as the reference. Note in particular that frozen-deep, frozen-mirror,
frozen-mid and frozen-mid-off are matched on
everything a capacity argument can see — 4.86M of 8.38M parameters frozen (58.0%), five trainable
blocks each, and final accuracies inside a 0.0016 band — and differ only in *where* the
trainable blocks sit. That four-way comparison is the cleanest in the series. Four of the ten frozen
conditions were later repeated from a second initialization (frozen-early, frozen-deep, blocks 6–10 and
blocks 0–4, all four again above the reference at 0.5629, 0.5730, 0.5730 and 0.5750 final accuracy), so
every comparison the conclusions rest on has a measured seed spread under at least one of its sides,
and the positional ones under both.

![raw interpolation curves, transition widths, injection-depth profile and validation accuracy for the reference and eight frozen-block runs](plots/frozen_blocks.png)

**Figure 23.** Frozen-block training test, 150 character pairs, interpolation block 0. **Top row:** raw
`d(t)` (y, relative distance to endpoint A vs B) against interpolation position `t` (x) for the same 20
pairs under twelve models — reference untrained (step 0), reference at step 2500, reference trained
(step 30000), and blocks 1–4, 8–11, 1–7, 0–3&9–11, 0–4&8–11, 0–1&7–11, 0&6–11, 5–11 and 1–10 frozen (each at
its final step 30000); the eleventh frozen group, blocks 0–5&11 (trainable 6–10), appears in the panels
below at its matched-accuracy checkpoint, which is this section's primary comparison axis (its
step-30000 width, 0.328, is reported in the text below and is sharper still).
Thin lines are individual pairs, the thick dashed line is their median, and the gray dashed diagonal is
the straight-line (no-plateau) reference `d = t`; each panel title gives that model's median width. The
tenth panel — blocks 0 and 6–11 frozen, i.e. five trainable blocks at 1–5 — has the sharpest
median in the figure, sharper than the fully trained reference seven panels to its left.
**Bottom left:** median transition width `w_10→90` (y) per condition (x), bars = interquartile range;
the gray dashed horizontal line is the untrained value 0.803 and the black dotted line the trained
reference's 0.351. **Bottom middle (three panels):** median width (y) against the interpolation block at
which the
path is injected (x: 0, 2, 4, 8, 10, 11); a drop between two injection points means the blocks in
between are what sharpen the path. The runs are split across three panels so that no panel carries
more than three hues. Left: the five-trainable-block runs whose window sits in the upper stack —
blocks 0–5&11 frozen (trainable 6–10; dash-dot-dash, hexagons), 0–3&9–11 frozen (dash-dot-dot,
down-triangles) and 0–1&7–11 frozen (long-dashed, left-triangles). Middle: the five-trainable-block
runs at the bottom of the stack, plus the one whose trainable set is not a window — blocks 0&6–11
frozen (fine-dotted, right-triangles), 5–11 frozen (long-dash-dot, plus-markers) and 1–7 frozen
(trainable 0 and 8–11; dotted, diamonds). Right: the other freeze sizes —
blocks 1–4 frozen (dashed, squares), 8–11 frozen (dash-dot, triangles), 0–4&8–11 frozen (fine-dotted,
crosses) and 1–10 frozen (dash-dot-dash, stars), all four in gray at four lightnesses. The
trained reference (black, solid, circles) appears in all three panels as the anchor.
**Bottom right:** validation next-character accuracy (y) against optimization step (x, symlog,
linear below 100) for the ten runs, same line styles — they are nearly coincident, which is the point.
The black dotted line is the reference run's
final accuracy and the open markers are each run's matched-accuracy checkpoint (steps 2,750 / 2,500 /
3,000 / 3,750 / 7,000 / 3,500 / 3,500 / 2,750 / 7,000 for blocks 1–4, 8–11, 1–7, 0–3&9–11, 0–4&8–11,
0–1&7–11, 0&6–11, 5–11 and 1–10
frozen).

- **Freezing four blocks falsified the first prediction; freezing seven confirmed the successor.**
  Frozen-early ends at median width **0.471** (IQR 0.403–0.524), nowhere near the predicted untrained
  0.803, and frozen-late — the group the ablations said contributes nothing — ends at **0.484**, i.e.
  the same. Freezing *any* four blocks costs about the same 0.11–0.12 of width (paired median `Δw` vs
  the trained reference **+0.107** early and **+0.120** late; 94% and 96% of pairs widen). Frozen-deep
  then tested the successor prediction and confirmed it: with 7 of 12 blocks held at initialization the
  paths still sharpen to **0.558** (IQR 0.471–0.621), narrower than untrained for 149/150 pairs
  (Wilcoxon p = 2e-26).
- **The computation relocates every time, and the depth control shows exactly where it goes.**
  Re-running Experiment 5's depth control — inject the interpolated activation at block 0, 2, 4, 8, 10
  or 11 instead of only block 0 — gives 0.351 / 0.646 / 0.761 / 0.805 / 0.806 / 0.805 for the trained
  reference: the sharpening is made in blocks 1–4, front-loaded into blocks 1–2, and nothing above
  block 4 does anything. Frozen-late reproduces that profile (0.484 / 0.739 / 0.793 / 0.806 / 0.806 /
  0.806). Frozen-early gives 0.471 / **0.471** / **0.471** / 0.788 / 0.804 / 0.809 — injecting anywhere
  inside its frozen group changes the width by 0.000, so the computation has moved down to blocks 5–8.
  Frozen-deep gives 0.558 / **0.558** / **0.557** / 0.695 / 0.767 / 0.805: its frozen blocks again
  contribute nothing, and the whole 0.248 of sharpening is spread across the four trainable blocks —
  0.139 across blocks 5–8 (of which only block 8 can train), 0.071 across blocks 9–10 and 0.039 in
  block 11. Frozen-mirror relocates it back to the bottom: 0.626 / 0.764 / **0.805** / 0.806 / 0.806 /
  0.806, i.e. every bit of its sharpening happens in blocks 1–4 (0.138 in blocks 1–2, 0.042 in 3–4) and
  injecting at block 4 already gives the untrained straight line. Frozen-mid puts it in the middle:
  0.331 / 0.342 / 0.525 / **0.802** / 0.802 / 0.803 — flat above block 8, with 0.277 of its total 0.471
  over blocks 5–8, 0.183 over blocks 3–4 (of which only block 4 can train) and 0.011 left for the
  frozen blocks 1–2. Frozen-mid-off shifts it one step down, into the window 2–6:
  0.355 / 0.525 / 0.737 / **0.807** / 0.807 / 0.807 — flat above block 8 again, with 0.382 of its total
  0.452 across blocks 1–4 and only 0.070 above block 4. Frozen-mid3 concentrates it further still:
  0.427 / 0.436 / 0.443 / **0.806** / 0.806 / 0.806 — 0.363 of its total 0.380 between injection blocks
  4 and 8, with the frozen blocks 1–4 and 9–11 contributing 0.017 between them. Frozen-two is the
  limiting case:
  0.726 / 0.725 / 0.724 / 0.725 / 0.725 / **0.803** — flat all the way up, with the *entire* 0.077 of
  sharpening appearing between injection blocks 10 and 11, i.e. produced by block 11 alone. Eight runs,
  eight different sites, the same phenomenon.
- **The count of trainable blocks does not fully determine the width — where they sit matters too.**
  Frozen-deep and frozen-mirror freeze the *same* 58.0% of parameters and leave the *same*
  five trainable blocks, differing only in whether those blocks abut the readout (8–11) or the
  embedding (0–4). They do not come out equal: **0.558** vs **0.626**, a paired median `Δw` of +0.063
  (81% of pairs wider, p = 6e-17), i.e. 54% vs **39%** of the reference sharpening recovered. So the
  count-only prediction is falsified at the margin, and a second frozen-deep seed (below) confirms the
  gap is larger than seed noise. Yet with eight trainable blocks the same contrast
  is nearly nil (frozen-early 0.471 vs frozen-late 0.484, Δ = 0.015). Ranked by width, those five runs
  read **0.351** (12 trainable) → **0.471 / 0.484** (8 trainable) → **0.558 / 0.626** (5
  trainable) → **0.726** (2 trainable, and only 1 of them downstream of the injection) — which invites
  the reading that the count of trainable blocks is the first-order term and their position a small
  correction on top of it. The third five-block position, run last and reported below, breaks that
  reading outright.
- **One trainable block can still bend the path, but only barely — and this is where the plateau
  finally degrades.** Frozen-two is the strongest test of the depth account because its two trainable
  blocks are 0 and 11, and injecting at block 0 *overwrites* block 0's output, so block 11 is the only
  trainable block the measurement can see. It lands at **0.726** (IQR 0.642–0.802), far from the ≈0.56
  a "one block beside the readout is enough" account predicts and close to the ≈0.70 the trainable-depth
  account predicts — paired `Δw` **+0.160** vs frozen-deep (97% of pairs, p = 7e-26) and **+0.094** vs
  frozen-mirror (89%, p = 3e-21). It recovers only **17%** of the reference sharpening, and for the
  first time in this series the plateau is genuinely damaged rather than merely blunted: **26%** of its
  pairs are *wider* than the untrained network's (versus 0–1.3% in the other seven runs), the boundary
  drifts off the prediction flip (median `|t* − t_flip|` 0.146 vs 0.043 reference), and the
  plausibility association mostly collapses (partial ρ = −0.18 vs −0.63). It is also the only run that
  needed materially longer to reach the reference's accuracy (step 7,000 vs 2,500–3,000).
- **Five of the ten frozen runs lose the sharpest tail; the five mid-stack-window runs keep it.** The
  strict plateau rule
  (`w ≤ 0.25`, both margins ≥ 0.10, near-monotone) is met by 10% of reference pairs but 0.7%
  (frozen-early) and **0%** (frozen-late, frozen-deep, frozen-mirror, frozen-two) — against **28.0%**
  for blocks 6–10, the highest rate of any model in this study, 24.7%
  for frozen-mid, 21.3% for frozen-mid-off, 19.3% for
  frozen-mid-low and 9.3% for frozen-mid3 (all below).
  Against the reference *at the
  matched-accuracy step* (2500, width
  0.443) the four-block gaps almost vanish (+0.033, +0.038) while the seven-block gaps are +0.110 and
  +0.171 and frozen-two's is +0.276 — freezing a third of the stack mostly *slows* the sharpening;
  freezing 58% of it also caps how far it can go; freezing 83% of it nearly removes it.
- **The rest of the geometry is unchanged in the seven runs that retain a plateau.** The boundary still
  sits mid-path (median `t*` 0.491 / 0.495 / 0.486 / 0.499 / 0.475 / 0.483 / 0.471 vs 0.488), the
  endpoints still
  predict different characters for 84% / 93% / 87% / 87% / 92% / 88% / 83% of pairs (86.7% reference), the
  boundary stays
  glued to the prediction flip (median `|t* − t_flip|` 0.062 / 0.059 / 0.092 / 0.085 / 0.045 / 0.072 /
  0.059 vs
  0.043 — contrast 0.214 under the MLP ablation), and the plausibility association survives
  (partial ρ = −0.61 / −0.60 / −0.62 / −0.54 / −0.61 / −0.56 / −0.51 vs −0.634). Frozen-two is the exception on
  the last two, as noted above.

- **Depth, not parameter count: the confound is now broken.** Every frozen run removes trainable
  blocks *and* trainable parameters at once, so "width tracks trainable depth" and "width tracks
  trainable capacity" fit them all equally well. The narrow run separates them: `n_embd` 192 instead of
  240, **nothing frozen**, so all 12 blocks train but the network holds only **5,375,808** parameters —
  **4.0% fewer** than frozen-early's 5,601,360 trainable parameters (both counted the same way, with
  the tied embedding/unembedding weight counted once), so on the capacity axis the narrow run is if
  anything handicapped rather than favoured. The capacity account predicts
  frozen-early's ≈0.47; the depth account predicts the reference's ≈0.35–0.44. At matched accuracy
  (step 2,750, val 0.5543) it lands at **0.397** (IQR 0.311–0.526): paired `Δw` **−0.073** vs
  frozen-early (only 23% of pairs wider, p = 2.5e-15) and **−0.092** vs frozen-late (13%, p = 1.8e-19),
  while against the reference at *its* matched step it is not worse but slightly **sharper**, −0.014
  (39% of pairs wider, p = 1.9e-4). Cutting a third of the parameters costs nothing; cutting a third of
  the trainable blocks costs 0.11–0.12. The rest of the narrow run's geometry matches the reference
  too — front-loaded depth profile (0.397 / 0.569 / 0.686 / 0.763 / 0.807 / 0.832 at injection blocks
  0 / 2 / 4 / 8 / 10 / 11, the reference's shape), partial ρ = −0.65 (reference −0.634), median
  `|t* − t_flip|` 0.061 — and it keeps the sharpest tail too, meeting the
  strict plateau rule on **13.3%** of pairs against the reference's 12.7% and 0–0.7% for the five
  frozen runs known at the time (the four mid-stack-window runs, below, later reached 24.7%, 21.3%,
  19.3% and 9.3%).

- **Both ends of the load-bearing comparison also carry a second seed, and seed noise is small.** The
  depth conclusion rests on runs with 12 trainable blocks coming out sharper than runs with 8, and
  until now every point on that axis was a single initialization — so a 0.397-vs-0.476 gap had no error
  bar under it. Both ends were therefore retrained from a fresh model seed (2024; identical data order,
  schedule, freeze mask and matched-accuracy rule). The **narrow** run repeats at median width
  **0.437** (IQR 0.326–0.514, strict rule 10.7%) against seed 1337's 0.397 — a small but detectable
  shift (paired `Δw` +0.015, p = 0.015). **Frozen-early** repeats almost exactly: **0.498** against
  0.476, and its per-pair widths are statistically indistinguishable from the first seed's (paired `Δw`
  **+0.001**, exactly half the pairs shifting each way, p = 0.40), with the two distributions agreeing
  decile by decile (10th/50th/90th percentile 0.286 / 0.498 / 0.653 against 0.308 / 0.476 / 0.647).
  Seed noise on this measure is therefore **≤ 0.04**, comfortably inside the 0.06–0.10 step it is being
  asked to resolve, and it has no consistent direction: trained on to step 30,000 (val accuracy 0.5629,
  against seed 1337's 0.5625) the second frozen-early seed comes out at **0.445** where the first gave
  0.471, i.e. 0.027 in the *opposite* direction (paired `Δw` −0.030, p = 3.3e-5). The second seed also
  reproduces the *relocation*: injecting at blocks 0, 2 and 4 gives 0.498 / 0.498 / 0.501 at matched
  accuracy and 0.445 / 0.444 / 0.443 at step 30,000 — its frozen group contributes nothing at either
  checkpoint — with the sharpening spread over blocks 5–8 and 9–10, the same signature as seed 1337.

- **With two seeds a side the depth step is a clean separation, not an overlap.** Ranking the six
  matched-accuracy runs by median width puts all three with 12 trainable blocks (reference 0.443,
  narrow 0.397 and 0.437) below all three with 8 (frozen-early 0.476 and 0.498, frozen-late 0.500):
  the two groups are **disjoint**, which is the smallest one-sided rank-sum p a 3-versus-3 comparison
  can produce (**p = 0.05**). All four narrow-versus-frozen-early seed combinations agree pair by pair
  as well — `Δw` −0.073, −0.067, −0.044 and −0.063, each with p ≤ 2.7e-8 — so the gap does not depend
  on which initialization is placed on which side. The one sub-claim a second seed **retracts** is that
  the narrow run beats the full-width reference at matched accuracy: narrow seed 2 is statistically
  indistinguishable from the reference's 0.443 (paired `Δw` −0.004, 46% of pairs wider, p = 0.17), so
  the honest statement is that removing a third of the parameters costs nothing measurable, not that it
  helps.

- **Trained on to the end, the narrow run is if anything sharper than the full-width reference.** The
  matched-accuracy comparison above is the primary one, because it is the only axis on which runs of
  different capacity are directly comparable; but the same conclusion holds at the end of training.
  Left to run, the narrow model reaches median width **0.332** (IQR 0.288–0.389) at step 27,143 (val
  accuracy 0.5639), against the reference's fully-trained **0.351**: paired over the same 150 pairs
  that is **−0.010**, with only 43% of pairs wider (p = 2.1e-4), i.e. indistinguishable-to-slightly-
  sharper. Against the fully-trained frozen runs the gap is large and one-sided — **−0.124** vs
  frozen-early (0.471; just 1.3% of pairs wider, p = 2.6e-26), **−0.098** vs frozen-early's second seed
  (0.445; 11%, p = 6.5e-24) and **−0.146** vs frozen-late (0.484; 3.3%, p = 3.6e-26). Its depth profile stays front-loaded (0.332 / 0.626 / 0.746 / 0.794 / 0.802 /
  0.808 at injection blocks 0 / 2 / 4 / 8 / 10 / 11), 12.0% of pairs meet the strict rule (reference
  10.0%), and the plausibility association holds (partial ρ = −0.51). One caveat on this row only: the
  harness time budget stopped it at 27,143 of the planned 30,000 steps, so its cosine schedule had
  annealed to lr 1.2e-4 rather than 1.0e-4 — a truncation that can only *understate* how sharp it
  would end up, since the run was still sharpening (0.397 at its matched step → 0.332 here, p = 3.1e-14).

- **Five trainable blocks beside the readout really do beat five at the bottom — the position term
  replicates.** This was the last sub-claim resting on a single pair of runs, and its 0.068 gap is
  small enough that seed noise could have produced it. Frozen-deep was therefore retrained from a
  second initialization (seed 2024, everything else fixed); the prediction fixed beforehand was that it
  lands within the measured ≈0.04 seed spread of 0.558 and stays below frozen-mirror's 0.626. It does,
  at both framings. At matched accuracy (step 3,000, val 0.5503 — the same step as seed 1337) it gives
  **0.559** against seed 1337's 0.590, and at step 30,000 (val 0.5730) **0.579** against 0.558: a
  within-condition spread of 0.031 and 0.021, again with no consistent sign (paired `Δw` −0.016 at
  matched accuracy, p = 8.9e-4, and +0.023 at the end of training, p = 4.5e-5). Both seeds sit below
  frozen-mirror on both axes — the *worst* frozen-deep seed is still 0.039 (matched) and 0.046 (final)
  narrower — and pair by pair the replicate is **−0.060** against frozen-mirror at matched accuracy
  (only 21% of pairs wider, p = 5.9e-14) and **−0.040** at the end of training (29%, p = 3.4e-8). The
  relocation signature reproduces exactly: injecting at blocks 0, 2 and 4 gives 0.559 / 0.558 / 0.557
  at matched accuracy and 0.579 / 0.578 / 0.577 at step 30,000, so the frozen blocks 1–7 again
  contribute nothing and all the sharpening sits in the trainable blocks 8–11 (0.683 and 0.714 by
  injection block 8). Geometry unchanged as well (median `t*` 0.486, endpoints differ for 88% of pairs,
  3 `argmax` regions, `|t* − t_flip|` 0.084, partial ρ = −0.58, strict rate 0). What those two runs
  cannot say is how large the position effect gets, because between them they sample only the two *ends*
  of the stack.

- **The third position overturns the count-first reading: five trainable blocks in mid-stack install
  the whole plateau.** Frozen-mid puts the same five trainable blocks at the one untested position —
  blocks 4–8, freezing 0–3 and 9–11, so the frozen fraction (58.0%) and the trainable count match both
  siblings exactly. The prediction on record was 0.58–0.60, between them. It lands at **0.365** (IQR
  0.253–0.471) at matched accuracy (step 3,750, val 0.5519) and **0.331** (IQR 0.258–0.428) at step
  30,000 (val 0.5728) — not between them but far below both, below every eight-block run, and below the
  untouched 12-block reference at its own matched checkpoint (0.443; paired `Δw` −0.056, only 25% of
  pairs wider, p = 2.7e-14). Against its positional siblings the gaps are the largest anywhere in the
  series: **−0.211** vs frozen-deep seed 1 (1.3% of pairs wider, p = 3.3e-26), **−0.188** vs seed 2
  (**0%** of pairs wider, p = 2.3e-26) and **−0.240** vs frozen-mirror (0.7%, p = 2.3e-26). At the end
  of training it is if anything sharper than the fully trained reference (−0.023, 37% of pairs wider,
  p = 0.004), and it is the only frozen run to keep the sharpest tail — **24.7%** of pairs meet the
  strict rule at matched accuracy and 22.7% at the end, against the reference's 10.0% and 0–0.7% for
  every other frozen run. Its geometry is an ordinary plateau (median `t*` 0.501, endpoints differ for
  89% of pairs, 3 `argmax` regions, `|t* − t_flip|` 0.048 against the reference's 0.043, partial
  ρ = −0.47 at matched accuracy and −0.61 at the end).

- **What that replaces.** Three things follow. (i) The number of trainable blocks is **not** the
  first-order term: the three five-block runs span 0.365–0.629, a wider range than the whole 12→5 block
  series, and five mid-stack blocks beat all three eight-block runs (0.476–0.500) and the 12-block
  reference. (ii) Position is **not** a gradient toward the readout — it has an interior optimum that
  the two-point contrast could not see, because that contrast sampled only the ends. (iii) What the
  three positions differ in is how the seven frozen blocks are *distributed* around the trainable
  window, not how many there are: frozen-mid splits them into two short runs of three either side
  (blocks 1–3 and 9–11, all downstream of the block-0 injection), while frozen-deep stacks all seven
  before the window and frozen-mirror all seven after it. The run with no long frozen stretch beside
  its window is much the sharpest (0.365 vs 0.558–0.629), and of the two that have one, the stretch
  *after* the window costs more than the same stretch before it (0.626 vs 0.558). That reading
  described three points rather than testing a law, so we tested it.

- **Three trainable blocks in mid-stack match the full network and beat five blocks at either end.**
  The test keeps the mid-stack site and shrinks the window from five blocks to three (freeze 0–4 and
  8–11, leaving blocks 5–7 — **74.6%** of the parameters frozen, more than any run but frozen-two). The
  prediction on record was 0.40–0.50 if position dominates, ≥ 0.558 if the block count reasserts
  itself. Outcome: **0.446** (IQR 0.344–0.559) at matched accuracy (step 7,000, val 0.5518) and
  **0.427** (IQR 0.324–0.541) at step 30,000 (val 0.5711) — inside the predicted window and
  **statistically indistinguishable from the 12-block reference** at the matched checkpoint (0.443;
  paired `Δw` +0.009, 55% of pairs wider, p = 0.17). Against the five-block windows at the two ends it
  wins on both framings: **−0.121** vs frozen-deep seed 1 (9.3% of pairs wider, p = 7.2e-23),
  **−0.090** vs seed 2 (11%, p = 1.4e-21), **−0.154** vs frozen-mirror (4.7%, p = 1.3e-25); at the end
  of training **−0.111** and **−0.184**. Strict plateau rate 9.3% (matched) / 10.0% (final), matching
  the reference's 12.7% / 10.0%. Its relocation is the tightest in the series — 96% of the sharpening
  falls between injection blocks 4 and 8 — and its geometry is an ordinary plateau (median `t*` 0.479,
  endpoints differ for 91% of pairs, 3 `argmax` regions, `|t* − t_flip|` 0.079, partial ρ = −0.46).
  Window size is not irrelevant: dropping from five mid-stack blocks to three costs **+0.086** of width
  (85% of pairs wider, p = 3.0e-17) and the run needs 7,000 steps to reach the reference's accuracy
  against frozen-mid's 3,750. But it is dominated by position — **three** trainable blocks in the middle
  beat **five** at either end by 0.09–0.18 and reproduce a 12-block network's plateau geometry while
  nine of twelve blocks never leave their initialization.

- **A fourth position falsifies the "adjacent frozen stretch" description and leaves a sharper one.**
  That description predicts that sliding the five-block window one step off centre — freeze 0–1 and
  7–11, leaving blocks 2–6, so five frozen blocks sit below the window instead of three — costs
  something, landing between frozen-mid and frozen-deep (prediction on record: **0.40–0.45**, with
  anything at or below 0.365 counting against it). It costs nothing. The run matches the reference's
  accuracy at step 3,500 (val 0.5507), ends at the highest validation accuracy in the study (0.5744),
  and gives **0.365** (IQR 0.271–0.468) at matched accuracy and **0.355** (IQR 0.275–0.405) at step
  30,000 — **identical to frozen-mid** (paired `Δw` +0.014, p = 0.06; +0.007, p = 0.23), sharper than
  the reference at its matched checkpoint (−0.050, 25% of pairs wider, p = 4.2e-12) and level with the
  fully trained reference (−0.009, p = 0.29). Against the two end windows it is 0.17–0.26 sharper on
  every comparison (all p ≤ 3e-25), and 21.3% of its pairs meet the strict rule. So the distribution of
  frozen blocks is not what governs the width, and that description is withdrawn.

- **A ninth run, run to test the interior/end split, refuted it.** After eight runs the widths
  separated exactly on whether a run's *usable* window — its trainable blocks intersected with 1–11,
  since patching at block 0 overwrites block 0's output — touched an end of the stack. That rule was
  written down with the prediction it makes before the test was run: a five-block window at blocks 1–5
  (freeze block 0 and blocks 6–11) touches block 1, so it had to land above **0.47** with the blunt
  group; near 0.365 would refute the rule. It matched the reference's accuracy at step 3,500 (val
  0.5503), finished at 0.5732, and gives **0.363** (IQR 0.280–0.447) at matched accuracy and
  **0.326** (IQR 0.258–0.416) at step 30,000 — the sharpest final width of any of the fourteen models
  in this study. It is indistinguishable from the two mid-stack windows (paired `Δw` +0.008, p = 0.27
  against blocks 4–8; −0.009, p = 0.23 against 2–6) and 0.10–0.23 sharper than every end window
  (p ≤ 2e-21). The interior/end rule is therefore **withdrawn**, and it is the second post-hoc
  description of this series to die on its first test.

- **What the ninth run establishes on its own does not depend on any rule.** Its trainable blocks 1–5
  are a strict *subset* of frozen-late's trainable 0–7, and it is 0.118 sharper (4.7% of pairs wider,
  p = 2.2e-25). Removing two trainable blocks from a network makes its interpolation paths sharper.
  No account in which the number of trainable blocks sets the width survives that, and it needs no
  regularity fitted to nine points.

- **A tenth run refuted the replacement description too, and it is the sharpest network in the
  study.** The coverage description — sharp exactly when the usable window covers mid-stack **block
  5** — was recorded with its prediction before the test: a five-block window at blocks 6–10 (freeze
  0–5 and 11) excludes block 5 while touching neither end, so it had to land at **0.55 or above**. It
  reached the reference's accuracy at step 3,750 (val 0.5523) and gives **0.342** (IQR 0.240–0.446),
  the sharpest matched-accuracy width of the fifteen models here — sharper even than the two mid-stack
  windows it was supposed to beat by 0.19 (paired `Δw` −0.014, p = 0.025 against blocks 4–8; −0.024,
  p = 9e-4 against blocks 1–5) and 0.14–0.25 sharper than every end window (p ≤ 1.2e-25). Its strict
  plateau rate, 28.0%, is the highest of any condition measured. **Coverage is withdrawn.** Two
  post-hoc geometric descriptions have now each died on the first experiment aimed at it, and the
  honest reading is that ten runs support no geometric summary of which blocks must be trainable.

- **What replaces the rule is a second rule-free network-to-network fact.** Blocks 6–10 alone, with
  58.0% of the parameters left at their random initialization, are **0.072 sharper than the untouched
  12-block reference** at the same validation accuracy (18.7% of pairs wider, p = 8.5e-18). The
  comparison holds at the end of training as well, where both networks have run the full 30,000 steps:
  blocks 6–10 give median `w` **0.328** (IQR 0.252–0.395, strict plateau rate 24.0%) against the
  reference's 0.351, a paired **−0.037** with 36.7% of pairs wider. The run also kept sharpening after
  the matched-accuracy checkpoint (0.342 → 0.328), so the matched-accuracy comparison is the
  conservative one. Together
  with the ninth run's subset comparison, this is what the frozen-block series establishes without
  fitting anything: training fewer blocks does not blunt the plateau and can sharpen it, so width is
  not read off trainable count, trainable capacity, or window geometry.

- **That fact now carries a second initialization, and the pre-registered prediction held exactly.**
  Blocks 6–10 was retrained from model seed 2024, with corpus, split, data order, optimizer, schedule,
  batch size, checkpoint grid and freeze mask unchanged; the prediction went into `PLAN.md` while the
  run was still training and before it was scored — stay within ≈0.04 of 0.342 and clearly below the
  reference's 0.443, with a replicate at or above 0.443 retracting the claim. It repeats at **0.344**
  (IQR 0.278–0.418; step 3,750, val 0.5530), a seed spread of **0.002** — the smallest measured
  anywhere in this study — with per-pair widths indistinguishable from the first seed's (paired
  `Δw` +0.007, 53% of pairs wider, p = 0.65), and **0.335** against 0.328 at step 30,000 (spread
  0.007, p = 0.068). The comparison itself reproduces on both axes: **−0.071** against the reference at
  matched accuracy (18.0% of pairs wider, p = 1.9e-16) and **−0.021** at step 30,000 (37.3% wider,
  p = 7.5e-4); it is also statistically indistinguishable from the sharpest mid-stack window
  (−0.005 against blocks 4–8, p = 0.18). The relocation signature is the first seed's block for block —
  injecting at 0 / 2 / 4 gives 0.344 / 0.356 / 0.365 and at 8 / 10 / 11 gives 0.722 / 0.821 / 0.821, so
  the frozen blocks 0–5 again contribute nothing and every bit of sharpening sits in the trainable
  6–10 — and so is the geometry (median `t*` 0.466, endpoints differ for 88% of pairs, 3 `argmax`
  regions, `|t* − t_flip|` 0.048, partial ρ = −0.57, strict plateau rate 18.0% at matched accuracy and
  20.7% at the end).

- **The position ordering also survives a second seed a side — but at the end of training its margin
  is the size of seed noise.** Blocks 0–4 (freeze 5–11) is the blunt end of that contrast and was the
  last single-seed run under a load-bearing comparison. Its pre-registered prediction: within ≈0.04 of
  0.629 and above *both* frozen-deep seeds (0.559 and 0.590), with a replicate at or below 0.590
  retracting the ordering. It lands at **0.624** at matched accuracy (step 2,750, val 0.5504; spread
  0.006, paired `Δw` +0.004, p = 0.35) and **0.590** at step 30,000 (spread 0.036, `Δw` −0.021,
  p = 1.2e-4). All four seed pairings keep five trainable blocks beside the readout sharper than five
  at the bottom, on both axes: at matched accuracy the replicate is **+0.031** wider than frozen-deep
  seed 1 (72% of pairs wider, p = 3.4e-10) and **+0.053** wider than seed 2 (83%, p = 1.8e-16); at step
  30,000 it is **+0.038** (72%, p = 6.0e-9) and **+0.022** (68%, p = 3.1e-3). What the replicate does
  change is how large that ordering may be called: the closest median pairing is 0.033 at matched
  accuracy and only **0.010** at the end of training, both at or inside the 0.040 largest seed spread
  measured here, so the ordering rests on the per-pair tests rather than on a median gap that clears
  seed noise by itself. Everything else reproduces — sharpening confined to the trainable 0–4
  (0.624 / 0.748 / 0.823 at injection blocks 0 / 2 / 4) and the same blunt geometry as seed 1337
  (2 `argmax` regions, `|t* − t_flip|` 0.120, strict plateau rate 0).

To show that neither candidate variable orders the runs, we plot each run's median width against both
at once, with every run shown at the same validation accuracy and again at the end of its training.

![median transition width against trainable blocks and against trainable parameters, for seven runs at matched validation accuracy](plots/capacity_vs_depth.png)

**Figure 24.** Trainable depth versus trainable capacity, 150 character pairs, interpolation block 0.
y (both panels): median transition width `w_10→90` (lower = sharper plateau), bars = interquartile
range; the gray dashed horizontal line is the untrained value 0.803. **Left:** x = number of trainable
transformer blocks (axis reversed, 12 → 2). **Right:** x = trainable parameters in millions. Runs that
share an x on either axis are nudged apart so they can be told apart; where a label says "(2 seeds)",
the two adjacent markers of that style are the same run trained from two model seeds. Large filled
circles are the three runs with all 12 blocks trainable (the 240-wide reference and the two seeds of
the 192-wide narrow run); large open diamonds are the fourteen runs with blocks frozen at
initialization (ten frozen groups, with two seeds each of frozen 1–4, frozen 1–7, frozen 0–5 & 11, and
frozen 5–11); labels for the crowded five-block column are parked in free space and joined to their
marker by a thin gray line. Each large marker is that run's first checkpoint to
reach the reference's final validation accuracy 0.550. The small open square joined to it by a dotted
line is the same run at the end of training. Neither axis orders the runs. On the right, at 5.4–5.6M
trainable parameters both narrow seeds (filled) are sharper than all three eight-block frozen runs
(open) while the 8.4M reference is no sharper than the 5.4M narrow runs, so capacity explains nothing.
On the left, the extremes are ordered by trainable depth but the middle of the axis is not: the six
diamonds at x = 5 run from 0.342 (frozen 0, 6–11 and the two other mid-stack windows, which land on top
of each other) to 0.629
(frozen 5–11), a spread wider than the whole 12-to-5 trend, and all four mid-stack windows — three at
x = 5 and one at x = 3 — sit below every eight-block run, with the three-block one level with the
12-block reference. The end-of-training squares preserve both patterns, so neither
is an artifact of the matching rule. Markers of one condition that sit adjacent are its two seeds, and
the gap between them is the across-seed spread: 0.397 vs 0.437 at 12 blocks, 0.476 vs 0.498 at 8, and
0.590 vs 0.559, 0.342 vs 0.344 and 0.629 vs 0.624 at 5 — every spread smaller than the positional
gaps it has to resolve, and two of the three five-block spreads are under 0.01.

**What this settles.** "Blocks 1–4 build the sharpness" is true of *this trained network at inference*
— deleting their MLPs still flattens `d(t)` completely — but false as a claim about training. The sharp
transition is a **relocatable** computation: denied blocks 1–4 it moves to 5–8; denied 1–7 it moves into
8–11; denied 5–11 it moves back into 1–4; denied 0–3 and 9–11 it settles in the middle window 4–8;
denied 0–1 and 7–11 it settles in 2–6; denied block 0 and 6–11 it settles in 1–5; denied all but blocks
5–7 it packs 96% of itself into that three-block window;
denied everything but block 11 it crams into block 11. In
every case the network reaches at least the reference's validation accuracy and still bends the path.
What freezing costs is *how* sharp the transition gets, and that cost is governed by **where** the
surviving trainable blocks sit rather than how many there are. The five runs that leave exactly five
trainable blocks span 0.363 (blocks 1–5), 0.365 (blocks 4–8), 0.365 (blocks 2–6), 0.558–0.590 (beside
the readout) and 0.626–0.629 (at the bottom)
— a wider range than the entire 12→5 block series — and all three mid-stack windows beat the untouched
12-block reference on both framings, one of them meeting the strict plateau rule on 24.7% of pairs
against the
reference's 10.0%. The sharpest statement of the point needs no fitted rule at all: blocks 1–5 alone are
0.118 sharper than blocks 0–7, which contain them, so *removing* trainable blocks can sharpen the
plateau. Shrinking the mid-stack window to **three** blocks, with 74.6% of the parameters
frozen, still ties the 12-block reference (0.446 vs 0.443, p = 0.17) and stays 0.09–0.18 clear of every
five-block window at either end. Three trainable transformer blocks, placed inside the stack, install
the whole phenomenon. What no longer holds is any tidy geometric summary: the interior/end split that
fit the first eight runs was refuted by the ninth, and the coverage description that replaced it was
refuted by the tenth, which excluded the block coverage called essential and came out sharper than
every run that contained it (0.342). The plateau is therefore not tied to particular weights or a particular depth; it is
something this architecture and objective produce wherever there is room for it. But "wherever there is
room" has a floor: with one usable block the shape survives only as a 17%-strength remnant whose
boundary no longer tracks the prediction flip, so some trainable depth is what makes a plateau a
plateau at all.

**How much of this is initialization luck.** Every claim above is a difference between two runs' median
widths, so the number that decides whether any of them means anything is how far a fresh
initialization moves that median on its own. Five conditions — the two ends of the depth comparison
(the narrow run and frozen-early) and all three runs carrying a positional claim (frozen-deep, blocks
6–10 and blocks 0–4) — have now been trained twice under identical data order, schedule and freeze
mask. Figure 25 puts that spread and the section's six load-bearing gaps on one scale.

![two seeds of each twice-trained condition, and the size of each reported gap against the largest seed spread](plots/seed_replication.png)

**Figure 25.** Seed replication, 150 character pairs, interpolation block 0, context `"The house was "`.
**Left:** median transition width `w_10→90` (y, lower = sharper) for the five conditions trained twice
(x); circles are the matched-accuracy checkpoint (validation accuracy 0.550), squares the step-30,000
checkpoint, filled = model seed 1337 and open = seed 2024, the two seeds of a checkpoint joined by a
gray line with their absolute difference printed above. **Right:** the six between-run gaps the
section's conclusions rest on (y, one bar each) measured as the difference in median `w` (x); the
dashed vertical line is the largest seed spread measured (0.040) and the two hatched gray bars are the
gaps that do not exceed it. Where a condition has two seeds the bar shows the *smallest* gap over all
seed pairings, so each claim is credited only with the margin its worst pair of initializations gives.
Retraining moves the median by 0.002–0.040, and four of the six gaps — including every one that
carries a positional conclusion about mid-stack windows — are 2.5–6.5 times that spread. The two that
are not (the 12-versus-8 trainable-block step, 0.033, and the near-readout-versus-bottom ordering,
0.033) are the two the text flags as resting on the paired per-pair tests rather than on their median
gap.

**What the seed work settles and what it leaves.** Both replicates were pre-registered and both
predictions held: the study's sharpest network is not an initialization artefact (0.342 → 0.344, a
0.002 spread), and every one of the four deep-versus-mirror seed pairings keeps five trainable blocks
beside the readout sharper than five at the bottom. The honest qualification is on the *size* of that
last ordering rather than its direction — 0.033 at matched accuracy and 0.010 at the end of training,
against a 0.040 largest spread. The six remaining conditions (frozen-late, the three mid-stack windows
at blocks 4–8, 2–6 and 1–5, the three-block window, and frozen-two) still have one seed each; the
gaps they carry are 0.14–0.26, three to six times the spread, so a second initialization would have to
be far more variable than any measured here to reach them.

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

Because this set uses 127-character natural prefixes rather than one shared context, it is the widest
test that the plateau shape is not an artifact of the short shared prompt; Figure 26 shows every
frozen pair individually.

![exploratory 40-pair raw curves](plots/pair_curves_logits.png)

**Figure 26.** *(Exploratory.)* Raw `d(t)` (y) vs interpolation position `t` (x) in final-logit space,
one panel per frozen pair; panel titles give the pair ID, the two endpoint characters and the width
`w`. Gray dashed = the straight-line reference `d = t`. Most curves hug `d ≈ 0`, cross rapidly near
`t ≈ 0.5`, then hug `d ≈ 1`; two (#10, #19) track the straight line.

Figure 27 shows the same pairs read at successively deeper recording points, which is the layerwise
signature Matthew predicts.

![exploratory layerwise emergence](plots/layerwise_emergence.png)

**Figure 27.** *(Exploratory.)* Layerwise emergence for four fixed pairs (IDs 0–3): `d(t)` (y) vs
interpolation position `t` (x). Thin lines are the recording blocks on the cividis scale (dark = early
block, light = late); the thick black line is the final logits and the gray dashed line the
straight-line reference. Curves start near-straight and sharpen into plateaus by the logits — the
plateau is formed by the downstream stack, not present in the patched activation.

Figure 28 is the converse control: moving the patch later leaves fewer blocks to build the plateau.

![exploratory interpolation-block comparison](plots/interpolation_layer_comparison.png)

**Figure 28.** *(Exploratory.)* Left: median final-logit `d(t)` (y) vs interpolation position `t` (x)
per interpolation block, cividis scale (dark = block 0 → light = block 10) as labelled in the legend;
the block-0 curve is sigmoid and later blocks approach the gray dashed straight line. Right: median
width `w_10→90` (y, inter-quartile-range bars, solid line with circle markers) vs interpolation block
(x); black dotted = strict rule 0.25, gray dashed = straight-line value 0.80.

## Implementation checks (all passed)

- `t=0` / `t=1` patched forwards reproduce the direct unpatched endpoint forwards (max logit error
  < 1e-3); `d(0) < 1e-4`, `d(1) > 1 − 1e-4` for every pair/checkpoint, including all 2,080 all-pairs
  runs (max `d(0)` = 3e-6, min `d(1)` = 0.999998).
- Prefix positions differ only at the final character; all earlier-position activations of A and B
  match at every block (max abs diff < 1e-4; exactly 0.0 in the all-pairs sweep).
- Batched interpolation matches a single-example reference to < 1e-5.
- Synthetic step path detected (w = 0.089); synthetic linear path rejected (w = 0.800).
- Slerp endpoints exact; norms interpolate linearly; documented near-collinear fallback.
- Endpoint-swap symmetry re-verified on 100 all-pairs runs (median and max |Δw| = 0.000).
- Both deliverables are checked to render on GitHub by `experiments/check_render.py` (KaTeX-compiles
  every equation, rejects macros GitHub's math renderer blocks such as `\operatorname`, and confirms
  every figure is an embedded image rather than a bare path).

## Headline

Both character runs reproduce *Deep Networks Always Grok* Figure 9's ordering (fresh char: LC
1940 → 491 @ 15 → 989 @ 36 → 8.1 on a 24-checkpoint grid, robustness 0.53 still rising after clean
accuracy saturates), but the
BPE run — the only one carrying Matthew's exact `big/in`, `big/large` tokens — **FAILs**, so the
**primary Grokking↔plateau relationship remains not testable** (PLAN case 5). The Matthew-faithful
char-token controls (`b↔i`, `b↔l`) do give a checkpoint-aligned answer for the character analogues: the
plateau is **absent at initialization**, sharpens from width 0.80 to 0.33 between steps 56 and 831 —
**inside** the second-descent window and straddling the sustained robustness onset — and is flat to 30k.
That is **temporal association** (PLAN case 1), bounded by one run, six checkpoints, and a second
descent that opens early enough (step 36) to overlap ordinary initial fitting.
Sweeping the comma against all 64 other characters — and repeating that sweep in 8 further held-out
contexts, 576 pairs in all — shows the same shape everywhere (**0/576** curves near the straight line)
while making clear that sharpness is graded: only 11/576 clear the strict ≤ 0.25 bar. The exhaustive
**all-pairs sweep (2,080 pairs)** then says what the plateaus *are*: **every character owns a basin**
(`flat_frac` ≥ 0.86 for all 65), **78%** of the variance in transition width is explained by
per-character terms rather than pair chemistry, **91%** of the model's next-character prediction
changes fall inside the transition window, and the whole structure is **learned** (median width
0.803 at init → 0.355 trained) and **built by blocks 1–4** (0.34 at block 0 vs 0.81 at block 8).
Four interventions then bound the mechanism: no readout bias can move `d(t)` at all; scaling the
block-1–4 MLPs sets the sharpness monotonically (0.80 at gain 0 → 0.31 at gain 1.5) while the same
scaling of blocks 8–11 does nothing; deleting those MLPs one at a time shows the effect is distributed
(41/28/18/11%) and tracks **neither** the next-character decision (which survives) nor endpoint
plausibility; and five retraining runs with a block group **frozen at initialization** all reach or
beat the reference validation accuracy and still bend the paths, relocating the sharpening into
whichever blocks stay trainable (blocks 5–8 at width 0.471, blocks 8–11 at 0.558, blocks 1–4 at 0.626,
and block 11 alone at 0.726). So the plateau is a decision *basin* by description, produced by a
**relocatable** computation that we have not yet characterised, whose sharpness is set by how much
trainable depth is left rather than by any particular weights or site — and which degrades into a
17%-strength remnant once only one usable block remains. Both orderings now carry a seed check on both
sides: with two initializations at each end of the depth step, the three runs with 12 trainable blocks
(0.397–0.443) stay disjoint from the three with 8 (0.476–0.500); and with two initializations of each
five-block end condition, both near-readout runs (0.559, 0.590) stay below both bottom-of-stack runs
(0.624, 0.629). Retraining from a second initialization moves the median width by 0.002–0.040 across
the five twice-trained conditions — smaller than either step, though the bottom-of-stack ordering is
only 0.033 wide at matched accuracy and 0.010 at the end of training, so it rests on the per-pair
tests rather than on the medians alone.
