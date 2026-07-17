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

## Models actually tested

| Model | Tokenizer | Params | Trained to | Role |
|---|---|---|---|---|
| Fresh character GPT | char (vocab 65) | 8.38M | 30,000 steps, val acc 0.554 (peak 0.568) | Figure-9 control + **Matthew char-control assay** |
| Fresh BPE GPT | GPT-2 BPE (vocab 50257) | — | 10,000 steps (killed; overfit) | primary Matthew bridge (`big/in`, `big/large`) — **gate FAILs** |
| Pilot character GPT | char (vocab 65) | 8.38M | 3,500 steps, val acc 0.560 | pilot only |

All are 12-layer/12-head GeLU GPTs (`d_model=240`, context 128). Provenance, seeds, corpus SHA-256 in
`results/train_meta*.json`; confirmed-vs-reconstructed fields in `MODEL_SPEC.md`. The paper's exact
GPT code/checkpoint is **not public** (repo audited 2026-07-15), so these are faithful reconstructions.

![Character-model training curves: cross-entropy loss falls to ~1.49 on validation (left); next-char accuracy rises to 0.56 (right); x = training step.](plots/training_curves.png)

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

![Pilot char (3.5k) Figure-9 curves. Left y-axis: local complexity (sign-crossings summed over 12 GeLU layers) for train (blue), test (orange), random (green), 99% CI bands. Right y-axis: next-token accuracy — black = clean test accuracy, red dashed = ε=0.03 PGD adversarial accuracy. x-axis: training step (log scale, step 0 at 1). LC monotone to the 3,500-step horizon (no second descent); adv accuracy climbs to 0.33; verdict FAIL.](plots/grokking_pilot_char.png)
![Fresh char (30k) Figure-9 gate. Same axes as the pilot panel. LC monotone to 8.1 while adversarial accuracy climbs to 0.53; no second descent → FAIL.](plots/grokking_fresh_char.png)
![Fresh BPE (10k) Figure-9 gate. Same axes. LC monotone to 95 while adversarial accuracy climbs to 0.19; no second descent → FAIL.](plots/grokking_fresh_bpe.png)

**Joint timeline (S7).** On one training-step axis the Grokking side is uniform across runs — LC falls
monotonically and PGD robustness rises, no second descent anywhere. Because the BPE bridge to Matthew's
exact tokens does not reproduce Figure 9 (nor does either character run), the primary Matthew-exact
relationship is not testable: that window never opens.

![Joint checkpoint timeline. Left: test local complexity (y) vs training step (x, log) for pilot char (gray), fresh char (blue), fresh BPE (red); legend gives each run's Figure-9 verdict. Middle: ε=0.03 PGD adversarial accuracy (y) vs step (x, log), same colors; dashed = 0.05 robustness threshold. Right: text summary of the three FAIL verdicts and the bounded relationship verdict. No second LC descent in any run → primary relationship not testable.](plots/joint_timeline.png)

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

![Matthew-faithful char-control d(t), interpolation block 0, final logits, one panel per frozen checkpoint (steps 0→30000); blue = b↔i, orange = b↔l, gray dashed = diagonal d=t. Curves are diagonal at init/step 56 and become sharp plateau–boundary–plateau sigmoids by step 831, stable thereafter.](plots/matthew_char_ctrl_by_checkpoint.png)

**Same timeline, both phenomena overlaid.** Top: Figure-9 grokking metrics (LC train/test/random +
clean/adv accuracy) for the fresh char run. Bottom: the plateau transition width for both controls.
The width drops to its floor by step ~831 while LC is still in its *first* descent and robustness has
not yet risen — no temporal coupling to a (non-existent) second descent.

![Grokking metrics vs plateau width on one timeline (fresh char run). Top: left y = local complexity for LC train (blue) / test (orange) / random (green) with 99% CI; right y = next-token accuracy, black = clean, red dashed = ε=0.03 PGD adv; x = training step (log). Bottom: transition width w_10→90 (y) for b↔i (blue) and b↔l (orange) vs step (log); gray dashed = diagonal 0.8, red dotted = plateau bar 0.25. Width collapses to its floor by step ~831 — during the first LC descent, before robustness rises.](plots/joint_timeline_char_ctrl.png)

**Depth control also holds for the Matthew-faithful assay.** Within the final checkpoint, moving the
interpolation point later (fewer downstream layers) widens the boundary back toward the diagonal — e.g.
`b↔i` at step 30000: width 0.33 (block 0) → 0.72 (block 3) → 0.80 (block 11). So the plateau is built
by the downstream stack, matching Matthew's layerwise prediction. Raw `d(t)` for every (step, pair,
interp-layer, hook) in `results/matthew_char_ctrl_raw.npz`; widths in
`results/matthew_char_ctrl_summary.json`.

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
![Exploratory 40-pair layerwise emergence for four fixed pairs (IDs 0–3): d(t) (y) vs t (x); line color = recording block (dark early → light late); red = final logits; gray dashed = diagonal. Curves start near-diagonal and sharpen into plateaus by the logits.](plots/layerwise_emergence.png)
![Exploratory 40-pair depth comparison. Left: median final-logit d(t) (y) vs t (x) per interpolation block (dark=0 → light=10); block-0 sigmoid, later blocks approach the gray dashed diagonal. Right: median width w_10→90 (y, IQR bars) vs interpolation block (x); red dashed = plateau bar 0.25, gray dashed = diagonal 0.8.](plots/interpolation_layer_comparison.png)

## Implementation checks (all passed)

- `t=0` / `t=1` patched forwards reproduce the direct unpatched endpoint forwards (max logit error
  < 1e-3); `d(0) < 1e-4`, `d(1) > 1 − 1e-4` for every pair/checkpoint.
- Prefix positions differ only at the final character; all earlier-position activations of A and B
  match at every block (max abs diff < 1e-4; `prefix_err` logged per checkpoint).
- Batched interpolation matches a single-example reference to < 1e-5.
- Synthetic step path detected (w = 0.089); synthetic linear path rejected (w = 0.800).
- Slerp endpoints exact; norms interpolate linearly; documented near-collinear fallback.

## Headline

No model we trained reproduces *Deep Networks Always Grok* Figure 9 (all three FAIL the mandatory
second-descent gate; the BPE bridge to Matthew's exact tokens fails too), so the **primary
Grokking↔plateau relationship is not testable** (PLAN case 5). But the Matthew-faithful char-token
controls (`b↔i`, `b↔l`) let us time the plateau: it is **absent at initialization**, **emerges during
the first LC descent** (width 0.80 → 0.33 by step ~831), and is **fully formed before** adversarial
robustness saturates — so, in this non-grokking model, the plateau is an early property of the trained
downstream stack with **no visible temporal coupling** to the grokking signature (which never occurs).
